import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.domain.entities import GroundedAnswer, DocumentChunk
from src.domain.enums import RouteChoice
from src.domain.interfaces import (
    IRouterService,
    IGraphRepository,
    IVectorRepository,
    IGeneratorService,
    IEntityResolverService
)

logger = logging.getLogger(__name__)

class QueryPipelineUseCase:
    """Orchestrates query execution using parameterized Cypher templates for secure, deterministic graph traversal."""

    def __init__(
        self,
        router: IRouterService,
        graph_repo: IGraphRepository,
        vector_repo: IVectorRepository,
        generator: IGeneratorService,
        resolver: IEntityResolverService
    ):
        self.router = router
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.generator = generator
        self.resolver = resolver
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def execute(self, query: str) -> GroundedAnswer:
        # 1. Classify query intent & extract target entities via cheap Router
        decision = self.router.route_query(query)
        route_choice = decision.route

        logger.info(f"Query Pipeline: route={route_choice}, confidence={decision.confidence}, target_entities={decision.target_entities}")

        graph_paths: List[Dict[str, Any]] = []
        vector_passages: List[DocumentChunk] = []

        # 2. Secure Parameterized Graph Path Execution
        if route_choice in (RouteChoice.GRAPH, RouteChoice.HYBRID):
            target_entities = decision.target_entities

            # Case A: Two entities extracted -> Parameterized shared_dependencies query
            if len(target_entities) >= 2:
                canon_a = self.resolver.resolve(target_entities[0])
                canon_b = self.resolver.resolve(target_entities[1])
                logger.info(f"Executing Cypher Template 'shared_dependencies' for ({canon_a}, {canon_b})")
                paths = self.graph_repo.execute_cypher_template(
                    template_name="shared_dependencies",
                    params={"entity_a": canon_a, "entity_b": canon_b, "limit": 15}
                )
                graph_paths.extend(paths)

            # Case B: Single/Multiple individual entities -> Parameterized two_hop_neighborhood query
            if not graph_paths and target_entities:
                for entity_name in target_entities:
                    canon_id = self.resolver.resolve(entity_name)
                    logger.info(f"Executing Cypher Template 'two_hop_neighborhood' for node_id='{canon_id}'")
                    paths = self.graph_repo.execute_cypher_template(
                        template_name="two_hop_neighborhood",
                        params={"entity_id": canon_id, "limit": 15}
                    )
                    graph_paths.extend(paths)

        # 3. Vector Path Execution (for VECTOR, HYBRID, or fallback)
        if route_choice in (RouteChoice.VECTOR, RouteChoice.HYBRID) or not graph_paths:
            query_embedding = self.encoder.encode(query).tolist()
            vector_passages = self.vector_repo.similarity_search(query_embedding, top_k=5)

            # Cross-Retrieval: Retrieve graph neighborhood around retrieved vector passage chunks
            if route_choice == RouteChoice.HYBRID and vector_passages:
                retrieved_chunk_ids = [v.chunk_id for v in vector_passages]
                extra_graph_paths = self.graph_repo.get_neighborhood_by_chunk_ids(retrieved_chunk_ids)
                graph_paths.extend(extra_graph_paths)

        # 4. Generate grounded answer with citation verification
        return self.generator.generate_grounded_answer(
            query=query,
            graph_paths=graph_paths,
            vector_passages=vector_passages,
            route_choice=route_choice
        )

import time
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
from src.infrastructure.routing.routing_logger import RoutingLogger

logger = logging.getLogger(__name__)

class QueryPipelineUseCase:
    """Orchestrates query execution using parameterized Cypher templates and records persistent routing decision logs for Phase 5 benchmarking."""

    def __init__(
        self,
        router: IRouterService,
        graph_repo: IGraphRepository,
        vector_repo: IVectorRepository,
        generator: IGeneratorService,
        resolver: IEntityResolverService,
        routing_logger: Optional[RoutingLogger] = None
    ):
        self.router = router
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.generator = generator
        self.resolver = resolver
        self.routing_logger = routing_logger or RoutingLogger()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def execute(self, query: str) -> GroundedAnswer:
        start_time = time.time()

        # 1. Classify query intent & extract target entities via cheap Router
        decision = self.router.route_query(query)
        route_choice = decision.route

        logger.info(f"Query Pipeline: route={route_choice}, confidence={decision.confidence}, target_entities={decision.target_entities}")

        graph_paths: List[Dict[str, Any]] = []
        vector_passages: List[DocumentChunk] = []

        # 2. Secure Parameterized Graph Path Execution
        if route_choice in (RouteChoice.GRAPH, RouteChoice.HYBRID):
            target_entities = decision.target_entities

            if len(target_entities) >= 2:
                canon_a = self.resolver.resolve(target_entities[0])
                canon_b = self.resolver.resolve(target_entities[1])
                logger.info(f"Executing Cypher Template 'shared_dependencies' for ({canon_a}, {canon_b})")
                paths = self.graph_repo.execute_cypher_template(
                    template_name="shared_dependencies",
                    params={"entity_a": canon_a, "entity_b": canon_b, "limit": 15}
                )
                graph_paths.extend(paths)

            if not graph_paths and target_entities:
                for entity_name in target_entities:
                    canon_id = self.resolver.resolve(entity_name)
                    logger.info(f"Executing Cypher Template 'two_hop_neighborhood' for node_id='{canon_id}'")
                    paths = self.graph_repo.execute_cypher_template(
                        template_name="two_hop_neighborhood",
                        params={"entity_id": canon_id, "limit": 15}
                    )
                    graph_paths.extend(paths)

        # 3. Vector Path & Shared Key Bridge Execution
        if graph_paths:
            # Cross from graph path chunk_ids back to original text passages via shared chunk_ids
            graph_chunk_ids = []
            for p in graph_paths:
                for k in ("chunks", "chunks1", "chunks2", "chunks_a", "chunks_b"):
                    if k in p and isinstance(p[k], list):
                        graph_chunk_ids.extend(p[k])
            if graph_chunk_ids:
                cross_passages = self.vector_repo.get_chunks_by_ids(list(set(graph_chunk_ids)))
                vector_passages.extend(cross_passages)

        if route_choice in (RouteChoice.VECTOR, RouteChoice.HYBRID) or not graph_paths:
            query_embedding = self.encoder.encode(query).tolist()
            sim_passages = self.vector_repo.similarity_search(query_embedding, top_k=5)
            vector_passages.extend(sim_passages)

            if route_choice == RouteChoice.HYBRID and sim_passages:
                retrieved_chunk_ids = [v.chunk_id for v in sim_passages]
                extra_graph_paths = self.graph_repo.get_neighborhood_by_chunk_ids(retrieved_chunk_ids)
                graph_paths.extend(extra_graph_paths)

        # 4. Generate grounded answer with citation verification
        grounded_answer = self.generator.generate_grounded_answer(
            query=query,
            graph_paths=graph_paths,
            vector_passages=vector_passages,
            route_choice=route_choice
        )

        latency_ms = (time.time() - start_time) * 1000

        # 5. Persistently log routing decision and execution outcome for Phase 5
        self.routing_logger.log_decision(
            question=query,
            route=route_choice.value if hasattr(route_choice, 'value') else str(route_choice),
            confidence=decision.confidence,
            is_fallback=decision.is_fallback,
            target_entities=decision.target_entities,
            reasoning=decision.reasoning,
            graph_paths_count=len(graph_paths),
            vector_passages_count=len(vector_passages),
            retrieved_chunk_ids=grounded_answer.retrieved_chunk_ids,
            citations=grounded_answer.citations,
            latency_ms=latency_ms
        )

        return grounded_answer

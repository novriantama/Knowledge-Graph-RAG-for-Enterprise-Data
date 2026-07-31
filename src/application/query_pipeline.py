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

class QueryPipelineUseCase:
    """Orchestrates query execution across dynamic router, graph traversal, vector search, and grounded generator."""

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
        # 1. Classify query intent via router
        decision = self.router.route_query(query)
        route_choice = decision.route

        graph_paths: List[Dict[str, Any]] = []
        vector_passages: List[DocumentChunk] = []

        # 2. Execute Graph path if GRAPH or HYBRID route chosen
        if route_choice in (RouteChoice.GRAPH, RouteChoice.HYBRID):
            for entity_name in decision.target_entities:
                canonical_id = self.resolver.resolve(entity_name)
                paths = self.graph_repo.execute_cypher_template(
                    template_name="two_hop_neighborhood",
                    params={"entity_id": canonical_id, "limit": 10}
                )
                graph_paths.extend(paths)

        # 3. Execute Vector path if VECTOR or HYBRID route chosen
        if route_choice in (RouteChoice.VECTOR, RouteChoice.HYBRID) or not graph_paths:
            query_embedding = self.encoder.encode(query).tolist()
            vector_passages = self.vector_repo.similarity_search(query_embedding, top_k=5)

        # 4. Generate grounded answer with citation verification
        return self.generator.generate_grounded_answer(
            query=query,
            graph_paths=graph_paths,
            vector_passages=vector_passages,
            route_choice=route_choice
        )

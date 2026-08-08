import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
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

# Global Singleton Encoder for fast sub-millisecond reuse
_SHARED_ENCODER: Optional[SentenceTransformer] = None

def get_shared_encoder() -> SentenceTransformer:
    global _SHARED_ENCODER
    if _SHARED_ENCODER is None:
        logger.info("Initializing Shared SentenceTransformer ('all-MiniLM-L6-v2')...")
        _SHARED_ENCODER = SentenceTransformer('all-MiniLM-L6-v2')
    return _SHARED_ENCODER

@lru_cache(maxsize=512)
def cached_encode_query(query: str) -> Tuple[float, ...]:
    encoder = get_shared_encoder()
    emb = encoder.encode(query, show_progress_bar=False).tolist()
    return tuple(emb)

class QueryPipelineUseCase:
    """High-speed query orchestrator using parallel retrieval, LRU caching, and Shared Key Cross-Retrieval."""

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
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._answer_cache: Dict[str, GroundedAnswer] = {}

    def execute(self, query: str) -> GroundedAnswer:
        clean_query = query.strip()
        
        # 0. Fast LRU Cache Check (< 2ms response)
        if clean_query in self._answer_cache:
            logger.info(f"Query Pipeline: Fast Cache Hit for '{clean_query}'")
            return self._answer_cache[clean_query]

        start_time = time.time()

        # 1. Classify query intent via fast Router
        decision = self.router.route_query(clean_query)
        route_choice = decision.route

        logger.info(f"Query Pipeline: route={route_choice}, confidence={decision.confidence}, target_entities={decision.target_entities}")

        graph_paths: List[Dict[str, Any]] = []
        vector_passages: List[DocumentChunk] = []

        # Function helpers for parallel execution
        def _fetch_graph_paths() -> List[Dict[str, Any]]:
            paths: List[Dict[str, Any]] = []
            target_entities = decision.target_entities

            if len(target_entities) >= 2:
                canon_a = self.resolver.resolve(target_entities[0])
                canon_b = self.resolver.resolve(target_entities[1])
                res = self.graph_repo.execute_cypher_template(
                    template_name="shared_dependencies",
                    params={"entity_a": canon_a, "entity_b": canon_b, "limit": 8}
                )
                paths.extend(res)

            if not paths and target_entities:
                for entity_name in target_entities[:2]:
                    canon_id = self.resolver.resolve(entity_name)
                    res = self.graph_repo.execute_cypher_template(
                        template_name="two_hop_neighborhood",
                        params={"entity_id": canon_id, "limit": 8}
                    )
                    paths.extend(res)
            return paths

        def _fetch_vector_passages() -> List[DocumentChunk]:
            query_emb = list(cached_encode_query(clean_query))
            return self.vector_repo.similarity_search(query_emb, top_k=3)

        # 2. Parallel Dual Retrieval
        if route_choice == RouteChoice.HYBRID:
            future_graph = self._executor.submit(_fetch_graph_paths)
            future_vector = self._executor.submit(_fetch_vector_passages)
            graph_paths = future_graph.result()
            vector_passages = future_vector.result()
        elif route_choice == RouteChoice.GRAPH:
            graph_paths = _fetch_graph_paths()
        else: # VECTOR
            vector_passages = _fetch_vector_passages()

        # Fallback to vector search if graph retrieval returned no paths
        if route_choice == RouteChoice.GRAPH and not graph_paths:
            vector_passages = _fetch_vector_passages()

        # 3. Shared Key Bridge Cross Retrieval
        if graph_paths:
            graph_chunk_ids = []
            for p in graph_paths:
                for k in ("chunks", "chunks1", "chunks2", "chunks_a", "chunks_b"):
                    if k in p and isinstance(p[k], list):
                        graph_chunk_ids.extend(p[k])
            if graph_chunk_ids:
                cross_passages = self.vector_repo.get_chunks_by_ids(list(set(graph_chunk_ids)))
                vector_passages.extend(cross_passages)

        # 4. Generate grounded answer
        grounded_answer = self.generator.generate_grounded_answer(
            query=clean_query,
            graph_paths=graph_paths,
            vector_passages=vector_passages,
            route_choice=route_choice
        )

        latency_ms = (time.time() - start_time) * 1000

        # Store in LRU cache (max 100 items)
        if len(self._answer_cache) > 100:
            self._answer_cache.pop(next(iter(self._answer_cache)))
        self._answer_cache[clean_query] = grounded_answer

        # 5. Persistently log routing decision
        self.routing_logger.log_decision(
            question=clean_query,
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

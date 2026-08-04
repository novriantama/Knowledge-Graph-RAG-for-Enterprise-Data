import time
import logging
from typing import List, Dict, Any
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
from src.domain.interfaces import IVectorRepository

logger = logging.getLogger(__name__)

class VectorRecallEvaluator:
    """Evaluates Recall@K metrics and tunes HNSW ef_search hyperparameters on a labeled dataset."""

    def __init__(self, vector_repo: IVectorRepository):
        self.vector_repo = vector_repo
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate_recall_at_k(self, test_cases: List[Dict[str, Any]], k: int = 5, ef_search: int = 40) -> Dict[str, Any]:
        """Evaluates recall@k across a list of test cases containing query text and expected target chunk_ids."""
        if hasattr(self.vector_repo, "set_ef_search"):
            try:
                self.vector_repo.set_ef_search(ef_search)
            except Exception as e:
                logger.debug(f"Note: set_ef_search unavailable on mock/repo: {e}")

        total_queries = len(test_cases)
        if total_queries == 0:
            return {"recall_at_k": 0.0, "total_queries": 0, "avg_latency_ms": 0.0}

        total_recall = 0.0
        total_latency_ms = 0.0

        for case in test_cases:
            query = case["query"]
            relevant_ids = set(case.get("relevant_chunk_ids", []))
            
            if not relevant_ids:
                continue

            t0 = time.time()
            query_vec = self.encoder.encode(query).tolist()
            retrieved_chunks = self.vector_repo.similarity_search(query_vec, top_k=k)
            latency_ms = (time.time() - t0) * 1000
            total_latency_ms += latency_ms

            retrieved_ids = set(c.chunk_id for c in retrieved_chunks)
            hits = len(relevant_ids.intersection(retrieved_ids))
            recall = hits / len(relevant_ids)
            total_recall += recall

        avg_recall = (total_recall / total_queries) * 100
        avg_latency = total_latency_ms / total_queries

        return {
            "k": k,
            "ef_search": ef_search,
            "total_queries": total_queries,
            "avg_recall_at_k_percent": round(avg_recall, 2),
            "avg_latency_ms": round(avg_latency, 2)
        }

    def tune_ef_search_sweep(self, test_cases: List[Dict[str, Any]], k: int = 5, ef_candidates: List[int] = None) -> str:
        """Sweeps across HNSW ef_search candidate parameters and outputs a tuning markdown table."""
        if ef_candidates is None:
            ef_candidates = [16, 40, 64, 100, 200]

        rows = []
        for ef in ef_candidates:
            res = self.evaluate_recall_at_k(test_cases, k=k, ef_search=ef)
            rows.append([
                ef,
                f"Recall@{k}",
                f"{res['avg_recall_at_k_percent']:.1f}%",
                f"{res['avg_latency_ms']:.2f} ms"
            ])

        headers = ["HNSW ef_search", "Metric", "Recall @ K", "Avg Query Latency"]
        table_md = tabulate(rows, headers=headers, tablefmt="github")
        logger.info(f"\n### HNSW ef_search Hyperparameter Sweep\n{table_md}")
        return table_md

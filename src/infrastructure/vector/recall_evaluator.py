import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.domain.interfaces import IVectorRepository

logger = logging.getLogger(__name__)

class VectorRecallEvaluator:
    """Evaluates Recall@K and Precision@K metrics for pgvector retrieval on a labeled dataset."""

    def __init__(self, vector_repo: IVectorRepository):
        self.vector_repo = vector_repo
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate_recall_at_k(self, test_cases: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
        """Evaluates recall@k across a list of test cases containing query text and expected target chunk_ids.
        
        Test case structure:
        {
            "query": "What framework is used by User Auth?",
            "relevant_chunk_ids": ["01_core_architecture_overview_chunk_0", "02_service_dependencies_chunk_1"]
        }
        """
        total_queries = len(test_cases)
        if total_queries == 0:
            return {"recall_at_k": 0.0, "total_queries": 0}

        total_recall = 0.0

        for case in test_cases:
            query = case["query"]
            relevant_ids = set(case.get("relevant_chunk_ids", []))
            
            if not relevant_ids:
                continue

            query_vec = self.encoder.encode(query).tolist()
            retrieved_chunks = self.vector_repo.similarity_search(query_vec, top_k=k)
            retrieved_ids = set(c.chunk_id for c in retrieved_chunks)

            hits = len(relevant_ids.intersection(retrieved_ids))
            recall = hits / len(relevant_ids)
            total_recall += recall

        avg_recall = (total_recall / total_queries) * 100

        report = {
            "k": k,
            "total_queries": total_queries,
            "avg_recall_at_k_percent": round(avg_recall, 2)
        }
        
        logger.info(f"Vector Index Evaluation: Recall@{k} = {avg_recall:.2f}% across {total_queries} queries.")
        return report

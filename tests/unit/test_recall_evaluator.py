import unittest
from unittest.mock import MagicMock
from src.domain.entities import DocumentChunk
from src.infrastructure.vector.recall_evaluator import VectorRecallEvaluator

class TestVectorRecallEvaluator(unittest.TestCase):
    def setUp(self):
        self.mock_repo = MagicMock()
        self.evaluator = VectorRecallEvaluator(self.mock_repo)

    def test_recall_at_k_calculation(self):
        # Mock vector search results
        mock_chunks = [
            DocumentChunk(
                chunk_id="doc1_chunk_0",
                document_id="doc1",
                content="User Auth Service uses FastAPI.",
                entity_ids=["User Auth Service", "FastAPI"]
            )
        ]
        self.mock_repo.similarity_search.return_value = mock_chunks

        test_cases = [
            {
                "query": "What framework is used by User Auth?",
                "relevant_chunk_ids": ["doc1_chunk_0"]
            }
        ]

        res = self.evaluator.evaluate_recall_at_k(test_cases, k=5, ef_search=40)
        self.assertEqual(res["k"], 5)
        self.assertEqual(res["avg_recall_at_k_percent"], 100.0)

    def test_ef_search_sweep_markdown_output(self):
        self.mock_repo.similarity_search.return_value = []
        test_cases = [
            {
                "query": "Sample test query",
                "relevant_chunk_ids": ["doc1_chunk_1"]
            }
        ]

        table_md = self.evaluator.tune_ef_search_sweep(test_cases, k=5, ef_candidates=[16, 40])
        self.assertIn("HNSW ef_search", table_md)
        self.assertIn("16", table_md)
        self.assertIn("40", table_md)

if __name__ == "__main__":
    unittest.main()

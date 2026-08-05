import os
import tempfile
import unittest
from src.infrastructure.routing.routing_logger import RoutingLogger

class TestRoutingLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = os.path.join(self.temp_dir.name, "test_decisions.jsonl")
        self.logger = RoutingLogger(log_file_path=self.log_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_log_and_read_decisions(self):
        entry = self.logger.log_decision(
            question="Which packages depend on Supplier-X?",
            route="GRAPH",
            confidence=0.95,
            is_fallback=False,
            target_entities=["Supplier-X"],
            reasoning="Multi-hop dependency chain query",
            graph_paths_count=3,
            vector_passages_count=0,
            retrieved_chunk_ids=["chunk_101", "chunk_102"],
            citations=["chunk_101"],
            latency_ms=125.5
        )

        self.assertEqual(entry["question"], "Which packages depend on Supplier-X?")
        self.assertEqual(entry["route"], "GRAPH")
        self.assertEqual(entry["latency_ms"], 125.5)

        logged_records = self.logger.load_logged_decisions()
        self.assertEqual(len(logged_records), 1)
        self.assertEqual(logged_records[0]["question"], "Which packages depend on Supplier-X?")
        self.assertEqual(logged_records[0]["route"], "GRAPH")
        self.assertEqual(logged_records[0]["citations"], ["chunk_101"])

if __name__ == "__main__":
    unittest.main()

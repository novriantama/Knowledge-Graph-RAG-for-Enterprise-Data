import unittest
from unittest.mock import MagicMock
from src.domain.entities import GroundedAnswer
from src.domain.enums import RouteChoice
from src.application.benchmark_runner import BenchmarkRunnerUseCase

class TestBenchmarkRunner(unittest.TestCase):
    def setUp(self):
        self.mock_kg_pipeline = MagicMock()
        self.mock_vector_repo = MagicMock()
        self.mock_generator = MagicMock()

        self.runner = BenchmarkRunnerUseCase(
            kg_pipeline=self.mock_kg_pipeline,
            vector_repo=self.mock_vector_repo,
            generator=self.mock_generator
        )

    def test_benchmark_execution_and_summary_table(self):
        # Mock Vector RAG answer (single-fact accurate, multi-hop inaccurate)
        self.mock_generator.generate_grounded_answer.side_effect = [
            GroundedAnswer(
                question="What framework is used by User Auth?",
                answer="User Auth uses FastAPI.",
                citations=["doc1_chunk_0"],
                route_used=RouteChoice.VECTOR
            ),
            GroundedAnswer(
                question="Which open-source maintainers affect EU CRA?",
                answer="Unknown.",
                citations=[],
                route_used=RouteChoice.VECTOR
            )
        ]

        # Mock Hybrid KG-RAG answer (both accurate)
        self.mock_kg_pipeline.execute.side_effect = [
            GroundedAnswer(
                question="What framework is used by User Auth?",
                answer="User Auth uses FastAPI.",
                citations=["doc1_chunk_0"],
                route_used=RouteChoice.VECTOR
            ),
            GroundedAnswer(
                question="Which open-source maintainers affect EU CRA?",
                answer="Supplier-X and AnyIO impact EU CRA compliance for Acme EU GmbH.",
                citations=["doc4_chunk_1"],
                route_used=RouteChoice.GRAPH
            )
        ]

        # Single-hop test case & Three-hop test case
        sample_results = [
            {
                "id": "q1",
                "category": "single_hop",
                "hop_count": 1,
                "vec_correct": True,
                "vec_latency": 0.85,
                "kg_correct": True,
                "kg_latency": 0.92
            },
            {
                "id": "q2",
                "category": "three_hop",
                "hop_count": 3,
                "vec_correct": False, # Plain vector fails on 3-hop
                "vec_latency": 0.82,
                "kg_correct": True,   # KG-RAG succeeds on 3-hop
                "kg_latency": 1.45
            }
        ]

        table_md, delta_report = self.runner._build_summary_table(sample_results)

        self.assertIn("1-Hop (Single Fact)", table_md)
        self.assertIn("3-Hop (Transitive Chain)", table_md)
        self.assertEqual(delta_report["single_hop"]["delta_acc_percent"], 0.0)  # Parity on 1-hop
        self.assertEqual(delta_report["three_hop"]["delta_acc_percent"], 100.0) # Widening gap on 3-hop!

if __name__ == "__main__":
    unittest.main()

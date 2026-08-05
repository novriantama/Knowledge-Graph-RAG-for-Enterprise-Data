import unittest
from unittest.mock import MagicMock
from src.domain.enums import RouteChoice
from src.domain.entities import RouterDecision, GroundedAnswer, DocumentChunk
from src.application.query_pipeline import QueryPipelineUseCase

class TestSecureParameterizedCypherExecution(unittest.TestCase):

    def setUp(self):
        self.mock_router = MagicMock()
        self.mock_graph_repo = MagicMock()
        self.mock_vector_repo = MagicMock()
        self.mock_generator = MagicMock()
        self.mock_resolver = MagicMock()

        self.pipeline = QueryPipelineUseCase(
            router=self.mock_router,
            graph_repo=self.mock_graph_repo,
            vector_repo=self.mock_vector_repo,
            generator=self.mock_generator,
            resolver=self.mock_resolver
        )

    def test_parameterized_shared_dependencies_execution(self):
        # 1. Mock Router decision for 2 entities
        self.mock_router.route_query.return_value = RouterDecision(
            route=RouteChoice.GRAPH,
            confidence=0.95,
            reasoning="Cross entity dependency query",
            target_entities=["User Auth Service", "Supplier-X"]
        )

        # 2. Mock entity resolution to canonical node IDs
        self.mock_resolver.resolve.side_effect = lambda name: f"CANONICAL_{name.upper()}"
        self.mock_graph_repo.execute_cypher_template.return_value = [
            {"source": "CANONICAL_USER AUTH SERVICE", "rel_a": "USES_TECH", "shared_entity": "FastAPI"}
        ]
        self.mock_generator.generate_grounded_answer.return_value = GroundedAnswer(
            question="Which packages depend on Supplier-X?",
            answer="FastAPI depends on Supplier-X.",
            citations=[],
            route_used=RouteChoice.GRAPH
        )

        answer = self.pipeline.execute("Which packages used by User Auth Service depend on Supplier-X?")

        # Assert entity resolution called for target entities
        self.mock_resolver.resolve.assert_any_call("User Auth Service")
        self.mock_resolver.resolve.assert_any_call("Supplier-X")

        # Assert parameterized Cypher template called safely (zero raw Cypher from model)
        self.mock_graph_repo.execute_cypher_template.assert_called_once_with(
            template_name="shared_dependencies",
            params={"entity_a": "CANONICAL_USER AUTH SERVICE", "entity_b": "CANONICAL_SUPPLIER-X", "limit": 15}
        )

if __name__ == "__main__":
    unittest.main()

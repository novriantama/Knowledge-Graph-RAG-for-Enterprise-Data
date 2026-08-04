import unittest
from unittest.mock import MagicMock
from src.domain.enums import RouteChoice
from src.domain.entities import RouterDecision
from src.infrastructure.routing.claude_router import ClaudeRouter

class TestClaudeRouterFallback(unittest.TestCase):
    def setUp(self):
        self.router = ClaudeRouter(confidence_threshold=0.70)

    def test_low_confidence_triggers_hybrid_fallback(self):
        # Mock low confidence router decision
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(input={
                "route": "VECTOR",
                "confidence": 0.45, # Below 0.70 threshold
                "reasoning": "Uncertain about query intent",
                "target_entities": ["Acme"]
            })
        ]
        mock_client.messages.create.return_value = mock_response
        self.router.client = mock_client

        decision = self.router.route_query("Ambiguous enterprise question")
        
        self.assertEqual(decision.route, RouteChoice.HYBRID)
        self.assertTrue(decision.is_fallback)
        self.assertEqual(decision.confidence, 0.45)

    def test_high_confidence_retains_original_route(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(input={
                "route": "GRAPH",
                "confidence": 0.95,
                "reasoning": "Multi-hop dependency query",
                "target_entities": ["User Auth Service", "Supplier-X"]
            })
        ]
        mock_client.messages.create.return_value = mock_response
        self.router.client = mock_client

        decision = self.router.route_query("Which packages used by User Auth depend on Supplier-X?")
        
        self.assertEqual(decision.route, RouteChoice.GRAPH)
        self.assertFalse(decision.is_fallback)
        self.assertEqual(decision.target_entities, ["User Auth Service", "Supplier-X"])

if __name__ == "__main__":
    unittest.main()

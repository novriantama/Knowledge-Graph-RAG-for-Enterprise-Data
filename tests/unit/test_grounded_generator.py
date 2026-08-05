import unittest
from unittest.mock import MagicMock
from src.domain.enums import RouteChoice
from src.domain.entities import DocumentChunk
from src.infrastructure.generation.claude_generator import ClaudeGenerator

class TestGroundedAnswerGeneratorCitationValidation(unittest.TestCase):
    def setUp(self):
        self.generator = ClaudeGenerator()

    def test_rejection_and_regeneration_on_invalid_citation(self):
        mock_client = MagicMock()

        # Attempt 1: Returns invalid citation "fake_chunk_99"
        attempt1_response = MagicMock()
        attempt1_response.content = [
            MagicMock(input={
                "answer": "Acme EU GmbH operates in Frankfurt.",
                "citations": ["doc3_chunk_0", "fake_chunk_99"] # "fake_chunk_99" is invalid
            })
        ]

        # Attempt 2 (Regeneration): Returns clean, valid citation "doc3_chunk_0"
        attempt2_response = MagicMock()
        attempt2_response.content = [
            MagicMock(input={
                "answer": "Acme EU GmbH operates in Frankfurt.",
                "citations": ["doc3_chunk_0"]
            })
        ]

        mock_client.messages.create.side_effect = [attempt1_response, attempt2_response]
        self.generator.client = mock_client

        vector_passages = [
            DocumentChunk(
                chunk_id="doc3_chunk_0",
                document_id="doc3",
                content="Acme EU GmbH operates in Frankfurt."
            )
        ]

        result = self.generator.generate_grounded_answer(
            query="Where does Acme EU GmbH operate?",
            graph_paths=[],
            vector_passages=vector_passages,
            route_choice=RouteChoice.VECTOR
        )

        # Assert two calls made (Attempt 1 rejected, Attempt 2 regenerated)
        self.assertEqual(mock_client.messages.create.call_count, 2)
        self.assertEqual(result.citations, ["doc3_chunk_0"])

if __name__ == "__main__":
    unittest.main()

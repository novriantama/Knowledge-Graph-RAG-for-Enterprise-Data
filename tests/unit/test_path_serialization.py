import unittest
from src.domain.entities import DocumentChunk
from src.infrastructure.generation.claude_generator import ClaudeGenerator

class TestPathSerializationAndDeduplication(unittest.TestCase):
    def setUp(self):
        self.generator = ClaudeGenerator()

    def test_two_hop_path_serialization(self):
        raw_paths = [
            {
                "source": "Service-102",
                "rel1": "USES_TECH",
                "intermediate": "FastAPI",
                "rel2": "DEPENDS_ON",
                "target": "Starlette",
                "chunks1": ["doc2_chunk_0"],
                "chunks2": ["doc2_chunk_1"]
            }
        ]

        serialized, chunk_ids = self.generator.serialize_graph_paths(raw_paths)
        self.assertIn("Service-102 uses technology FastAPI", serialized)
        self.assertIn("which depends on Starlette", serialized)
        self.assertIn("[Source Chunks: doc2_chunk_0, doc2_chunk_1]", serialized)
        self.assertEqual(chunk_ids, {"doc2_chunk_0", "doc2_chunk_1"})

    def test_shared_entity_serialization(self):
        raw_paths = [
            {
                "entity_a": "Service-102",
                "rel_a": "USES_TECH",
                "shared_entity": "Supplier-X",
                "rel_b": "MAINTAINED_BY",
                "entity_b": "Service-104",
                "chunks_a": ["doc2_chunk_0"],
                "chunks_b": ["doc2_chunk_1"]
            }
        ]

        serialized, chunk_ids = self.generator.serialize_graph_paths(raw_paths)
        self.assertIn("Service-102 uses technology Supplier-X", serialized)
        self.assertIn("and Service-104 is maintained by Supplier-X", serialized)

    def test_deduplicate_and_assemble_context(self):
        raw_paths = [
            {
                "source": "Service-102",
                "relation": "USES_TECH",
                "target": "FastAPI",
                "chunks": ["doc2_chunk_0"]
            },
            {
                "source": "Service-102",
                "relation": "USES_TECH",
                "target": "FastAPI",
                "chunks": ["doc2_chunk_0"]
            }
        ]

        vector_passages = [
            DocumentChunk(
                chunk_id="doc2_chunk_0",
                document_id="doc2",
                section_path="1. Tech Stack",
                content="User Auth Service uses FastAPI."
            ),
            DocumentChunk(
                chunk_id="doc2_chunk_0", # Duplicate passage chunk
                document_id="doc2",
                section_path="1. Tech Stack",
                content="User Auth Service uses FastAPI."
            )
        ]

        context, valid_ids = self.generator.deduplicate_and_assemble_context(raw_paths, vector_passages)

        self.assertIn("=== GRAPH DERIVED FACTS ===", context)
        self.assertIn("=== VECTOR TEXT PASSAGES ===", context)

        self.assertEqual(context.count("--- Chunk ID: doc2_chunk_0 Section: 1. Tech Stack ---"), 1)
        self.assertEqual(valid_ids, {"doc2_chunk_0"})

if __name__ == "__main__":
    unittest.main()

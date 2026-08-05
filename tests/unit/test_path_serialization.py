import unittest
from src.infrastructure.generation.claude_generator import ClaudeGenerator

class TestPathSerialization(unittest.TestCase):
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

        serialized = self.generator.serialize_graph_paths(raw_paths)
        self.assertIn("Service-102 uses technology FastAPI", serialized)
        self.assertIn("which depends on Starlette", serialized)
        self.assertIn("[Source Chunks: doc2_chunk_0, doc2_chunk_1]", serialized)

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

        serialized = self.generator.serialize_graph_paths(raw_paths)
        self.assertIn("Service-102 uses technology Supplier-X", serialized)
        self.assertIn("and Service-104 is maintained by Supplier-X", serialized)

if __name__ == "__main__":
    unittest.main()

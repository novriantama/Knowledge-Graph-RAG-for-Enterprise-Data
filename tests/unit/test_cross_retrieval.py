import unittest
from src.domain.entities import DocumentChunk

class TestCrossRetrievalContract(unittest.TestCase):

    def test_document_chunk_shared_key_structure(self):
        chunk = DocumentChunk(
            chunk_id="01_core_arch_chunk_0",
            document_id="01_core_arch",
            section_path="1. Overview",
            created_at="2026-08-04T12:00:00Z",
            content="Acme Corp operates Service-101 API Gateway.",
            entity_ids=["Acme Corp", "Service-101"]
        )

        self.assertEqual(chunk.chunk_id, "01_core_arch_chunk_0")
        self.assertEqual(chunk.entity_ids, ["Acme Corp", "Service-101"])
        self.assertEqual(chunk.section_path, "1. Overview")

if __name__ == "__main__":
    unittest.main()

import unittest
from pydantic import ValidationError
from src.domain.entities import ChunkExtractionResult, ExtractedEntity, ExtractedRelationship
from src.domain.enums import EntityType, RelationType

class TestExtractionSchemaValidation(unittest.TestCase):

    def test_valid_schema_instantiation(self):
        entity = ExtractedEntity(
            canonical_name="Acme Corp",
            entity_type=EntityType.COMPANY,
            aliases=["Acme", "Acme Corporation"]
        )
        relationship = ExtractedRelationship(
            source_entity="Service-101",
            target_entity="FastAPI",
            relation_type=RelationType.USES_TECH,
            source_chunk_id="doc1_chunk_0",
            confidence=0.95
        )
        result = ChunkExtractionResult(
            chunk_id="doc1_chunk_0",
            entities=[entity],
            relationships=[relationship]
        )

        self.assertEqual(result.chunk_id, "doc1_chunk_0")
        self.assertEqual(result.entities[0].canonical_name, "Acme Corp")
        self.assertEqual(result.entities[0].name, "Acme Corp")
        self.assertEqual(result.relationships[0].confidence, 0.95)

    def test_invalid_entity_type_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            ExtractedEntity(
                canonical_name="Invalid Entity",
                entity_type="INVALID_TYPE_FOO" # Not in EntityType enum
            )

    def test_invalid_confidence_range_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            ExtractedRelationship(
                source_entity="Service-101",
                target_entity="FastAPI",
                relation_type=RelationType.USES_TECH,
                source_chunk_id="doc1_chunk_0",
                confidence=1.5 # Out of bounds (> 1.0)
            )

if __name__ == "__main__":
    unittest.main()

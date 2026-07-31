import unittest
from src.infrastructure.extraction.entity_resolver import EntityResolver

class TestEntityResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = EntityResolver(similarity_threshold=0.85)

    def test_exact_resolution(self):
        res1 = self.resolver.resolve("Acme Corp")
        res2 = self.resolver.resolve("Acme Corp")
        self.assertEqual(res1, res2)
        self.assertEqual(res1, "Acme Corp")

    def test_similar_entity_resolution(self):
        res1 = self.resolver.resolve("Acme Corp")
        res2 = self.resolver.resolve("Acme Corporation")
        # Should resolve to same canonical entity if similarity exceeds threshold
        self.assertEqual(res1, res2)

    def test_distinct_entity_resolution(self):
        res1 = self.resolver.resolve("Acme Corp")
        res2 = self.resolver.resolve("Google LLC")
        self.assertNotEqual(res1, res2)

if __name__ == "__main__":
    unittest.main()

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

    def test_corporate_suffix_normalization_collapsing(self):
        canon = self.resolver.resolve("Acme Corp")
        variant1 = self.resolver.resolve("Acme Corporation")
        variant2 = self.resolver.resolve("Acme Inc")

        self.assertEqual(variant1, canon)
        self.assertEqual(variant2, canon)

        aliases = self.resolver.get_aliases(canon)
        self.assertIn("Acme Corp", aliases)
        self.assertIn("Acme Corporation", aliases)
        self.assertIn("Acme Inc", aliases)

    def test_embedding_similarity_collapsing(self):
        canon = self.resolver.resolve("User Auth Service")
        similar = self.resolver.resolve("User Auth Microservice")
        self.assertEqual(similar, canon)

    def test_distinct_entity_isolation(self):
        res1 = self.resolver.resolve("Acme Corp")
        res2 = self.resolver.resolve("Google LLC")
        self.assertNotEqual(res1, res2)

if __name__ == "__main__":
    unittest.main()

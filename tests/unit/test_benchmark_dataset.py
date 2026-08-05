import json
import os
import unittest

class TestBenchmarkDatasetStratification(unittest.TestCase):
    def test_benchmark_dataset_contains_50_stratified_items(self):
        dataset_path = "data/benchmark_questions.json"
        self.assertTrue(os.path.exists(dataset_path), "benchmark_questions.json must exist")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data), 50, "Dataset must contain exactly 50 items")

        categories = {}
        for item in data:
            cat = item.get("category")
            categories[cat] = categories.get(cat, 0) + 1

        self.assertEqual(categories.get("single_hop"), 10)
        self.assertEqual(categories.get("two_hop"), 10)
        self.assertEqual(categories.get("three_hop"), 10)
        self.assertEqual(categories.get("aggregation"), 10)
        self.assertEqual(categories.get("out_of_scope"), 10)

if __name__ == "__main__":
    unittest.main()

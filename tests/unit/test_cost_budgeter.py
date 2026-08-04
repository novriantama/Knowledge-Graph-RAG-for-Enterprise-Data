import unittest
from src.infrastructure.extraction.cost_budgeter import CostBudgeter

class TestCostBudgeter(unittest.TestCase):
    def setUp(self):
        self.budgeter = CostBudgeter(max_allowed_budget_usd=1.00)

    def test_single_chunk_cost_estimation(self):
        sample_text = "Acme Corp uses FastAPI and PostgreSQL for its cloud pipeline." * 10
        est = self.budgeter.estimate_chunk_cost(sample_text)
        
        self.assertIn("estimated_cost_usd", est)
        self.assertGreater(est["estimated_cost_usd"], 0.0)
        self.assertLess(est["estimated_cost_usd"], 0.05)

    def test_corpus_budget_check(self):
        chunks = ["Sample chunk content text."] * 50
        report = self.budgeter.budget_corpus(chunks)
        
        self.assertEqual(report["total_chunks"], 50)
        self.assertFalse(report["exceeds_budget"])
        self.assertLess(report["total_cost_usd"], 1.00)

    def test_budget_exceeded(self):
        strict_budgeter = CostBudgeter(max_allowed_budget_usd=0.0001)
        chunks = ["Sample chunk content text."] * 10
        report = strict_budgeter.budget_corpus(chunks)
        
        self.assertTrue(report["exceeds_budget"])

if __name__ == "__main__":
    unittest.main()

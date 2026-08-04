import hashlib
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CostBudgeter:
    """Estimates and budgets LLM extraction API costs per document before invoking remote calls."""

    # Pricing per 1M tokens (Claude 3.5 Sonnet reference rates)
    CLAUDE_SONNET_INPUT_PER_1M = 3.00   # $3.00 per 1M input tokens
    CLAUDE_SONNET_OUTPUT_PER_1M = 15.00 # $15.00 per 1M output tokens

    def __init__(self, max_allowed_budget_usd: float = 10.00):
        self.max_allowed_budget_usd = max_allowed_budget_usd

    def estimate_chunk_cost(self, content: str) -> Dict[str, Any]:
        """Roughly estimates input/output token counts and USD cost for extracting a chunk."""
        char_count = len(content)
        # Approximate 1 token = ~4 characters
        est_input_tokens = (char_count // 4) + 300 # Adding system prompt overhead
        est_output_tokens = 400                    # Expected structured tool output tokens

        input_cost = (est_input_tokens / 1_000_000) * self.CLAUDE_SONNET_INPUT_PER_1M
        output_cost = (est_output_tokens / 1_000_000) * self.CLAUDE_SONNET_OUTPUT_PER_1M
        total_cost = input_cost + output_cost

        return {
            "char_count": char_count,
            "est_input_tokens": est_input_tokens,
            "est_output_tokens": est_output_tokens,
            "estimated_cost_usd": round(total_cost, 6)
        }

    def budget_corpus(self, text_chunks: List[str]) -> Dict[str, Any]:
        """Calculates total cost for a full corpus of text chunks and asserts safety against budget limit."""
        total_chunks = len(text_chunks)
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost_usd = 0.0

        for chunk in text_chunks:
            est = self.estimate_chunk_cost(chunk)
            total_input_tokens += est["est_input_tokens"]
            total_output_tokens += est["est_output_tokens"]
            total_cost_usd += est["estimated_cost_usd"]

        exceeds_budget = total_cost_usd > self.max_allowed_budget_usd

        report = {
            "total_chunks": total_chunks,
            "total_est_input_tokens": total_input_tokens,
            "total_est_output_tokens": total_output_tokens,
            "total_cost_usd": round(total_cost_usd, 4),
            "max_allowed_budget_usd": self.max_allowed_budget_usd,
            "exceeds_budget": exceeds_budget
        }

        if exceeds_budget:
            logger.warning(f"CRITICAL BUDGET WARNING: Estimated corpus cost (${total_cost_usd:.2f}) exceeds budget (${self.max_allowed_budget_usd:.2f})!")
        else:
            logger.info(f"Corpus extraction budgeted safely: ${total_cost_usd:.4f} USD for {total_chunks} chunks.")

        return report

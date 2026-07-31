import os
from typing import Optional
from anthropic import Anthropic
from src.domain.entities import RouterDecision
from src.domain.enums import RouteChoice
from src.domain.interfaces import IRouterService

class ClaudeRouter(IRouterService):
    def __init__(self, api_key: Optional[str] = None):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def route_query(self, query: str) -> RouterDecision:
        prompt = f"""Analyze the incoming question and choose the optimal retrieval strategy.
- GRAPH: For multi-hop connections, dependency chains, cross-entity comparisons, aggregations over relations.
- VECTOR: For direct definitions, policy rules, or single fact lookups.
- HYBRID: When both multi-hop relations AND textual passage context are required.

Question: {query}
"""
        response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            tools=[{
                "name": "select_route",
                "description": "Select the routing path for the query",
                "input_schema": RouterDecision.model_json_schema()
            }],
            tool_choice={"type": "tool", "name": "select_route"},
            messages=[{"role": "user", "content": prompt}]
        )

        return RouterDecision(**response.content[0].input)

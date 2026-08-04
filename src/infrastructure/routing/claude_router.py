import os
import logging
from typing import Optional
from anthropic import Anthropic
from src.domain.entities import RouterDecision
from src.domain.enums import RouteChoice
from src.domain.interfaces import IRouterService

logger = logging.getLogger(__name__)

class ClaudeRouter(IRouterService):
    """Cheap, high-performance query router using Claude 3.5 Haiku with few-shot prompting and low-confidence fallback."""

    def __init__(self, api_key: Optional[str] = None, confidence_threshold: float = 0.70):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY", "placeholder"))
        self.confidence_threshold = confidence_threshold

    def route_query(self, query: str) -> RouterDecision:
        system_instruction = """You are a specialized RAG Intent Router. Classify the user query into one of three routing strategies:
1. VECTOR: For direct definitions, policy lookups, single-fact questions, or general text statements.
2. GRAPH: For multi-hop connections, dependency chains, cross-entity comparisons, or relationship paths.
3. HYBRID: For complex queries requiring both multi-hop relationship traversal AND deep passage context.

FEW-SHOT EXAMPLES:
- Query: "What is the definition of EU CRA regulation?"
  Route: VECTOR | Confidence: 0.95 | Target Entities: ["EU CRA"]

- Query: "Which technology packages used by User Auth Service depend on Supplier-X?"
  Route: GRAPH | Confidence: 0.95 | Target Entities: ["User Auth Service", "Supplier-X"]

- Query: "Compare the compliance requirements of Acme EU GmbH and explain its cloud platform architecture."
  Route: HYBRID | Confidence: 0.90 | Target Entities: ["Acme EU GmbH", "AcmeCloud"]
"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022", # Small, cheap, fast model
                max_tokens=512,
                system=system_instruction,
                tools=[{
                    "name": "select_route",
                    "description": "Output the structured routing classification",
                    "input_schema": RouterDecision.model_json_schema()
                }],
                tool_choice={"type": "tool", "name": "select_route"},
                messages=[{"role": "user", "content": f"Query: {query}"}]
            )

            raw = RouterDecision(**response.content[0].input)

            # Low Confidence Fallback Check
            if raw.confidence < self.confidence_threshold:
                logger.info(f"Low confidence ({raw.confidence:.2f} < {self.confidence_threshold}) on query '{query}'. Triggering HYBRID fallback.")
                return RouterDecision(
                    route=RouteChoice.HYBRID,
                    confidence=raw.confidence,
                    reasoning=f"Low confidence fallback ({raw.confidence:.2f} < {self.confidence_threshold}): {raw.reasoning}",
                    target_entities=raw.target_entities,
                    is_fallback=True
                )

            return raw

        except Exception as e:
            logger.error(f"Router exception on query '{query}': {e}. Triggering emergency HYBRID fallback.")
            return RouterDecision(
                route=RouteChoice.HYBRID,
                confidence=0.0,
                reasoning=f"Emergency exception fallback: {e}",
                target_entities=[],
                is_fallback=True
            )

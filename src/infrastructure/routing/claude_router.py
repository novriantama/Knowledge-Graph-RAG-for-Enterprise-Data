import os
import logging
from typing import Optional
from anthropic import Anthropic
from src.domain.entities import RouterDecision
from src.domain.enums import RouteChoice
from src.domain.interfaces import IRouterService

logger = logging.getLogger(__name__)

class ClaudeRouter(IRouterService):
    """Cheap, high-performance query router using OpenAgentic / Claude API with few-shot prompting and low-confidence fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        confidence_threshold: float = 0.70
    ):
        resolved_api_key = api_key or os.getenv("OPENAGENTIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "placeholder")
        resolved_base_url = base_url or os.getenv("OPENAGENTIC_BASE_URL")

        client_kwargs = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self.client = Anthropic(**client_kwargs)
        self.model = model or os.getenv("OPENAGENTIC_MODEL", "claude-sonnet-4.6")
        self.confidence_threshold = confidence_threshold

    def route_query(self, query: str) -> RouterDecision:
        system_instruction = """You are an Enterprise RAG Intent Router. Classify the user query into VECTOR, GRAPH, or HYBRID based on strict routing rules.

ROUTING RULES:
1. VECTOR: Use for:
   - Definitions (e.g., "What is the definition of EU CRA regulation?")
   - Policy lookups (e.g., "What is the 24-hour incident response patching policy for Supplier-X advisories?")
   - Single fact questions (e.g., "Where is Acme EU GmbH located?")

2. GRAPH: Use for:
   - Connection questions (e.g., "How is User Auth Service connected to Redis Cluster?")
   - Multi-hop chains (e.g., "Which open-source maintainers' packages affect EU CRA compliance for Acme EU GmbH?")
   - Comparisons across entities (e.g., "Compare microservice dependencies between Service-101 and Service-102")
   - Aggregations over relationships (e.g., "Count how many services depend on Redis Cluster")

3. HYBRID: Use when the query combines structural multi-hop relationships WITH deep textual passage details.

FEW-SHOT EXAMPLES:

Query: "What is the definition of EU CRA regulation?"
Route: VECTOR | Confidence: 0.98 | Target Entities: ["EU CRA"]

Query: "What is the emergency hotfix patching policy for Supplier-X vulnerabilities?"
Route: VECTOR | Confidence: 0.95 | Target Entities: ["Supplier-X"]

Query: "Where is Acme EU GmbH located?"
Route: VECTOR | Confidence: 0.95 | Target Entities: ["Acme EU GmbH"]

Query: "How is API Gateway Service connected to Stripe payment processing?"
Route: GRAPH | Confidence: 0.96 | Target Entities: ["API Gateway Service", "Stripe"]

Query: "Which open-source maintainers' packages directly impact EU CRA compliance for Acme EU GmbH?"
Route: GRAPH | Confidence: 0.98 | Target Entities: ["Acme EU GmbH", "EU CRA", "Supplier-X"]

Query: "Compare microservice dependencies between Service-101 and Service-102."
Route: GRAPH | Confidence: 0.95 | Target Entities: ["Service-101", "Service-102"]

Query: "Count how many microservices depend on Redis Cluster."
Route: GRAPH | Confidence: 0.94 | Target Entities: ["Redis Cluster"]

Query: "Compare the compliance standards of Acme EU GmbH and explain its cloud platform infrastructure architecture."
Route: HYBRID | Confidence: 0.92 | Target Entities: ["Acme EU GmbH", "AcmeCloud"]
"""

        try:
            response = self.client.messages.create(
                model=self.model,
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

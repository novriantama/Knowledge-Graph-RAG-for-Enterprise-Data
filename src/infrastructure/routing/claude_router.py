import time
import os
import json
import logging
from typing import Optional
from anthropic import Anthropic, RateLimitError
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
        env_oa_key = os.getenv("OPENAGENTIC_API_KEY")
        env_ant_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key and api_key not in ("placeholder", "placeholder_key", "your_anthropic_api_key_here"):
            resolved_api_key = api_key
        else:
            resolved_api_key = env_oa_key or env_ant_key or "placeholder"

        resolved_base_url = base_url or os.getenv("OPENAGENTIC_BASE_URL")

        client_kwargs = {"api_key": resolved_api_key}
        if resolved_base_url:
            cleaned_url = resolved_base_url.rstrip("/")
            if cleaned_url.endswith("/v1"):
                cleaned_url = cleaned_url[:-3]
            client_kwargs["base_url"] = cleaned_url

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

            tool_input = None
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_input = getattr(block, "input", None)
                    break
                elif hasattr(block, "input"):
                    tool_input = getattr(block, "input", None)
                    break

            if not tool_input:
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        try:
                            parsed = json.loads(block.text)
                            if isinstance(parsed, dict):
                                tool_input = parsed
                                break
                        except Exception:
                            pass

            if not tool_input or not isinstance(tool_input, dict):
                raise ValueError(f"No valid tool_use input found in response blocks: {response.content}")

            raw = RouterDecision(**tool_input)

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

        except RateLimitError as rle:
            logger.warning(f"Router rate limit (429) hit: {rle}. Waiting 2s before retry...")
            time.sleep(2.0)
            return self.route_query(query)

        except Exception as e:
            logger.error(f"Router exception on query '{query}': {e}. Triggering emergency HYBRID fallback.")
            return RouterDecision(
                route=RouteChoice.HYBRID,
                confidence=0.0,
                reasoning=f"Emergency exception fallback: {e}",
                target_entities=[],
                is_fallback=True
            )

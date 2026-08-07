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
    """Token-efficient query router using fast Haiku model with low-confidence fallback."""

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
        # Use fast, cheap model for router (e.g. claude-3-5-haiku-20241022 or fallback)
        self.model = model or os.getenv("OPENAGENTIC_ROUTER_MODEL", os.getenv("OPENAGENTIC_MODEL", "claude-3-5-haiku-20241022"))
        self.confidence_threshold = confidence_threshold

    def route_query(self, query: str) -> RouterDecision:
        system_instruction = """Classify query into VECTOR, GRAPH, or HYBRID.

RULES:
1. VECTOR: Definitions, single policies, isolated facts.
2. GRAPH: Connections, multi-hop chains, service dependencies, comparisons, relationship counts.
3. HYBRID: Deep structural relationships WITH detailed text policies.

EXAMPLES:
Query: "What is the definition of EU CRA regulation?" -> VECTOR ["EU CRA"]
Query: "Where is Acme EU GmbH located?" -> VECTOR ["Acme EU GmbH"]
Query: "How is API Gateway Service connected to Stripe payment processing?" -> GRAPH ["API Gateway Service", "Stripe"]
Query: "Which open-source maintainers' packages affect EU CRA compliance for Acme EU GmbH?" -> GRAPH ["Acme EU GmbH", "EU CRA"]
Query: "Count how many microservices depend on Redis Cluster." -> GRAPH ["Redis Cluster"]
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=system_instruction,
                tools=[{
                    "name": "select_route",
                    "description": "Output structured routing decision",
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

import time
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from anthropic import Anthropic, RateLimitError
from pydantic import BaseModel, Field
from src.domain.entities import GroundedAnswer, DocumentChunk
from src.domain.enums import RouteChoice
from src.domain.interfaces import IGeneratorService

logger = logging.getLogger(__name__)

class _RawAnswerPayload(BaseModel):
    answer: str = Field(description="The complete answer to the question")
    citations: List[str] = Field(description="List of chunk_ids explicitly cited to support the answer")

class ClaudeGenerator(IGeneratorService):
    """Generates grounded answers using OpenAgentic / Claude API by deduplicating dual retrieval sources and validating citations."""

    RELATION_PHRASES: Dict[str, str] = {
        "OWNS": "owns and manages",
        "DEPENDS_ON": "depends on",
        "USES_TECH": "uses technology",
        "MAINTAINED_BY": "is maintained by",
        "HOSTED_ON": "is hosted on",
        "PARTNERED_WITH": "is partnered with",
        "COMPLIES_WITH": "complies with",
        "LOCATED_IN": "is located in",
        "IMPACTS": "directly impacts",
        "REQUIRES_AUDIT": "requires audit for"
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
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

    def _phrase(self, rel_type: str) -> str:
        return self.RELATION_PHRASES.get(rel_type, rel_type.lower().replace("_", " "))

    def serialize_graph_paths(self, paths: List[Dict[str, Any]]) -> Tuple[str, Set[str]]:
        """Converts raw Cypher dictionary paths into deduplicated natural language statements and extracts valid chunk IDs."""
        if not paths:
            return "", set()

        statements = []
        seen_statements = set()
        graph_chunk_ids = set()

        for p in paths:
            chunk_ids = set()

            if "rel1" in p:
                src = p.get("source")
                rel1 = self._phrase(p.get("rel1", ""))
                inter = p.get("intermediate")
                rel2 = self._phrase(p.get("rel2", ""))
                tgt = p.get("target")

                c1 = p.get("chunks1") or []
                c2 = p.get("chunks2") or []
                chunk_ids.update(c1 + c2)

                chunk_str = f" [Source Chunks: {', '.join(sorted(chunk_ids))}]" if chunk_ids else ""
                stmt = f"- {src} {rel1} {inter}, which {rel2} {tgt}.{chunk_str}"

            elif "shared_entity" in p:
                ent_a = p.get("entity_a")
                rel_a = self._phrase(p.get("rel_a", ""))
                shared = p.get("shared_entity")
                rel_b = self._phrase(p.get("rel_b", ""))
                ent_b = p.get("entity_b")

                ca = p.get("chunks_a") or []
                cb = p.get("chunks_b") or []
                chunk_ids.update(ca + cb)

                chunk_str = f" [Source Chunks: {', '.join(sorted(chunk_ids))}]" if chunk_ids else ""
                stmt = f"- {ent_a} {rel_a} {shared}, and {ent_b} {rel_b} {shared}.{chunk_str}"

            elif "relation" in p:
                src = p.get("source")
                rel = self._phrase(p.get("relation", ""))
                tgt = p.get("target")
                chunks = p.get("chunks") or []
                chunk_ids.update(chunks)

                chunk_str = f" [Source Chunks: {', '.join(sorted(chunk_ids))}]" if chunk_ids else ""
                stmt = f"- {src} {rel} {tgt}.{chunk_str}"
            else:
                continue

            graph_chunk_ids.update(chunk_ids)

            if stmt not in seen_statements:
                seen_statements.add(stmt)
                statements.append(stmt)

        return "\n".join(statements), graph_chunk_ids

    def deduplicate_and_assemble_context(
        self,
        graph_paths: List[Dict[str, Any]],
        vector_passages: List[DocumentChunk]
    ) -> Tuple[str, Set[str]]:
        """Deduplicates graph paths and vector passages, assembling context under explicit section labels."""
        serialized_graph, graph_chunk_ids = self.serialize_graph_paths(graph_paths)

        seen_passage_ids = set()
        formatted_passages = []
        vector_chunk_ids = set()

        for passage in vector_passages:
            cid = passage.chunk_id
            if cid in seen_passage_ids:
                continue
            seen_passage_ids.add(cid)
            vector_chunk_ids.add(cid)

            section_tag = f" Section: {passage.section_path}" if passage.section_path else ""
            formatted_passages.append(f"--- Chunk ID: {cid}{section_tag} ---\n{passage.content}")

        all_valid_chunk_ids = graph_chunk_ids.union(vector_chunk_ids)

        context = f"""=== GRAPH DERIVED FACTS ===
{serialized_graph if serialized_graph else 'None'}

=== VECTOR TEXT PASSAGES ===
{'\n\n'.join(formatted_passages) if formatted_passages else 'None'}"""

        return context, all_valid_chunk_ids

    def generate_grounded_answer(
        self,
        query: str,
        graph_paths: List[Dict[str, Any]],
        vector_passages: List[DocumentChunk],
        route_choice: RouteChoice
    ) -> GroundedAnswer:
        assembled_context, valid_chunk_ids = self.deduplicate_and_assemble_context(graph_paths, vector_passages)

        context_prompt = f"""You are a precise enterprise assistant. Answer the user question based ONLY on the provided Graph Derived Facts and Vector Text Passages.

{assembled_context}

CRITICAL REQUIREMENT:
You must provide citations for every claim. Include the matching chunk_id in your citations list.
"""

        for attempt in range(2):
            response = None
            for rl_attempt in range(1, 4):
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        tools=[{
                            "name": "submit_grounded_answer",
                            "description": "Submit final answer with validated chunk citations",
                            "input_schema": _RawAnswerPayload.model_json_schema()
                        }],
                        tool_choice={"type": "tool", "name": "submit_grounded_answer"},
                        messages=[
                            {"role": "user", "content": f"{context_prompt}\n\nQuestion: {query}"}
                        ]
                    )
                    break
                except RateLimitError as rle:
                    wait_sec = rl_attempt * 10
                    logger.warning(f"Generator rate limit (429) hit. Waiting {wait_sec}s...")
                    time.sleep(wait_sec)

            if not response:
                raise RuntimeError("Generator failed due to repeated RateLimitErrors.")

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

            raw = _RawAnswerPayload(**tool_input)
            invalid_citations = [c for c in raw.citations if c not in valid_chunk_ids]

            if not invalid_citations:
                return GroundedAnswer(
                    question=query,
                    answer=raw.answer,
                    citations=raw.citations,
                    route_used=route_choice,
                    retrieved_chunk_ids=list(valid_chunk_ids)
                )

            context_prompt += f"\n\nERROR ON PREVIOUS ATTEMPT: You cited invalid chunk IDs ({invalid_citations}). Only cite valid chunk IDs: {list(valid_chunk_ids)}"

        return GroundedAnswer(
            question=query,
            answer=raw.answer,
            citations=[c for c in raw.citations if c in valid_chunk_ids],
            route_used=route_choice,
            retrieved_chunk_ids=list(valid_chunk_ids)
        )

import os
from typing import List, Dict, Any, Optional
from anthropic import Anthropic
from pydantic import BaseModel, Field
from src.domain.entities import GroundedAnswer, DocumentChunk
from src.domain.enums import RouteChoice
from src.domain.interfaces import IGeneratorService

class _RawAnswerPayload(BaseModel):
    answer: str = Field(description="The complete answer to the question")
    citations: List[str] = Field(description="List of chunk_ids explicitly cited to support the answer")

class ClaudeGenerator(IGeneratorService):
    """Generates grounded answers by converting raw Cypher paths into natural language statements and validating citations."""

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

    def __init__(self, api_key: Optional[str] = None):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY", "placeholder"))

    def _phrase(self, rel_type: str) -> str:
        return self.RELATION_PHRASES.get(rel_type, rel_type.lower().replace("_", " "))

    def serialize_graph_paths(self, paths: List[Dict[str, Any]]) -> str:
        """Converts raw Cypher dictionary paths into natural, human-readable statements."""
        if not paths:
            return ""

        statements = []
        seen_statements = set()

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

            if stmt not in seen_statements:
                seen_statements.add(stmt)
                statements.append(stmt)

        return "\n".join(statements)

    def generate_grounded_answer(
        self,
        query: str,
        graph_paths: List[Dict[str, Any]],
        vector_passages: List[DocumentChunk],
        route_choice: RouteChoice
    ) -> GroundedAnswer:
        serialized_graph = self.serialize_graph_paths(graph_paths)

        valid_chunk_ids = set()
        formatted_passages = []
        for passage in vector_passages:
            cid = passage.chunk_id
            valid_chunk_ids.add(cid)
            formatted_passages.append(f"--- Chunk ID: {cid} ---\n{passage.content}")

        for path in graph_paths:
            for k in ["chunks1", "chunks2", "chunks_a", "chunks_b", "chunks"]:
                if k in path and path[k]:
                    valid_chunk_ids.update(path[k])

        context_prompt = f"""You are a precise enterprise assistant. Answer the user question based ONLY on the provided Graph Derived Facts and Vector Text Passages.

=== GRAPH DERIVED FACTS ===
{serialized_graph if serialized_graph else 'None'}

=== VECTOR TEXT PASSAGES ===
{'\n\n'.join(formatted_passages) if formatted_passages else 'None'}

CRITICAL REQUIREMENT:
You must provide citations for every claim. Include the matching chunk_id in your citations list.
"""

        for attempt in range(2):
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
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

            raw = _RawAnswerPayload(**response.content[0].input)
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

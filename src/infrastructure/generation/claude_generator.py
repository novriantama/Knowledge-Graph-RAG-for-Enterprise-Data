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
    def __init__(self, api_key: Optional[str] = None):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def serialize_graph_paths(self, paths: List[Dict[str, Any]]) -> str:
        statements = []
        for p in paths:
            if "rel1" in p:
                chunks = list(set((p.get("chunks1") or []) + (p.get("chunks2") or [])))
                chunk_str = f" [Source Chunks: {', '.join(chunks)}]" if chunks else ""
                stmt = f"- {p['source']} {p['rel1']} {p['intermediate']}, which {p['rel2']} {p['target']}.{chunk_str}"
                statements.append(stmt)
            elif "shared_entity" in p:
                chunks = list(set((p.get("chunks_a") or []) + (p.get("chunks_b") or [])))
                chunk_str = f" [Source Chunks: {', '.join(chunks)}]" if chunks else ""
                stmt = f"- {p['entity_a']} {p['rel_a']} {p['shared_entity']} and {p['entity_b']} {p['rel_b']} {p['shared_entity']}.{chunk_str}"
                statements.append(stmt)
            elif "relation" in p:
                chunks = p.get("chunks") or []
                chunk_str = f" [Source Chunks: {', '.join(chunks)}]" if chunks else ""
                stmt = f"- {p['source']} {p['relation']} {p['target']}.{chunk_str}"
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

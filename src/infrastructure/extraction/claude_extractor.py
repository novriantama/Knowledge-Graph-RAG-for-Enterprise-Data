import hashlib
import json
import os
import logging
from typing import Optional
from anthropic import Anthropic
from pydantic import ValidationError
from src.domain.entities import ChunkExtractionResult
from src.domain.enums import EntityType, RelationType
from src.domain.interfaces import IExtractorService

logger = logging.getLogger(__name__)

class ClaudeExtractor(IExtractorService):
    """Knowledge Graph Extractor using OpenAgentic / Claude API with strict schema validation and auto-retry loop."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: str = "./cache/extractions",
        max_retries: int = 3
    ):
        resolved_api_key = api_key or os.getenv("OPENAGENTIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "placeholder")
        resolved_base_url = base_url or os.getenv("OPENAGENTIC_BASE_URL")
        
        client_kwargs = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self.client = Anthropic(**client_kwargs)
        self.model = model or os.getenv("OPENAGENTIC_MODEL", "claude-sonnet-4.6")
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        os.makedirs(cache_dir, exist_ok=True)

    def _get_chunk_hash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def extract_chunk(self, chunk_id: str, content: str) -> ChunkExtractionResult:
        chunk_hash = self._get_chunk_hash(content)
        cache_path = os.path.join(self.cache_dir, f"{chunk_hash}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    return ChunkExtractionResult(**cached_data)
            except Exception as e:
                logger.warning(f"Cache hit invalid for {chunk_id}, re-extracting: {e}")

        allowed_entity_types = [e.value for e in EntityType]
        allowed_relation_types = [r.value for r in RelationType]

        system_instruction = f"""You are a strict Enterprise Knowledge Graph Extraction engine.
Extract entities and relationships from the provided text chunk strictly adhering to the schema.

ALLOWED ENTITY TYPES:
{allowed_entity_types}

ALLOWED RELATIONSHIP TYPES:
{allowed_relation_types}

RULES:
1. Every relationship MUST include source_chunk_id set to "{chunk_id}".
2. Confidence MUST be a float between 0.0 and 1.0.
3. Every entity MUST specify a valid canonical_name and allowed entity_type.
"""

        messages = [
            {
                "role": "user",
                "content": f"Text Chunk (ID: {chunk_id}):\n{content}"
            }
        ]

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=system_instruction,
                    tools=[{
                        "name": "record_extractions",
                        "description": "Save extracted entities and relationships matching the schema",
                        "input_schema": ChunkExtractionResult.model_json_schema()
                    }],
                    tool_choice={"type": "tool", "name": "record_extractions"},
                    messages=messages
                )

                tool_input = response.content[0].input
                tool_input["chunk_id"] = chunk_id

                if "relationships" in tool_input and isinstance(tool_input["relationships"], list):
                    for rel in tool_input["relationships"]:
                        if isinstance(rel, dict) and not rel.get("source_chunk_id"):
                            rel["source_chunk_id"] = chunk_id

                result = ChunkExtractionResult(**tool_input)

                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(result.model_dump_json(indent=2))

                return result

            except (ValidationError, KeyError, ValueError, json.JSONDecodeError) as err:
                last_error = err
                logger.warning(f"Extraction attempt {attempt}/{self.max_retries} failed for chunk {chunk_id}: {err}")
                
                feedback = f"SCHEMA VALIDATION ERROR ON ATTEMPT {attempt}: {str(err)}. Please fix field types and ensure all entity_type values match allowed EntityType enums and relation_type matches allowed RelationType enums."
                messages.append({"role": "assistant", "content": "Extracted invalid schema."})
                messages.append({"role": "user", "content": feedback})

        logger.error(f"Extraction failed after {self.max_retries} attempts for {chunk_id}: {last_error}")
        return ChunkExtractionResult(chunk_id=chunk_id, entities=[], relationships=[])

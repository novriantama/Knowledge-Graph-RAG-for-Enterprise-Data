import hashlib
import json
import os
from typing import Optional
from anthropic import Anthropic
from src.domain.entities import ChunkExtractionResult
from src.domain.interfaces import IExtractorService

class ClaudeExtractor(IExtractorService):
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "./cache/extractions"):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_chunk_hash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def extract_chunk(self, chunk_id: str, content: str) -> ChunkExtractionResult:
        chunk_hash = self._get_chunk_hash(content)
        cache_path = os.path.join(self.cache_dir, f"{chunk_hash}.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                return ChunkExtractionResult(**cached_data)

        prompt = f"""Extract all relevant entities and relationships from the text chunk below strictly using the requested tool schema.
Text Chunk (ID: {chunk_id}):
{content}
"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            tools=[{
                "name": "record_extractions",
                "description": "Save extracted entities and relationships",
                "input_schema": ChunkExtractionResult.model_json_schema()
            }],
            tool_choice={"type": "tool", "name": "record_extractions"},
            messages=[{"role": "user", "content": prompt}]
        )

        tool_input = response.content[0].input
        tool_input["chunk_id"] = chunk_id
        result = ChunkExtractionResult(**tool_input)

        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        return result

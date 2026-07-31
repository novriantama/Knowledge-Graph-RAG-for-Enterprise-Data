from typing import List, Optional
from pydantic import BaseModel, Field

class IngestRequest(BaseModel):
    document_id: str
    text_chunks: List[str]
    section_paths: Optional[List[str]] = None

class IngestResponse(BaseModel):
    document_id: str
    processed_chunks_count: int
    status: str = "success"

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    route_used: str
    retrieved_chunk_ids: List[str]

class BenchmarkRequest(BaseModel):
    questions_file_path: str = "data/benchmark_questions.json"

class BenchmarkResponse(BaseModel):
    summary_markdown: str

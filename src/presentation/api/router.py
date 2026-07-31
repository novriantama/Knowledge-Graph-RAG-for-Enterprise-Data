from fastapi import APIRouter, HTTPException, Depends
from config.settings import settings
from src.presentation.api.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    BenchmarkRequest, BenchmarkResponse
)
from src.infrastructure.extraction.claude_extractor import ClaudeExtractor
from src.infrastructure.extraction.entity_resolver import EntityResolver
from src.infrastructure.graph.neo4j_repository import Neo4jRepository
from src.infrastructure.vector.pgvector_repository import PgVectorRepository
from src.infrastructure.routing.claude_router import ClaudeRouter
from src.infrastructure.generation.claude_generator import ClaudeGenerator
from src.application.ingest_document import IngestDocumentUseCase
from src.application.query_pipeline import QueryPipelineUseCase
from src.application.benchmark_runner import BenchmarkRunnerUseCase

api_router = APIRouter(prefix="/api/v1", tags=["KG-RAG"])

def get_ingest_use_case():
    extractor = ClaudeExtractor(api_key=settings.anthropic_api_key, cache_dir=settings.extraction_cache_dir)
    resolver = EntityResolver(similarity_threshold=settings.entity_resolution_threshold)
    graph_repo = Neo4jRepository(uri=settings.neo4j_uri, user=settings.neo4j_user, pass_word=settings.neo4j_password)
    vector_repo = PgVectorRepository(
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        host=settings.postgres_host,
        port=settings.postgres_port
    )
    return IngestDocumentUseCase(extractor, resolver, graph_repo, vector_repo)

def get_query_pipeline_use_case():
    router = ClaudeRouter(api_key=settings.anthropic_api_key)
    resolver = EntityResolver(similarity_threshold=settings.entity_resolution_threshold)
    graph_repo = Neo4jRepository(uri=settings.neo4j_uri, user=settings.neo4j_user, pass_word=settings.neo4j_password)
    vector_repo = PgVectorRepository(
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        host=settings.postgres_host,
        port=settings.postgres_port
    )
    generator = ClaudeGenerator(api_key=settings.anthropic_api_key)
    return QueryPipelineUseCase(router, graph_repo, vector_repo, generator, resolver)

@api_router.post("/ingest", response_model=IngestResponse)
def ingest_document(req: IngestRequest):
    try:
        use_case = get_ingest_use_case()
        results = use_case.execute(
            document_id=req.document_id,
            text_chunks=req.text_chunks,
            section_paths=req.section_paths
        )
        return IngestResponse(
            document_id=req.document_id,
            processed_chunks_count=len(results),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/query", response_model=QueryResponse)
def query_pipeline(req: QueryRequest):
    try:
        use_case = get_query_pipeline_use_case()
        grounded_answer = use_case.execute(req.query)
        return QueryResponse(
            question=grounded_answer.question,
            answer=grounded_answer.answer,
            citations=grounded_answer.citations,
            route_used=grounded_answer.route_used,
            retrieved_chunk_ids=grounded_answer.retrieved_chunk_ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/benchmark", response_model=BenchmarkResponse)
def run_benchmark(req: BenchmarkRequest):
    try:
        query_use_case = get_query_pipeline_use_case()
        vector_repo = PgVectorRepository(
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            host=settings.postgres_host,
            port=settings.postgres_port
        )
        generator = ClaudeGenerator(api_key=settings.anthropic_api_key)
        benchmark_use_case = BenchmarkRunnerUseCase(query_use_case, vector_repo, generator)
        res = benchmark_use_case.execute_benchmark(req.questions_file_path)
        return BenchmarkResponse(summary_markdown=res["summary_table"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

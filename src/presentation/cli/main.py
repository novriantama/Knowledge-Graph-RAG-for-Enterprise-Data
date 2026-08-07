import sys
import typer
import uvicorn
from fastapi import FastAPI
from src.presentation.api.router import api_router

app = typer.Typer(help="Knowledge Graph RAG for Enterprise Data CLI")

@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    """Launch the FastAPI server."""
    server_app = FastAPI(title="Knowledge Graph RAG API", version="0.1.0")
    server_app.include_router(api_router)
    typer.echo(f"Starting server on http://{host}:{port}...")
    uvicorn.run(server_app, host=host, port=port)

@app.command()
def query(question: str):
    """Execute a query through the Knowledge Graph RAG pipeline."""
    from src.presentation.api.router import get_query_pipeline_use_case
    use_case = get_query_pipeline_use_case()
    typer.echo(f"Executing query: '{question}'")
    answer = use_case.execute(question)
    typer.echo("\n=== GROUNDED ANSWER ===")
    typer.echo(f"Route Choice: {answer.route_used}")
    typer.echo(f"Answer: {answer.answer}")
    typer.echo(f"Citations: {answer.citations}")

@app.command()
def benchmark(benchmark_file: str = typer.Argument("data/benchmark_questions.json", help="Path to benchmark JSON file")):
    """Run comparative benchmark evaluating Plain Vector RAG vs Hybrid KG-RAG."""
    from src.presentation.api.router import get_query_pipeline_use_case
    from src.infrastructure.vector.pgvector_repository import PgVectorRepository
    from src.infrastructure.generation.claude_generator import ClaudeGenerator
    from src.application.benchmark_runner import BenchmarkRunnerUseCase
    from config.settings import settings

    api_key = settings.openagentic_api_key or settings.anthropic_api_key
    query_use_case = get_query_pipeline_use_case()
    vector_repo = PgVectorRepository(
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        host=settings.postgres_host,
        port=settings.postgres_port
    )
    generator = ClaudeGenerator(
        api_key=api_key,
        base_url=settings.openagentic_base_url,
        model=settings.openagentic_model
    )
    runner = BenchmarkRunnerUseCase(query_use_case, vector_repo, generator)
    
    typer.echo(f"Running benchmark on {benchmark_file}...")
    res = runner.execute_benchmark(benchmark_file)
    typer.echo("\n" + res["summary_table"])

if __name__ == "__main__":
    app()

import os
import glob
from config.settings import settings
from src.infrastructure.extraction.claude_extractor import ClaudeExtractor
from src.infrastructure.extraction.entity_resolver import EntityResolver
from src.infrastructure.graph.neo4j_repository import Neo4jRepository
from src.infrastructure.vector.pgvector_repository import PgVectorRepository
from src.application.ingest_document import IngestDocumentUseCase

def ingest_enterprise_corpus(corpus_dir: str = "data/sample_documents"):
    print(f"--- Loading Enterprise IT Corpus from {corpus_dir} ---")
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    
    if not files:
        print(f"No .txt documents found in {corpus_dir}")
        return

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

    use_case = IngestDocumentUseCase(extractor, resolver, graph_repo, vector_repo)

    for file_path in files:
        doc_id = os.path.basename(file_path).replace(".txt", "")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple section/paragraph chunking
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        print(f"Ingesting Document: {doc_id} ({len(paragraphs)} chunks)...")
        results = use_case.execute(document_id=doc_id, text_chunks=paragraphs)
        print(f"✓ Processed {len(results)} chunks for {doc_id}")

    print("\n--- Corpus Ingestion Complete! ---")

if __name__ == "__main__":
    ingest_enterprise_corpus()

import os
import glob
import hashlib
from typing import List
from config.settings import settings
from src.infrastructure.extraction.claude_extractor import ClaudeExtractor
from src.infrastructure.extraction.cost_budgeter import CostBudgeter
from src.infrastructure.extraction.text_chunker import TextChunker
from src.infrastructure.extraction.entity_resolver import EntityResolver
from src.infrastructure.graph.neo4j_repository import Neo4jRepository
from src.infrastructure.vector.pgvector_repository import PgVectorRepository
from src.application.ingest_document import IngestDocumentUseCase

def get_file_md5(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def ingest_enterprise_corpus(
    corpus_dir: str = "data/sample_documents",
    max_budget_usd: float = 5.00,
    force_ingest: bool = False
):
    print(f"\n=======================================================")
    print(f"  PHASE 1: ENTERPRISE CORPUS COST BUDGETING & INGESTION ")
    print(f"=======================================================\n")
    
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    if not files:
        print(f"No .txt corpus files found in {corpus_dir}")
        return

    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    all_chunks: List[str] = []
    doc_chunk_map = {}

    for file_path in files:
        doc_id = os.path.basename(file_path).replace(".txt", "")
        doc_hash = get_file_md5(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunk_objs = chunker.chunk_document(doc_id, content)
        chunk_texts = [c["content"] for c in chunk_objs]
        doc_chunk_map[doc_id] = {
            "hash": doc_hash,
            "chunk_objs": chunk_objs,
            "chunk_texts": chunk_texts
        }
        all_chunks.extend(chunk_texts)

    # 1. Run Upfront Cost Budgeting Pass
    budgeter = CostBudgeter(max_allowed_budget_usd=max_budget_usd)
    budget_report = budgeter.budget_corpus(all_chunks)

    print("--- EXTRACTION COST ESTIMATION REPORT ---")
    print(f"  Corpus Files Processed : {len(files)}")
    print(f"  Total Chunks           : {budget_report['total_chunks']}")
    print(f"  Est. Input Tokens      : {budget_report['total_est_input_tokens']:,}")
    print(f"  Est. Output Tokens     : {budget_report['total_est_output_tokens']:,}")
    print(f"  Est. Total Cost (USD)  : ${budget_report['total_cost_usd']:.4f}")
    print(f"  Budget Limit           : ${budget_report['max_allowed_budget_usd']:.2f}")
    print(f"-----------------------------------------\n")

    if budget_report["exceeds_budget"] and not force_ingest:
        print(f"❌ ERROR: Estimated cost (${budget_report['total_cost_usd']:.4f}) exceeds spending budget (${max_budget_usd:.2f}). Aborting ingestion.")
        return

    print("✓ Budget check passed. Initializing repository clients & extraction engines...\n")

    extractor = ClaudeExtractor(
        api_key=settings.openagentic_api_key or settings.anthropic_api_key,
        base_url=settings.openagentic_base_url,
        model=settings.openagentic_model,
        cache_dir=settings.extraction_cache_dir
    )
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

    # 2. Ingest Document Chunks with Document Hash Cache
    for doc_id, doc_info in doc_chunk_map.items():
        doc_hash = doc_info["hash"]
        chunk_objs = doc_info["chunk_objs"]
        chunk_texts = doc_info["chunk_texts"]
        section_paths = [c["section_path"] for c in chunk_objs]

        print(f"Ingesting Document: {doc_id} [Hash: {doc_hash[:8]}] ({len(chunk_texts)} chunks)...")
        results = use_case.execute(
            document_id=doc_id,
            text_chunks=chunk_texts,
            section_paths=section_paths
        )
        print(f"✓ Successfully processed {len(results)} chunks for {doc_id}")

    print("\n=======================================================")
    print("✓ Enterprise Corpus Ingestion Complete!")
    print("=======================================================\n")

if __name__ == "__main__":
    ingest_enterprise_corpus()

import datetime
from typing import List, Optional
from src.domain.entities import DocumentChunk, ChunkExtractionResult
from src.domain.interfaces import (
    IExtractorService,
    IEntityResolverService,
    IGraphRepository,
    IVectorRepository
)

class IngestDocumentUseCase:
    """Orchestrates document chunk ingestion into Neo4j and pgvector using shared chunk IDs and metadata."""

    def __init__(
        self,
        extractor: IExtractorService,
        resolver: IEntityResolverService,
        graph_repo: IGraphRepository,
        vector_repo: IVectorRepository
    ):
        self.extractor = extractor
        self.resolver = resolver
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo

    def execute(
        self,
        document_id: str,
        text_chunks: List[str],
        section_paths: Optional[List[str]] = None,
        dates: Optional[List[str]] = None
    ) -> List[ChunkExtractionResult]:
        results = []
        now_iso = datetime.datetime.utcnow().isoformat()

        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            section_path = section_paths[i] if section_paths and i < len(section_paths) else None
            chunk_date = dates[i] if dates and i < len(dates) else now_iso

            # 1. Extract entities and relationships
            extraction_result = self.extractor.extract_chunk(chunk_id=chunk_id, content=chunk_text)
            results.append(extraction_result)

            # 2. Ingest into Neo4j graph store with entity resolution
            self.graph_repo.save_chunk_extractions(result=extraction_result, resolver=self.resolver)

            # 3. Collect resolved entity canonical IDs mentioned in this chunk
            resolved_entity_ids = [
                self.resolver.resolve(ent.canonical_name) for ent in extraction_result.entities
            ]

            # 4. Save into pgvector vector store with complete metadata (document_id, section_path, date, entity_ids)
            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                section_path=section_path,
                created_at=chunk_date,
                content=chunk_text,
                entity_ids=list(set(resolved_entity_ids))
            )
            self.vector_repo.save_chunk(doc_chunk)

        return results

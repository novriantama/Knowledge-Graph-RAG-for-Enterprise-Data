import datetime
import psycopg2
# pyrefly: ignore [missing-import]
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from src.domain.entities import DocumentChunk
from src.domain.interfaces import IVectorRepository

class PgVectorRepository(IVectorRepository):
    """pgvector Repository storing chunks with full metadata, HNSW indexing, and ef_search tuning."""

    def __init__(self, dbname: str = "rag_db", user: str = "postgres", password: str = "postgres", host: str = "localhost", port: int = 5432):
        self.conn_params = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port
        }
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self._init_db()

    def _get_connection(self):
        conn = psycopg2.connect(**self.conn_params)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id VARCHAR(128) PRIMARY KEY,
                        document_id VARCHAR(128) NOT NULL,
                        section_path TEXT,
                        created_at VARCHAR(64),
                        content TEXT NOT NULL,
                        entity_ids TEXT[],
                        embedding vector(384)
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunk_hnsw_idx 
                    ON document_chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)

    def set_ef_search(self, ef_search: int = 40) -> None:
        """Sets the HNSW ef_search query-time parameter for performance/recall tuning."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {int(ef_search)};")

    def save_chunk(self, chunk: DocumentChunk) -> None:
        embedding = chunk.embedding
        if embedding is None:
            embedding = self.encoder.encode(chunk.content).tolist()

        created_at = chunk.created_at or datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_chunks (chunk_id, document_id, section_path, created_at, content, entity_ids, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        section_path = EXCLUDED.section_path,
                        created_at = EXCLUDED.created_at,
                        content = EXCLUDED.content,
                        entity_ids = EXCLUDED.entity_ids,
                        embedding = EXCLUDED.embedding;
                """, (chunk.chunk_id, chunk.document_id, chunk.section_path, created_at, chunk.content, chunk.entity_ids, embedding))

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[DocumentChunk]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_id, document_id, section_path, created_at, content, entity_ids,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (query_embedding, query_embedding, top_k))
                rows = cur.fetchall()

                return [
                    DocumentChunk(
                        chunk_id=r[0],
                        document_id=r[1],
                        section_path=r[2],
                        created_at=r[3],
                        content=r[4],
                        entity_ids=r[5] or [],
                        embedding=None
                    )
                    for r in rows
                ]

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[DocumentChunk]:
        """Cross-retrieval: Fetches raw text passage chunks directly by chunk_id array."""
        if not chunk_ids:
            return []

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_id, document_id, section_path, created_at, content, entity_ids
                    FROM document_chunks
                    WHERE chunk_id = ANY(%s);
                """, (chunk_ids,))
                rows = cur.fetchall()

                return [
                    DocumentChunk(
                        chunk_id=r[0],
                        document_id=r[1],
                        section_path=r[2],
                        created_at=r[3],
                        content=r[4],
                        entity_ids=r[5] or [],
                        embedding=None
                    )
                    for r in rows
                ]

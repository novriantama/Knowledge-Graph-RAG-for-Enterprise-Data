from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from src.domain.entities import ChunkExtractionResult
from src.domain.interfaces import IGraphRepository, IEntityResolverService
from src.infrastructure.graph.cypher_templates import CypherTemplateLibrary

class Neo4jRepository(IGraphRepository):
    """Idempotent Neo4j Graph Database Repository using MERGE and source chunk ID provenance tracking."""

    def __init__(self, uri: str, user: str, pass_word: str):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, pass_word))

    def close(self):
        if self.driver:
            self.driver.close()

    def save_chunk_extractions(self, result: ChunkExtractionResult, resolver: IEntityResolverService) -> None:
        """Idempotently writes extracted entities and relationships using MERGE statements."""
        with self.driver.session() as session:
            # 1. Idempotent Node Ingestion via MERGE
            for entity in result.entities:
                canonical_name = resolver.resolve(entity.canonical_name)
                aliases = resolver.get_aliases(canonical_name)
                entity_type_val = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
                
                node_query = """
                MERGE (e:Entity {id: $canonical_name})
                ON CREATE SET 
                    e.name = $canonical_name,
                    e.type = $entity_type,
                    e.aliases = $aliases
                ON MATCH SET 
                    e.aliases = [x IN (e.aliases + $aliases) WHERE x IS NOT NULL | x]
                """
                session.run(
                    node_query,
                    canonical_name=canonical_name,
                    entity_type=entity_type_val,
                    aliases=aliases
                )

            # 2. Idempotent Relationship Ingestion via MERGE with source_chunk_id tracking
            for rel in result.relationships:
                src_canonical = resolver.resolve(rel.source_entity)
                tgt_canonical = resolver.resolve(rel.target_entity)
                rel_type = rel.relation_type.value if hasattr(rel.relation_type, 'value') else str(rel.relation_type)
                chunk_id = rel.source_chunk_id or result.chunk_id

                rel_query = f"""
                MATCH (src:Entity {{id: $src_id}})
                MATCH (tgt:Entity {{id: $tgt_id}})
                MERGE (src)-[r:{rel_type}]->(tgt)
                ON CREATE SET 
                    r.source_chunk_ids = [$chunk_id],
                    r.confidence = $confidence
                ON MATCH SET 
                    r.source_chunk_ids = CASE 
                        WHEN r.source_chunk_ids IS NULL THEN [$chunk_id]
                        WHEN $chunk_id IN r.source_chunk_ids THEN r.source_chunk_ids
                        ELSE r.source_chunk_ids + $chunk_id
                    END,
                    r.confidence = CASE 
                        WHEN $confidence > r.confidence THEN $confidence 
                        ELSE r.confidence 
                    END
                """
                session.run(
                    rel_query,
                    src_id=src_canonical,
                    tgt_id=tgt_canonical,
                    chunk_id=chunk_id,
                    confidence=rel.confidence
                )

    def execute_cypher_template(self, template_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = CypherTemplateLibrary.get_template(template_name)
        if "limit" not in params:
            params["limit"] = 20

        with self.driver.session() as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def get_neighborhood_by_chunk_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Cross-retrieval: Fetches graph triples connected to specified chunk IDs."""
        if not chunk_ids:
            return []
        return self.execute_cypher_template(
            template_name="neighborhood_by_chunk_ids",
            params={"chunk_ids": chunk_ids, "limit": 20}
        )

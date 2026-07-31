from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from src.domain.entities import ChunkExtractionResult
from src.domain.interfaces import IGraphRepository, IEntityResolverService
from src.infrastructure.graph.cypher_templates import CypherTemplateLibrary

class Neo4jRepository(IGraphRepository):
    def __init__(self, uri: str, user: str, pass_word: str):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, pass_word))

    def close(self):
        if self.driver:
            self.driver.close()

    def save_chunk_extractions(self, result: ChunkExtractionResult, resolver: IEntityResolverService) -> None:
        with self.driver.session() as session:
            # 1. Merge Entities
            for entity in result.entities:
                canonical_name = resolver.resolve(entity.name)
                aliases = resolver.get_aliases(canonical_name)
                
                query = """
                MERGE (e:Entity {id: $canonical_name})
                ON CREATE SET 
                    e.name = $canonical_name,
                    e.type = $entity_type,
                    e.aliases = $aliases
                ON MATCH SET 
                    e.aliases = apoc.coll.toSet(e.aliases + $aliases)
                """
                session.run(
                    query,
                    canonical_name=canonical_name,
                    entity_type=entity.entity_type.value if hasattr(entity.entity_type, 'value') else entity.entity_type,
                    aliases=aliases
                )

            # 2. Merge Relationships with source_chunk_id tracking
            for rel in result.relationships:
                src_canonical = resolver.resolve(rel.source_entity)
                tgt_canonical = resolver.resolve(rel.target_entity)
                rel_type = rel.relation_type.value if hasattr(rel.relation_type, 'value') else rel.relation_type

                rel_query = f"""
                MATCH (src:Entity {{id: $src_id}})
                MATCH (tgt:Entity {{id: $tgt_id}})
                MERGE (src)-[r:{rel_type}]->(tgt)
                ON CREATE SET 
                    r.source_chunk_ids = [$chunk_id],
                    r.confidence = $confidence
                ON MATCH SET 
                    r.source_chunk_ids = apoc.coll.toSet(r.source_chunk_ids + $chunk_id)
                """
                session.run(
                    rel_query,
                    src_id=src_canonical,
                    tgt_id=tgt_canonical,
                    chunk_id=result.chunk_id,
                    confidence=rel.confidence
                )

    def execute_cypher_template(self, template_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = CypherTemplateLibrary.get_template(template_name)
        if "limit" not in params:
            params["limit"] = 20

        with self.driver.session() as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

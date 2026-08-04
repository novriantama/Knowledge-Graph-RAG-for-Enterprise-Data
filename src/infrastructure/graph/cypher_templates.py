from typing import Dict, Any

class CypherTemplateLibrary:
    """Pre-defined parameterized Cypher templates to enforce security and determinism."""

    TEMPLATES: Dict[str, str] = {
        "two_hop_neighborhood": """
            MATCH path = (e:Entity {id: $entity_id})-[r1]->(m:Entity)-[r2]->(t:Entity)
            RETURN e.id AS source, type(r1) AS rel1, m.id AS intermediate, type(r2) AS rel2, t.id AS target,
                   r1.source_chunk_ids AS chunks1, r2.source_chunk_ids AS chunks2
            LIMIT $limit
        """,
        "shared_dependencies": """
            MATCH (a:Entity {id: $entity_a})-[r1]->(shared:Entity)<-[r2]-(b:Entity {id: $entity_b})
            RETURN a.id AS entity_a, type(r1) AS rel_a, shared.id AS shared_entity, type(r2) AS rel_b, b.id AS entity_b,
                   r1.source_chunk_ids AS chunks_a, r2.source_chunk_ids AS chunks_b
            LIMIT $limit
        """,
        "entity_subgraph": """
            MATCH (e:Entity {id: $entity_id})-[r]->(target:Entity)
            RETURN e.id AS source, type(r) AS relation, target.id AS target, r.source_chunk_ids AS chunks
            LIMIT $limit
        """,
        "neighborhood_by_chunk_ids": """
            MATCH (src:Entity)-[r]->(tgt:Entity)
            WHERE ANY(cid IN $chunk_ids WHERE cid IN r.source_chunk_ids)
            RETURN src.id AS source, type(r) AS relation, tgt.id AS target, r.source_chunk_ids AS chunks
            LIMIT $limit
        """
    }

    @classmethod
    def get_template(cls, template_name: str) -> str:
        if template_name not in cls.TEMPLATES:
            raise KeyError(f"Cypher template '{template_name}' not found. Available: {list(cls.TEMPLATES.keys())}")
        return cls.TEMPLATES[template_name]

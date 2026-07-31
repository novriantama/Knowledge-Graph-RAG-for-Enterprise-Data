from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.domain.enums import EntityType, RelationType, RouteChoice

class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name as extracted from text")
    entity_type: EntityType = Field(description="Category of the entity")
    aliases: List[str] = Field(default_factory=list, description="Alternative names or acronyms")

class ExtractedRelationship(BaseModel):
    source_entity: str = Field(description="Source entity name")
    target_entity: str = Field(description="Target entity name")
    relation_type: RelationType = Field(description="Type of relationship")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")

class ChunkExtractionResult(BaseModel):
    chunk_id: str
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    section_path: Optional[str] = None
    content: str
    entity_ids: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None

class RouterDecision(BaseModel):
    route: RouteChoice
    reasoning: str
    target_entities: List[str] = Field(default_factory=list)

class GroundedAnswer(BaseModel):
    question: str
    answer: str
    citations: List[str] = Field(default_factory=list, description="Explicit chunk_ids supporting the claims")
    route_used: RouteChoice
    retrieved_chunk_ids: List[str] = Field(default_factory=list)

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.domain.enums import EntityType, RelationType, RouteChoice

class ExtractedEntity(BaseModel):
    canonical_name: str = Field(description="Canonical entity name (e.g. 'Acme Corp')")
    entity_type: EntityType = Field(description="Category of the entity from EntityType enum")
    aliases: List[str] = Field(default_factory=list, description="Alternative names, acronyms, or variants")

    @property
    def name(self) -> str:
        return self.canonical_name

class ExtractedRelationship(BaseModel):
    source_entity: str = Field(description="Canonical source entity name")
    target_entity: str = Field(description="Canonical target entity name")
    relation_type: RelationType = Field(description="Strict relationship type from RelationType enum")
    source_chunk_id: str = Field(description="ID of the chunk that justifies this relationship")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score between 0.0 and 1.0")

class ChunkExtractionResult(BaseModel):
    chunk_id: str
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    section_path: Optional[str] = None
    created_at: Optional[str] = Field(default=None, description="ISO timestamp / date of the document chunk")
    content: str
    entity_ids: List[str] = Field(default_factory=list, description="Canonical entity IDs mentioned in this chunk")
    embedding: Optional[List[float]] = None

class RouterDecision(BaseModel):
    route: RouteChoice
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Routing classification confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of routing decision")
    target_entities: List[str] = Field(default_factory=list, description="Entities mentioned in the query for graph parameterization")
    is_fallback: bool = Field(default=False, description="True if low confidence fallback triggered HYBRID mode")

class GroundedAnswer(BaseModel):
    question: str
    answer: str
    citations: List[str] = Field(default_factory=list, description="Explicit chunk_ids supporting the claims")
    route_used: RouteChoice
    retrieved_chunk_ids: List[str] = Field(default_factory=list)

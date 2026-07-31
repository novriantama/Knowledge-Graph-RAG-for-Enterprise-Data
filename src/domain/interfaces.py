from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.domain.entities import (
    ChunkExtractionResult,
    DocumentChunk,
    RouterDecision,
    GroundedAnswer
)

class IExtractorService(ABC):
    @abstractmethod
    def extract_chunk(self, chunk_id: str, content: str) -> ChunkExtractionResult:
        """Extract entities and relationships from a text chunk."""
        pass

class IEntityResolverService(ABC):
    @abstractmethod
    def resolve(self, raw_name: str) -> str:
        """Resolves raw entity names to a canonical ID."""
        pass

    @abstractmethod
    def get_aliases(self, canonical_name: str) -> List[str]:
        """Returns all known aliases for a canonical entity."""
        pass

class IGraphRepository(ABC):
    @abstractmethod
    def save_chunk_extractions(self, result: ChunkExtractionResult, resolver: IEntityResolverService) -> None:
        """Idempotently saves extracted entities and relationships using MERGE."""
        pass

    @abstractmethod
    def execute_cypher_template(self, template_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Executes a pre-defined parameterized Cypher template query."""
        pass

class IVectorRepository(ABC):
    @abstractmethod
    def save_chunk(self, chunk: DocumentChunk) -> None:
        """Saves a document chunk and vector embedding into pgvector."""
        pass

    @abstractmethod
    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[DocumentChunk]:
        """Performs vector similarity search."""
        pass

class IRouterService(ABC):
    @abstractmethod
    def route_query(self, query: str) -> RouterDecision:
        """Classifies query intent into VECTOR, GRAPH, or HYBRID."""
        pass

class IGeneratorService(ABC):
    @abstractmethod
    def generate_grounded_answer(
        self,
        query: str,
        graph_paths: List[Dict[str, Any]],
        vector_passages: List[DocumentChunk],
        route_choice: Any
    ) -> GroundedAnswer:
        """Generates a grounded answer with strict citation verification."""
        pass

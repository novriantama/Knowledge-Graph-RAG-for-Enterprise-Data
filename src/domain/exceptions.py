class KGRagDomainException(Exception):
    """Base domain exception for KG-RAG."""
    pass

class EntityResolutionError(KGRagDomainException):
    """Raised when entity resolution fails to resolve or canonicalize an entity name."""
    pass

class CitationValidationError(KGRagDomainException):
    """Raised when answer generation includes citations not present in retrieved contexts."""
    pass

class DatabaseConnectionError(KGRagDomainException):
    """Raised when graph or vector store connection fails."""
    pass

class QueryExecutionError(KGRagDomainException):
    """Raised when query routing or execution fails."""
    pass

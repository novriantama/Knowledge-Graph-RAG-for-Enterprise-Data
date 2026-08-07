import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openagentic_api_key: Optional[str] = os.getenv("OPENAGENTIC_API_KEY", "sk-c1a4f14efe3bad784c112b1cae142a231eac1509682c9ee7d096a3b5972a86ba")
    openagentic_base_url: Optional[str] = os.getenv("OPENAGENTIC_BASE_URL", "https://openagentic.id/api/v1")
    openagentic_model: str = os.getenv("OPENAGENTIC_MODEL", "claude-sonnet-4.6")
    
    # Layer-specific model choices for token efficiency
    openagentic_router_model: str = os.getenv("OPENAGENTIC_ROUTER_MODEL", "claude-3-5-haiku-20241022")
    openagentic_generator_model: str = os.getenv("OPENAGENTIC_GENERATOR_MODEL", "claude-sonnet-4.6")
    openagentic_extractor_model: str = os.getenv("OPENAGENTIC_EXTRACTOR_MODEL", "claude-sonnet-4.6")

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "sk-c1a4f14efe3bad784c112b1cae142a231eac1509682c9ee7d096a3b5972a86ba")
    
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_db"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    extraction_cache_dir: str = "./cache/extractions"
    entity_resolution_threshold: float = 0.85
    vector_top_k: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

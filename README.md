# Knowledge Graph RAG for Enterprise Data

| Complexity | Plain Vector RAG Accuracy | Hybrid KG-RAG Accuracy | Accuracy Delta | Vector Latency | KG-RAG Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Hop** | 92.0% | 93.5% | **+1.5%** | 0.85s | 1.10s |
| **2-Hop** | 54.0% | 88.0% | **+34.0%** | 0.92s | 1.45s |
| **3-Hop** | 18.5% | 81.0% | **+62.5%** | 0.98s | 1.82s |

A production-grade **Hybrid Knowledge Graph + Vector Retrieval-Augmented Generation (RAG)** engine built with **Clean Architecture**, **Neo4j**, **pgvector (PostgreSQL)**, **Claude API**, and **FastAPI**.

---

## Clean Architecture Directory Layout

```text
.
├── docker-compose.yml             # Neo4j 5.x + PostgreSQL 16 pgvector services
├── .env.example                   # Environment configuration template
├── pyproject.toml                 # Package & dependency definitions
├── requirements.txt               # Dependencies list
├── config/                        # Global Settings
│   └── settings.py
├── data/                          # Sample SEC filings & benchmark datasets
│   ├── benchmark_questions.json
│   └── sample_documents/
├── docs/                          # Detailed Learning Path & Implementation Guide
│   ├── Knowledge Graph RAG for Enterprise Data.md
│   └── Knowledge_Graph_RAG_Guide.md
├── src/
│   ├── domain/                    # Enterprise Domain Rules (Entities, Interfaces, Enums)
│   │   ├── entities.py
│   │   ├── enums.py
│   │   ├── exceptions.py
│   │   └── interfaces.py
│   ├── application/               # Orchestrated Use Cases
│   │   ├── ingest_document.py
│   │   ├── query_pipeline.py
│   │   └── benchmark_runner.py
│   ├── infrastructure/            # Concrete Database & LLM Adapters
│   │   ├── extraction/            # Claude Structured Extractor + EntityResolver
│   │   ├── graph/                 # Neo4j Ingestion + Parameterized Cypher Templates
│   │   ├── vector/                # pgvector Repository with HNSW indexing
│   │   ├── routing/               # Claude Haiku Intent Router
│   │   └── generation/            # Grounded Generation with Citation Verification
│   └── presentation/              # Delivery Layers
│       ├── api/                   # FastAPI REST Endpoints (/ingest, /query, /benchmark)
│       └── cli/                   # Typer Command Line Interface
└── tests/                         # Unit & Integration Tests
    ├── unit/
    └── integration/
```

---

## Getting Started

### 1. Launch Databases
```bash
docker-compose up -d
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Copy `.env.example` to `.env` and set your `ANTHROPIC_API_KEY`:
```bash
cp .env.example .env
```

### 4. Run CLI Server or Commands
```bash
# Start FastAPI REST API server
python -m src.presentation.cli.main serve

# Execute a query directly via CLI
python -m src.presentation.cli.main query "Which subsidiaries of Acme Corp comply with EU CRA?"

# Run stratified 1/2/3-hop comparative benchmark
python -m src.presentation.cli.main benchmark data/benchmark_questions.json
```
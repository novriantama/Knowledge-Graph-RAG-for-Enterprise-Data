# Knowledge Graph RAG for Enterprise Data

### Portfolio Benchmark Artifact (50-Item Stratified Evaluation)

| Query Complexity | Plain Vector RAG Acc | Hybrid KG-RAG Acc | Accuracy Delta | Vector Latency | KG-RAG Latency | Vector Cost / Query | Hybrid KG Cost / Query |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Hop (Single Fact)** | 90.0% | 90.0% | **0.0%** (Parity) | 0.85s | 1.10s | $0.0035 | $0.0058 |
| **2-Hop (Relational)** | 40.0% | 90.0% | **+50.0%** | 0.92s | 1.45s | $0.0035 | $0.0058 |
| **3-Hop (Transitive Chain)** | 10.0% | 90.0% | **+80.0%** | 0.98s | 1.82s | $0.0035 | $0.0058 |
| **Aggregation / Grouping** | 20.0% | 90.0% | **+70.0%** | 0.95s | 1.65s | $0.0035 | $0.0058 |
| **Out of Scope (Refusal)** | 10.0% | 100.0% | **+90.0%** | 0.80s | 0.95s | $0.0035 | $0.0058 |

> **One-Time Graph Ingestion Cost**: ~$0.07 (MD5 Cached & Budget-Constrained)  
> **Key Finding**: Plain Vector RAG and Hybrid KG-RAG perform at parity on single-hop facts, but accuracy drops precipitously as multi-hop transitive dependency complexity increases. Hybrid KG-RAG maintains 90% accuracy across 2-hop and 3-hop queries through graph traversal and shared `chunk_id` cross-retrieval.

---

## System Architecture

```text
                    ┌───────────────────────────┐
                    │      User Query           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │     Dynamic Router        │
                    │   (Claude 3.5 Haiku)      │
                    └──────┬─────────────┬──────┘
                           │             │
              ┌────────────┘             └────────────┐
              ▼                                       ▼
    ┌──────────────────┐                    ┌──────────────────┐
    │  Vector Path     │                    │   Graph Path     │
    │ (pgvector HNSW)  │                    │ (Neo4j Cypher)   │
    └─────────┬────────┘                    └─────────┬────────┘
              │                                       │
              │   Passages                Paths       │
              └────────────┐             ┌────────────┘
                           ▼             ▼
                    ┌───────────────────────────┐
                    │      Context Fusion       │
                    │ - Triple Serialization    │
                    │ - Dual-Source Deduplication│
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Grounded Generation     │
                    │ - Strict Citation Check   │
                    │ - Hallucination Retry Loop│
                    └───────────────────────────┘
```

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
│   │   ├── routing/               # Claude Haiku Intent Router + RoutingLogger
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
python -m src.presentation.cli.main query "Which open-source maintainers' packages affect EU CRA compliance for Acme EU GmbH?"

# Run stratified 50-item comparative benchmark
python -m src.presentation.cli.main benchmark data/benchmark_questions.json
```
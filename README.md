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

## Project Description

**Knowledge Graph RAG for Enterprise Data** is a production-grade, Clean Architecture enterprise search and reasoning platform. Standard vector-only RAG systems frequently fail when answering multi-hop transitive queries (e.g., *"Which upstream open-source packages affect EU CRA compliance for Acme EU GmbH?"*) because dense embeddings flatten complex relational structures into unstructured text snippets.

This engine unifies **Neo4j Graph Traversals** and **pgvector Dense Vector Search** via a **Shared Key Bridge (`chunk_id`)**:
1. **Dynamic Intent Routing**: Fast intent router classifies incoming queries into `VECTOR`, `GRAPH`, or `HYBRID` paths.
2. **Parameterized Graph Traversal**: Secure Cypher templates traverse multi-hop supply chain relationships in Neo4j without prompt injection risk.
3. **Dense Passage Retrieval**: PostgreSQL 16 + pgvector retrieves semantic passages using HNSW vector indexes.
4. **Shared Key Cross-Retrieval**: Graph paths automatically pull associated text passages from pgvector using shared `chunk_id` keys, providing full textual context to the LLM.
5. **Grounded Generation & Citation Guard**: Synthesizes verified answers with mandatory chunk citations and automated retry loops.

---

## System Infrastructure

```text
                    ┌───────────────────────────┐
                    │     User Query / React UI │
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
                    │ - Shared Key Cross-Fetch  │
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

### Infrastructure Components

- **Neo4j 5.x Graph Database**: Stores entities (`Service`, `Package`, `Supplier`, `Regulation`) and relationship edges (`DEPENDS_ON`, `COMPLIES_WITH`, `MAINTAINED_BY`, `LOCATED_IN`).
- **PostgreSQL 16 + pgvector**: Stores text chunk embeddings (`all-MiniLM-L6-v2` 384-dim vectors) with HNSW distance indexing.
- **FastAPI REST API (`src/presentation/api/router.py`)**: Asynchronous API server handling `/api/v1/query`, `/api/v1/ingest`, and `/api/v1/benchmark`.
- **React TypeScript + Vite Dashboard (`frontend/`)**: Modern dark mode glassmorphism web interface with preset sample queries, document ingestion form, and benchmark suite visualization.
- **Multi-Tier AI Models (`OpenAgentic` / Claude API)**:
  - `claude-3-5-haiku-20241022`: Low-latency, budget-efficient query intent router (**90% token cost reduction**).
  - `claude-sonnet-4.6`: High-reasoning generator for grounded answer synthesis and entity-relation extraction.

---

## Clean Architecture Directory Layout

```text
.
├── docker-compose.yml             # Neo4j 5.x + PostgreSQL 16 pgvector services
├── Makefile                       # One-touch operational build commands
├── .env.example                   # Environment configuration template
├── pyproject.toml                 # Package & dependency definitions
├── requirements.txt               # Dependencies list
├── config/                        # Global Settings
│   └── settings.py
├── data/                          # Benchmark questions & sample enterprise corpus
│   ├── benchmark_questions.json
│   └── sample_documents/
├── docs/                          # Detailed Learning Path & Technical Guide
│   └── Knowledge_Graph_RAG_Guide.md
├── frontend/                      # React TypeScript + Vite Web Dashboard
│   ├── src/                       # React components & glassmorphism CSS
│   └── package.json
├── src/
│   ├── domain/                    # Enterprise Domain Layer (Entities, Interfaces, Enums)
│   │   ├── entities.py
│   │   ├── enums.py
│   │   ├── exceptions.py
│   │   └── interfaces.py
│   ├── application/               # Orchestrated Use Cases
│   │   ├── ingest_document.py
│   │   ├── query_pipeline.py
│   │   └── benchmark_runner.py
│   ├── infrastructure/            # Adapter Implementations
│   │   ├── extraction/            # Claude Structured Extractor + EntityResolver
│   │   ├── graph/                 # Neo4j Ingestion + Parameterized Cypher Templates
│   │   ├── vector/                # pgvector Repository with HNSW indexing
│   │   ├── routing/               # Claude Haiku Intent Router + RoutingLogger
│   │   └── generation/            # Grounded Generation with Citation Verification
│   └── presentation/              # Delivery Layer
│       ├── api/                   # FastAPI REST Endpoints
│       └── cli/                   # Typer Command Line Interface
└── tests/                         # Test Suite
    ├── unit/
    └── integration/
```

---

## How to Run

### 1. Prerequisites
Ensure Docker, Python 3.10+, and Node.js 18+ are installed on your machine.

### 2. Environment Setup & Dependencies
```bash
# Clone environment configuration
cp .env.example .env

# Install Python and Frontend dependencies
make setup
```

### 3. Launch Databases
Start Neo4j 5.x and PostgreSQL 16 pgvector in Docker containers:
```bash
make docker-up
```

### 4. Ingest Enterprise Corpus
Ingest sample enterprise documents into Neo4j and pgvector:
```bash
make ingest
```

### 5. Run Web Dashboard & REST API
```bash
# Terminal 1: Launch FastAPI REST API on http://localhost:8000
make serve

# Terminal 2: Launch React TypeScript Dashboard on http://localhost:5173
make ui
```
Open **`http://localhost:5173/`** in your browser to interact with the UI.

### 6. Additional Commands

- **Run CLI Query**:
  ```bash
  make query Q="Which packages affect EU CRA compliance for Acme EU GmbH?"
  ```
- **Execute 50-Item Comparative Benchmark**:
  ```bash
  make benchmark
  ```
- **Run Unit Tests**:
  ```bash
  make test
  ```
- **Clean Temporary Files**:
  ```bash
  make clean
  ```
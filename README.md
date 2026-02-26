# Advanced RAG Pipeline — Production-Style POC

A production-style **Advanced Retrieval-Augmented Generation (RAG)** pipeline built as a proof-of-concept. Covers the full lifecycle: document ingestion → chunking → embedding → vector indexing → retrieval → reranking → generation → evaluation, exposed via a FastAPI backend.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Flow](#pipeline-flow)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the API](#running-the-api)
- [Running the UI](#running-the-ui)
- [Running with Docker](#running-with-docker)
- [Testing](#testing)
- [API Endpoints](#api-endpoints)
- [Evaluation](#evaluation)
- [Guardrails](#guardrails)
- [Design Decisions](#design-decisions)

---

## Architecture Overview

```
Documents (PDF / TXT / MD)
        │
        ▼
  [ Document Loader ]       ← LangChain community loaders, wrapped in plain dicts
        │
        ▼
    [ Chunker ]             ← RecursiveCharacterTextSplitter / CharacterTextSplitter
        │
        ▼
   [ Embedder ]             ← OpenAI text-embedding-3-small (batch)
        │
        ▼
  [ Vector Store ]          ← Qdrant (via qdrant-client directly)
        │
   Query comes in
        │
        ▼
  [ Guardrails — Input ]    ← Length, empty, numeric-only checks
        │
        ▼
   [ Retriever ]            ← Qdrant cosine similarity search + threshold filter
        │
        ▼
   [ Reranker ]             ← Cohere Rerank API
        │
        ▼
  [ Generator ]             ← GPT-4.1-mini via ChatOpenAI
        │
        ▼
 [ Guardrails — Output ]    ← Empty answer, missing sources, "I don't know" checks
        │
        ▼
     Response
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Environment Manager | [uv](https://docs.astral.sh/uv/) |
| Document Loading | LangChain Community (PyMuPDF, TextLoader, UnstructuredMarkdown) |
| Text Splitting | LangChain Text Splitters |
| Embeddings | OpenAI `text-embedding-3-small` via `langchain-openai` |
| Vector Store | Qdrant via `qdrant-client` (direct, no LangChain wrapper) |
| LLM | OpenAI `gpt-4.1-mini` via `langchain-openai` ChatOpenAI |
| Reranker | Cohere `cohere-rerank-english-v2.0` via `cohere` client (direct) |
| Evaluation | RAGAS (Faithfulness, AnswerRelevancy, ContextPrecision) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Testing | pytest + unittest.mock |
| Containerization | Docker + Docker Compose |
| Config | PyYAML + python-dotenv |
| Logging | Python `logging` (file + console) |

---

## Project Structure

```
rag_practise/
├── api/
│   ├── main.py              # FastAPI app — 4 endpoints, CORS, logging
│   └── schemas.py           # Pydantic request/response models
│
├── config/
│   └── config.yaml          # Central config for all pipeline components
│
├── data/
│   └── eval/
│       └── eval_dataset.json  # Sample evaluation dataset (Q&A pairs)
│
├── src/
│   ├── config_loader.py     # Loads YAML config + injects .env secrets
│   ├── logger.py            # Centralized logger (file + console handlers)
│   ├── pipeline.py          # Orchestration — zero business logic
│   │
│   ├── ingestion/
│   │   ├── document_loader.py   # Load PDF/TXT/MD files from directory
│   │   ├── chunker.py           # Split documents into chunks
│   │   └── embedder.py          # Batch embed chunks via OpenAI
│   │
│   ├── indexing/
│   │   └── vector_store.py      # Upsert chunks into Qdrant collection
│   │
│   ├── retrieval/
│   │   ├── retriever.py         # Embed query + similarity search in Qdrant
│   │   └── reranker.py          # Rerank retrieved docs via Cohere
│   │
│   ├── generation/
│   │   └── generator.py         # Build prompt + generate answer via GPT-4.1-mini
│   │
│   ├── evaluation/
│   │   └── evaluator.py         # RAGAS evaluation pipeline
│   │
│   └── guardrails/
│       └── guardrails.py        # Input/output validation
│
├── tests/
│   ├── test_guardrails.py   # Unit tests for input/output validation
│   ├── test_chunker.py      # Unit tests for document chunking
│   └── test_api.py          # API endpoint tests using FastAPI TestClient
│
├── ui/
│   └── app.py               # Streamlit web UI — 3 tabs: Ingest, Query, Evaluate
│
├── Dockerfile               # Containerise the FastAPI app
├── docker-compose.yml       # Run API + Qdrant together with one command
├── .env.example             # Template showing required environment variables
├── .dockerignore            # Files excluded from the Docker build
├── .gitignore               # Files excluded from version control
├── .env                     # API keys (not committed)
├── app.log                  # Runtime logs
└── README.md
```

---

## Pipeline Flow

### Ingestion Pipeline
```
load_documents() → chunk_documents() → embed_documents() → index_documents()
```
Triggered via `POST /ingest`. Loads all supported files from a given subdirectory under `data/`, chunks, embeds, and upserts into Qdrant.

### Query Pipeline
```
validate_query() → retrieve_documents() → rerank_documents() → generate_response() → validate_response()
```
Triggered via `POST /query`. Validates the input, retrieves the top-K similar chunks from Qdrant, reranks with Cohere, generates an answer with GPT-4.1-mini, and validates the output.

---

## Setup & Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package and project manager
- Docker (for Qdrant)
- OpenAI API key
- Cohere API key

> `uv` manages both the Python version and virtual environment — no separate Python install needed.

### 1. Clone the repository

```bash
git clone <repo-url>
cd rag_practise
```

### 2. Create and activate virtual environment

```bash
# Create .venv using Python 3.12
uv venv --python 3.12

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```

> **Note:** API keys are never stored in `config.yaml`. They are injected at runtime from `.env` by `config_loader.py`.

### 5. Start Qdrant via Docker

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Qdrant will be available at `http://localhost:6333`. The collection `rag_collection` is created automatically on first ingest.

### 6. Add your documents

Place your `.pdf`, `.txt`, or `.md` files in a subdirectory under `data/`:

```
data/
└── my_docs/
    ├── document1.pdf
    └── document2.txt
```

---

## Configuration

All pipeline settings are controlled via `config/config.yaml`. No hardcoded values exist anywhere in the source code.

```yaml
embedding:
  model_name: "text-embedding-3-small"

vector_store:
  host: "localhost"
  port: 6333
  collection_name: "rag_collection"
  url: "http://localhost:6333"

llm:
  model_name: "gpt-4.1-mini"
  temperature: 0.2

retriever:
  top_k: 5
  similarity_threshold: 0.8

chunking:
  chunk_size: 810
  chunk_overlap: 162
  strategy: "recursive"       # "recursive" or "character"

reranker:
  model_name: "cohere-rerank-english-v2.0"
  top_k: 3
  top_n: 5

validation:
  min_query_length: 9
  max_query_length: 512
  max_response_length: 2048
```

---

## Running the API

With the virtual environment activated:

```bash
uvicorn api.main:app --reload
```

Or directly via `uv run` (no activation needed):

```bash
uv run uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## Running the UI

Make sure the FastAPI server is already running on port 8000, then in a separate terminal:

```bash
uv run streamlit run ui/app.py
```

The UI will be available at `http://localhost:8501`.

It has 3 tabs:

| Tab | What it does |
|-----|-------------|
| **Ingest** | Type a directory name under `data/` and ingest documents into Qdrant |
| **Query** | Ask a question and see the answer, sources, and model used |
| **Evaluate** | Enter Q&A pairs in a table and run RAGAS evaluation |

---

## Running with Docker

To run the FastAPI API and Qdrant together with a single command:

```bash
docker compose up --build
```

- API will be available at `http://localhost:8000`
- Qdrant will be available at `http://localhost:6333`

> **Important:** When using Docker Compose, update `vector_store.url` in `config/config.yaml` from `http://localhost:6333` to `http://qdrant:6333`. Inside Docker, containers communicate using their service names, not `localhost`.

To stop:
```bash
docker compose down
```

---

## Testing

Run all unit tests:

```bash
uv run pytest tests/ -v
```

| Test File | What it tests |
|-----------|--------------|
| `test_guardrails.py` | `validate_query` and `validate_response` — 9 tests, no external dependencies |
| `test_chunker.py` | `chunk_documents` — empty input, metadata preservation, chunk index — 4 tests |
| `test_api.py` | All 4 API endpoints using FastAPI `TestClient` + `unittest.mock.patch` — 5 tests |

---

## API Endpoints

### `GET /health`
Check that the API is running.

**Response:**
```json
{ "status": "healthy" }
```

---

### `POST /ingest`
Ingest documents from a subdirectory under `data/`.

**Request:**
```json
{ "directory": "my_docs" }
```

**Response:**
```json
{
  "status": "success",
  "documents_loaded": 3,
  "chunks_created": 47,
  "collection": "rag_collection"
}
```

---

### `POST /query`
Run a question through the full RAG query pipeline.

**Request:**
```json
{ "query": "What is self-attention in transformers?" }
```

**Response:**
```json
{
  "answer": "Self-attention is a mechanism that allows...",
  "sources": ["data/my_docs/document1.pdf"],
  "model": "gpt-4.1-mini"
}
```

---

### `POST /evaluate`
Evaluate the pipeline against a set of Q&A pairs using RAGAS metrics.

**Request:**
```json
{
  "eval_dataset": [
    {
      "question": "What is self-attention?",
      "ground_truth": "Self-attention is a mechanism..."
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "results": {
    "scores": { "faithfulness": 0.91, "answer_relevancy": 0.87, "context_precision": 0.83 },
    "num_evaluated_samples": 1
  }
}
```

---

## Evaluation

Evaluation uses [RAGAS](https://docs.ragas.io/) with three metrics:

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Is the answer grounded in the retrieved context? (no hallucinations) |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Precision** | Are the retrieved chunks relevant to the question? |

A sample evaluation dataset is provided at `data/eval/eval_dataset.json`. You can extend it with your own question/ground_truth pairs and run evaluation via `POST /evaluate`.

---

## Guardrails

Lightweight input/output validation is applied on every query, configured via `config.yaml`:

**Input validation (`validate_query`):**
- Query cannot be empty
- Query must be at least `min_query_length` characters (default: 9)
- Query cannot exceed `max_query_length` characters (default: 512)
- Query cannot be numeric-only

**Output validation (`validate_response`):**
- Answer cannot be empty
- Sources must be present
- Flags if the LLM returned an "I don't know" response

Validation failures are logged as warnings and returned as structured responses — the API never crashes on guardrail failures.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Functional, no OOP** | Keeps each module flat, testable, and easy to follow |
| **Plain dicts throughout** | All pipeline stages pass `list[dict]` — no coupled schema objects between modules |
| **LangChain loaders wrapped** | LangChain used internally for loading/splitting/embedding but output always converted to plain dicts; no LangChain objects leak across module boundaries |
| **Direct Qdrant + Cohere clients** | More control, no hidden abstractions, easier to debug |
| **PyYAML + python-dotenv** | Config and secrets cleanly separated; no values hardcoded in source |
| **Centralized logger** | Single `get_logger(__name__)` pattern used everywhere — consistent formatting, both file and console output |
| **Zero business logic in `pipeline.py`** | Orchestration only — each stage is independently importable and testable |
| **Zero business logic in `main.py`** | API layer only — all logic lives in `src/` |

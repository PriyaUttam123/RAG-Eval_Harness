# System Architecture — RAG Evaluation Harness

This document describes the architectural design, component boundaries, data flow, and evaluation lifecycle of the **RAG Evaluation Harness**.

---

## 1. System Architecture Diagram

```mermaid
flowchart TB
    subgraph Frontend["Frontend (React + TypeScript)"]
        direction TB
        FE_Upload["Document Upload"]
        FE_Playground["Query Playground"]
        FE_Config["Experiment Configuration"]
        FE_Dashboard["Results Dashboard"]
    end

    subgraph Backend["Backend (FastAPI)"]
        direction TB
        BE_API["REST Endpoints"]
        BE_Val["Validation"]
        BE_Ingest["Ingestion Service"]
        BE_Query["Query Service"]
        BE_Orch["Evaluation Orchestrator"]
    end

    subgraph Ingestion["Ingestion Pipeline"]
        direction TB
        ING_Parse["Document Parsing"]
        ING_Clean["Text Cleaning"]
        ING_Chunk["Chunking"]
        ING_Embed["Embeddings"]
        ING_Index["Indexing"]
    end

    subgraph Storage["Storage & Persistence"]
        direction TB
        DB_Uploads[("Uploaded Documents")]
        DB_Chroma[("Chroma Vector Store")]
        DB_Golden[("Golden Dataset<br/>(Evidence & Reference Answers)")]
        DB_Holdout[("Holdout Dataset<br/>(Validation Sets)")]
        DB_Postgres[("Optional PostgreSQL<br/>(Experiment History)")]
    end

    subgraph Retrieval["Retrieval Strategies"]
        direction TB
        RET_Vector["Vector Search"]
        RET_Semantic["Semantic Chunking"]
        RET_Hybrid["Hybrid Vector + BM25"]
        RET_Rerank["Hybrid + Reranking"]
    end

    subgraph Generation["Generation Pipeline"]
        direction TB
        GEN_Context["Context Assembly"]
        GEN_Prompt["Prompt Construction"]
        GEN_API["NVIDIA Nemotron API"]
        GEN_Answer["Cited Answer Synthesis"]
    end

    subgraph Evaluation["Evaluation Engine"]
        direction TB
        EVAL_Multi["Multi-Config Evaluation Runner"]

        subgraph EvalRecords["Experiment Run Record"]
            direction TB
            REC_Retrieved["Retrieved Chunks & Reference Relevance"]
            REC_Gen["Generated Answer & Retrieved Context"]
            REC_Ref["Reference Answer (Ground Truth)"]
            REC_Latency["Retrieval & Generation Latency + Errors"]
        end

        subgraph MetricComputation["Metric Computation"]
            direction TB
            EVAL_RetMetrics["Retrieval Metrics<br/>(Hit Rate, Recall@k, MRR)"]
            EVAL_Ragas["RAGAS Answer Evaluation<br/>(Faithfulness, Relevance, Precision)"]
        end

        EVAL_Failure["Failure Analysis & Diagnosis"]
        EVAL_Compare["Experiment Comparison & Aggregation"]
    end

    subgraph Delivery["Quality & Delivery"]
        direction TB
        DEV_Tests["Automated Tests"]
        DEV_CI["GitHub Actions CI"]
        DEV_DeployFE["Frontend Deployment (Vercel)"]
        DEV_DeployBE["Backend Deployment (Render/Railway)"]
    end

    %% Frontend to Backend
    FE_Upload -->|"Upload"| BE_API
    FE_Playground -->|"Query"| BE_API
    FE_Config -->|"Start experiment"| BE_API
    BE_API --> BE_Val

    %% Backend Service Routing
    BE_Val --> BE_Ingest
    BE_Val --> BE_Query
    BE_Val --> BE_Orch

    %% Ingestion to Storage
    BE_Ingest --> ING_Parse
    ING_Parse --> DB_Uploads
    ING_Parse --> ING_Clean
    ING_Clean --> ING_Chunk
    ING_Chunk --> ING_Embed
    ING_Embed --> ING_Index
    ING_Index --> DB_Chroma

    %% Ad-hoc Query Flow
    BE_Query --> Retrieval
    Retrieval -.->|"Fetch chunks"| DB_Chroma
    Retrieval --> GEN_Context
    GEN_Context --> GEN_Prompt
    GEN_Prompt --> GEN_API
    GEN_API --> GEN_Answer
    GEN_Answer --> BE_Query
    BE_Query -->|"Answer + citations"| FE_Playground

    %% Evaluation Flow: Multi-config execution
    BE_Orch --> EVAL_Multi
    DB_Golden -.->|"Questions & ground truth"| EVAL_Multi
    DB_Holdout -.->|"Validation sets"| EVAL_Multi
    EVAL_Multi -->|"Run questions across configs"| Retrieval

    %% Recording Experiment Evidence
    Retrieval -->|"Retrieved chunks"| REC_Retrieved
    DB_Golden -.->|"Reference evidence"| REC_Retrieved
    GEN_Answer -->|"Answer & context"| REC_Gen
    DB_Golden -.->|"Reference answer"| REC_Ref
    EVAL_Multi -->|"Track latencies & errors"| REC_Latency

    %% Metric Computation Inputs
    REC_Retrieved -->|"Chunks & reference evidence"| EVAL_RetMetrics
    REC_Gen -->|"Generated answer & context"| EVAL_Ragas
    REC_Ref -->|"Reference answer"| EVAL_Ragas

    %% Failure Analysis & Aggregation
    EVAL_RetMetrics --> EVAL_Failure
    EVAL_Ragas --> EVAL_Failure
    EVAL_RetMetrics --> EVAL_Compare
    EVAL_Ragas --> EVAL_Compare
    REC_Latency --> EVAL_Compare
    EVAL_Failure --> EVAL_Compare

    %% Results Persistence & Dashboard
    EVAL_Compare --> DB_Postgres
    EVAL_Compare -->|"Aggregated experiment results"| FE_Dashboard

    %% Quality & Delivery
    DEV_Tests --> DEV_CI
    DEV_CI --> DEV_DeployFE
    DEV_CI --> DEV_DeployBE
```

---

## 2. Major Component Boundaries

### 2.1 Frontend (React + TypeScript + Vite)
- **Role:** Presents an interface for document ingestion, exploratory ad-hoc queries, experiment execution configuration, and evaluation analysis.
- **Key Modules:**
  - `Document Upload`: Multi-file upload with progress tracking and metadata assignment.
  - `Query Playground`: Interactive single-query testing allowing runtime selection of retrieval strategy, top-k, and temperature.
  - `Experiment Configuration`: Form to launch batch evaluations across defined strategies over selected test datasets.
  - `Results Dashboard`: Visualizes comparative metrics and item-level failure inspections.
- **Boundaries:** Completely decoupled from RAG and LLM logic; communicates strictly via asynchronous JSON REST APIs.

### 2.2 Backend (Python + FastAPI + Pydantic)
- **Role:** Central application server coordinating validation, document processing, RAG orchestration, and evaluation runs.
- **Key Modules:**
  - `API Routes & Middleware`: CORS, error handling, rate limiting, and endpoint routing.
  - `Pydantic Models`: Request validation schemas, configuration data classes, and serialized metric responses.
  - `Ingestion Service`: Coordinates document parsing, chunking, embedding generation, and vector indexing.
  - `Query Service`: Executes single-turn RAG queries across configurable retrieval strategies.
  - `Evaluation & Experiment Orchestrator`: Manages batch benchmark execution, metrics calculation, and results aggregation.

### 2.3 Document Ingestion Pipeline
- **Role:** Transforms unstructured documents into searchable vector and lexical representations.
- **Steps:**
  1. *Parsing:* Extracts text and structural metadata from PDF, TXT, or Markdown documents.
  2. *Cleaning:* Normalizes whitespace, strips extraction artifacts, and preserves headings.
  3. *Chunking:* Implements configurable chunking strategies (fixed-window with overlap vs. semantic boundary chunking).
  4. *Embedding Generation:* Computes dense embeddings using configurable models (e.g., local Sentence-Transformers).
  5. *Indexing:* Writes chunk embeddings and metadata to ChromaDB and constructs lexical indices for BM25.

### 2.4 Storage & Persistence
- **Chroma Vector Store:** Embedded vector database storing chunk embeddings, chunk text, document identifiers, and chunk index metadata.
- **Uploaded Documents:** Local directory (`data/uploads/`) holding raw source files.
- **Golden Evaluation Dataset (`data/golden/`):** Fixed set of curated evaluation queries, reference contexts, and ground-truth answers for development iteration.
- **Holdout Test Dataset (`data/holdout/`):** Separate validation set reserved for unbiased evaluation and regression checks.
- **Optional PostgreSQL:** Relational store reserved for enterprise experiment persistence, multi-user runs, and historical audit logs.

### 2.5 Configurable RAG Retrieval
Supports four distinct strategies tested under identical conditions:
- **Strategy A (Basic Vector Search):** Standard cosine-similarity dense retrieval over fixed chunks.
- **Strategy B (Semantic Chunking + Retrieval):** Dense retrieval over chunks determined by semantic breakpoint distances.
- **Strategy C (Hybrid Vector + BM25 Search):** Reciprocal Rank Fusion (RRF) combining dense semantic vectors and sparse lexical BM25 scores.
- **Strategy D (Hybrid Retrieval + Reranking):** Two-stage retrieval retrieving candidate chunks via hybrid search followed by a cross-encoder reranker.
- **Context Assembly:** De-duplicates chunks, formats context blocks with source provenance, and bounds context by token budget.

### 2.6 Answer Generation
- **Prompt Construction:** Strict grounding prompt instructing the model to answer exclusively from retrieved context and cite source chunks.
- **NVIDIA-hosted Nemotron API:** External model endpoint queried via an OpenAI-compatible REST client.
- **Answer Synthesis:** Produces grounded answers with explicit chunk citations, tracking latency and token usage.

### 2.7 Evaluation Engine
- **Orchestration:** Executes the exact same query set across all active retrieval configurations.
- **Retrieval Metrics (Deterministic):**
  - *Hit Rate@k:* Proportion of queries where at least one ground-truth context is retrieved in top-k.
  - *Recall@k:* Proportion of all relevant ground-truth contexts retrieved in top-k.
  - *Mean Reciprocal Rank (MRR):* Rank position of the first relevant chunk.
- **Answer-Quality Metrics (RAGAS + LLM Judge):**
  - *Faithfulness:* Groundedness of the generated answer against retrieved context.
  - *Answer Relevance:* Semantic relevance of the answer to the user's question.
  - *Context Precision:* Signal-to-noise ratio of retrieved chunks.
- **Failure Analysis:** Categorizes failures into *Retrieval Failure* (relevant context missed), *Context Overflow* (irrelevant context distracting model), or *Generation Hallucination* (context present but answer incorrect).

### 2.8 Quality & Delivery
- **Testing:** Unit tests for chunking and metrics via Pytest; component tests for frontend.
- **GitHub Actions:** Runs linting, tests, and automated evaluation regression thresholds on pull requests.
- **Deployment Targets:** Vercel for frontend SPA hosting; Render/Railway for backend containerized hosting with attached storage volume.

---

## 3. End-to-End Data & Execution Flows

### 3.1 Document Ingestion Flow
1. User uploads a document via the UI.
2. Ingestion service writes the raw file to `data/uploads/`.
3. Ingestion pipeline extracts text, cleans formatting, applies selected chunking strategy, generates embeddings, and persists chunks and vectors to ChromaDB.
4. Ingestion summary (chunk count, token statistics) returns to the UI.

### 3.2 Evaluation & Experiment Flow
1. User selects benchmark dataset (Golden or Holdout) and chooses retrieval configurations to compare (e.g., Strategy A vs. Strategy C vs. Strategy D).
2. Evaluation Orchestrator loads questions and ground-truth contexts.
3. For each configuration:
   - Queries are dispatched through the retrieval pipeline.
   - Deterministic retrieval metrics (Hit Rate, Recall@k, MRR) are calculated against ground-truth contexts.
   - Retrieved contexts and questions are passed to Nemotron for answer generation.
   - Generated answers are evaluated via RAGAS metrics for faithfulness and relevance.
   - Latency per stage is captured.
4. Experiment Aggregator compiles comparative tables and failure logs.
5. Frontend dashboard renders side-by-side performance charts and highlights failure cases.

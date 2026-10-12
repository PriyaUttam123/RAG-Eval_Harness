# Architectural Decision Records (ADR)

This document tracks the foundational architectural decisions for the **RAG Evaluation Harness**. All early decisions are **provisional** and subject to empirical review as the harness matures through successive phases.

---

## ADR-001: Frontend Framework — React with TypeScript and Vite

- **Status:** Accepted (Provisional)
- **Context:** The application requires an interactive evaluation dashboard, query playground, experiment configurator, and dynamic metrics/charts visualization.
- **Decision:** Use **React** with **TypeScript** and **Vite** as the build tool, styled with **Tailwind CSS**, and **Recharts** for metric visualization.
- **Rationale:**
  - Rapid hot-module replacement (HMR) via Vite improves developer feedback loops.
  - Strict typing via TypeScript reduces runtime API integration bugs when handling multi-metric evaluation schemas.
  - Recharts provides declarative, responsive charting for retrieval metrics (Hit Rate, Recall@k, MRR) and score distributions.
- **Alternatives Considered:**
  - *Next.js:* Offers SSR/SSG, but adds unnecessary complexity and hosting requirements for an internal benchmarking harness where client-side rendering (SPA) is sufficient.
  - *Streamlit / Gradio:* Enables rapid Python prototyping, but lacks fine-grained UI controls, modular state handling for complex side-by-side experiment comparisons, and responsive component isolation.
- **Trade-offs:** Requires maintaining separate frontend and backend toolchains, requiring CORS configuration during local development.

---

## ADR-002: Backend Framework — Python with FastAPI and Pydantic

- **Status:** Implemented (Phase 1 — Foundation)
- **Context:** The backend must handle file ingestion, vector chunking, retrieval orchestration, LLM API communication, and metric computation.
- **Decision:** Build the backend using **Python 3.11+** with **FastAPI** and **Pydantic v2**.
- **Rationale:**
  - Python is the primary ecosystem for modern AI/RAG tooling (LangChain, Chroma, RAGAS, Hugging Face).
  - FastAPI provides native asynchronous request processing (`async`/`await`), auto-generated OpenAPI documentation, and strict runtime request/response validation through Pydantic.
  - High performance for async I/O operations when dispatching concurrent requests to external LLM APIs.
- **Alternatives Considered:**
  - *Flask:* Simpler, but lacks native async handling and automated OpenAPI schema validation out of the box.
  - *Node.js / Express:* Strong async performance, but lacks native integration with leading RAG evaluation and NLP libraries (RAGAS, semantic chunking toolkits).
- **Trade-offs:** GIL limitations for CPU-bound tasks like local embeddings; mitigated by offloading heavy chunking/embeddings to worker threads or optimized C-extensions (e.g., ONNX, PyTorch).

---

## ADR-003: Vector Store — ChromaDB

- **Status:** Accepted (Provisional)
- **Context:** The harness requires vector storage to store chunk embeddings, persist document collections locally, and allow rapid reconfiguration across experiments.
- **Decision:** Use **Chroma** as the initial embedded vector store.
- **Rationale:**
  - Zero-friction local setup: Runs embedded inside Python without requiring dedicated Docker containers or cloud infrastructure during early development.
  - Native metadata filtering and persistence to disk.
  - Straightforward migration path to client/server mode if workload grows.
- **Alternatives Considered:**
  - *Qdrant / Milvus / Weaviate:* Enterprise-grade performance and filtering, but requires separate service deployment and container orchestration overhead during early phases.
  - *pgvector (PostgreSQL):* Combines relational and vector storage, but increases initial local setup complexity before retrieval baseline requirements are finalized.
- **Trade-offs:** Embedded SQLite-backed Chroma can encounter concurrency constraints during heavy multi-threaded bulk inserts; acceptable for benchmark test sets.

---

## ADR-004: Answer Generation LLM — NVIDIA-Hosted Nemotron API

- **Status:** Accepted (Provisional)
- **Context:** Answer generation requires an enterprise-grade instruction-tuned model accessible via a standard API with predictable rate limits.
- **Decision:** Integrate the **NVIDIA-hosted Nemotron API** (e.g., `nvidia/nemotron-4-340b-instruct`) using its OpenAI-compatible REST endpoint.
- **Rationale:**
  - NVIDIA provides high-throughput hosted endpoints with OpenAI client compatibility, minimizing custom SDK lock-in.
  - Nemotron models have strong synthetic data generation and reasoning capabilities suitable for synthetic evaluation and question answering.
- **Alternatives Considered:**
  - *OpenAI GPT-4o / GPT-3.5:* High capability, but higher ongoing token costs for bulk multi-configuration evaluation runs.
  - *Local Ollama / vLLM:* Zero API cost, but introduces steep GPU hardware requirements for local developer machines.
- **Trade-offs:** Network dependency on NVIDIA's hosted endpoints and susceptibility to rate limits during large batch evaluations. Requires robust retry/backoff logic.

---

## ADR-005: Evaluation Framework — RAGAS with Custom Retrieval Metrics

- **Status:** Accepted (Provisional)
- **Context:** Evaluating RAG requires measuring both retrieval quality (did we find the right context?) and generation quality (is the answer grounded and helpful?).
- **Decision:** Combine **RAGAS** (Retrieval Augmented Generation Assessment) for generation quality with **custom deterministic metrics** for retrieval quality.
- **Rationale:**
  - *Retrieval Metrics:* Deterministic metrics (Hit Rate@k, Recall@k, Mean Reciprocal Rank - MRR) provide objective, reproducible benchmarks independent of LLM variance.
  - *Generation Metrics:* RAGAS measures faithfulness, answer relevance, and context precision using LLM-as-a-judge patterns.
  - Decoupling retrieval evaluation from generation evaluation allows diagnosing whether failures originate in retrieval or generation.
- **Alternatives Considered:**
  - *Pure LLM-as-a-Judge without retrieval metrics:* Subjective and costly; fails to identify whether hallucinations stem from missing context or generation drift.
  - *TruLens / DeepEval:* Comprehensive frameworks, but RAGAS provides a lightweight, widely adopted baseline that integrates well with custom datasets.
- **Trade-offs:** LLM-as-a-judge metrics incur API token costs and non-deterministic variability; requires fixed seed/temperature and prompt stability.

---

## ADR-006: Evaluation Dataset Strategy — Golden Dataset vs. Separate Holdout Dataset

- **Status:** Accepted (Provisional)
- **Context:** A common failure mode in RAG tuning is overfitting retrieval parameters (chunk size, top-k, similarity thresholds) to a small, familiar set of queries.
- **Decision:** Maintain two distinct evaluation splits:
  1. **Golden Evaluation Dataset:** Curated queries with verified ground-truth contexts and reference answers used during iterative tuning and prompt engineering.
  2. **Holdout Test Dataset:** Unseen queries with ground-truth labels reserved for regression testing and validation of finalized configurations.
- **Rationale:**
  - Prevents data leakage and hyperparameter overfitting to specific test queries.
  - Mirrors standard machine learning evaluation practices.
- **Alternatives Considered:**
  - *Single evaluation dataset:* Faster to assemble, but provides no safeguard against over-optimizing chunk boundaries or top-k parameters to known questions.
- **Trade-offs:** Requires additional effort to curate or synthetically generate two distinct high-quality question-context pairs.

---

## ADR-007: Retrieval Configurations — Multi-Strategy Comparable Experiments

- **Status:** Accepted (Provisional)
- **Context:** The core differentiator of this harness is testing hypotheses on how retrieval architecture affects downstream quality.
- **Decision:** Support four pluggable retrieval strategies:
  - **Strategy A:** Basic Vector Search (standard dense retrieval).
  - **Strategy B:** Semantic Chunking + Dense Retrieval (dynamic breakpoint chunking based on semantic distance).
  - **Strategy C:** Hybrid Search (Dense Vector + Sparse BM25 via Reciprocal Rank Fusion).
  - **Strategy D:** Hybrid Search with Cross-Encoder Reranking (Retrieve top-N candidates, rerank to top-K).
- **Rationale:**
  - Enables direct, apples-to-apples performance comparisons across identical evaluation datasets.
  - Demonstrates practical engineering trade-offs between latency, compute cost, and retrieval accuracy.
- **Alternatives Considered:**
  - *Single fixed retrieval pipeline:* Simpler to implement, but fails to address the evaluation purpose of the harness.
- **Trade-offs:** Higher engineering complexity to maintain multiple indexing and search paths.

---

## ADR-008: Experiment History Persistence — Optional PostgreSQL

- **Status:** Accepted (Provisional)
- **Context:** Users need to track historical experiment runs across configurations and compare metrics over time.
- **Decision:** Use file-based JSON/SQLite storage for MVP experiment runs, with an **optional PostgreSQL schema** for persistent history and multi-user tracking.
- **Rationale:**
  - Keeps local setup friction low for Phase 0 and Phase 1.
  - Provides a clean abstraction boundary (Repository pattern) allowing seamless connection to PostgreSQL when running in production or team environments.
- **Alternatives Considered:**
  - *Mandatory PostgreSQL from Day 1:* Adds container dependencies and setup friction before core pipelines are proven.
- **Trade-offs:** Requires an abstraction layer to support both file-based and SQL-backed repositories.

---

## ADR-009: Deployment Targets — Vercel (Frontend) and Render/Railway (Backend)

- **Status:** Accepted (Provisional)
- **Context:** The application must eventually be deployable to cloud infrastructure with low maintenance.
- **Decision:** Target **Vercel** for the Vite/React static SPA frontend, and **Render or Railway** for the FastAPI backend and persistent storage.
- **Rationale:**
  - Vercel provides seamless global CDN distribution, preview deployments, and zero-config Vite hosting.
  - Render/Railway provides straightforward containerized Python hosting with persistent disk support (for Chroma and local datasets) without full Kubernetes complexity.
- **Alternatives Considered:**
  - *Single monolithic container on AWS ECS / DigitalOcean:* Viable, but higher operational overhead.
- **Trade-offs:** Requires managing CORS and environment variable synchronization across two distinct hosting providers.

---

## ADR-010: Backend Structure, Pydantic Settings, and Health Endpoint Scope

- **Status:** Implemented (Phase 1)
- **Context:** The backend foundation requires a clean, testable entry point, modular settings management, CORS security, and a health endpoint prior to building RAG pipelines.
- **Decision:**
  - Use an application factory (`create_app()`) in `backend/app/main.py`.
  - Encapsulate configuration in `backend/app/config.py` using `pydantic-settings` (`BaseSettings`), supporting both comma-separated strings and JSON arrays for origins with a cached `get_settings()` singleton.
  - Reject wildcard (`*`) CORS origins when `environment == "production"`.
  - Scope `GET /health` to verifying that the FastAPI web service itself is operational (`{"status": "ok", "environment": ..., "version": ...}`). Do not report downstream components (databases, vector stores, model APIs) as healthy before they are implemented and probed.
- **Rationale:**
  - Application factory prevents import side-effects and enables parameterized testing via `TestClient`.
  - Explicit CORS configuration protects against unrestricted cross-origin access in deployed environments while easing local Vite development (`localhost:5173`).
  - Truthful health reporting maintains observability integrity and avoids misleading monitoring checks.
- **Alternatives Considered:**
  - *Monolithic single-file app:* Fast for prototyping, but mixes route handlers, config, and middleware, making test isolation harder.
  - *Reporting databases/models as healthy:* Misleading since those components do not yet exist.
- **Trade-offs:** Introduces separate modules (`config.py`, `main.py`, `health.py`), requiring import resolution when running uvicorn from different working directories.

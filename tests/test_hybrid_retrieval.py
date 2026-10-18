"""Tests for HybridRetriever reciprocal rank fusion."""

import pytest

from backend.app.bm25_store import BM25Result
from backend.app.hybrid_retrieval import HybridResult, HybridRetriever


class FakeEmbeddingService:
    def __init__(self, embedding=None):
        self.embedding = embedding or [0.1, 0.2, 0.3]
        self.received_query = None

    def embed_text(self, text: str):
        self.received_query = text
        return self.embedding


class FakeVectorStore:
    def __init__(self, ids=None, documents=None, metadatas=None, distances=None):
        self.ids = ids if ids is not None else [[]]
        self.documents = documents if documents is not None else [[]]
        self.metadatas = metadatas if metadatas is not None else [[]]
        self.distances = distances if distances is not None else [[]]
        self.received_embedding = None
        self.received_top_k = None

    def search(self, query_embedding, top_k=None):
        self.received_embedding = query_embedding
        self.received_top_k = top_k
        return {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "distances": self.distances,
        }


class FakeBM25Store:
    def __init__(self, results=None):
        self.results = results if results is not None else []
        self.received_query = None
        self.received_top_k = None

    def search(self, query: str, top_k: int):
        self.received_query = query
        self.received_top_k = top_k
        return self.results[:top_k]


def test_hybrid_retrieval_equal_weighted_fusion():
    vec_store = FakeVectorStore(
        ids=[["chunk-A", "chunk-B"]],
        documents=[["Text A", "Text B"]],
        metadatas=[[{"source": "vecA"}, {"source": "vecB"}]],
    )
    bm25_store = FakeBM25Store(
        results=[
            BM25Result(chunk_id="chunk-B", text="Text B", metadata={"source": "bm25B"}, score=2.5),
            BM25Result(chunk_id="chunk-C", text="Text C", metadata={"source": "bm25C"}, score=1.8),
        ]
    )
    embed_svc = FakeEmbeddingService()

    retriever = HybridRetriever(
        vector_store=vec_store,
        bm25_store=bm25_store,
        embedding_service=embed_svc,
    )

    results = retriever.retrieve("query", top_k=5, vector_weight=0.5, bm25_weight=0.5)

    # Weights normalized: vec_w = 0.5, bm25_w = 0.5
    # chunk-A: vector rank 0 -> score = 0.5 / (0 + 1) = 0.5
    # chunk-B: vector rank 1 -> 0.5 / (1 + 1) = 0.25; bm25 rank 0 -> 0.5 / (0 + 1) = 0.5 => total = 0.75
    # chunk-C: bm25 rank 1 -> 0.5 / (1 + 1) = 0.25
    assert len(results) == 3
    assert results[0].chunk_id == "chunk-B"
    assert pytest.approx(results[0].score, rel=1e-5) == 0.75
    assert results[0].vector_rank == 1
    assert results[0].bm25_rank == 0

    assert results[1].chunk_id == "chunk-A"
    assert pytest.approx(results[1].score, rel=1e-5) == 0.5
    assert results[1].vector_rank == 0
    assert results[1].bm25_rank is None

    assert results[2].chunk_id == "chunk-C"
    assert pytest.approx(results[2].score, rel=1e-5) == 0.25
    assert results[2].vector_rank is None
    assert results[2].bm25_rank == 1


def test_hybrid_retrieval_vector_only_contribution():
    vec_store = FakeVectorStore(
        ids=[["chunk-V1", "chunk-V2"]],
        documents=[["Doc 1", "Doc 2"]],
        metadatas=[[{}, {}]],
    )
    bm25_store = FakeBM25Store(
        results=[
            BM25Result(chunk_id="chunk-B1", text="Doc B", score=10.0),
        ]
    )
    retriever = HybridRetriever(
        vector_store=vec_store,
        bm25_store=bm25_store,
        embedding_service=FakeEmbeddingService(),
    )

    # vector_weight = 1.0, bm25_weight = 0.0
    results = retriever.retrieve("query", top_k=3, vector_weight=1.0, bm25_weight=0.0)

    # BM25 weight is 0, so chunk-B1 gets score = 0.0 / (0 + 1) = 0.0
    assert results[0].chunk_id == "chunk-V1"
    assert pytest.approx(results[0].score, rel=1e-5) == 1.0
    assert results[1].chunk_id == "chunk-V2"
    assert pytest.approx(results[1].score, rel=1e-5) == 0.5


def test_hybrid_retrieval_bm25_only_contribution():
    vec_store = FakeVectorStore(
        ids=[["chunk-V1"]],
        documents=[["Doc 1"]],
        metadatas=[[{}]],
    )
    bm25_store = FakeBM25Store(
        results=[
            BM25Result(chunk_id="chunk-B1", text="Doc B1", score=10.0),
            BM25Result(chunk_id="chunk-B2", text="Doc B2", score=5.0),
        ]
    )
    retriever = HybridRetriever(
        vector_store=vec_store,
        bm25_store=bm25_store,
        embedding_service=FakeEmbeddingService(),
    )

    # vector_weight = 0.0, bm25_weight = 1.0
    results = retriever.retrieve("query", top_k=3, vector_weight=0.0, bm25_weight=1.0)

    assert results[0].chunk_id == "chunk-B1"
    assert pytest.approx(results[0].score, rel=1e-5) == 1.0
    assert results[1].chunk_id == "chunk-B2"
    assert pytest.approx(results[1].score, rel=1e-5) == 0.5


def test_hybrid_retrieval_custom_weights_and_normalization():
    vec_store = FakeVectorStore(
        ids=[["chunk-1"]],
        documents=[["Text 1"]],
        metadatas=[[{}]],
    )
    bm25_store = FakeBM25Store(
        results=[
            BM25Result(chunk_id="chunk-2", text="Text 2", score=5.0),
        ]
    )
    retriever = HybridRetriever(
        vector_store=vec_store,
        bm25_store=bm25_store,
        embedding_service=FakeEmbeddingService(),
    )

    # Weights 3.0 and 1.0 should normalize to 0.75 and 0.25
    results = retriever.retrieve("query", top_k=2, vector_weight=3.0, bm25_weight=1.0)

    assert results[0].chunk_id == "chunk-1"
    assert pytest.approx(results[0].score, rel=1e-5) == 0.75
    assert results[1].chunk_id == "chunk-2"
    assert pytest.approx(results[1].score, rel=1e-5) == 0.25


def test_hybrid_retrieval_zero_total_weight_rejected():
    retriever = HybridRetriever(
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        embedding_service=FakeEmbeddingService(),
    )

    with pytest.raises(ValueError, match="Sum of vector_weight and bm25_weight must be greater than 0"):
        retriever.retrieve("query", top_k=5, vector_weight=0.0, bm25_weight=0.0)


def test_hybrid_retrieval_negative_weight_rejected():
    retriever = HybridRetriever(
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        embedding_service=FakeEmbeddingService(),
    )

    with pytest.raises(ValueError, match="Weights must be non-negative"):
        retriever.retrieve("query", top_k=5, vector_weight=-0.5, bm25_weight=0.5)

    with pytest.raises(ValueError, match="Weights must be non-negative"):
        retriever.retrieve("query", top_k=5, vector_weight=0.5, bm25_weight=-0.5)


def test_hybrid_retrieval_invalid_query():
    retriever = HybridRetriever(
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        embedding_service=FakeEmbeddingService(),
    )

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.retrieve("", top_k=5)

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.retrieve("   ", top_k=5)

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.retrieve(None, top_k=5)  # type: ignore


def test_hybrid_retrieval_invalid_top_k():
    retriever = HybridRetriever(
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        embedding_service=FakeEmbeddingService(),
    )

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("query", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("query", top_k=-1)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("query", top_k=True)  # type: ignore


def test_hybrid_retrieval_deterministic_tie_breaking():
    # Two chunks with identical fusion scores
    vec_store = FakeVectorStore(
        ids=[["chunk-Z", "chunk-A"]],
        documents=[["Doc Z", "Doc A"]],
        metadatas=[[{}, {}]],
    )
    # BM25 is empty, so chunk-Z rank 0 (score=0.5), chunk-A rank 1 (score=0.25)
    # Let's make both have equal scores:
    # chunk-Z has vector rank 0 (score 0.5)
    # chunk-A has BM25 rank 0 (score 0.5)
    vec_store_tie = FakeVectorStore(
        ids=[["chunk-Z"]],
        documents=[["Doc Z"]],
        metadatas=[[{"source": "vec"}]],
    )
    bm25_store_tie = FakeBM25Store(
        results=[
            BM25Result(chunk_id="chunk-A", text="Doc A", metadata={"source": "bm25"}, score=1.0)
        ]
    )

    retriever = HybridRetriever(
        vector_store=vec_store_tie,
        bm25_store=bm25_store_tie,
        embedding_service=FakeEmbeddingService(),
    )

    # Both get score 0.5. Alphabetical tie-breaking puts chunk-A before chunk-Z
    results = retriever.retrieve("query", top_k=2, vector_weight=0.5, bm25_weight=0.5)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk-A"
    assert results[1].chunk_id == "chunk-Z"
    assert results[0].score == results[1].score


def test_hybrid_retrieval_top_k_limit():
    vec_store = FakeVectorStore(
        ids=[[f"chunk-{i}" for i in range(10)]],
        documents=[[f"Doc {i}" for i in range(10)]],
        metadatas=[[{} for _ in range(10)]],
    )
    retriever = HybridRetriever(
        vector_store=vec_store,
        bm25_store=FakeBM25Store(),
        embedding_service=FakeEmbeddingService(),
    )

    results = retriever.retrieve("query", top_k=3)
    assert len(results) == 3


def test_hybrid_retrieval_empty_sources():
    retriever = HybridRetriever(
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        embedding_service=FakeEmbeddingService(),
    )
    results = retriever.retrieve("query", top_k=5)
    assert results == []


def test_hybrid_retrieval_metadata_preservation():
    vec_store = FakeVectorStore(
        ids=[["chunk-1"]],
        documents=[["Vector document content"]],
        metadatas=[[{"author": "Alice", "tags": ["rag", "ai"]}]],
    )
    retriever = HybridRetriever(
        vector_store=vec_store,
        bm25_store=FakeBM25Store(),
        embedding_service=FakeEmbeddingService(),
    )

    results = retriever.retrieve("query", top_k=1)
    assert len(results) == 1
    assert results[0].text == "Vector document content"
    assert results[0].metadata["author"] == "Alice"
    assert results[0].metadata["tags"] == ["rag", "ai"]

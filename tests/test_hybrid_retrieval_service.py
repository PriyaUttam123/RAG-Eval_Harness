"""Tests for HybridRetrievalService managing dual indexing and retrieval delegation."""

import pytest

from backend.app.chunker import Chunk
from backend.app.experiment_config import ExperimentConfig
from backend.app.hybrid_retrieval import HybridResult
from backend.app.hybrid_retrieval_service import HybridRetrievalService


class FakeEmbeddingService:
    def __init__(self):
        self.received_query = None

    def embed_text(self, text: str):
        self.received_query = text
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self):
        self.added_chunks = None
        self.added_embeddings = None
        self.assigned_ids = ["v-chunk-1", "v-chunk-2"]

    def add_chunks(self, chunks, embeddings):
        self.added_chunks = chunks
        self.added_embeddings = embeddings
        return self.assigned_ids


class FakeBM25Store:
    def __init__(self):
        self.added_chunks = None
        self.added_chunk_ids = None

    def add_chunks(self, chunks, chunk_ids=None):
        self.added_chunks = chunks
        self.added_chunk_ids = chunk_ids
        return chunk_ids or ["b-chunk-1"]


class FakeHybridRetriever:
    def __init__(self):
        self.called_query = None
        self.called_top_k = None
        self.called_vector_weight = None
        self.called_bm25_weight = None
        self.return_results = [
            HybridResult(
                chunk_id="v-chunk-1",
                text="Delegated result text",
                metadata={"key": "val"},
                score=0.85,
                vector_rank=0,
                bm25_rank=1,
            )
        ]

    def retrieve(self, query: str, top_k: int, vector_weight: float = 0.5, bm25_weight: float = 0.5):
        self.called_query = query
        self.called_top_k = top_k
        self.called_vector_weight = vector_weight
        self.called_bm25_weight = bm25_weight
        return self.return_results


def _create_sample_chunks():
    return [
        Chunk(
            text="First chunk of the policy manual.",
            chunk_index=0,
            start_offset=0,
            end_offset=33,
            source_filename="manual.pdf",
            metadata={"chapter": 1},
        ),
        Chunk(
            text="Second chunk on employee guidelines.",
            chunk_index=1,
            start_offset=34,
            end_offset=70,
            source_filename="manual.pdf",
            metadata={"chapter": 2},
        ),
    ]


def test_hybrid_retrieval_service_index_chunks_sends_to_both_stores():
    vec_store = FakeVectorStore()
    bm25_store = FakeBM25Store()
    embed_svc = FakeEmbeddingService()

    service = HybridRetrievalService(
        vector_store=vec_store,
        bm25_store=bm25_store,
        embedding_service=embed_svc,
    )

    chunks = _create_sample_chunks()
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    assigned_ids = service.index_chunks(chunks=chunks, embeddings=embeddings)

    assert assigned_ids == ["v-chunk-1", "v-chunk-2"]
    assert vec_store.added_chunks == chunks
    assert vec_store.added_embeddings == embeddings
    assert bm25_store.added_chunks == chunks
    # Chunk IDs in BM25 must match chunk IDs from VectorStore
    assert bm25_store.added_chunk_ids == ["v-chunk-1", "v-chunk-2"]


def test_hybrid_retrieval_service_retrieve_delegates_to_retriever():
    retriever = FakeHybridRetriever()
    service = HybridRetrievalService(
        retriever=retriever,
    )

    results = service.retrieve(
        query="what are the guidelines?",
        top_k=3,
        vector_weight=0.7,
        bm25_weight=0.3,
    )

    assert retriever.called_query == "what are the guidelines?"
    assert retriever.called_top_k == 3
    assert retriever.called_vector_weight == 0.7
    assert retriever.called_bm25_weight == 0.3
    assert len(results) == 1
    assert results[0].chunk_id == "v-chunk-1"
    assert results[0].score == 0.85


def test_hybrid_retrieval_service_dependency_injection():
    vec_store = FakeVectorStore()
    bm25_store = FakeBM25Store()
    embed_svc = FakeEmbeddingService()
    retriever = FakeHybridRetriever()

    service = HybridRetrievalService(
        vector_store=vec_store,
        bm25_store=bm25_store,
        embedding_service=embed_svc,
        retriever=retriever,
    )

    assert service.vector_store is vec_store
    assert service.bm25_store is bm25_store
    assert service.embedding_service is embed_svc
    assert service.retriever is retriever


def test_hybrid_retrieval_service_compatible_with_experiment_config():
    config = ExperimentConfig(
        name="hybrid-benchmark-v1",
        chunk_size=500,
        chunk_overlap=50,
        retrieval_top_k=5,
        evaluation_k=3,
        retrieval_strategy="hybrid",
        embedding_model="test-embedding-model",
    )

    assert config.retrieval_strategy == "hybrid"

    # Verify service can be used with parameters from ExperimentConfig
    retriever = FakeHybridRetriever()
    service = HybridRetrievalService(retriever=retriever)
    service.retrieve(
        query="test query",
        top_k=config.retrieval_top_k,
    )
    assert retriever.called_top_k == 5

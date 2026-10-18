"""Tests for BM25Store lexical search index."""

import pytest

from backend.app.bm25_store import BM25Result, BM25Store
from backend.app.chunker import Chunk


def _create_chunk(text: str, index: int = 0, filename: str = "doc.txt") -> Chunk:
    return Chunk(
        text=text,
        chunk_index=index,
        start_offset=0,
        end_offset=len(text),
        source_filename=filename,
        metadata={"category": "testing", "section": f"sec_{index}"},
    )


def test_bm25_store_init_empty():
    store = BM25Store()
    assert store.chunk_ids == []
    assert store.documents == []
    assert store.metadatas == []
    assert store.search("anything", top_k=5) == []


def test_bm25_store_indexing_and_lexical_matching():
    store = BM25Store()
    chunk1 = _create_chunk("Apples and oranges are delicious fruits.", index=0)
    chunk2 = _create_chunk("Deep learning models require GPUs for training.", index=1)
    chunk3 = _create_chunk("Apples are rich in fiber and vitamins.", index=2)

    ids = store.add_chunks([chunk1, chunk2, chunk3], chunk_ids=["c1", "c2", "c3"])
    assert ids == ["c1", "c2", "c3"]

    results = store.search("apples", top_k=2)
    assert len(results) == 2
    # chunk1 and chunk3 contain "apples"
    matched_ids = [r.chunk_id for r in results]
    assert "c1" in matched_ids
    assert "c3" in matched_ids
    assert "c2" not in matched_ids
    assert all(r.score > 0 for r in results)


def test_bm25_store_ranking_order():
    store = BM25Store()
    # chunk2 has higher frequency of "refund"
    chunk1 = _create_chunk("The product is good and durable.", index=0)
    chunk2 = _create_chunk("Refund policy: To request a refund, contact refund support for a full refund.", index=1)
    chunk3 = _create_chunk("You can request a refund within 30 days.", index=2)

    store.add_chunks([chunk1, chunk2, chunk3], chunk_ids=["c1", "c2", "c3"])

    results = store.search("refund", top_k=3)
    assert len(results) == 2
    # c2 mentions refund multiple times so it should rank first
    assert results[0].chunk_id == "c2"
    assert results[1].chunk_id == "c3"
    assert results[0].score >= results[1].score


def test_bm25_store_metadata_and_text_preservation():
    store = BM25Store()
    chunk = _create_chunk(
        "Preserve my metadata perfectly.",
        index=4,
        filename="custom_source.md",
    )
    store.add_chunks([chunk], chunk_ids=["custom-id-42"])

    results = store.search("preserve", top_k=1)
    assert len(results) == 1
    res = results[0]
    assert res.chunk_id == "custom-id-42"
    assert res.text == "Preserve my metadata perfectly."
    assert res.metadata["category"] == "testing"
    assert res.metadata["section"] == "sec_4"
    assert res.metadata["source_filename"] == "custom_source.md"
    assert res.metadata["chunk_index"] == 4
    assert res.metadata["start_offset"] == 0
    assert res.metadata["end_offset"] == len("Preserve my metadata perfectly.")


def test_bm25_store_auto_generated_ids():
    store = BM25Store()
    chunk = _create_chunk("Auto ID generation test.")
    ids = store.add_chunks([chunk])
    assert len(ids) == 1
    assert isinstance(ids[0], str) and len(ids[0]) > 0

    results = store.search("test", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == ids[0]


def test_bm25_store_empty_query_rejection():
    store = BM25Store()
    chunk = _create_chunk("Some valid content.")
    store.add_chunks([chunk], chunk_ids=["c1"])

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        store.search("", top_k=3)

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        store.search("   ", top_k=3)

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        store.search(None, top_k=3)  # type: ignore


def test_bm25_store_invalid_top_k():
    store = BM25Store()
    chunk = _create_chunk("Valid content.")
    store.add_chunks([chunk], chunk_ids=["c1"])

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        store.search("valid", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        store.search("valid", top_k=-1)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        store.search("valid", top_k=True)  # type: ignore (bool check)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        store.search("valid", top_k="2")  # type: ignore


def test_bm25_store_empty_index_returns_empty_list():
    store = BM25Store()
    results = store.search("query", top_k=5)
    assert results == []


def test_bm25_store_query_with_no_tokens():
    store = BM25Store()
    chunk = _create_chunk("Valid content.")
    store.add_chunks([chunk], chunk_ids=["c1"])

    # Query with only non-alphanumeric punctuation
    results = store.search("??? !!!", top_k=5)
    assert results == []


def test_bm25_store_deterministic_results():
    store = BM25Store()
    chunks = [
        _create_chunk(f"Document segment number {i} discussing machine learning algorithms.", index=i)
        for i in range(10)
    ]
    ids = [f"doc_{i}" for i in range(10)]
    store.add_chunks(chunks, chunk_ids=ids)

    run1 = store.search("machine learning", top_k=5)
    run2 = store.search("machine learning", top_k=5)

    assert [r.chunk_id for r in run1] == [r.chunk_id for r in run2]
    assert [r.score for r in run1] == [r.score for r in run2]


def test_bm25_store_set_chunks_replaces_index():
    store = BM25Store()
    old_chunk = _create_chunk("Old topic about solar energy systems.", index=0)
    store.add_chunks([old_chunk], chunk_ids=["old_1"])

    # Old chunk matches
    assert len(store.search("solar", top_k=1)) == 1

    # Replace with new chunks
    new_chunk = _create_chunk("New topic regarding quantum computing.", index=0)
    store.set_chunks([new_chunk], chunk_ids=["new_1"])

    assert len(store.search("solar", top_k=1)) == 0
    new_res = store.search("quantum", top_k=1)
    assert len(new_res) == 1
    assert new_res[0].chunk_id == "new_1"


def test_bm25_store_clear():
    store = BM25Store()
    chunk = _create_chunk("Sample text.", index=0)
    store.add_chunks([chunk], chunk_ids=["c1"])
    assert len(store.chunk_ids) == 1

    store.clear()
    assert len(store.chunk_ids) == 0
    assert store.search("sample", top_k=1) == []


def test_bm25_store_add_chunks_validation():
    store = BM25Store()
    with pytest.raises(ValueError, match="Chunks list must not be empty"):
        store.add_chunks([])

    chunk = _create_chunk("Text")
    with pytest.raises(ValueError, match="Each chunk must have exactly one chunk ID"):
        store.add_chunks([chunk], chunk_ids=["id1", "id2"])

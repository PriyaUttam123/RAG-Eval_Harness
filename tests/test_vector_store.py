
import pytest

from backend.app.chunker import Chunk
from backend.app.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    return VectorStore(persist_directory=str(tmp_path / "chroma_test"))


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_index=0,
            source_filename="guide.txt",
            text="RAG combines retrieval with generation.",
            start_offset=0,
            end_offset=38,
            metadata={},
        ),
        Chunk(
            chunk_index=1,
            source_filename="guide.txt",
            text="Embeddings represent text as numerical vectors.",
            start_offset=39,
            end_offset=87,
            metadata={},
        ),
    ]



@pytest.fixture
def sample_embeddings():
    first = [1.0] + [0.0] * 383
    second = [0.0, 1.0] + [0.0] * 382
    return [first, second]


def test_add_chunks_returns_ids(
    vector_store, sample_chunks, sample_embeddings
):
    ids = vector_store.add_chunks(sample_chunks, sample_embeddings)

    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert vector_store.collection.count() == 2


def test_search_returns_matching_chunks(
    vector_store, sample_chunks, sample_embeddings
):
    vector_store.add_chunks(sample_chunks, sample_embeddings)

    results = vector_store.search(sample_embeddings[0], top_k=1)

    assert len(results["ids"][0]) == 1
    assert results["documents"][0][0] == sample_chunks[0].text
    assert results["metadatas"][0][0]["source_filename"] == "guide.txt"


def test_add_chunks_rejects_empty_list(vector_store):
    with pytest.raises(ValueError, match="must not be empty"):
        vector_store.add_chunks([], [])


def test_add_chunks_rejects_mismatched_counts(
    vector_store, sample_chunks, sample_embeddings
):
    with pytest.raises(ValueError, match="exactly one embedding"):
        vector_store.add_chunks(sample_chunks, sample_embeddings[:1])


def test_add_chunks_rejects_wrong_embedding_dimension(
    vector_store, sample_chunks
):
    with pytest.raises(ValueError, match="dimensions"):
        vector_store.add_chunks(sample_chunks, [[0.1] * 10, [0.2] * 384])


def test_search_rejects_wrong_embedding_dimension(vector_store):
    with pytest.raises(ValueError, match="dimensions"):
        vector_store.search([0.1] * 10)


def test_search_rejects_non_positive_top_k(vector_store):
    with pytest.raises(ValueError, match="positive"):
        vector_store.search([0.1] * 384, top_k=0)


def test_search_empty_collection_returns_empty_results(vector_store):
    results = vector_store.search([0.1] * 384)

    assert results["ids"] == [[]]
    assert results["documents"] == [[]]
    assert results["metadatas"] == [[]]
def test_add_chunks_rejects_non_finite_embedding(
    vector_store, sample_chunks, sample_embeddings
):
    invalid_embedding = [float("nan")] + [0.0] * 383
    embeddings = [invalid_embedding, sample_embeddings[1]]

    with pytest.raises(ValueError, match="finite"):
        vector_store.add_chunks(sample_chunks, embeddings)


def test_search_rejects_non_finite_embedding(vector_store):
    invalid_embedding = [float("inf")] + [0.0] * 383

    with pytest.raises(ValueError, match="finite"):
        vector_store.search(invalid_embedding)
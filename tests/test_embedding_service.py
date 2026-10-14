
import numpy as np
import pytest

from backend.app.embedding_service import EmbeddingService


class FakeEmbeddingModel:
    """Fake model that returns predictable 384-dimensional vectors."""

    def encode(self, texts, convert_to_numpy=True):
        return np.ones((len(texts), 384), dtype=float)


@pytest.fixture
def service(monkeypatch):
    instance = EmbeddingService()
    instance.settings.embedding_dimension = 384
    instance._model = FakeEmbeddingModel()
    return instance


def test_embed_single_text_returns_vector(service):
    vector = service.embed_text("What is RAG?")

    assert len(vector) == 384
    assert all(value == 1.0 for value in vector)


def test_embed_multiple_texts_returns_correct_count(service):
    vectors = service.embed_texts(["What is RAG?", "What is an embedding?"])

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)


def test_empty_text_is_rejected(service):
    with pytest.raises(ValueError, match="non-empty string"):
        service.embed_text("   ")


def test_empty_batch_is_rejected(service):
    with pytest.raises(ValueError, match="batch must not be empty"):
        service.embed_texts([])


def test_batch_with_empty_text_is_rejected(service):
    with pytest.raises(ValueError, match="Every text"):
        service.embed_texts(["valid text", "   "])


def test_wrong_embedding_dimension_is_rejected(service):
    class WrongDimensionModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.ones((len(texts), 100), dtype=float)

    service._model = WrongDimensionModel()

    with pytest.raises(ValueError, match="Expected 384 dimensions"):
        service.embed_text("Test text")


def test_non_finite_embedding_values_are_rejected(service):
    class InvalidValueModel:
        def encode(self, texts, convert_to_numpy=True):
            vectors = np.ones((len(texts), 384), dtype=float)
            vectors[0, 0] = np.nan
            return vectors

    service._model = InvalidValueModel()

    with pytest.raises(ValueError, match="non-finite"):
        service.embed_text("Test text")


def test_model_is_loaded_lazily():
    service = EmbeddingService()

    assert service._model is None


def test_model_is_reused(service):
    original_model = service._model

    service.embed_text("First text")
    service.embed_text("Second text")

    assert service._model is original_model

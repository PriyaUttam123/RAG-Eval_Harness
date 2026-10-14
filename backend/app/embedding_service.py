
from math import isfinite
from typing import Any, List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from backend.app.config import get_settings
except ImportError:
    from app.config import get_settings


class EmbeddingService:
    """Generate text embeddings using a local Sentence Transformers model."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any = None

    def _get_model(self) -> Any:
        """Load the model only when embeddings are first requested."""
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install project dependencies to generate embeddings."
                )

            self._model = SentenceTransformer(
                self.settings.embedding_model_name
            )

        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Convert one non-empty text into a numerical vector."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Convert multiple texts into numerical vectors."""
        if not texts:
            raise ValueError("Texts batch must not be empty.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError(
                "Every text must be a non-empty, non-whitespace string."
            )

        vectors = self._get_model().encode(
            texts,
            convert_to_numpy=True,
        )

        if len(vectors) != len(texts):
            raise ValueError("Embedding count does not match input count.")

        results: List[List[float]] = []

        for vector in vectors:
            values = [float(value) for value in vector]

            if len(values) != self.settings.embedding_dimension:
                raise ValueError(
                    f"Expected {self.settings.embedding_dimension} dimensions, "
                    f"got {len(values)}."
                )

            if not all(isfinite(value) for value in values):
                raise ValueError("Embedding contains non-finite values.")

            results.append(values)

        return results

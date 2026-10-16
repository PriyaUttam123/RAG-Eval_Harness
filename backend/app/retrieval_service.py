from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from backend.app.embedding_service import EmbeddingService
    from backend.app.vector_store import VectorStore
except ImportError:
    from app.embedding_service import EmbeddingService
    from app.vector_store import VectorStore


class RetrievedChunk(BaseModel):
    """A single chunk returned by similarity search."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    distance: float


class RetrievalResult(BaseModel):
    """Clean retrieval response for a user query."""

    query: str
    results: List[RetrievedChunk]


class RetrievalService:
    """Embed a query and retrieve the most relevant document chunks."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieve the most similar chunks for a non-empty query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        cleaned_query = query.strip()

        query_embedding = self.embedding_service.embed_text(cleaned_query)

        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        results: List[RetrievedChunk] = []

        for index, chunk_id in enumerate(ids):
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=documents[index],
                    metadata=metadatas[index] if metadatas else {},
                    distance=float(distances[index]),
                )
            )

        return RetrievalResult(
            query=cleaned_query,
            results=results,
        )
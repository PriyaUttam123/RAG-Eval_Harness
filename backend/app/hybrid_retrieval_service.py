"""Service coordinating dual indexing into vector and BM25 stores, and hybrid retrieval."""

from typing import List, Optional

try:
    from backend.app.bm25_store import BM25Store
    from backend.app.chunker import Chunk
    from backend.app.embedding_service import EmbeddingService
    from backend.app.hybrid_retrieval import HybridResult, HybridRetriever
    from backend.app.vector_store import VectorStore
except ImportError:
    from app.bm25_store import BM25Store
    from app.chunker import Chunk
    from app.embedding_service import EmbeddingService
    from app.hybrid_retrieval import HybridResult, HybridRetriever
    from app.vector_store import VectorStore


class HybridRetrievalService:
    """Service orchestrating dual indexing (Vector + BM25) and hybrid retrieval."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_store: Optional[BM25Store] = None,
        embedding_service: Optional[EmbeddingService] = None,
        retriever: Optional[HybridRetriever] = None,
    ) -> None:
        """Initialize the hybrid retrieval service with stores and retriever."""
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.embedding_service = embedding_service or EmbeddingService()
        self.retriever = retriever or HybridRetriever(
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
            embedding_service=self.embedding_service,
        )

    def index_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Index chunks and their embeddings into both VectorStore and BM25Store.

        Args:
            chunks: List of Chunk objects to index.
            embeddings: Corresponding embedding vectors.

        Returns:
            List of assigned chunk IDs.
        """
        chunk_ids = self.vector_store.add_chunks(chunks, embeddings)
        self.bm25_store.add_chunks(chunks, chunk_ids=chunk_ids)
        return chunk_ids

    def retrieve(
        self,
        query: str,
        top_k: int,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[HybridResult]:
        """Delegate hybrid retrieval to HybridRetriever.

        Args:
            query: Query string.
            top_k: Number of top results to return.
            vector_weight: Weight for vector similarity ranking.
            bm25_weight: Weight for BM25 lexical ranking.

        Returns:
            Ranked list of HybridResult objects.
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

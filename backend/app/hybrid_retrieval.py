"""Hybrid retrieval combining dense vector similarity with sparse BM25 lexical ranking."""

from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

try:
    from backend.app.bm25_store import BM25Result, BM25Store
    from backend.app.embedding_service import EmbeddingService
    from backend.app.vector_store import VectorStore
except ImportError:
    from app.bm25_store import BM25Result, BM25Store
    from app.embedding_service import EmbeddingService
    from app.vector_store import VectorStore


class HybridResult(BaseModel):
    """Result of hybrid vector and BM25 rank fusion."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


class HybridRetriever:
    """Combines vector search and BM25 search using weighted Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedding_service: EmbeddingService,
    ) -> None:
        """Initialize HybridRetriever with stores and embedding service.

        Args:
            vector_store: VectorStore instance for semantic similarity retrieval.
            bm25_store: BM25Store instance for lexical search.
            embedding_service: EmbeddingService for converting queries to vector space.
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedding_service = embedding_service

    def retrieve(
        self,
        query: str,
        top_k: int,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[HybridResult]:
        """Execute hybrid search using weighted reciprocal rank fusion.

        Args:
            query: The search query string.
            top_k: Maximum number of merged results to return.
            vector_weight: Non-negative weight for semantic vector ranking.
            bm25_weight: Non-negative weight for lexical BM25 ranking.

        Returns:
            Ranked list of HybridResult objects sorted by fused score descending.

        Raises:
            ValueError: If query is empty, top_k <= 0, weights are negative,
                or both weights sum to zero.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if vector_weight < 0.0 or bm25_weight < 0.0:
            raise ValueError("Weights must be non-negative.")

        total_weight = vector_weight + bm25_weight
        if total_weight <= 0.0:
            raise ValueError("Sum of vector_weight and bm25_weight must be greater than 0.")

        norm_vector_weight = vector_weight / total_weight
        norm_bm25_weight = bm25_weight / total_weight

        cleaned_query = query.strip()

        # 1. Vector retrieval
        query_embedding = self.embedding_service.embed_text(cleaned_query)
        vector_raw = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        vector_ids: List[str] = vector_raw.get("ids", [[]])[0]
        vector_docs: List[str] = vector_raw.get("documents", [[]])[0]
        vector_metas: List[Dict[str, Any]] = vector_raw.get("metadatas", [[]])[0]

        # 2. BM25 retrieval
        bm25_results = self.bm25_store.search(query=cleaned_query, top_k=top_k)

        # 3. Weighted Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict[str, Any]] = {}
        vector_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}
        all_chunk_ids: Set[str] = set()

        # Process vector results (rank starts at 0)
        for rank, cid in enumerate(vector_ids):
            all_chunk_ids.add(cid)
            vector_ranks[cid] = rank
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (norm_vector_weight / (rank + 1))
            chunk_data[cid] = {
                "text": vector_docs[rank] if rank < len(vector_docs) else "",
                "metadata": vector_metas[rank] if rank < len(vector_metas) else {},
            }

        # Process BM25 results (rank starts at 0)
        for rank, item in enumerate(bm25_results):
            cid = item.chunk_id
            all_chunk_ids.add(cid)
            bm25_ranks[cid] = rank
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (norm_bm25_weight / (rank + 1))
            if cid not in chunk_data:
                chunk_data[cid] = {
                    "text": item.text,
                    "metadata": item.metadata,
                }

        # Sort: 1. descending fused score, 2. ascending chunk_id for deterministic tie-breaking
        sorted_ids = sorted(
            all_chunk_ids,
            key=lambda cid: (-fused_scores[cid], cid),
        )

        results: List[HybridResult] = []
        for cid in sorted_ids[:top_k]:
            data = chunk_data[cid]
            results.append(
                HybridResult(
                    chunk_id=cid,
                    text=data["text"],
                    metadata=data["metadata"],
                    score=float(fused_scores[cid]),
                    vector_rank=vector_ranks.get(cid),
                    bm25_rank=bm25_ranks.get(cid),
                )
            )

        return results

"""In-memory BM25 lexical index for document chunks."""

import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from backend.app.chunker import Chunk
except ImportError:
    from app.chunker import Chunk


class BM25Result(BaseModel):
    """Result of a BM25 lexical search query."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float


class BM25Store:
    """In-memory BM25 lexical retrieval index over document chunks."""

    def __init__(self) -> None:
        """Initialize an empty BM25 store."""
        self.chunk_ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self._bm25: Optional[Any] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple and deterministic alphanumeric tokenization."""
        if not isinstance(text, str):
            return []
        return re.findall(r"\w+", text.lower())

    def clear(self) -> None:
        """Clear all indexed chunks and reset the BM25 model."""
        self.chunk_ids = []
        self.documents = []
        self.metadatas = []
        self.corpus_tokens = []
        self._bm25 = None

    def add_chunks(
        self,
        chunks: List[Chunk],
        chunk_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Index chunks into the BM25 store.

        Args:
            chunks: List of Chunk objects to index.
            chunk_ids: Optional aligned IDs. If not provided, generates UUID4 IDs.

        Returns:
            List of chunk IDs assigned to the indexed chunks.

        Raises:
            ValueError: If chunks list is empty or chunk_ids length mismatches chunks.
        """
        if not chunks:
            raise ValueError("Chunks list must not be empty.")

        if chunk_ids is not None and len(chunk_ids) != len(chunks):
            raise ValueError("Each chunk must have exactly one chunk ID.")

        assigned_ids = chunk_ids if chunk_ids is not None else [str(uuid4()) for _ in chunks]

        for chunk, cid in zip(chunks, assigned_ids):
            self.chunk_ids.append(cid)
            self.documents.append(chunk.text)
            self.metadatas.append(
                {
                    **chunk.metadata,
                    "source_filename": chunk.source_filename,
                    "chunk_index": chunk.chunk_index,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                }
            )
            self.corpus_tokens.append(self.tokenize(chunk.text))

        self._rebuild_index()
        return assigned_ids

    def set_chunks(
        self,
        chunks: List[Chunk],
        chunk_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Replace the existing index with a new set of chunks.

        Args:
            chunks: List of Chunk objects.
            chunk_ids: Optional aligned IDs.

        Returns:
            List of chunk IDs.
        """
        self.clear()
        return self.add_chunks(chunks=chunks, chunk_ids=chunk_ids)

    def _rebuild_index(self) -> None:
        """Rebuild the underlying BM25Okapi instance."""
        if not self.corpus_tokens:
            self._bm25 = None
            return

        if BM25Okapi is None:
            raise RuntimeError(
                "rank-bm25 package is not installed. "
                "Install rank-bm25 to use BM25Store."
            )

        self._bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self,
        query: str,
        top_k: int,
    ) -> List[BM25Result]:
        """Perform BM25 lexical search.

        Args:
            query: Non-empty query string.
            top_k: Positive integer maximum results to return.

        Returns:
            Ranked list of BM25Result items sorted by score descending.

        Raises:
            ValueError: If query is empty or top_k is non-positive.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if not self.chunk_ids or self._bm25 is None:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)
        matching_indices = [
            idx for idx in range(len(self.chunk_ids))
            if any(token in query_token_set for token in self.corpus_tokens[idx])
        ]
        if not matching_indices:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Deterministic sorting: highest score first, tie-break by chunk_id ascending
        ranked_indices = sorted(
            matching_indices,
            key=lambda idx: (-float(scores[idx]), self.chunk_ids[idx]),
        )

        results: List[BM25Result] = []
        for idx in ranked_indices[:top_k]:
            results.append(
                BM25Result(
                    chunk_id=self.chunk_ids[idx],
                    text=self.documents[idx],
                    metadata=self.metadatas[idx],
                    score=float(scores[idx]),
                )
            )

        return results

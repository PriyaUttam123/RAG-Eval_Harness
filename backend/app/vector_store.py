
from math import isfinite
from typing import Any, Dict, List, Optional
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection

try:
    from backend.app.config import get_settings
    from backend.app.chunker import Chunk
except ImportError:
    from app.config import get_settings
    from app.chunker import Chunk


class VectorStore:
    """Persist document chunk embeddings and retrieve similar chunks."""

    def __init__(self, persist_directory: Optional[str] = None) -> None:
        self.settings = get_settings()

        directory = persist_directory or self.settings.chroma_persist_dir
        self.client = chromadb.PersistentClient(path=directory)

        self.collection: Collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"hnsw:space": self.settings.chroma_distance_metric},
        )

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Store chunks with their corresponding embeddings."""
        if not chunks:
            raise ValueError("Chunks list must not be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        vectors: List[List[float]] = []

        for chunk, embedding in zip(chunks, embeddings):
            if len(embedding) != self.settings.embedding_dimension:
                raise ValueError(
                    f"Expected {self.settings.embedding_dimension} dimensions, "
                    f"got {len(embedding)}."
                )

            if not all(isfinite(value) for value in embedding):
                raise ValueError("Embedding values must all be finite.")

            ids.append(str(uuid4()))
            documents.append(chunk.text)
            vectors.append([float(value) for value in embedding])

            # Required provenance fields cannot be overwritten by custom metadata.
            metadatas.append(
                {
                    **chunk.metadata,
                    "source_filename": chunk.source_filename,
                    "chunk_index": chunk.chunk_index,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                }
            )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieve the most similar stored chunks for a query vector."""
        if len(query_embedding) != self.settings.embedding_dimension:
            raise ValueError(
                f"Expected {self.settings.embedding_dimension} dimensions, "
                f"got {len(query_embedding)}."
            )

        if not all(isfinite(float(value)) for value in query_embedding):
            raise ValueError("Query embedding values must be finite.")

        if not all(isfinite(value) for value in query_embedding):
            raise ValueError("Query embedding values must all be finite.")

        k = top_k if top_k is not None else self.settings.default_retrieval_k

        if k <= 0:
            raise ValueError("top_k must be positive.")

        count = self.collection.count()

        # Chroma does not accept n_results=0.
        if count == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        return self.collection.query(
            query_embeddings=[
                [float(value) for value in query_embedding]
            ],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )

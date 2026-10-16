from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

try:
    from backend.app.document_loader import DocumentLoader
    from backend.app.chunker import DocumentChunker
    from backend.app.embedding_service import EmbeddingService
    from backend.app.vector_store import VectorStore
except ImportError:
    from app.document_loader import DocumentLoader
    from app.chunker import DocumentChunker
    from app.embedding_service import EmbeddingService
    from app.vector_store import VectorStore


class IngestionResult(BaseModel):
    source_filename: str
    chunks_created: int = Field(ge=0)
    chunk_ids: List[str]
    status: str

class IngestionService:
    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[DocumentChunker] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()

    def ingest_file(self, file_path: str | Path) -> IngestionResult:
        document = self.loader.load(file_path)

        chunks = self.chunker.chunk(document)

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        chunk_ids = self.vector_store.add_chunks(chunks, embeddings)

        return IngestionResult(
            source_filename=document.source_filename,
            chunks_created=len(chunks),
            chunk_ids=chunk_ids,
            status="success",
        )    
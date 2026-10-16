from pathlib import Path
import pytest


from backend.app.ingestion_service import IngestionService


class FakeDocument:
    source_filename = "example.txt"


class FakeLoader:
    def load(self, file_path):
        self.received_path = file_path
        return FakeDocument()


class FakeChunk:
    text = "This is a test chunk."


class FakeChunker:
    def chunk(self, document):
        self.received_document = document
        return [FakeChunk()]


class FakeEmbeddingService:
    def embed_texts(self, texts):
        self.received_texts = texts
        return [[0.1, 0.2, 0.3]]


class FakeVectorStore:
    def add_chunks(self, chunks, embeddings):
        self.received_chunks = chunks
        self.received_embeddings = embeddings
        return ["chunk-id-1"]


def test_ingest_file_success():
    loader = FakeLoader()
    chunker = FakeChunker()
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    service = IngestionService(
        loader=loader,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = service.ingest_file(Path("example.txt"))

    assert result.source_filename == "example.txt"
    assert result.chunks_created == 1
    assert result.chunk_ids == ["chunk-id-1"]
    assert result.status == "success"

    assert loader.received_path == Path("example.txt")
    assert embedding_service.received_texts == ["This is a test chunk."]
    assert chunker.received_document is not None
    assert vector_store.received_chunks[0].text == "This is a test chunk."
    assert vector_store.received_embeddings == [[0.1, 0.2, 0.3]]


def test_ingest_file_propagates_loader_error():
    class FailingLoader:
        def load(self, file_path):
            raise FileNotFoundError("Document not found.")

    service = IngestionService(
        loader=FailingLoader(),
        chunker=FakeChunker(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(FileNotFoundError, match="Document not found."):
        service.ingest_file("missing.txt")    



def test_ingest_file_propagates_chunker_error():
    class FailingChunker:
        def chunk(self, document):
            raise ValueError("Chunking failed.")

    service = IngestionService(
        loader=FakeLoader(),
        chunker=FailingChunker(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="Chunking failed."):
        service.ingest_file("example.txt")


def test_ingest_file_propagates_embedding_error():
    class FailingEmbeddingService:
        def embed_texts(self, texts):
            raise RuntimeError("Embedding failed.")

    service = IngestionService(
        loader=FakeLoader(),
        chunker=FakeChunker(),
        embedding_service=FailingEmbeddingService(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(RuntimeError, match="Embedding failed."):
        service.ingest_file("example.txt")


def test_ingest_file_propagates_vector_store_error():
    class FailingVectorStore:
        def add_chunks(self, chunks, embeddings):
            raise RuntimeError("Vector storage failed.")

    service = IngestionService(
        loader=FakeLoader(),
        chunker=FakeChunker(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FailingVectorStore(),
    )

    with pytest.raises(RuntimeError, match="Vector storage failed."):
        service.ingest_file("example.txt")


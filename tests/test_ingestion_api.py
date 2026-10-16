
from io import BytesIO

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.routes import ingestion as ingestion_route


client = TestClient(app)


class FakeIngestionResult:
    def model_dump(self):
        return {
            "source_filename": "example.txt",
            "chunks_created": 2,
            "chunk_ids": ["chunk-1", "chunk-2"],
            "status": "success",
        }


class FakeIngestionService:
    def ingest_file(self, file_path):
        self.received_path = file_path
        return FakeIngestionResult()


def test_upload_txt_success(monkeypatch):
    monkeypatch.setattr(
        ingestion_route, "IngestionService", FakeIngestionService
    )

    response = client.post(
        "/ingestion/upload",
        files={"file": ("example.txt", BytesIO(b"Sample document text."), "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["source_filename"] == "example.txt"
    assert response.json()["chunks_created"] == 2
    assert response.json()["status"] == "success"


def test_upload_rejects_unsupported_file_type():
    response = client.post(
        "/ingestion/upload",
        files={"file": ("notes.docx", BytesIO(b"content"), "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_empty_file():
    response = client.post(
        "/ingestion/upload",
        files={"file": ("empty.txt", BytesIO(b""), "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."

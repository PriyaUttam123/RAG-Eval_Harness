
from fastapi import APIRouter, File, HTTPException, UploadFile

try:
    from backend.app.config import get_settings
    from backend.app.ingestion_service import IngestionService
except ImportError:
    from app.config import get_settings
    from app.ingestion_service import IngestionService


router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    settings = get_settings()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in {".txt", ".pdf"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a TXT or PDF file.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the maximum allowed size.",
        )

    # Temporary upload file: ingestion service currently accepts a file path.
    import tempfile
    from pathlib import Path

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        result = IngestionService().ingest_file(temp_path)
        return result.model_dump()

    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

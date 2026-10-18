"""RAG question answering API route."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from backend.app.rag_service import RAGService
except ImportError:
    from app.rag_service import RAGService


router = APIRouter(prefix="/rag", tags=["RAG"])


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, gt=0)


@router.post("/answer")
def answer_question(request: RAGRequest):
    """Answer a user question using grounded retrieval-augmented generation."""
    try:
        result = RAGService().answer(
            question=request.question,
            top_k=request.top_k,
        )
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

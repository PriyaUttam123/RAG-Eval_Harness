from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from backend.app.retrieval_service import RetrievalService
except ImportError:
    from app.retrieval_service import RetrievalService


router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, gt=0)


@router.post("/search")
def search_documents(request: RetrievalRequest):
    try:
        result = RetrievalService().retrieve(
            query=request.query,
            top_k=request.top_k,
        )
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
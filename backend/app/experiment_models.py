"""Models for executed RAG experiment results."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

try:
    from backend.app.evaluation_report import EvaluationReport
    from backend.app.experiment_config import ExperimentRunMetadata
except ImportError:
    from app.evaluation_report import EvaluationReport
    from app.experiment_config import ExperimentRunMetadata


class ExperimentResult(BaseModel):
    """Result of executing one configured RAG experiment."""

    run_metadata: ExperimentRunMetadata
    report: EvaluationReport
    duration_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
    )
    error_count: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

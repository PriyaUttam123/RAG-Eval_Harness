"""Evaluation runner for executing RAG benchmark datasets."""

from typing import List, Optional

from pydantic import BaseModel, Field

try:
    from backend.app.evaluation_models import EvaluationCase, EvaluationDataset
    from backend.app.rag_service import RAGAnswer, RAGService
except ImportError:
    from app.evaluation_models import EvaluationCase, EvaluationDataset
    from app.rag_service import RAGAnswer, RAGService


class EvaluationResult(BaseModel):
    """Execution result for a single evaluation case."""

    question: str
    expected_answer: str
    actual_answer: str
    retrieved_chunk_ids: List[str]
    relevant_chunk_ids: List[str]
    case_type: str
    error: Optional[str] = None
    retrieved_contexts: List[str] = Field(default_factory=list)


class EvaluationRunResult(BaseModel):
    """Aggregated results across an entire evaluation dataset run."""

    results: List[EvaluationResult]
    total_cases: int
    successful_cases: int
    failed_cases: int


class EvaluationRunner:
    """Executes evaluation datasets against a RAG pipeline."""

    def __init__(self, rag_service: Optional[RAGService] = None) -> None:
        """Initialize runner with an optional RAGService dependency.

        Args:
            rag_service: Optional RAGService instance. If not provided,
                a default RAGService is instantiated lazily.
        """
        self.rag_service = rag_service or RAGService()

    def run(
        self,
        dataset: EvaluationDataset,
        top_k: Optional[int] = None,
    ) -> EvaluationRunResult:
        """Execute all cases in an EvaluationDataset against the RAG pipeline.

        Args:
            dataset: Validated EvaluationDataset containing test cases.
            top_k: Optional number of chunks to retrieve for each question.

        Returns:
            EvaluationRunResult containing individual results and run statistics.

        Raises:
            ValueError: If dataset is not a valid EvaluationDataset instance.
        """
        if dataset is None or not isinstance(dataset, EvaluationDataset):
            raise ValueError("A valid EvaluationDataset instance must be provided.")

        results: List[EvaluationResult] = []
        successful_cases = 0
        failed_cases = 0

        for case in dataset.cases:
            try:
                rag_answer = self.rag_service.answer(
                    question=case.question,
                    top_k=top_k,
                )

                retrieved_ids = [
                    getattr(source, "chunk_id", "")
                    if hasattr(source, "chunk_id")
                    else (source.get("chunk_id", "") if isinstance(source, dict) else "")
                    for source in rag_answer.sources
                ]

                retrieved_contexts = [
                    getattr(source, "text", "")
                    if hasattr(source, "text")
                    else (source.get("text", "") if isinstance(source, dict) else "")
                    for source in rag_answer.sources
                ]

                results.append(
                    EvaluationResult(
                        question=case.question,
                        expected_answer=case.expected_answer,
                        actual_answer=rag_answer.answer,
                        retrieved_chunk_ids=retrieved_ids,
                        relevant_chunk_ids=list(case.relevant_chunk_ids),
                        case_type=case.case_type,
                        error=None,
                        retrieved_contexts=retrieved_contexts,
                    )
                )
                successful_cases += 1
            except Exception as exc:
                error_msg = str(exc) if str(exc).strip() else exc.__class__.__name__
                results.append(
                    EvaluationResult(
                        question=case.question,
                        expected_answer=case.expected_answer,
                        actual_answer="",
                        retrieved_chunk_ids=[],
                        relevant_chunk_ids=list(case.relevant_chunk_ids),
                        case_type=case.case_type,
                        error=error_msg,
                        retrieved_contexts=[],
                    )
                )
                failed_cases += 1

        return EvaluationRunResult(
            results=results,
            total_cases=len(dataset.cases),
            successful_cases=successful_cases,
            failed_cases=failed_cases,
        )

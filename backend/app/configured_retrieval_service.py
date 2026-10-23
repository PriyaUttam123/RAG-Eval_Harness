"""Bind experiment configuration to the experiment retrieval service."""

from typing import Any, Optional

try:
    from backend.app.experiment_config import ExperimentConfig
    from backend.app.experiment_retrieval_service import ExperimentRetrievalService
except ImportError:
    from app.experiment_config import ExperimentConfig
    from app.experiment_retrieval_service import ExperimentRetrievalService


class ConfiguredRetrievalService:
    """Expose experiment-configured retrieval through the RAG retrieval interface."""

    def __init__(
        self,
        experiment_retrieval_service: ExperimentRetrievalService,
        config: ExperimentConfig,
    ):
        self.experiment_retrieval_service = experiment_retrieval_service
        self.config = config

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Any:
        effective_top_k = (
            self.config.retrieval_top_k
            if top_k is None
            else top_k
        )

        return self.experiment_retrieval_service.retrieve(
            query=query,
            retrieval_strategy=self.config.retrieval_strategy,
            retrieval_top_k=effective_top_k,
            reranking_enabled=self.config.reranking_enabled,
        )
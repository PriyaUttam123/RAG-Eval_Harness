"""Experiment management service for creating configurations and run metadata."""

from datetime import datetime, timezone
from typing import Optional

try:
    from backend.app.experiment_config import (
        ExperimentConfig,
        ExperimentRunMetadata,
    )
except ImportError:
    from app.experiment_config import (
        ExperimentConfig,
        ExperimentRunMetadata,
    )


class ExperimentService:
    """Service for managing reproducible experiment configurations and run metadata."""

    def create_config(
        self,
        name: str,
        chunk_size: int,
        chunk_overlap: int,
        retrieval_top_k: int,
        evaluation_k: int,
        embedding_model: str,
        description: str = "",
        retrieval_strategy: str = "vector",
        reranking_enabled: bool = False,
        generation_model: str = "",
        **extra_kwargs,
    ) -> ExperimentConfig:
        """Create and validate an ExperimentConfig instance.

        Args:
            name: Non-empty experiment name.
            chunk_size: Character size for chunk splitting (> 0).
            chunk_overlap: Overlap between adjacent chunks (0 <= overlap < chunk_size).
            retrieval_top_k: Top chunks retrieved for queries (> 0).
            evaluation_k: Metric cutoff depth (> 0).
            embedding_model: Non-empty embedding model identifier.
            description: Optional experiment description.
            retrieval_strategy: 'vector' or 'hybrid'.
            reranking_enabled: Whether reranking is enabled.
            generation_model: Optional LLM model identifier.

        Returns:
            Validated ExperimentConfig instance.
        """
        return ExperimentConfig(
            name=name,
            description=description,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_top_k=retrieval_top_k,
            evaluation_k=evaluation_k,
            retrieval_strategy=retrieval_strategy,
            reranking_enabled=reranking_enabled,
            embedding_model=embedding_model,
            generation_model=generation_model,
            **extra_kwargs,
        )

    def create_run_metadata(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        dataset_name: str,
        created_at: Optional[datetime] = None,
    ) -> ExperimentRunMetadata:
        """Create and validate an ExperimentRunMetadata instance.

        Args:
            experiment_id: Non-empty unique experiment identifier.
            config: An ExperimentConfig instance.
            dataset_name: 'golden' or 'holdout'.
            created_at: Optional timezone-aware timestamp. Defaults to UTC now.

        Returns:
            Validated ExperimentRunMetadata instance.

        Raises:
            ValueError: If inputs are invalid or dataset_name is unsupported.
        """
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string.")

        if not isinstance(config, ExperimentConfig):
            raise ValueError("config must be a valid ExperimentConfig instance.")

        if not isinstance(dataset_name, str) or dataset_name.strip().lower() not in {"golden", "holdout"}:
            raise ValueError(
                f"Invalid dataset_name '{dataset_name}'. Must be 'golden' or 'holdout'."
            )

        kwargs = {
            "experiment_id": experiment_id.strip(),
            "config": config,
            "dataset_name": dataset_name.strip().lower(),
        }
        if created_at is not None:
            kwargs["created_at"] = created_at
        else:
            kwargs["created_at"] = datetime.now(timezone.utc)

        return ExperimentRunMetadata(**kwargs)

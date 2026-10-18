"""Unit tests for ExperimentService."""

from datetime import datetime, timezone, timedelta

import pytest

from backend.app.experiment_config import (
    ExperimentConfig,
    ExperimentRunMetadata,
)
from backend.app.experiment_service import ExperimentService


def test_create_config_returns_experiment_config():
    """Verify create_config constructs and returns a valid ExperimentConfig."""
    service = ExperimentService()

    config = service.create_config(
        name="Hybrid Rerank Test",
        chunk_size=512,
        chunk_overlap=50,
        retrieval_top_k=5,
        evaluation_k=3,
        embedding_model="all-MiniLM-L6-v2",
        description="Testing hybrid retrieval strategy.",
        retrieval_strategy="hybrid",
        reranking_enabled=True,
        generation_model="meta/llama-3.1-8b-instruct",
    )

    assert isinstance(config, ExperimentConfig)
    assert config.name == "Hybrid Rerank Test"
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.retrieval_top_k == 5
    assert config.evaluation_k == 3
    assert config.retrieval_strategy == "hybrid"
    assert config.reranking_enabled is True
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.generation_model == "meta/llama-3.1-8b-instruct"


def test_create_run_metadata_returns_metadata():
    """Verify create_run_metadata constructs and returns ExperimentRunMetadata."""
    service = ExperimentService()
    config = service.create_config(
        name="Test Experiment",
        chunk_size=256,
        chunk_overlap=25,
        retrieval_top_k=3,
        evaluation_k=2,
        embedding_model="bge-small-en-v1.5",
    )

    metadata = service.create_run_metadata(
        experiment_id="exp-2026-10-19-001",
        config=config,
        dataset_name="golden",
    )

    assert isinstance(metadata, ExperimentRunMetadata)
    assert metadata.experiment_id == "exp-2026-10-19-001"
    assert metadata.config == config
    assert metadata.dataset_name == "golden"
    assert metadata.created_at.tzinfo is not None


@pytest.mark.parametrize("invalid_dataset", ["unknown", "test", "", "   "])
def test_create_run_metadata_rejects_invalid_dataset(invalid_dataset):
    """Verify create_run_metadata raises ValueError for unsupported datasets."""
    service = ExperimentService()
    config = service.create_config(
        name="Test",
        chunk_size=256,
        chunk_overlap=25,
        retrieval_top_k=3,
        evaluation_k=2,
        embedding_model="all-MiniLM-L6-v2",
    )

    with pytest.raises(ValueError, match="dataset_name"):
        service.create_run_metadata(
            experiment_id="exp-1",
            config=config,
            dataset_name=invalid_dataset,
        )


@pytest.mark.parametrize("invalid_id", ["", "   ", " \n "])
def test_create_run_metadata_rejects_invalid_experiment_id(invalid_id):
    """Verify empty experiment_id raises ValueError."""
    service = ExperimentService()
    config = service.create_config(
        name="Test",
        chunk_size=256,
        chunk_overlap=25,
        retrieval_top_k=3,
        evaluation_k=2,
        embedding_model="all-MiniLM-L6-v2",
    )

    with pytest.raises(ValueError, match="experiment_id"):
        service.create_run_metadata(
            experiment_id=invalid_id,
            config=config,
            dataset_name="golden",
        )


def test_create_run_metadata_preserves_supplied_config():
    """Verify the exact supplied config object is preserved in metadata."""
    service = ExperimentService()
    config = service.create_config(
        name="Preserved Config",
        chunk_size=1024,
        chunk_overlap=100,
        retrieval_top_k=10,
        evaluation_k=5,
        embedding_model="text-embedding-3-small",
        description="Preserved description",
    )

    metadata = service.create_run_metadata(
        experiment_id="exp-preserve",
        config=config,
        dataset_name="holdout",
    )

    assert metadata.config is config
    assert metadata.config.name == "Preserved Config"
    assert metadata.config.chunk_size == 1024


def test_generated_metadata_timestamp_is_timezone_aware():
    """Verify generated metadata timestamp has timezone awareness."""
    service = ExperimentService()
    config = service.create_config(
        name="Timestamp Test",
        chunk_size=512,
        chunk_overlap=50,
        retrieval_top_k=3,
        evaluation_k=2,
        embedding_model="all-MiniLM-L6-v2",
    )

    metadata = service.create_run_metadata(
        experiment_id="exp-tz",
        config=config,
        dataset_name="golden",
    )

    assert metadata.created_at.tzinfo is not None
    assert metadata.created_at.tzinfo.utcoffset(metadata.created_at) is not None

    # Custom explicit timezone-aware timestamp
    custom_dt = datetime.now(timezone(timedelta(hours=1)))
    metadata_custom = service.create_run_metadata(
        experiment_id="exp-tz-custom",
        config=config,
        dataset_name="golden",
        created_at=custom_dt,
    )
    assert metadata_custom.created_at == custom_dt

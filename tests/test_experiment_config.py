"""Unit tests for ExperimentConfig and ExperimentRunMetadata models."""

from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

from backend.app.experiment_config import (
    ExperimentConfig,
    ExperimentRunMetadata,
)


def _make_valid_config_kwargs(**overrides):
    base = {
        "name": "Baseline Vector Search",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "retrieval_top_k": 4,
        "evaluation_k": 3,
        "embedding_model": "all-MiniLM-L6-v2",
    }
    base.update(overrides)
    return base


def test_valid_configuration_and_defaults():
    """Verify creating ExperimentConfig with valid fields and default values."""
    config = ExperimentConfig(**_make_valid_config_kwargs())

    assert config.name == "Baseline Vector Search"
    assert config.description == ""
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.retrieval_top_k == 4
    assert config.evaluation_k == 3
    assert config.retrieval_strategy == "vector"
    assert config.reranking_enabled is False
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.generation_model == ""


def test_name_and_strings_whitespace_normalization():
    """Verify name, description, and model strings are trimmed."""
    config = ExperimentConfig(
        **_make_valid_config_kwargs(
            name="   Trimmed Experiment Name   ",
            description="   Padded description.   ",
            embedding_model="   bge-small-en-v1.5   ",
            generation_model="   meta/llama-3.1-8b-instruct   ",
        )
    )

    assert config.name == "Trimmed Experiment Name"
    assert config.description == "Padded description."
    assert config.embedding_model == "bge-small-en-v1.5"
    assert config.generation_model == "meta/llama-3.1-8b-instruct"


@pytest.mark.parametrize("empty_name", ["", "   ", " \n\t "])
def test_invalid_empty_name_rejected(empty_name):
    """Verify empty or whitespace-only name raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(**_make_valid_config_kwargs(name=empty_name))


@pytest.mark.parametrize("invalid_chunk_size", [0, -1, -100])
def test_invalid_chunk_size_rejected(invalid_chunk_size):
    """Verify non-positive chunk_size raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(**_make_valid_config_kwargs(chunk_size=invalid_chunk_size))


@pytest.mark.parametrize("invalid_overlap", [-1, -50])
def test_invalid_overlap_rejected(invalid_overlap):
    """Verify negative chunk_overlap raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(**_make_valid_config_kwargs(chunk_overlap=invalid_overlap))


def test_overlap_equal_to_chunk_size_rejected():
    """Verify overlap equal to chunk_size raises ValidationError."""
    with pytest.raises(ValidationError, match="strictly less than"):
        ExperimentConfig(**_make_valid_config_kwargs(chunk_size=200, chunk_overlap=200))


def test_overlap_greater_than_chunk_size_rejected():
    """Verify overlap strictly greater than chunk_size raises ValidationError."""
    with pytest.raises(ValidationError, match="strictly less than"):
        ExperimentConfig(**_make_valid_config_kwargs(chunk_size=200, chunk_overlap=250))


@pytest.mark.parametrize("invalid_top_k", [0, -1])
def test_invalid_retrieval_top_k_rejected(invalid_top_k):
    """Verify non-positive retrieval_top_k raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(**_make_valid_config_kwargs(retrieval_top_k=invalid_top_k))


@pytest.mark.parametrize("invalid_eval_k", [0, -1])
def test_invalid_evaluation_k_rejected(invalid_eval_k):
    """Verify non-positive evaluation_k raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(**_make_valid_config_kwargs(evaluation_k=invalid_eval_k))


@pytest.mark.parametrize("invalid_strategy", ["bm25", "semantic", "random", ""])
def test_invalid_retrieval_strategy_rejected(invalid_strategy):
    """Verify unsupported retrieval strategy raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(**_make_valid_config_kwargs(retrieval_strategy=invalid_strategy))


@pytest.mark.parametrize("empty_model", ["", "   "])
def test_empty_embedding_model_rejected(empty_model):
    """Verify empty embedding_model raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(**_make_valid_config_kwargs(embedding_model=empty_model))


def test_generation_model_may_be_empty():
    """Verify generation_model is optional and can be an empty string."""
    config = ExperimentConfig(**_make_valid_config_kwargs(generation_model=""))
    assert config.generation_model == ""


def test_valid_hybrid_configuration():
    """Verify valid hybrid retrieval strategy with normalized casing."""
    config = ExperimentConfig(**_make_valid_config_kwargs(retrieval_strategy="  HYBRID  "))
    assert config.retrieval_strategy == "hybrid"


def test_reranking_enabled_flag():
    """Verify reranking_enabled boolean toggle."""
    config_disabled = ExperimentConfig(**_make_valid_config_kwargs(reranking_enabled=False))
    assert config_disabled.reranking_enabled is False

    config_enabled = ExperimentConfig(**_make_valid_config_kwargs(reranking_enabled=True))
    assert config_enabled.reranking_enabled is True


# ============================================================================
# ExperimentRunMetadata tests
# ============================================================================


def test_valid_experiment_run_metadata():
    """Verify creating valid ExperimentRunMetadata with explicit and default fields."""
    config = ExperimentConfig(**_make_valid_config_kwargs())
    now = datetime.now(timezone.utc)

    metadata = ExperimentRunMetadata(
        experiment_id="exp-001",
        config=config,
        dataset_name="golden",
        created_at=now,
    )

    assert metadata.experiment_id == "exp-001"
    assert metadata.config == config
    assert metadata.dataset_name == "golden"
    assert metadata.created_at == now


@pytest.mark.parametrize("empty_id", ["", "   ", " \n "])
def test_metadata_experiment_id_non_empty_required(empty_id):
    """Verify empty experiment_id raises ValidationError."""
    config = ExperimentConfig(**_make_valid_config_kwargs())
    with pytest.raises(ValidationError):
        ExperimentRunMetadata(
            experiment_id=empty_id,
            config=config,
            dataset_name="golden",
        )


def test_metadata_valid_golden_and_holdout_datasets():
    """Verify both golden and holdout are accepted (case-insensitive)."""
    config = ExperimentConfig(**_make_valid_config_kwargs())

    meta_golden = ExperimentRunMetadata(
        experiment_id="exp-1",
        config=config,
        dataset_name="  GOLDEN  ",
    )
    assert meta_golden.dataset_name == "golden"

    meta_holdout = ExperimentRunMetadata(
        experiment_id="exp-2",
        config=config,
        dataset_name="Holdout",
    )
    assert meta_holdout.dataset_name == "holdout"


@pytest.mark.parametrize("invalid_dataset", ["unknown", "test", "val", ""])
def test_metadata_invalid_dataset_name_rejected(invalid_dataset):
    """Verify unsupported dataset names raise ValidationError."""
    config = ExperimentConfig(**_make_valid_config_kwargs())
    with pytest.raises(ValidationError):
        ExperimentRunMetadata(
            experiment_id="exp-3",
            config=config,
            dataset_name=invalid_dataset,
        )


def test_metadata_created_at_timezone_aware():
    """Verify created_at must be timezone-aware and defaults to UTC."""
    config = ExperimentConfig(**_make_valid_config_kwargs())

    # Default created_at should be timezone-aware
    meta_default = ExperimentRunMetadata(
        experiment_id="exp-4",
        config=config,
        dataset_name="golden",
    )
    assert meta_default.created_at.tzinfo is not None

    # Custom timezone-aware datetime accepted
    custom_tz = timezone(timedelta(hours=5, minutes=30))
    custom_dt = datetime(2026, 10, 19, 12, 0, tzinfo=custom_tz)
    meta_custom = ExperimentRunMetadata(
        experiment_id="exp-5",
        config=config,
        dataset_name="golden",
        created_at=custom_dt,
    )
    assert meta_custom.created_at == custom_dt

    # Naive datetime must be rejected
    naive_dt = datetime(2026, 10, 19, 12, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        ExperimentRunMetadata(
            experiment_id="exp-6",
            config=config,
            dataset_name="golden",
            created_at=naive_dt,
        )

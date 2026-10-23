from backend.app.configured_retrieval_service import ConfiguredRetrievalService
from backend.app.experiment_config import ExperimentConfig


class FakeExperimentRetrievalService:
    def __init__(self):
        self.calls = []

    def retrieve(
        self,
        query,
        retrieval_strategy,
        retrieval_top_k,
        reranking_enabled=False,
        reranking_top_k=None,
    ):
        self.calls.append(
            {
                "query": query,
                "retrieval_strategy": retrieval_strategy,
                "retrieval_top_k": retrieval_top_k,
                "reranking_enabled": reranking_enabled,
                "reranking_top_k": reranking_top_k,
            }
        )
        return ["retrieved"]


def make_config():
    return ExperimentConfig(
        name="baseline",
        chunk_size=512,
        chunk_overlap=50,
        retrieval_top_k=5,
        evaluation_k=3,
        retrieval_strategy="hybrid",
        reranking_enabled=True,
        embedding_model="test-embedding",
        generation_model="test-generation",
    )


def test_retrieve_forwards_experiment_configuration():
    retrieval_service = FakeExperimentRetrievalService()
    service = ConfiguredRetrievalService(
        experiment_retrieval_service=retrieval_service,
        config=make_config(),
    )

    result = service.retrieve("What is RAG?")

    assert result == ["retrieved"]
    assert retrieval_service.calls == [
        {
            "query": "What is RAG?",
            "retrieval_strategy": "hybrid",
            "retrieval_top_k": 5,
            "reranking_enabled": True,
            "reranking_top_k": None,
        }
    ]


def test_explicit_top_k_overrides_experiment_default():
    retrieval_service = FakeExperimentRetrievalService()
    service = ConfiguredRetrievalService(
        experiment_retrieval_service=retrieval_service,
        config=make_config(),
    )

    service.retrieve("What is RAG?", top_k=2)

    assert retrieval_service.calls[0]["retrieval_top_k"] == 2
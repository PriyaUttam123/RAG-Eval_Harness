"""Unit tests for GenerationService using fake OpenAI-compatible clients."""

from typing import Any

import pytest

from backend.app.generation_service import GenerationService
from backend.app.prompt_builder import PromptContext, RAGPromptBuilder


class FakeChatMessage:
    def __init__(self, content: Any) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, message: Any) -> None:
        self.message = message


class FakeChatCompletionResponse:
    def __init__(self, choices: Any) -> None:
        self.choices = choices


class FakeCompletions:
    def __init__(self, response: Any = None) -> None:
        self.response = response
        self.last_kwargs: dict = {}

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, response: Any = None) -> None:
        self.completions = FakeCompletions(response=response)
        self.chat = FakeChat(completions=self.completions)


def make_fake_client(content: Any = "Test answer") -> FakeOpenAIClient:
    response = FakeChatCompletionResponse(
        choices=[FakeChoice(message=FakeChatMessage(content=content))]
    )
    return FakeOpenAIClient(response=response)


def test_successful_generation(monkeypatch):
    """Test successful answer generation returning clean answer string."""
    fake_client = make_fake_client("Paris is the capital of France.")
    service = GenerationService(client=fake_client)
    monkeypatch.setattr(service.settings, "nvidia_model_name", "meta/llama-3.1-8b-instruct")

    contexts = [
        PromptContext(
            text="Paris is the capital of France.",
            source_filename="geography.txt",
            chunk_index=0,
        )
    ]

    answer = service.generate(
        question="What is the capital of France?",
        contexts=contexts,
    )

    assert answer == "Paris is the capital of France."


def test_generation_request_parameters(monkeypatch):
    """Verify request parameters passed to client.chat.completions.create."""
    fake_client = make_fake_client("Answer")
    service = GenerationService(client=fake_client)
    monkeypatch.setattr(service.settings, "nvidia_model_name", "meta/llama-3.1-70b-instruct")
    monkeypatch.setattr(service.settings, "generation_temperature", 0.4)
    monkeypatch.setattr(service.settings, "generation_max_tokens", 256)

    contexts = [
        PromptContext(
            text="Relevant context snippet.",
            source_filename="doc.txt",
            chunk_index=1,
        )
    ]

    service.generate(
        question="What is the answer?",
        contexts=contexts,
    )

    kwargs = fake_client.completions.last_kwargs
    assert kwargs["model"] == "meta/llama-3.1-70b-instruct"
    assert kwargs["temperature"] == 0.4
    assert kwargs["max_tokens"] == 256

    messages = kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Relevant context snippet." in messages[1]["content"]
    assert "What is the answer?" in messages[1]["content"]


def test_generation_strips_answer_whitespace(monkeypatch):
    """Verify surrounding whitespace in the provider response is trimmed."""
    fake_client = make_fake_client("   \n\t Paris is the capital.  \n ")
    service = GenerationService(client=fake_client)
    monkeypatch.setattr(service.settings, "nvidia_model_name", "meta/llama-3.1-8b-instruct")

    answer = service.generate(
        question="What is the capital?",
        contexts=[],
    )

    assert answer == "Paris is the capital."


def test_empty_provider_answer_raises_runtime_error(monkeypatch):
    """Verify whitespace-only or empty provider response raises RuntimeError."""
    fake_client = make_fake_client("   \n\t  ")
    service = GenerationService(client=fake_client)
    monkeypatch.setattr(service.settings, "nvidia_model_name", "meta/llama-3.1-8b-instruct")

    with pytest.raises(RuntimeError, match="non-empty string"):
        service.generate(
            question="What is the refund period?",
            contexts=[],
        )


@pytest.mark.parametrize(
    "malformed_response",
    [
        None,
        object(),
        FakeChatCompletionResponse(choices=[]),
        FakeChatCompletionResponse(choices=[FakeChoice(message=None)]),
        FakeChatCompletionResponse(choices=[FakeChoice(message=FakeChatMessage(content=None))]),
        FakeChatCompletionResponse(choices=[FakeChoice(message=FakeChatMessage(content=123))]),
        {"choices": []},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": ""}}]},
    ],
)
def test_invalid_provider_response_raises_runtime_error(monkeypatch, malformed_response):
    """Verify malformed responses lacking choices, message, or content raise RuntimeError."""
    fake_client = FakeOpenAIClient(response=malformed_response)
    service = GenerationService(client=fake_client)
    monkeypatch.setattr(service.settings, "nvidia_model_name", "meta/llama-3.1-8b-instruct")

    with pytest.raises(RuntimeError):
        service.generate(
            question="What is the refund period?",
            contexts=[],
        )


@pytest.mark.parametrize("empty_model", ["", "   "])
def test_missing_nvidia_model_name_raises_runtime_error(monkeypatch, empty_model):
    """Verify missing NVIDIA model name raises RuntimeError without requiring NVIDIA key."""
    fake_client = make_fake_client("Valid answer")
    service = GenerationService(client=fake_client)
    monkeypatch.setattr(service.settings, "nvidia_model_name", empty_model)
    monkeypatch.setattr(service.settings, "nvidia_api_key", "")

    with pytest.raises(RuntimeError, match="NVIDIA_MODEL_NAME"):
        service.generate(
            question="What is the refund period?",
            contexts=[],
        )


@pytest.mark.parametrize("empty_key", ["", "   "])
def test_missing_nvidia_api_key_when_creating_real_client(monkeypatch, empty_key):
    """Verify _get_client raises RuntimeError when NVIDIA_API_KEY is missing."""
    service = GenerationService(client=None)
    monkeypatch.setattr(service.settings, "nvidia_api_key", empty_key)

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        service._get_client()


def test_missing_openai_package_raises_runtime_error(monkeypatch):
    """Verify _get_client raises RuntimeError when the openai package is missing."""
    service = GenerationService(client=None)
    monkeypatch.setattr("backend.app.generation_service.OpenAI", None)
    monkeypatch.setattr(service.settings, "nvidia_api_key", "mock-nvapi-key")

    with pytest.raises(RuntimeError, match="openai"):
        service._get_client()


def test_real_client_lazy_instantiation_uses_settings(monkeypatch):
    """Verify real client is created lazily with configured key and base URL, and is reused."""
    created_clients = []

    class MockOpenAI:
        def __init__(self, api_key: str, base_url: str):
            self.api_key = api_key
            self.base_url = base_url
            created_clients.append(self)

    monkeypatch.setattr("backend.app.generation_service.OpenAI", MockOpenAI)

    service = GenerationService(client=None)
    monkeypatch.setattr(service.settings, "nvidia_api_key", "test-api-key-value")
    monkeypatch.setattr(service.settings, "nvidia_base_url", "https://custom.nvidia.com/v1")

    # Client is not created on initialization
    assert service._client is None
    assert len(created_clients) == 0

    # Client is created lazily on first call
    client1 = service._get_client()
    assert len(created_clients) == 1
    assert client1.api_key == "test-api-key-value"
    assert client1.base_url == "https://custom.nvidia.com/v1"

    # Subsequent call reuses cached client instance
    client2 = service._get_client()
    assert client2 is client1
    assert len(created_clients) == 1


def test_custom_prompt_builder_injection(monkeypatch):
    """Verify custom prompt builder is respected when injected."""
    class CustomPromptBuilder:
        def build(self, question: str, contexts: list) -> list:
            return [{"role": "user", "content": f"Custom prompt: {question}"}]

    fake_client = make_fake_client("Custom answer")
    service = GenerationService(
        client=fake_client,
        prompt_builder=CustomPromptBuilder(),
    )
    monkeypatch.setattr(service.settings, "nvidia_model_name", "meta/llama-3.1-8b-instruct")

    service.generate("Hello?", [])

    messages = fake_client.completions.last_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["content"] == "Custom prompt: Hello?"


def test_empty_question_propagates_prompt_builder_value_error(monkeypatch):
    """Verify invalid empty question raises ValueError from prompt builder."""
    fake_client = make_fake_client("Answer")
    service = GenerationService(client=fake_client)
    monkeypatch.setattr(service.settings, "nvidia_model_name", "meta/llama-3.1-8b-instruct")

    with pytest.raises(ValueError, match="non-empty string"):
        service.generate("   ", [])

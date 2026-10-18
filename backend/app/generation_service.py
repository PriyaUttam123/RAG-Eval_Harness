"""Generation service for RAG response synthesis via OpenAI-compatible API."""

from typing import Any, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from backend.app.config import Settings, get_settings
    from backend.app.prompt_builder import PromptContext, RAGPromptBuilder
except ImportError:
    from app.config import Settings, get_settings
    from app.prompt_builder import PromptContext, RAGPromptBuilder


class GenerationService:
    """Service for generating grounded answers using an LLM provider."""

    def __init__(
        self,
        client: Optional[Any] = None,
        prompt_builder: Optional[RAGPromptBuilder] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize GenerationService with optional dependencies.

        Args:
            client: Optional OpenAI-compatible client for dependency injection.
            prompt_builder: Optional prompt builder instance.
            settings: Optional application settings instance.
        """
        self.settings = settings or get_settings()
        self._client = client
        self.prompt_builder = prompt_builder or RAGPromptBuilder()

    def _get_client(self) -> Any:
        """Lazily instantiate and return an OpenAI-compatible client.

        Returns:
            An OpenAI-compatible client instance.

        Raises:
            RuntimeError: If the openai package is missing or NVIDIA_API_KEY is unset.
        """
        if self._client is not None:
            return self._client

        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. "
                "Install project dependencies to enable LLM generation."
            )

        api_key = self.settings.nvidia_api_key
        if not api_key or not api_key.strip():
            raise RuntimeError(
                "NVIDIA_API_KEY is not configured. "
                "Set NVIDIA_API_KEY to instantiate the generation client."
            )

        self._client = OpenAI(
            api_key=api_key.strip(),
            base_url=self.settings.nvidia_base_url,
        )
        return self._client

    def generate(
        self,
        question: str,
        contexts: List[PromptContext],
    ) -> str:
        """Generate an answer using retrieved contexts and configured LLM.

        Args:
            question: The user question to answer.
            contexts: Retrieved document chunks providing grounding context.

        Returns:
            The generated response string with surrounding whitespace removed.

        Raises:
            RuntimeError: If NVIDIA_MODEL_NAME is missing, client creation fails,
                or the provider response is invalid or empty.
            ValueError: If the question is empty or invalid.
        """
        messages = self.prompt_builder.build(question=question, contexts=contexts)

        model_name = self.settings.nvidia_model_name
        if not model_name or not model_name.strip():
            raise RuntimeError(
                "NVIDIA_MODEL_NAME is not configured. "
                "Configure a model name in settings to generate answers."
            )

        client = self._get_client()

        response = client.chat.completions.create(
            model=model_name.strip(),
            messages=messages,
            temperature=self.settings.generation_temperature,
            max_tokens=self.settings.generation_max_tokens,
        )

        return self._extract_answer(response)

    def _extract_answer(self, response: Any) -> str:
        """Extract and validate the generated text from provider response.

        Args:
            response: Chat completion response from the provider.

        Returns:
            The trimmed text content of the first choice.

        Raises:
            RuntimeError: If the response structure is malformed or content is empty.
        """
        if response is None:
            raise RuntimeError("Invalid provider response: received None.")

        if isinstance(response, dict):
            choices = response.get("choices")
        else:
            choices = getattr(response, "choices", None)

        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise RuntimeError(
                "Invalid provider response: missing or empty 'choices' list."
            )

        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
        else:
            message = getattr(first_choice, "message", None)

        if message is None:
            raise RuntimeError(
                "Invalid provider response: missing 'message' in first choice."
            )

        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "Invalid provider response: answer content must be a non-empty string."
            )

        return content.strip()

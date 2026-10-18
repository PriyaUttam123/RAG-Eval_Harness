"""Prompt construction for grounded RAG generation."""

from typing import List

from pydantic import BaseModel


class PromptContext(BaseModel):
    text: str
    source_filename: str = ""
    chunk_index: int = -1


class RAGPromptBuilder:
    """Builds a grounded prompt from a user question and retrieved context."""

    SYSTEM_PROMPT = """You are a helpful question-answering assistant.

Answer the user's question using only the provided context.

Rules:
1. Use only information supported by the context.
2. Do not invent facts or fill gaps with outside knowledge.
3. If the context does not contain enough information to answer, say:
   "I don't have enough information in the provided context."
4. Keep the answer concise and directly answer the question.
5. When useful, mention the source filename associated with the supporting context."""

    def build(
        self,
        question: str,
        contexts: List[PromptContext],
    ) -> List[dict]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        cleaned_question = question.strip()

        context_sections = []

        for index, context in enumerate(contexts, start=1):
            source = context.source_filename or "unknown source"

            context_sections.append(
                f"[Context {index} | Source: {source}]\n"
                f"{context.text}"
            )

        context_text = "\n\n".join(context_sections)

        if not context_text:
            context_text = "[No relevant context was retrieved.]"

        user_prompt = (
            f"Context:\n"
            f"{context_text}\n\n"
            f"Question:\n"
            f"{cleaned_question}\n\n"
            f"Answer:"
        )

        return [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

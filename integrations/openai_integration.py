"""Simplified OpenAI chat integration used in tests.

This module provides a minimal wrapper around the OpenAI chat API that
integrates with the :class:`~core.memory_engine.MemoryEngine`.  The
original project contains a much more feature rich implementation that
relies on external packages such as LangChain.  Those heavy dependencies
are intentionally avoided here so that the test-suite can run in a
lightweight execution environment.

Only the functionality exercised by the unit tests is implemented:

* initialization wires together the OpenAI client, an embedding provider
  and the :class:`ContextBuilder`
* ``chat_with_memory`` sends a message to the OpenAI API and optionally
  stores the conversation in memory
* ``add_memory_with_embedding`` delegates to ``MemoryEngine``
* basic management of a small in-memory conversation buffer

The implementation purposefully keeps the surface area small and has no
runtime dependency on LangChain.  When the real project is executed the
full featured version of this module should be used instead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from openai import OpenAI

from core.conversation_buffer import ConversationBuffer
from core.context_builder import ContextBuilder
from core.memory_engine import Memory, MemoryEngine
from core import logging_config
from core.logging_config import monitor_performance

from .embeddings import OpenAIEmbeddings


class OpenAIIntegration:
    """Light-weight OpenAI chat helper with memory support."""

    def __init__(
        self,
        api_key: str,
        memory_engine: MemoryEngine,
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        conversation_buffer_size: int = 20,
    ) -> None:
        # Create OpenAI client and helper components
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.memory_engine = memory_engine
        self.embeddings = OpenAIEmbeddings(api_key, embedding_model)
        self.context_builder = ContextBuilder(
            memory_engine, profile_query="Jeremy wife Ashley kids dogs age"
        )
        self.conversation_buffer = ConversationBuffer(
            max_messages=conversation_buffer_size
        )
        self.logger = logging_config.get_logger("openai_integration")

        self.logger.info(
            "OpenAI integration initialized",
            extra={"model": model, "embedding_model": embedding_model},
        )

    # ------------------------------------------------------------------
    @monitor_performance("chat_with_memory")
    def chat_with_memory(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        include_recent: int = 5,
        include_relevant: int = 5,
        remember_response: bool = True,
        thread_id: str = "default",
    ) -> str:
        """Send a chat message and optionally store the conversation."""

        # Build context from existing memories
        context = self.context_builder.build_context(
            query=message, include_recent=include_recent, include_relevant=include_relevant
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            # Include previous context as a system message so tests can assert
            messages.append({"role": "system", "content": f"Previous context:\n{context}"})
        messages.append({"role": "user", "content": message})

        self.logger.debug(
            "Sending chat completion request",
            extra={"model": self.model, "message_length": len(message)},
        )

        response = self.client.chat.completions.create(
        model=self.model, messages=messages
        )
        answer = response.choices[0].message.content

        if remember_response:
            # Store both user message and assistant response in memory engine
            self.memory_engine.add_memory(
                f"User: {message}",
                {"type": "user_message", "thread_id": thread_id},
            )
            self.memory_engine.add_memory(
                f"Assistant: {answer}",
                {"type": "assistant_response", "thread_id": thread_id},
            )

            # Also keep short-term buffer for building context strings
            self.conversation_buffer.add_message("user", message)
            self.conversation_buffer.add_message("assistant", answer)

        self.logger.debug(
            "Received chat completion",
            extra={"response_length": len(answer)},
        )

        return answer

    # ------------------------------------------------------------------
    def add_memory_with_embedding(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """Convenience helper used in tests.

        Delegates to :meth:`MemoryEngine.add_memory`.  The memory engine is
        responsible for generating embeddings when an ``embedding_provider``
        is configured, which is the behaviour exercised by the tests.
        """

        return self.memory_engine.add_memory(content, metadata)

    # ------------------------------------------------------------------
    def clear_conversation_buffer(self) -> None:
        """Remove all messages from the internal conversation buffer."""

        self.conversation_buffer.clear()

    # ------------------------------------------------------------------
    def get_conversation_buffer_info(self) -> Dict[str, Any]:
        """Return basic statistics about the conversation buffer.

        This mirrors a tiny subset of the functionality of the original
        implementation and is primarily useful for debugging.
        """

        return {
            "custom_message_count": self.conversation_buffer.get_message_count(),
            "custom_max_messages": self.conversation_buffer.max_messages,
        }


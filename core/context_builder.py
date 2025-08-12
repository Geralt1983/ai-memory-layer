from typing import Optional
from .memory_engine import MemoryEngine


class ContextBuilder:
    def __init__(
        self,
        memory_engine: MemoryEngine,
        max_context_length: int = 4000,
        profile_query: Optional[str] = None,
    ):
        """Initialize the context builder.

        Args:
            memory_engine: Engine used for memory retrieval.
            max_context_length: Maximum length of the returned context string.
            profile_query: Query used to retrieve persistent profile information.
                If ``None``, no profile memories are included.
        """
        self.memory_engine = memory_engine
        self.max_context_length = max_context_length
        self.profile_query = profile_query

    def build_context(
        self,
        query: Optional[str] = None,
        include_recent: int = 5,
        include_relevant: int = 5,
    ) -> str:
        context_parts = []

        # Include profile information if a query is provided
        if self.profile_query:
            profile_memories = self.memory_engine.search_memories(
                self.profile_query, k=10
            )
            important_info = [
                memory.content
                for memory in profile_memories
                if memory.relevance_score > 0.5
            ]

            if important_info:
                context_parts.append("## Profile Information:")
                context_parts.extend(f"- {info}" for info in important_info[:5])

        # Add recent memories
        if include_recent > 0:
            recent_memories = self.memory_engine.get_recent_memories(include_recent)
            if recent_memories:
                context_parts.append("## Recent Context:")
                for memory in recent_memories:
                    context_parts.append(f"- {memory.content}")

        # Add relevant memories based on query
        if query and include_relevant > 0:
            relevant_memories = self.memory_engine.search_memories(
                query, include_relevant
            )
            if relevant_memories:
                context_parts.append("## Relevant Context:")
                for memory in relevant_memories:
                    context_parts.append(
                        f"- {memory.content} (relevance: {memory.relevance_score:.2f})"
                    )

        # Join and truncate if necessary
        context = "\n".join(context_parts)
        if len(context) > self.max_context_length:
            context = context[: self.max_context_length] + "..."

        return context

    def format_for_prompt(self, context: str, query: str) -> str:
        return f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

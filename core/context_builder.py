from typing import Optional
from .memory_engine import MemoryEngine


class ContextBuilder:
    def __init__(self, memory_engine: MemoryEngine, max_context_length: int = 4000):
        self.memory_engine = memory_engine
        self.max_context_length = max_context_length

    def build_context(
        self,
        query: Optional[str] = None,
        include_recent: int = 5,
        include_relevant: int = 5,
    ) -> str:
        context_parts = []

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
                context_parts.append("\n## Relevant Context:")
                for memory in relevant_memories:
                    context_parts.append(
                        f"- {memory.content} "
                        f"(relevance: {memory.relevance_score:.2f})"
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

"""LLM port — minimal interface the application uses to talk to any model.

Tool calling support is exposed via a generic ``tools`` parameter so we can pass
Anthropic-style tool schemas (Phase 2) or, later, an MCP / OpenAI style schema
without changing the application layer. The return value stays a plain string
(the LLM's final textual answer); tool-execution control flow lives in the
application layer so it can be tested without the LLM.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    """Generate a textual completion, optionally with tool calling."""

    def generate(
        self,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return the model's text response.

        ``tools`` is the provider-specific schema list (e.g. Anthropic tools).
        Implementations should raise their own ``LLMError`` on API failure.
        """
        ...

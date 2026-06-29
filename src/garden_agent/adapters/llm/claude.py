"""Anthropic Claude implementation of :class:`LLMPort`.

================================================================================
Read this header before the code — the *why* matters more than the *how* here.
================================================================================

The job of this adapter
-----------------------
Be the only file in the codebase that knows about the Anthropic SDK. Everywhere
else in the app, the LLM is just a port: "give it a prompt, get back a string".
Swap this file for ``lora_adapter.py`` in Phase 7 and *nothing else changes*.

The one design decision worth understanding
-------------------------------------------
Anthropic's tool-use feature is a **multi-turn conversation**: model emits a
``tool_use`` block, you run the tool, you send the result back as a
``tool_result`` message, the model emits another ``tool_use`` block, and so on
until it gives you a final text answer.

We could have hidden that whole conversation inside ``generate()`` (one call in,
final answer out). We deliberately did **not**. Reasons:

1. The :class:`LLMPort` interface is provider-agnostic — its signature is
   ``generate(prompt, tools) -> str``. If the adapter ran a loop internally,
   every other adapter (LoRA, OpenAI, MCP) would have to reimplement that loop
   with its own quirks. By making the adapter *single-shot*, the loop lives
   in the application layer where every provider shares it.
2. Single-shot is trivial to test: one fake response → one assertion. A
   multi-turn loop would need scripted SDK fakes for every iteration.
3. Observability (Phase 5) becomes easier: each ``generate`` = one trace span.

So the contract is:

    one call to ``generate``  →  one Anthropic ``messages.create`` call
                              →  one envelope returned to the planner

The planner then decides whether to run a tool and call ``generate`` again, or
stop because the model produced a final plan.

The envelope (a 2-key JSON dictionary)
--------------------------------------
The planner already understands a tiny JSON dialect (defined in Slice 1):

* ``{"tool_calls": [{"id": "...", "name": "...", "input": {...}}, ...]}``
* ``{"final_plan": <WateringPlan JSON>}``

Anthropic's response has its own format ("content blocks" — a list of
``tool_use`` / ``text`` / ``thinking`` items). This adapter is the bridge:

    Anthropic content blocks  ──translate──▶  planner envelope (JSON string)

That's *the* trick. The planner has no idea Claude exists.

References worth bookmarking
----------------------------
* Anthropic tool use overview:
  https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
* Anthropic messages API reference:
  https://docs.anthropic.com/en/api/messages
* Hexagonal architecture (Cockburn 2005):
  https://alistair.cockburn.us/hexagonal-architecture/
* Why we don't use ``unittest.mock`` here — fakes vs. mocks (Fowler):
  https://martinfowler.com/articles/mocksArentStubs.html
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import anthropic


class LLMError(RuntimeError):
    """Wraps any Anthropic-side failure.

    Why have our own exception type at all? So application code can write
    ``except LLMError`` once instead of chasing every concrete exception the
    SDK might raise (``APIConnectionError``, ``RateLimitError``,
    ``AuthenticationError``, …). The boundary translation pattern: convert
    foreign error hierarchies into one local type.
    """


class _AnthropicLike(Protocol):
    """The slice of the SDK we actually touch.

    We only ever call ``client.messages.create(...)``. Declaring that as a
    Protocol lets tests pass a 20-line dataclass-based fake (see
    ``tests/unit/adapters/test_claude_adapter.py``) instead of constructing
    a real ``anthropic.Anthropic`` — which would refuse to instantiate without
    a valid-looking API key. Same duck-typing trick we use for the ports.
    """

    messages: Any


class ClaudeAdapter:
    """Concrete :class:`LLMPort` over the Anthropic Python SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
        system_prompt: str | None = None,
        client: _AnthropicLike | None = None,
    ) -> None:
        """Two construction modes.

        Production::

            ClaudeAdapter(api_key="sk-ant-...")     # builds its own client

        Tests::

            ClaudeAdapter(client=FakeAnthropic())   # plugs in a fake

        The ``*`` after ``api_key`` forces every other parameter to be passed
        by name — that makes call sites self-documenting (no mystery
        ``ClaudeAdapter("k", "claude-3", 1024, None, None)`` lines).
        """
        if client is None:
            if not api_key:
                # Fail fast at construction time, not deep inside generate().
                # "Errors should surface at the boundary closest to their
                # cause" — keeps stack traces short and obvious.
                raise LLMError("Anthropic API key is required")
            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------ #
    # Port surface                                                        #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Send ONE prompt to Claude, return ONE envelope string.

        The whole conversation history is already baked into ``prompt`` by the
        planner (it concatenates ``SYSTEM/USER/ASSISTANT/TOOL_RESULT`` lines).
        We do not maintain message history ourselves — that's the planner's
        job. We just forward whatever string we're given.
        """
        # Build the call kwargs incrementally so we only send keys that have
        # a value. The Anthropic SDK is strict: passing ``tools=None`` or
        # ``system=None`` is different from omitting the key.
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._system_prompt is not None:
            kwargs["system"] = self._system_prompt
        if tools:
            kwargs["tools"] = tools

        # ---- Boundary translation: SDK exceptions → our LLMError ----------
        # Two excepts on purpose: the first catches every Anthropic-specific
        # error (rate limits, auth, overload, …). The second is a safety net
        # for anything genuinely unexpected (e.g. JSON encoding of huge inputs
        # that fails before the request goes out). Both end up wrapped in
        # ``LLMError`` so the application sees one error type.
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Anthropic API call failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — defensive boundary
            raise LLMError(f"Unexpected error talking to Anthropic: {exc}") from exc

        return self._translate_response(response)

    # ------------------------------------------------------------------ #
    # Translation                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _translate_response(response: Any) -> str:
        """Anthropic response → planner envelope (JSON string).

        Anthropic's ``response.content`` is a list of "content blocks". Each
        block has a ``type``: ``"text"``, ``"tool_use"``, or ``"thinking"``
        (the last one is from extended-thinking models). Example shape::

            response.content = [
                Block(type="text",     text="Let me check the weather."),
                Block(type="tool_use", id="toolu_01", name="get_weather_forecast",
                                       input={"location": "Paris", "days": 7}),
            ]

        Rules we apply:

        * **If ANY ``tool_use`` block is present**, we ignore the text and
          emit a ``{"tool_calls": [...]}`` envelope. The model is asking us
          to run tools — the surrounding chatter ("Let me check the weather.")
          is just narrative we don't need.
        * **Otherwise** we concatenate all ``text`` blocks and return that
          raw string. The system prompt instructs the model that, when not
          calling tools, its text MUST already be a ``{"final_plan": ...}``
          JSON object — so the planner can ``json.loads`` it directly.
        * **``thinking`` blocks** are silently ignored for now. In Phase 5
          (observability) we will surface them in Langfuse traces instead of
          dropping them on the floor.
        """
        blocks = getattr(response, "content", None)
        if blocks is None:
            # Defensive: the SDK has always returned a content list, but if
            # an upgrade ever changes that we want a clear error, not an
            # ``AttributeError`` deep in a comprehension.
            raise LLMError("Anthropic response has no 'content' field")

        tool_calls: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for block in blocks:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        # SDK gives us ``input`` as an already-parsed dict.
                        # ``or {}`` normalises the (theoretical) ``None`` case
                        # so the planner can always do ``call["input"][key]``.
                        "input": getattr(block, "input", {}) or {},
                    }
                )
            elif block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            # Unknown types fall through silently (forward-compat).

        if tool_calls:
            # Tool calls win over text: the model wants to act, not to talk.
            return json.dumps({"tool_calls": tool_calls})

        if not text_parts:
            # Empty response (no tool_use, no text) — almost certainly a bug
            # in the prompt or the SDK. Fail loudly instead of returning "".
            raise LLMError("Anthropic response had no text or tool_use blocks")

        return "".join(text_parts)

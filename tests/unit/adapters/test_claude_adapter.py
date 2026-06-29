"""Unit tests for :class:`ClaudeAdapter`.

These tests use a tiny **fake Anthropic client** rather than ``unittest.mock``
to honour the project rule (see CLAUDE.md / spec §17). The fake is a 20-line
duck-typed object that satisfies the bits of the SDK the adapter actually
touches — which is also the most realistic regression test you can write
against a third-party API: if Anthropic changes the response shape, this
fake stops mirroring reality and you'll see it in code review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic
import pytest

from garden_agent.adapters.llm.claude import ClaudeAdapter, LLMError
from garden_agent.ports.llm import LLMPort

# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    """Mimics one entry in ``response.content``."""

    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Response:
    content: list[_Block]


class _FakeMessages:
    def __init__(self, response: _Response | Exception):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@dataclass
class _FakeAnthropic:
    messages: _FakeMessages


def _client_returning(blocks: list[_Block]) -> _FakeAnthropic:
    return _FakeAnthropic(messages=_FakeMessages(_Response(content=blocks)))


def _client_raising(exc: Exception) -> _FakeAnthropic:
    return _FakeAnthropic(messages=_FakeMessages(exc))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_adapter_satisfies_port_protocol() -> None:
    adapter = ClaudeAdapter(client=_client_returning([_Block(type="text", text="ok")]))
    assert isinstance(adapter, LLMPort)


def test_constructor_requires_api_key_when_no_client() -> None:
    with pytest.raises(LLMError, match="API key"):
        ClaudeAdapter(api_key=None)


# ---------------------------------------------------------------------------
# Translation: text-only responses
# ---------------------------------------------------------------------------


def test_generate_returns_concatenated_text_blocks() -> None:
    blocks = [_Block(type="text", text="hello "), _Block(type="text", text="world")]
    client = _client_returning(blocks)
    adapter = ClaudeAdapter(client=client, model="claude-test", max_tokens=42)

    out = adapter.generate("ping")

    assert out == "hello world"
    # The adapter sent the prompt as a single user message and respected the knobs.
    call = client.messages.calls[0]
    assert call["model"] == "claude-test"
    assert call["max_tokens"] == 42
    assert call["messages"] == [{"role": "user", "content": "ping"}]
    assert "tools" not in call  # we didn't pass any
    assert "system" not in call  # no system prompt configured


def test_generate_forwards_tools_and_system_prompt() -> None:
    blocks = [_Block(type="text", text='{"final_plan": {}}')]
    client = _client_returning(blocks)
    adapter = ClaudeAdapter(client=client, system_prompt="be terse")

    tools = [{"name": "noop", "description": "", "input_schema": {"type": "object"}}]
    adapter.generate("prompt", tools=tools)

    call = client.messages.calls[0]
    assert call["tools"] == tools
    assert call["system"] == "be terse"


# ---------------------------------------------------------------------------
# Translation: tool_use blocks → planner envelope
# ---------------------------------------------------------------------------


def test_tool_use_blocks_become_tool_calls_envelope() -> None:
    blocks = [
        _Block(type="text", text="thinking..."),  # text mixed with tool calls
        _Block(
            type="tool_use",
            id="toolu_01",
            name="get_garden_state",
            input={"garden_id": "g-001"},
        ),
        _Block(
            type="tool_use",
            id="toolu_02",
            name="get_weather_forecast",
            input={"location": "Paris, France", "days": 7},
        ),
    ]
    client = _client_returning(blocks)
    adapter = ClaudeAdapter(client=client)

    raw = adapter.generate("plan it")
    envelope = json.loads(raw)

    # When tool_use is present, the envelope MUST be tool_calls (no text leakage).
    assert set(envelope) == {"tool_calls"}
    assert envelope["tool_calls"] == [
        {"id": "toolu_01", "name": "get_garden_state", "input": {"garden_id": "g-001"}},
        {
            "id": "toolu_02",
            "name": "get_weather_forecast",
            "input": {"location": "Paris, France", "days": 7},
        },
    ]


def test_tool_use_with_none_input_normalises_to_empty_dict() -> None:
    blocks = [_Block(type="tool_use", id="t", name="get_garden_state", input=None)]  # type: ignore[arg-type]
    adapter = ClaudeAdapter(client=_client_returning(blocks))

    envelope = json.loads(adapter.generate("plan it"))
    assert envelope["tool_calls"][0]["input"] == {}


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def test_anthropic_error_is_wrapped_as_llm_error() -> None:
    client = _client_raising(anthropic.AnthropicError("rate limit"))
    adapter = ClaudeAdapter(client=client)

    with pytest.raises(LLMError, match="Anthropic API call failed"):
        adapter.generate("ping")


def test_unexpected_error_is_also_wrapped() -> None:
    client = _client_raising(RuntimeError("network died"))
    adapter = ClaudeAdapter(client=client)

    with pytest.raises(LLMError, match="Unexpected error"):
        adapter.generate("ping")


def test_response_without_content_raises() -> None:
    @dataclass
    class _Empty:
        pass  # no .content

    client = _FakeAnthropic(messages=_FakeMessages(_Empty()))  # type: ignore[arg-type]
    adapter = ClaudeAdapter(client=client)

    with pytest.raises(LLMError, match="no 'content'"):
        adapter.generate("ping")


def test_response_with_only_unknown_blocks_raises() -> None:
    blocks = [_Block(type="thinking", text="ignored")]
    adapter = ClaudeAdapter(client=_client_returning(blocks))
    with pytest.raises(LLMError, match="no text or tool_use"):
        adapter.generate("ping")

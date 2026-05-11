"""Unit tests for LLMClient JSON-extraction robustness.

The Anthropic API frequently returns JSON wrapped in markdown fences,
prefaced with prose, or as plain JSON. Verify the regex-based extractor
handles all common shapes and raises cleanly on unparseable input.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.llm_client import LLMClient


def _mock_response(text: str) -> MagicMock:
    """Build a fake Anthropic response carrying a single text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


@pytest.mark.parametrize(
    "raw_text,expected",
    [
        # Pure JSON, no fence
        ('{"relevance_score": 9, "reason": "core"}', {"relevance_score": 9, "reason": "core"}),
        # JSON in a ```json fence with newline
        (
            '```json\n{"relevance_score": 7, "reason": "ok"}\n```',
            {"relevance_score": 7, "reason": "ok"},
        ),
        # JSON in a bare ``` fence
        ('```\n{"a": 1}\n```', {"a": 1}),
        # Single-line fence without internal newlines (the prior IndexError edge case)
        ('```{"x": 2}```', {"x": 2}),
        # Prose preceding the JSON block
        ('Here is the result: {"relevance_score": 5}', {"relevance_score": 5}),
        # JSON followed by prose
        ('{"k": "v"} -- explanation follows', {"k": "v"}),
        # Whitespace and trailing newline
        ('  {"k": "v"}  \n', {"k": "v"}),
    ],
)
@pytest.mark.asyncio
async def test_call_json_extracts_json_from_various_shapes(monkeypatch, raw_text, expected):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    client = LLMClient(model="test-model")
    client.client = MagicMock()
    client.client.messages = MagicMock()
    client.client.messages.create = AsyncMock(return_value=_mock_response(raw_text))

    result = await client.call_json("system", "user")
    assert result == expected


@pytest.mark.asyncio
async def test_call_json_raises_on_unparseable_text(monkeypatch):
    """Truly malformed responses still raise — surfaces real LLM failures
    rather than silently returning {}."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    client = LLMClient(model="test-model")
    client.client = MagicMock()
    client.client.messages = MagicMock()
    client.client.messages.create = AsyncMock(return_value=_mock_response("no json here at all"))

    # tenacity wraps the JSONDecodeError; we accept either the raw error or a RetryError
    with pytest.raises((json.JSONDecodeError, Exception)):
        await client.call_json("system", "user")

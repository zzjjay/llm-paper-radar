"""Async Anthropic client wrapper with prompt-cache support and retries."""
from __future__ import annotations

import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
    async def call_json(self, system: str, user: str, max_tokens: int = 1024) -> dict:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)


def load_prompt(path: Path) -> str:
    return Path(path).read_text()

"""Async Anthropic client wrapper with prompt-cache support and retries."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


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
        # Tolerate markdown fences and inline prose: extract first {...} block
        m = _JSON_BLOCK_RE.search(text)
        if m:
            text = m.group(0)
        return json.loads(text)


def load_prompt(path: Path) -> str:
    return Path(path).read_text()

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

    async def call_json(self, system: str, user: str, max_tokens: int = 1024) -> dict:
        # First try with prompt cache (cheap path). If all retries fail —
        # most commonly because the cached system block returned a corrupt
        # or empty response — fall back to one more retry-burst with the
        # cache disabled.
        try:
            return await self._call_json_attempt(system, user, max_tokens, use_cache=True)
        except Exception:
            return await self._call_json_attempt(system, user, max_tokens, use_cache=False)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=20))
    async def _call_json_attempt(
        self, system: str, user: str, max_tokens: int, use_cache: bool
    ) -> dict:
        system_block: dict = {"type": "text", "text": system}
        if use_cache:
            system_block["cache_control"] = {"type": "ephemeral"}
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[system_block],
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

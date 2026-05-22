from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceName = Literal[
    "arxiv",
    "arxiv_authors",
    "hf_daily",
    "openreview",
    "reddit",
]


class SourceRecord(BaseModel):
    name: SourceName
    fetched_at: datetime
    extras: dict = Field(default_factory=dict)


class Paper(BaseModel):
    id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str | None
    published_at: datetime
    primary_category: str
    categories: list[str]
    code_url: str | None = None

    sources: list[SourceRecord]

    relevance_score: int | None = None
    relevance_reason: str | None = None
    relevance_breakdown: dict | None = None
    summary: str | None = None
    highlights: list[str] = Field(default_factory=list)
    # Up to 3 prior methods the paper directly compares against / builds on,
    # extracted from the abstract by the summarize step. Each entry:
    # {"name": str, "relation": str, "arxiv_id": str | None}.
    related_methods: list[dict] = Field(default_factory=list)
    seen_before: bool = False


class Source(ABC):
    """Base class. Each concrete source implements `fetch()` returning Papers."""

    name: SourceName

    @abstractmethod
    async def fetch(self, target_date: datetime) -> list[Paper]:
        """Fetch papers for the given date (UTC)."""
        ...

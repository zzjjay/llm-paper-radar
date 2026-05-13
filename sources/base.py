from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceName = Literal[
    "arxiv",
    "hf_daily",
    "reddit",
    "semantic_scholar",
    "papers_with_code",
    "twitter_rsshub",
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
    seen_before: bool = False


class Source(ABC):
    """Base class. Each concrete source implements `fetch()` returning Papers."""

    name: SourceName

    @abstractmethod
    async def fetch(self, target_date: datetime) -> list[Paper]:
        """Fetch papers for the given date (UTC)."""
        ...

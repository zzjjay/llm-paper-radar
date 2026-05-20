from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ArxivConfig(BaseModel):
    enabled: bool = True
    categories: list[str] = ["cs.CL", "cs.LG", "cs.AR"]


class SimpleSourceConfig(BaseModel):
    enabled: bool = True


class RedditConfig(BaseModel):
    enabled: bool = True
    subreddit: str = "LocalLLaMA"
    top_window: str = "day"


class SemanticScholarConfig(BaseModel):
    enabled: bool = True
    seeds_file: str = "seeds.yaml"
    citation_window_days: int = 7


class TwitterConfig(BaseModel):
    enabled: bool = True
    accounts: list[str] = []


class WatchedAuthor(BaseModel):
    name: str
    affiliation: str = ""


class ArxivAuthorsConfig(BaseModel):
    enabled: bool = True
    window_days: int = 7
    # Restrict author search to these arXiv categories. Defaults cover the
    # compression/quant/diffusion space; widen if you want everything.
    categories: list[str] = ["cs.LG", "cs.CL", "cs.CV", "cs.AR", "stat.ML"]
    authors: list[WatchedAuthor] = []


class SourcesConfig(BaseModel):
    arxiv: ArxivConfig = ArxivConfig()
    hf_daily: SimpleSourceConfig = SimpleSourceConfig()
    reddit: RedditConfig = RedditConfig()
    semantic_scholar: SemanticScholarConfig = SemanticScholarConfig()
    twitter_rsshub: TwitterConfig = TwitterConfig()
    arxiv_authors: ArxivAuthorsConfig = ArxivAuthorsConfig()


class KeywordRule(BaseModel):
    pattern: str
    weight: int


class PrefilterConfig(BaseModel):
    """Cheap keyword prefilter that runs before the LLM judge. Papers with
    no whitelist hits AND >= `max_blacklist_hits` blacklist matches are
    hard-gated locally without burning a Haiku call.

    Word-boundary matching is applied — `QuIP` will not match inside
    `equipping`, `MIT` will not match inside `Amit`."""

    enabled: bool = True
    whitelist: list[KeywordRule] = Field(default_factory=list)
    blacklist: list[KeywordRule] = Field(default_factory=list)
    max_blacklist_hits: int = 2


class FilterConfig(BaseModel):
    model: str = "claude-haiku-4-5-20251001"
    concurrency: int = 50
    prefilter: PrefilterConfig = PrefilterConfig()


class SummarizeConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    concurrency: int = 20


class RenderConfig(BaseModel):
    truncate_after: int = 10
    # New bucket enum: [ptq, qat, low_bits, kv_cache, pruning_distill, diffusion]
    topic_caps: dict[str, int] = Field(
        default_factory=lambda: {
            "ptq": 8,
            "low_bits": 5,
            "qat": 5,
            "kv_cache": 5,
            "pruning_distill": 3,
            "diffusion": 3,
            "_default": 2,
        }
    )


class DedupeConfig(BaseModel):
    cross_day_strategy: Literal["strict", "lenient"] = "lenient"
    source_priority: list[str] = []


class Config(BaseModel):
    sources: SourcesConfig = SourcesConfig()
    filter: FilterConfig = FilterConfig()
    summarize: SummarizeConfig = SummarizeConfig()
    render: RenderConfig = RenderConfig()
    dedupe: DedupeConfig = DedupeConfig()


def load_config(path: Path = Path("config.yaml")) -> Config:
    data = yaml.safe_load(Path(path).read_text())
    return Config(**data)

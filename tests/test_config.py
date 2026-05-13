from pathlib import Path

from pipeline.config import Config, load_config


def test_load_config_from_yaml(tmp_path: Path):
    yaml_text = """
sources:
  arxiv:
    enabled: true
    categories: [cs.CL, cs.LG]
  hf_daily:
    enabled: true
filter:
  model: claude-haiku-4-5-20251001
  threshold: 7
  concurrency: 50
summarize:
  model: claude-sonnet-4-6
  concurrency: 20
render:
  truncate_after: 10
  topic_caps:
    compression: 5
    _default: 3
dedupe:
  cross_day_strategy: lenient
  source_priority: [hf_daily, arxiv]
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.filter.threshold == 7
    assert cfg.sources.arxiv.enabled is True
    assert cfg.sources.arxiv.categories == ["cs.CL", "cs.LG"]
    assert cfg.dedupe.source_priority[0] == "hf_daily"
    assert cfg.render.topic_caps == {"compression": 5, "_default": 3}

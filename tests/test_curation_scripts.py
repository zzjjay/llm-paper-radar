"""Tests for scripts/seed_add.py and scripts/seed_reject.py.

These cover the deterministic paths only: name → arxiv id fuzzy lookup,
bucket lookup from scored cache, idempotent dedup, blacklist append,
log-line schema. The arXiv-API fetch and Sonnet fallback are not exercised
here (they need network + an API key); seed_add's main() guards those
paths separately.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch):
    """Build a temp repo skeleton (seeds.yaml + config.yaml + data dirs) and
    patch the scripts to look at it instead of the real repo. Run the
    scripts via subprocess with cwd=tmp_path so REPO_ROOT inside them
    resolves correctly."""
    # ---- seeds.yaml ----
    (tmp_path / "seeds.yaml").write_text(
        textwrap.dedent("""
        seeds:
          # ---- PTQ (primary) ----
          - { id: arXiv:2210.17323, name: GPTQ,        category: ptq }
          - { id: arXiv:2306.00978, name: AWQ,         category: ptq }

          # ---- Low-bit (≤ 2 bits, primary) ----
          - { id: arXiv:2402.17764, name: "BitNet b1.58", category: low_bits }

          # ---- QAT (secondary) ----
          - { id: arXiv:2305.17888, name: LLM-QAT,     category: qat }

          # ---- KV cache (secondary) ----
          - { id: arXiv:2402.02750, name: KIVI,        category: kv_cache }

          # ---- Pruning & distillation (low priority, merged bucket) ----
          - { id: arXiv:2306.11695, name: Wanda,       category: pruning_distill }

          # ---- Diffusion (low priority) ----
          - { id: arXiv:2302.04304, name: Q-Diffusion, category: diffusion }
        """).lstrip()
    )

    # ---- config.yaml (minimal but valid for the schema) ----
    (tmp_path / "config.yaml").write_text(
        textwrap.dedent("""
        filter:
          model: claude-sonnet-4-6
          concurrency: 50
          prefilter:
            enabled: true
            max_blacklist_hits: 2
            whitelist:
              - { pattern: "PTQ",  weight: 6 }
            blacklist:
              - { pattern: "ImageNet",          weight: -3 }
              - { pattern: "federated learning", weight: -4 }
        """).lstrip()
    )

    # ---- data dirs with a fake scored entry and a fake summarized entry ----
    scored = tmp_path / "data" / "scored"
    scored.mkdir(parents=True)
    (scored / "2026-05-20.json").write_text(json.dumps([
        {
            "id": "2605.99001",
            "title": "NewQuant: A snazzy W4A4 PTQ recipe for Llama-70B",
            "relevance_score": 9,
            "relevance_breakdown": {"hard_gate": False, "topic_bucket": "ptq"},
        },
        {
            "id": "2605.99002",
            "title": "Off-topic: An ImageNet study",
            "relevance_score": 0,
            "relevance_breakdown": {"hard_gate": True, "topic_bucket": "unknown"},
        },
    ]))
    summarized = tmp_path / "data" / "summarized"
    summarized.mkdir(parents=True)
    (summarized / "2026-05-20.json").write_text(json.dumps([
        {
            "id": "2605.99001",
            "title": "NewQuant: A snazzy W4A4 PTQ recipe for Llama-70B",
            "relevance_breakdown": {"topic_bucket": "ptq"},
        },
        {
            "id": "2605.99003",
            "title": "NewQuant-v2: an extension of NewQuant",  # shares 'NewQuant'
            "relevance_breakdown": {"topic_bucket": "ptq"},
        },
    ]))

    return tmp_path


def _run(repo: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    """Run a script against the tmp repo. Use LLM_RADAR_REPO_ROOT to point
    the script at the tmp dir (so its REPO_ROOT doesn't resolve to the real
    one via __file__)."""
    import os as _os
    env = {**_os.environ, "LLM_RADAR_REPO_ROOT": str(repo)}
    src = REPO_ROOT / "scripts" / script
    return subprocess.run(
        [sys.executable, str(src), *args],
        cwd=repo, capture_output=True, text=True, env=env,
    )


# ---------------------------------------------------------------------------
# seed_add.py


def test_seed_add_dedup_is_noop(isolated_repo: Path):
    """Adding an already-present arxiv id should print 'already in' and exit 0."""
    before = (isolated_repo / "seeds.yaml").read_text()
    r = _run(isolated_repo, "seed_add.py", "--arxiv-id", "2210.17323")
    assert r.returncode == 0, r.stderr
    assert "already in seeds.yaml" in r.stdout
    after = (isolated_repo / "seeds.yaml").read_text()
    assert before == after, "dedup must not touch the file"


def test_seed_add_uses_scored_cache_bucket(isolated_repo: Path):
    """arxiv id new to seeds.yaml but already judged → bucket comes from cache."""
    r = _run(isolated_repo, "seed_add.py", "--arxiv-id", "2605.99001")
    assert r.returncode == 0, r.stderr
    assert "via scored_cache" in r.stdout
    assert "bucket=ptq" in r.stdout
    seeds = yaml.safe_load((isolated_repo / "seeds.yaml").read_text())["seeds"]
    matching = [s for s in seeds if s["id"] == "arXiv:2605.99001"]
    assert len(matching) == 1
    assert matching[0]["category"] == "ptq"
    log = (isolated_repo / "data" / "curation" / "accepted.jsonl").read_text().strip()
    entry = json.loads(log)
    assert entry["arxiv_id"] == "2605.99001"
    assert entry["bucket_source"] == "scored_cache"


def test_seed_add_dry_run_writes_nothing(isolated_repo: Path):
    before = (isolated_repo / "seeds.yaml").read_text()
    r = _run(isolated_repo, "seed_add.py", "--arxiv-id", "2605.99001", "--dry-run")
    assert r.returncode == 0
    assert "--dry-run: no files changed" in r.stdout
    assert (isolated_repo / "seeds.yaml").read_text() == before
    assert not (isolated_repo / "data" / "curation" / "accepted.jsonl").exists()


def test_seed_add_name_unique_match(isolated_repo: Path):
    """A needle present in only one summarized title resolves cleanly."""
    r = _run(isolated_repo, "seed_add.py", "--name", "snazzy")
    assert r.returncode == 0, r.stderr
    assert "via scored_cache" in r.stdout
    seeds = yaml.safe_load((isolated_repo / "seeds.yaml").read_text())["seeds"]
    assert any(s["id"] == "arXiv:2605.99001" for s in seeds)


def test_seed_add_name_ambiguous(isolated_repo: Path):
    """'NewQuant' substring appears in both fixture titles → must refuse and
    print the candidates."""
    r = _run(isolated_repo, "seed_add.py", "--name", "NewQuant")
    assert r.returncode != 0
    assert "disambiguate" in r.stderr
    assert "2605.99001" in r.stderr and "2605.99003" in r.stderr


def test_seed_add_bucket_override(isolated_repo: Path):
    """--bucket flag overrides everything, skips Sonnet fallback even if cache
    lookup would have worked."""
    r = _run(isolated_repo, "seed_add.py", "--arxiv-id", "2605.99001",
             "--bucket", "low_bits")
    assert r.returncode == 0
    assert "via cli" in r.stdout
    assert "bucket=low_bits" in r.stdout
    seeds = yaml.safe_load((isolated_repo / "seeds.yaml").read_text())["seeds"]
    match = [s for s in seeds if s["id"] == "arXiv:2605.99001"]
    assert match[0]["category"] == "low_bits"


def test_seed_add_preserves_section_layout(isolated_repo: Path):
    """New PTQ seed should land inside the PTQ section, not at the end of the file."""
    _run(isolated_repo, "seed_add.py", "--arxiv-id", "2605.99001")
    lines = (isolated_repo / "seeds.yaml").read_text().splitlines()
    ptq_header = next(i for i, ln in enumerate(lines) if "PTQ (primary)" in ln)
    lowbits_header = next(i for i, ln in enumerate(lines) if "Low-bit" in ln)
    # New line must be between PTQ header and Low-bit header
    new_line_idx = next(i for i, ln in enumerate(lines) if "2605.99001" in ln)
    assert ptq_header < new_line_idx < lowbits_header


# ---------------------------------------------------------------------------
# seed_reject.py


def test_seed_reject_logs_minimal(isolated_repo: Path):
    r = _run(isolated_repo, "seed_reject.py", "--arxiv-id", "2605.99002",
             "--reason", "off-topic, ImageNet")
    assert r.returncode == 0, r.stderr
    log = (isolated_repo / "data" / "curation" / "rejected.jsonl").read_text().strip()
    entry = json.loads(log)
    assert entry["arxiv_id"] == "2605.99002"
    assert entry["reason"] == "off-topic, ImageNet"
    assert entry["action"] == "reject"
    # No blacklist mutation when --add-blacklist not given
    cfg = yaml.safe_load((isolated_repo / "config.yaml").read_text())
    patterns = {e["pattern"] for e in cfg["filter"]["prefilter"]["blacklist"]}
    assert patterns == {"ImageNet", "federated learning"}


def test_seed_reject_adds_blacklist_patterns(isolated_repo: Path):
    r = _run(isolated_repo, "seed_reject.py", "--arxiv-id", "2605.99002",
             "--reason", "diffusion-only",
             "--add-blacklist", "stable diffusion, FID metric")
    assert r.returncode == 0, r.stderr
    cfg = yaml.safe_load((isolated_repo / "config.yaml").read_text())
    patterns = {e["pattern"] for e in cfg["filter"]["prefilter"]["blacklist"]}
    assert "stable diffusion" in patterns
    assert "FID metric" in patterns
    # Log entry should record what was added
    log = (isolated_repo / "data" / "curation" / "rejected.jsonl").read_text().strip()
    entry = json.loads(log)
    assert entry["blacklist_added"] == ["stable diffusion", "FID metric"]


def test_seed_reject_blacklist_dedup(isolated_repo: Path):
    """Re-adding an existing blacklist pattern should be a no-op for that pattern."""
    r = _run(isolated_repo, "seed_reject.py", "--arxiv-id", "2605.99002",
             "--reason", "test",
             "--add-blacklist", "ImageNet, new pattern X")
    assert r.returncode == 0, r.stderr
    cfg = yaml.safe_load((isolated_repo / "config.yaml").read_text())
    bl = cfg["filter"]["prefilter"]["blacklist"]
    # ImageNet should appear exactly once
    assert sum(1 for e in bl if e["pattern"] == "ImageNet") == 1
    # new pattern should be present
    assert any(e["pattern"] == "new pattern X" for e in bl)

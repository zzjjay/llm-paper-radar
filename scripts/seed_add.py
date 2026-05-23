#!/usr/bin/env python3
"""Append a positive seed to seeds.yaml.

Usage (either flag is enough; if both, --arxiv-id wins):

    python scripts/seed_add.py --arxiv-id 2605.19561
    python scripts/seed_add.py --name TORQ
    python scripts/seed_add.py --arxiv-id 2605.19561 --name TORQ

Bucket is auto-detected in this order:

  1. If the paper appears in any data/scored/*.json with a real (non-hard-gated)
     topic_bucket, that value is used. This is the cheapest path and matches
     whatever the daily Haiku judge decided.
  2. Otherwise, fetch title+abstract from arXiv, call Haiku with the daily
     filter rubric (prompts/relevance.md). If Haiku hard-gates, refuse to add
     (the paper does not belong in seeds.yaml) unless --bucket is given.
  3. If --bucket is given on the command line, it overrides everything.

Dedup is by arXiv id: re-running with the same id is a no-op.

Title resolution when only --name is given:
  - fuzzy substring match against data/summarized/*.json titles
  - exactly one hit → use its arXiv id + bucket
  - multiple hits → print them and exit non-zero, asking the user to disambiguate
  - zero hits → exit non-zero, ask for --arxiv-id
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("LLM_RADAR_REPO_ROOT") or Path(__file__).resolve().parent.parent)
SEEDS_PATH = REPO_ROOT / "seeds.yaml"
SCORED_DIR = REPO_ROOT / "data" / "scored"
SUMMARIZED_DIR = REPO_ROOT / "data" / "summarized"

VALID_BUCKETS = {
    "ptq", "low_bits", "qat", "kv_cache", "pruning_distill", "diffusion", "trending",
}

# Header comments mark the start of each bucket section in seeds.yaml.
# When inserting, we append to the LAST line of the matching section so
# the file's hand-curated layout survives.
SECTION_HEADER = {
    "ptq":             "# ---- PTQ (primary) ----",
    "low_bits":        "# ---- Low-bit (≤ 2 bits, primary) ----",
    "qat":             "# ---- QAT (secondary) ----",
    "kv_cache":        "# ---- KV cache (secondary) ----",
    "pruning_distill": "# ---- Pruning & distillation (low priority, merged bucket) ----",
    "diffusion":       "# ---- Diffusion (low priority) ----",
    "trending":        "# ---- Trending (hf_daily-popular, no other bucket fits) ----",
}

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def normalize_arxiv_id(raw: str) -> str:
    """Accept '2605.19561', 'arXiv:2605.19561', 'http://arxiv.org/abs/2605.19561v2'."""
    raw = raw.strip()
    m = re.search(r"(\d{4}\.\d{4,5})", raw)
    if not m:
        raise ValueError(f"can't parse arXiv id from {raw!r}")
    return m.group(1)


# ---------------------------------------------------------------------------
# Step 1: name → arxiv id (via local summarized cache)


def resolve_name_to_id(name: str) -> tuple[str, str | None]:
    """Return (arxiv_id, bucket_hint_or_None). Bucket hint comes from the
    same scored record so we don't have to re-judge."""
    if not SUMMARIZED_DIR.exists():
        sys.exit(f"name lookup needs {SUMMARIZED_DIR} but it doesn't exist")
    needle = name.lower().strip()
    hits: list[tuple[str, str, str | None]] = []  # (id, title, bucket)
    seen_ids: set[str] = set()
    for f in sorted(SUMMARIZED_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for p in data:
            title = (p.get("title") or "").strip()
            if not title or needle not in title.lower():
                continue
            pid = p.get("id", "")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            arxiv_id = (p.get("id") or "").split(":")[-1] if pid else ""
            try:
                arxiv_id = normalize_arxiv_id(arxiv_id)
            except ValueError:
                continue
            bd = p.get("relevance_breakdown") or {}
            bucket = bd.get("topic_bucket") if bd.get("topic_bucket") in VALID_BUCKETS else None
            hits.append((arxiv_id, title, bucket))
    if not hits:
        sys.exit(
            f"no paper title contains {name!r} in data/summarized/. "
            "Pass --arxiv-id directly."
        )
    if len(hits) > 1:
        print(f"name {name!r} matched {len(hits)} papers:", file=sys.stderr)
        for arxiv_id, title, bucket in hits:
            print(f"  {arxiv_id}  [{bucket or '?'}]  {title}", file=sys.stderr)
        sys.exit("disambiguate with --arxiv-id")
    arxiv_id, _title, bucket = hits[0]
    return arxiv_id, bucket


# ---------------------------------------------------------------------------
# Step 2: bucket auto-detection


def bucket_from_scored_cache(arxiv_id: str) -> str | None:
    """Search recent data/scored/*.json for this paper's already-judged bucket.
    Skips hard-gated entries — those have placeholder bucket values.
    Paper.id may be stored as `2605.19561` or `arxiv:2605.19561` depending
    on the source; match by the trailing arxiv id segment either way."""
    if not SCORED_DIR.exists():
        return None
    for f in sorted(SCORED_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for p in data:
            pid_tail = (p.get("id") or "").split(":")[-1].strip()
            if pid_tail != arxiv_id:
                continue
            bd = p.get("relevance_breakdown") or {}
            if bd.get("hard_gate"):
                continue
            bucket = bd.get("topic_bucket")
            if bucket in VALID_BUCKETS:
                return bucket
    return None


def fetch_arxiv_metadata(arxiv_id: str) -> tuple[str, str]:
    """Return (title, abstract) by querying the arXiv API."""
    url = (
        "http://export.arxiv.org/api/query?"
        + urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
    )
    req = urllib.request.Request(url, headers={"User-Agent": "llm-paper-radar/seed_add"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(body)
    entry = root.find("atom:entry", ns)
    if entry is None:
        raise RuntimeError(f"arXiv returned no entry for {arxiv_id}")
    title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
    abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
    if not title:
        raise RuntimeError(f"arXiv entry for {arxiv_id} has no title")
    return title, abstract


def bucket_from_haiku(arxiv_id: str) -> tuple[str | None, str | None]:
    """Fetch metadata + call Haiku with the daily filter rubric.
    Returns (bucket, title). Bucket is None if Haiku hard-gates."""
    title, abstract = fetch_arxiv_metadata(arxiv_id)
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_BASE_URL"):
        sys.exit(
            "Haiku fallback needs ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL. "
            "Either set it, or pass --bucket on the command line."
        )
    # Import lazily so the script works without anthropic installed when only
    # the scored-cache or --bucket paths are exercised.
    from pipeline.config import load_config
    from pipeline.llm_client import LLMClient, load_prompt

    cfg = load_config()
    client = LLMClient(model=cfg.filter.model)
    prompt = load_prompt(REPO_ROOT / "prompts" / "relevance.md")
    user_msg = f"Title: {title}\n\nAbstract: {abstract}"
    result = asyncio.run(client.call_json(prompt, user_msg, max_tokens=600))
    if result.get("hard_gate"):
        return None, title
    bucket = result.get("topic_bucket")
    if bucket not in VALID_BUCKETS:
        return None, title
    return bucket, title


# ---------------------------------------------------------------------------
# Step 3: insert into seeds.yaml


def is_already_seeded(arxiv_id: str) -> tuple[bool, str | None]:
    """Return (already_present, existing_bucket)."""
    if not SEEDS_PATH.exists():
        return False, None
    data = yaml.safe_load(SEEDS_PATH.read_text()) or {}
    for s in data.get("seeds", []):
        sid = (s.get("id") or "").split(":")[-1].strip()
        if sid == arxiv_id:
            return True, s.get("category")
    return False, None


def insert_seed_line(arxiv_id: str, name: str, bucket: str, note: str | None) -> None:
    """Append a seed line under the matching `# ---- ... ----` section.
    String-level edit (not yaml dump) so the existing layout / aligned
    `name:` columns survive."""
    header = SECTION_HEADER[bucket]
    lines = SEEDS_PATH.read_text().splitlines()

    # Find header, then find the end of that section: either the next blank
    # line (sections are blank-separated) or the next `# ---- ... ----` header.
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == header.strip())
    except StopIteration:
        sys.exit(f"could not find section header {header!r} in seeds.yaml")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i].strip()
        if ln.startswith("# ----"):
            end = i
            break
        if not ln:  # blank line ends the section
            end = i
            break

    # Format the new line. Padding `name` to the widest existing name in the
    # section keeps `category:` columns aligned.
    longest = 0
    for i in range(start + 1, end):
        m = re.search(r"name:\s*(\S+(?:\s+\S+)*?),\s*category:", lines[i])
        if m:
            longest = max(longest, len(m.group(1)))
    longest = max(longest, len(name))
    pad = " " * (longest - len(name))
    new_line = (
        f"  - {{ id: arXiv:{arxiv_id}, name: {name},{pad} category: {bucket} }}"
    )
    if note:
        new_line += f"   # {note}"

    # Insert at `end` (which is either the next header or the blank line).
    # If the line before `end` is the last seed line in the section, just
    # append after it.
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    new_lines = lines[:insert_at] + [new_line] + lines[insert_at:]
    SEEDS_PATH.write_text("\n".join(new_lines) + "\n")


# ---------------------------------------------------------------------------
# Step 4: log to accepted.jsonl


def log_accept(arxiv_id: str, name: str, bucket: str, source: str, note: str | None) -> None:
    log = REPO_ROOT / "data" / "curation" / "accepted.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    from datetime import UTC, datetime
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "arxiv_id": arxiv_id,
        "name": name,
        "bucket": bucket,
        "bucket_source": source,  # "scored_cache" | "haiku" | "cli"
        "action": "seed_add",
    }
    if note:
        entry["note"] = note
    with log.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arxiv-id", help="arXiv id; e.g. 2605.19561")
    ap.add_argument("--name", help="paper title or substring (used only when arxiv-id missing)")
    ap.add_argument("--bucket", choices=sorted(VALID_BUCKETS),
                    help="override bucket (skips auto-detection)")
    ap.add_argument("--note", help="inline YAML comment to append")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would change, do not write")
    args = ap.parse_args()

    if not args.arxiv_id and not args.name:
        ap.error("either --arxiv-id or --name is required")

    # ---- resolve arxiv id ----
    bucket_hint: str | None = None
    if args.arxiv_id:
        try:
            arxiv_id = normalize_arxiv_id(args.arxiv_id)
        except ValueError as e:
            sys.exit(str(e))
    else:
        arxiv_id, bucket_hint = resolve_name_to_id(args.name)

    # ---- already seeded? ----
    present, existing = is_already_seeded(arxiv_id)
    if present:
        print(f"arXiv:{arxiv_id} already in seeds.yaml under bucket {existing!r}; nothing to do")
        return 0

    # ---- resolve bucket ----
    bucket_source = "cli" if args.bucket else None
    bucket = args.bucket
    if not bucket:
        cached = bucket_from_scored_cache(arxiv_id)
        if cached:
            bucket = cached
            bucket_source = "scored_cache"

    # ---- resolve name (need it for the seed entry) ----
    # When user passed --name, trust it verbatim. When we derive a name from
    # a cached title (or from arXiv), shorten to the first colon-prefix
    # token group so the result is YAML-flow-safe and reads like a method
    # short-name (e.g. "TORQ" instead of "TORQ: Two-Level Orthogonal ...").
    name = args.name
    derived_title: str | None = None
    if not name:
        for f in sorted(SUMMARIZED_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            for p in data:
                if (p.get("id") or "").split(":")[-1] == arxiv_id:
                    derived_title = (p.get("title") or "").strip()
                    break
            if derived_title:
                break

    title_from_arxiv: str | None = None
    if not bucket:
        bucket, title_from_arxiv = bucket_from_haiku(arxiv_id)
        bucket_source = "haiku"
        if bucket is None:
            sys.exit(
                f"Haiku hard-gated arXiv:{arxiv_id} — it doesn't belong in seeds.yaml. "
                "If you disagree, re-run with --bucket to override."
            )

    if not name:
        if derived_title is None and title_from_arxiv is not None:
            derived_title = title_from_arxiv
        if derived_title is None:
            derived_title, _ = fetch_arxiv_metadata(arxiv_id)
        # Short-name heuristic: prefix before the first colon is usually the
        # method name; cap at 3 words and 40 chars. User can edit later.
        prefix = derived_title.split(":")[0].strip()
        words = prefix.split()
        name = " ".join(words[:3]) if words else derived_title[:40]
        name = name[:40]

    # ---- act ----
    print(f"adding seed: arXiv:{arxiv_id}  name={name!r}  bucket={bucket}  (via {bucket_source})")
    if args.dry_run:
        print("--dry-run: no files changed")
        return 0
    insert_seed_line(arxiv_id, name, bucket, args.note)
    log_accept(arxiv_id, name, bucket, bucket_source or "unknown", args.note)
    print(f"seeds.yaml updated; logged to data/curation/accepted.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())

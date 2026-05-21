#!/usr/bin/env python3
"""Log a paper rejection to data/curation/rejected.jsonl.

Optionally also append blacklist patterns to config.yaml so the cheap
prefilter kills similar papers next time without burning a Haiku call.

Usage:

    # Just log the reject + reason
    python scripts/seed_reject.py --arxiv-id 2605.XXXXX --reason "actually a survey"

    # Resolve by paper-name substring (must be unique in data/summarized/)
    python scripts/seed_reject.py --name OScaR --reason "..."

    # Log AND add comma-separated phrases to config.yaml prefilter.blacklist
    # (weight defaults to -3 per pattern; user-controlled, never auto-extracted)
    python scripts/seed_reject.py --arxiv-id 2605.XXXXX --reason "diffusion-only" \\
        --add-blacklist "stable diffusion,FID metric"

Why no auto-extraction of blacklist words from the rejected paper's text:
this is an irreversible-ish change (next day's filter sees it). Forcing the
user to explicitly type the phrases avoids overfitting on one-off rejects.

For tuning the LLM rubric (prompts/relevance.md few-shot anchors), do that
manually after eyeballing rejected.jsonl — too risky to script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("LLM_RADAR_REPO_ROOT") or Path(__file__).resolve().parent.parent)
CONFIG_PATH = REPO_ROOT / "config.yaml"
SUMMARIZED_DIR = REPO_ROOT / "data" / "summarized"
SCORED_DIR = REPO_ROOT / "data" / "scored"
LOG_PATH = REPO_ROOT / "data" / "curation" / "rejected.jsonl"

DEFAULT_BL_WEIGHT = -3


def normalize_arxiv_id(raw: str) -> str:
    raw = raw.strip()
    m = re.search(r"(\d{4}\.\d{4,5})", raw)
    if not m:
        raise ValueError(f"can't parse arXiv id from {raw!r}")
    return m.group(1)


def resolve_name_to_id(name: str) -> tuple[str, str | None]:
    """Same fuzzy lookup as seed_add.py."""
    if not SUMMARIZED_DIR.exists():
        sys.exit(f"name lookup needs {SUMMARIZED_DIR} but it doesn't exist")
    needle = name.lower().strip()
    hits: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
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
            if pid in seen:
                continue
            seen.add(pid)
            try:
                arxiv_id = normalize_arxiv_id(pid.split(":")[-1])
            except ValueError:
                continue
            bd = p.get("relevance_breakdown") or {}
            hits.append((arxiv_id, title, bd.get("topic_bucket")))
    if not hits:
        sys.exit(f"no paper title contains {name!r} in data/summarized/")
    if len(hits) > 1:
        print(f"name {name!r} matched {len(hits)} papers:", file=sys.stderr)
        for arxiv_id, title, bucket in hits:
            print(f"  {arxiv_id}  [{bucket or '?'}]  {title}", file=sys.stderr)
        sys.exit("disambiguate with --arxiv-id")
    arxiv_id, _title, _bucket = hits[0]
    return arxiv_id, hits[0][1]


def lookup_title(arxiv_id: str) -> str | None:
    """Best-effort title lookup from local cache; returns None if unknown."""
    for d in (SUMMARIZED_DIR, SCORED_DIR):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            for p in data:
                if (p.get("id") or "").split(":")[-1] == arxiv_id:
                    title = (p.get("title") or "").strip()
                    if title:
                        return title
    return None


def lookup_bucket_when_surfaced(arxiv_id: str) -> str | None:
    """Best-effort: what bucket was this paper in when the LLM judged it?
    Useful in the reject log for later debugging — was the issue the bucket
    routing, or the relevance call?"""
    if not SCORED_DIR.exists():
        return None
    for f in sorted(SCORED_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for p in data:
            if (p.get("id") or "").split(":")[-1] != arxiv_id:
                continue
            bd = p.get("relevance_breakdown") or {}
            bucket = bd.get("topic_bucket")
            if bucket:
                return bucket
    return None


def append_blacklist(patterns: list[str], weight: int) -> list[str]:
    """Append patterns to filter.prefilter.blacklist in config.yaml. Skips
    patterns that already exist (case-insensitive). Returns the patterns
    actually added."""
    raw = CONFIG_PATH.read_text()
    cfg = yaml.safe_load(raw)
    bl = cfg.setdefault("filter", {}).setdefault("prefilter", {}).setdefault("blacklist", [])
    existing = {(e.get("pattern") or "").lower() for e in bl if isinstance(e, dict)}
    added: list[str] = []
    new_entries: list[dict] = []
    for p in patterns:
        p_stripped = p.strip()
        if not p_stripped:
            continue
        if p_stripped.lower() in existing:
            continue
        new_entries.append({"pattern": p_stripped, "weight": weight})
        existing.add(p_stripped.lower())
        added.append(p_stripped)
    if not added:
        return added

    # String-level append so we don't rewrite the whole file and lose the
    # hand-curated comments / column alignment. Find the last blacklist
    # entry line and insert after it.
    lines = raw.splitlines()
    bl_section_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*blacklist:\s*$", ln):
            bl_section_idx = i
            break
    if bl_section_idx is None:
        sys.exit("could not locate `blacklist:` in config.yaml prefilter section")
    # Find the last consecutive line that looks like a blacklist entry.
    insert_at = bl_section_idx + 1
    while insert_at < len(lines):
        ln = lines[insert_at]
        if re.match(r"^\s*-\s*\{", ln) or re.match(r"^\s*-\s+pattern", ln):
            insert_at += 1
            continue
        if not ln.strip():
            break
        # First non-blacklist, non-blank line — stop.
        break
    new_lines = []
    for p in added:
        # Compact flow style matching the existing entries.
        # Pad pattern column for alignment with longest existing pattern.
        max_existing_pat_len = max(
            (len(e.get("pattern", "")) for e in bl if isinstance(e, dict)),
            default=0,
        )
        max_pat = max(max_existing_pat_len, len(p))
        padding = " " * (max_pat - len(p))
        new_lines.append(f'      - {{ pattern: "{p}",{padding}  weight: {weight} }}')
    lines = lines[:insert_at] + new_lines + lines[insert_at:]
    CONFIG_PATH.write_text("\n".join(lines) + "\n")
    return added


def log_reject(
    arxiv_id: str, title: str | None, reason: str, bucket_when_surfaced: str | None,
    blacklist_added: list[str],
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "arxiv_id": arxiv_id,
        "reason": reason,
        "action": "reject",
    }
    if title:
        entry["title"] = title
    if bucket_when_surfaced:
        entry["bucket_when_surfaced"] = bucket_when_surfaced
    if blacklist_added:
        entry["blacklist_added"] = blacklist_added
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arxiv-id", help="arXiv id; e.g. 2605.19561")
    ap.add_argument("--name", help="title substring fallback when arxiv-id missing")
    ap.add_argument("--reason", required=True,
                    help="short reason (≤100 chars). Used for active-learning review later.")
    ap.add_argument("--add-blacklist", default="",
                    help="comma-separated phrases to add to filter.prefilter.blacklist")
    ap.add_argument("--blacklist-weight", type=int, default=DEFAULT_BL_WEIGHT,
                    help=f"weight for each added blacklist phrase (default {DEFAULT_BL_WEIGHT})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.arxiv_id and not args.name:
        ap.error("either --arxiv-id or --name is required")
    if len(args.reason) > 200:
        ap.error("--reason too long (>200 chars); summarize")

    if args.arxiv_id:
        try:
            arxiv_id = normalize_arxiv_id(args.arxiv_id)
        except ValueError as e:
            sys.exit(str(e))
        title = lookup_title(arxiv_id)
    else:
        arxiv_id, title = resolve_name_to_id(args.name)

    bucket_when = lookup_bucket_when_surfaced(arxiv_id)

    patterns = [p for p in (args.add_blacklist.split(",") if args.add_blacklist else []) if p.strip()]

    print(f"reject: arXiv:{arxiv_id}")
    if title:
        print(f"  title: {title[:100]}")
    if bucket_when:
        print(f"  bucket when surfaced: {bucket_when}")
    print(f"  reason: {args.reason}")
    if patterns:
        print(f"  +blacklist (weight {args.blacklist_weight}): {patterns}")

    if args.dry_run:
        print("--dry-run: no files changed")
        return 0

    added: list[str] = []
    if patterns:
        added = append_blacklist(patterns, args.blacklist_weight)
        if added != patterns:
            skipped = [p for p in patterns if p not in added]
            print(f"  (already in blacklist, skipped: {skipped})")
        if added:
            print(f"  config.yaml updated; {len(added)} new blacklist entries")

    log_reject(arxiv_id, title, args.reason, bucket_when, added)
    print(f"  logged to {LOG_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

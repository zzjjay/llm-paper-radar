"""Resolve an arXiv id or fuzzy paper name into everything the radar knows.

This is the deterministic data-gathering layer under the `paper-interpret`
skill. Given `2607.01127` or `LogbQuant` or a chunk of a title, it:

  1. Locates the paper — local-first (`data/summarized/*.json`), falling back
     to a live arXiv metadata fetch when only an id is known and the radar
     never surfaced it.
  2. Emits the full radar record if we have one (bilingual summary, highlights,
     related methods, the whole `relevance_breakdown` — scores, format, largest
     model tested, perf, calibration cost, peak memory).
  3. Reports triage status: accepted seed (`seeds.yaml`) or rejected
     (`data/curation/rejected.jsonl`, with the human reason).
  4. Reports whether a `paper-river/*.org` lineage analysis already exists.
  5. Lists sibling papers in the same `topic_bucket` (for "is this new or a
     re-skin?" comparison) and a coarse trend of that bucket over recent days.

The skill reads this report as context; it does NOT re-grep the data dirs.
Output is a readable text report by default; `--json` emits a machine record.

Usage:
    uv run python scripts/resolve_paper.py 2607.01127
    uv run python scripts/resolve_paper.py LogbQuant
    uv run python scripts/resolve_paper.py "logarithmic space" --siblings 8
    uv run python scripts/resolve_paper.py 2607.01127 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUMMARIZED_DIR = ROOT / "data" / "summarized"
PAPER_RIVER_DIR = ROOT / "paper-river"
SEEDS_FILE = ROOT / "seeds.yaml"
REJECTED_FILE = ROOT / "data" / "curation" / "rejected.jsonl"

_ID_RE = re.compile(r"\d{4}\.\d{4,5}")
_ID_IN_STEM_RE = re.compile(r"(\d{4})[.-](\d{4,5})")


def _norm_id(raw: str) -> str:
    """Strip arXiv:/version/whitespace, return bare NNNN.NNNNN or ''."""
    m = _ID_RE.search(raw or "")
    return m.group(0) if m else ""


def _load_summarized() -> list[tuple[Path, dict]]:
    """Every summarized paper record, tagged with its source file (newest first)."""
    out: list[tuple[Path, dict]] = []
    if not SUMMARIZED_DIR.exists():
        return out
    for f in sorted(SUMMARIZED_DIR.glob("*.json"), reverse=True):
        try:
            for p in json.loads(f.read_text()):
                out.append((f, p))
        except Exception:
            continue
    return out


def _rec_id(rec: dict) -> str:
    return (rec.get("id") or "").split(":")[-1].strip()


def find_local(query: str) -> tuple[dict | None, Path | None, list[dict]]:
    """Return (exact_record, its_file, candidates).

    Resolution order for a name query:
      1. Exact acronym match (paper-river filenames / seed names) — takes
         precedence so a short acronym like `GPTQ` locks onto the classic seed
         instead of getting buried under titles that merely *contain* "gptq"
         (`MatGPTQ`, `GPTQ-intrinsic LoRA`). A collision (one acronym → several
         ids) comes back as candidates.
      2. Title substring fuzzy search, for free-text queries.
    An arXiv id always matches by id directly.
    """
    records = _load_summarized()
    qid = _norm_id(query)
    if qid:
        for f, rec in records:
            if _rec_id(rec) == qid:
                return rec, f, []
        return None, None, []

    q = query.strip().lower()
    acro_ids = _acronym_ids()  # acronym(lower) -> set of ids, from paper-river + seeds

    # 1. Exact acronym match wins over title substring.
    if q in acro_ids:
        ids = sorted(acro_ids[q])
        local: dict[str, tuple[Path, dict]] = {}
        for f, rec in records:
            rid = _rec_id(rec)
            if rid in ids and rid not in local:
                local[rid] = (f, rec)
        if len(ids) == 1:
            rid = ids[0]
            if rid in local:
                f, rec = local[rid]
                return rec, f, []
            # Known acronym, no summarized record (e.g. a classic seed) —
            # stub with the id so build_report() live-fetches it.
            return None, None, [{"id": rid, "title": f"(acronym {query!r} → {rid})"}]
        # One acronym maps to several ids — disambiguate.
        cands = [
            {"id": rid, "title": (local[rid][1].get("title", "") if rid in local else f"(acronym {query!r})")}
            for rid in ids
        ]
        return None, None, cands

    # 2. Title substring fuzzy search.
    hits: list[tuple[Path, dict]] = []
    seen_ids: set[str] = set()
    for f, rec in records:
        rid = _rec_id(rec)
        if rid in seen_ids:
            continue
        if q in (rec.get("title") or "").lower():
            hits.append((f, rec))
            seen_ids.add(rid)

    if len(hits) == 1:
        return hits[0][1], hits[0][0], []
    cands = [{"id": _rec_id(r), "title": r.get("title", ""), "file": str(f.name)} for f, r in hits[:12]]
    return None, None, cands


def _acronym_ids() -> dict[str, set[str]]:
    """Map lowercased acronym -> set of arxiv ids, harvested from paper-river
    filenames (`<Acronym>-<id>.org`) and seeds.yaml (`name` field). A set (not a
    single id) so `find_local` can tell a clean hit from a name collision."""
    out: dict[str, set[str]] = {}
    if PAPER_RIVER_DIR.exists():
        for f in PAPER_RIVER_DIR.glob("*.org"):
            stem = f.stem[:-3] if f.stem.endswith("_en") else f.stem
            m = _ID_IN_STEM_RE.search(stem)
            if not m:
                continue
            acro = stem[: m.start()].rstrip("-").strip()
            if acro:
                out.setdefault(acro.lower(), set()).add(f"{m.group(1)}.{m.group(2)}")
    # seeds.yaml: parse the simple `- { id: arXiv:XXXX, name: YYY, ... }` lines
    if SEEDS_FILE.exists():
        for line in SEEDS_FILE.read_text().splitlines():
            mid = _ID_RE.search(line)
            mname = re.search(r"name:\s*([^,}]+)", line)
            if mid and mname:
                out.setdefault(mname.group(1).strip().lower(), set()).add(mid.group(0))
    return out


def seed_status(arxiv_id: str) -> dict | None:
    """Return {name, category} if this id is an accepted seed, else None."""
    if not SEEDS_FILE.exists() or not arxiv_id:
        return None
    for line in SEEDS_FILE.read_text().splitlines():
        if arxiv_id in line and "name:" in line:
            name = re.search(r"name:\s*([^,}]+)", line)
            cat = re.search(r"category:\s*([^,}\s]+)", line)
            return {
                "name": name.group(1).strip() if name else "",
                "category": cat.group(1).strip() if cat else "",
            }
    return None


def reject_status(arxiv_id: str) -> dict | None:
    """Return the most recent rejection record for this id, else None."""
    if not REJECTED_FILE.exists() or not arxiv_id:
        return None
    last = None
    for line in REJECTED_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if _norm_id(rec.get("arxiv_id", "")) == arxiv_id:
            last = rec
    return last


def paper_river_files(arxiv_id: str) -> dict[str, str]:
    """Return {'zh': path, 'en': path} for existing lineage analyses (paths
    relative to repo root); missing keys mean not generated."""
    out: dict[str, str] = {}
    if not PAPER_RIVER_DIR.exists() or not arxiv_id:
        return out
    id_dash = arxiv_id.replace(".", "-")
    for f in PAPER_RIVER_DIR.glob("*.org"):
        if arxiv_id in f.stem or id_dash in f.stem:
            key = "en" if f.stem.endswith("_en") else "zh"
            out[key] = str(f.relative_to(ROOT))
    return out


def siblings_in_bucket(bucket: str, exclude_id: str, limit: int) -> list[dict]:
    """Top papers in the same topic_bucket, by relevance_score desc, deduped."""
    if not bucket:
        return []
    best: dict[str, dict] = {}
    for _f, rec in _load_summarized():
        bd = rec.get("relevance_breakdown") or {}
        if bd.get("topic_bucket") != bucket:
            continue
        rid = _rec_id(rec)
        if rid == exclude_id or not rid:
            continue
        prev = best.get(rid)
        if prev is None or (rec.get("relevance_score") or 0) > (prev.get("relevance_score") or 0):
            best[rid] = rec
    ranked = sorted(best.values(), key=lambda r: r.get("relevance_score") or 0, reverse=True)
    out = []
    for r in ranked[:limit]:
        out.append({
            "id": _rec_id(r),
            "title": r.get("title", ""),
            "score": r.get("relevance_score"),
            "format": (r.get("relevance_breakdown") or {}).get("format_or_method", ""),
            "published_at": r.get("published_at", ""),
        })
    return out


def bucket_trend(bucket: str, days: int) -> dict:
    """Count non-hard-gated papers in `bucket` per day over the last `days`."""
    if not bucket:
        return {}
    per_day: dict[str, int] = {}
    cutoff = datetime.now(UTC) - timedelta(days=days)
    for f, rec in _load_summarized():
        bd = rec.get("relevance_breakdown") or {}
        if bd.get("topic_bucket") != bucket or bd.get("hard_gate"):
            continue
        day = f.stem  # YYYY-MM-DD
        try:
            if datetime.fromisoformat(day + "T00:00:00+00:00") < cutoff:
                continue
        except ValueError:
            continue
        per_day[day] = per_day.get(day, 0) + 1
    return dict(sorted(per_day.items()))


async def fetch_live(arxiv_id: str) -> dict | None:
    """Fetch bare arXiv metadata for an id the radar never scored."""
    try:
        from sources._arxiv_lookup import fetch_arxiv_by_ids
    except Exception as e:  # pragma: no cover
        print(f"(live fetch unavailable: {e})", file=sys.stderr)
        return None
    try:
        papers = await fetch_arxiv_by_ids([arxiv_id], source_name="arxiv")
    except Exception as e:
        print(f"(live arXiv fetch failed: {e})", file=sys.stderr)
        return None
    if not papers:
        return None
    p = papers[0]
    d = p.model_dump()
    d["_live"] = True
    return d


def build_report(query: str, siblings: int, trend_days: int) -> dict:
    rec, src_file, cands = find_local(query)
    live = False
    if rec is None and not cands:
        qid = _norm_id(query)
        if qid:
            rec = asyncio.run(fetch_live(qid))
            live = bool(rec)
    elif rec is None and cands and len(cands) == 1 and _norm_id(cands[0]["id"]):
        # Single acronym stub with a known id but no local record: fetch it live.
        rec = asyncio.run(fetch_live(cands[0]["id"]))
        live = bool(rec)
        cands = []

    result: dict = {"query": query, "resolved": rec is not None, "candidates": cands}
    if rec is None:
        return result

    arxiv_id = _rec_id(rec) or _norm_id(rec.get("id", "")) or _norm_id(query)
    bd = rec.get("relevance_breakdown") or {}
    bucket = bd.get("topic_bucket") or ""
    result.update({
        "source": "arxiv-live" if live else "radar-local",
        "summarized_file": None if live else (src_file.name if src_file else None),
        "arxiv_id": arxiv_id,
        "paper": {
            "title": rec.get("title", ""),
            "authors": rec.get("authors", []),
            "published_at": rec.get("published_at", ""),
            "categories": rec.get("categories", []),
            "url": rec.get("url", f"https://arxiv.org/abs/{arxiv_id}"),
            "pdf_url": rec.get("pdf_url", ""),
            "code_url": rec.get("code_url"),
            "abstract": rec.get("abstract", ""),
        },
        "radar": None if live else {
            "relevance_score": rec.get("relevance_score"),
            "topic_bucket": bucket,
            "topic_relevance": bd.get("topic_relevance"),
            "practicality": bd.get("practicality"),
            "hard_gate": bd.get("hard_gate"),
            "format_or_method": bd.get("format_or_method"),
            "largest_model_tested": bd.get("largest_model_tested"),
            "accuracy_summary": bd.get("accuracy_summary"),
            "inference_perf": bd.get("inference_perf"),
            "calibration_cost": bd.get("calibration_cost"),
            "peak_memory": bd.get("peak_memory"),
            "relevance_reason": rec.get("relevance_reason"),
            "summary_zh": rec.get("summary"),
            "summary_en": rec.get("summary_en"),
            "highlights": rec.get("highlights", []),
            "related_methods": rec.get("related_methods", []),
        },
        "triage": {
            "accepted_seed": seed_status(arxiv_id),
            "rejected": reject_status(arxiv_id),
        },
        "paper_river": paper_river_files(arxiv_id),
        "siblings": siblings_in_bucket(bucket, arxiv_id, siblings) if not live else [],
        "bucket_trend": bucket_trend(bucket, trend_days) if not live else {},
    })
    return result


def render_text(r: dict) -> str:
    L: list[str] = []
    if not r["resolved"]:
        if r["candidates"]:
            L.append(f"AMBIGUOUS: {len(r['candidates'])} papers match {r['query']!r}. Ask the user which:")
            for c in r["candidates"]:
                L.append(f"  - {c['id']}  {c['title'][:90]}")
        else:
            L.append(f"NOT FOUND: no radar record for {r['query']!r}, and could not fetch it live.")
            L.append("If this is an arXiv id, check the id; if a name, the radar may never have surfaced it.")
            L.append("You can still interpret it by fetching the paper directly (WebFetch the arXiv abstract/PDF).")
        return "\n".join(L)

    p = r["paper"]
    L.append(f"RESOLVED [{r['source']}]  arXiv:{r['arxiv_id']}")
    if r.get("summarized_file"):
        L.append(f"  radar record from: data/summarized/{r['summarized_file']}")
    L.append("")
    L.append(f"# {p['title']}")
    L.append(f"authors: {', '.join(p['authors'][:8])}{' …' if len(p['authors']) > 8 else ''}")
    L.append(f"published: {p['published_at']}   categories: {', '.join(p['categories'])}")
    L.append(f"abs: {p['url']}   pdf: {p['pdf_url']}")
    if p.get("code_url"):
        L.append(f"code: {p['code_url']}")
    L.append("")
    L.append("## Abstract")
    L.append(p["abstract"])

    rd = r.get("radar")
    if rd:
        L.append("")
        L.append("## Radar scoring")
        L.append(f"composite={rd['relevance_score']}  bucket={rd['topic_bucket']}  "
                 f"topic_relevance={rd['topic_relevance']}  practicality={rd['practicality']}  "
                 f"hard_gate={rd['hard_gate']}")
        L.append(f"format/method: {rd['format_or_method']}")
        L.append(f"largest model tested: {rd['largest_model_tested']}")
        L.append(f"accuracy: {rd['accuracy_summary']}")
        L.append(f"inference perf: {rd['inference_perf']}")
        L.append(f"calibration cost: {rd['calibration_cost']}   peak memory: {rd['peak_memory']}")
        if rd.get("relevance_reason"):
            L.append(f"scorer reason: {rd['relevance_reason']}")
        if rd.get("summary_zh"):
            L.append("")
            L.append("### 中文摘要 (radar)")
            L.append(rd["summary_zh"])
        if rd.get("summary_en"):
            L.append("")
            L.append("### English summary (radar)")
            L.append(rd["summary_en"])
        if rd.get("highlights"):
            L.append("")
            L.append("### Highlights")
            for h in rd["highlights"]:
                L.append(f"  - {h}")
        if rd.get("related_methods"):
            L.append("")
            L.append("### Related methods (radar-extracted)")
            for m in rd["related_methods"]:
                aid = f" (arXiv:{m['arxiv_id']})" if m.get("arxiv_id") else ""
                L.append(f"  - {m.get('name','')}{aid}: {m.get('relation','')}")

    t = r.get("triage") or {}
    L.append("")
    L.append("## Triage status")
    if t.get("accepted_seed"):
        s = t["accepted_seed"]
        L.append(f"  ACCEPTED as seed: name={s['name']} category={s['category']}")
    if t.get("rejected"):
        rj = t["rejected"]
        L.append(f"  REJECTED on {rj.get('ts','?')}: {rj.get('reason','')}")
        if rj.get("blacklist_added"):
            L.append(f"    (blacklist added: {rj['blacklist_added']})")
    if not t.get("accepted_seed") and not t.get("rejected"):
        L.append("  not yet triaged (neither accepted nor rejected)")

    pr = r.get("paper_river") or {}
    L.append("")
    L.append("## Paper-river lineage")
    if pr:
        for k, v in pr.items():
            L.append(f"  {k}: {v}")
        L.append("  → a lineage analysis already exists; read it instead of regenerating.")
    else:
        L.append("  none — no paper-river/*.org for this id yet.")

    sib = r.get("siblings") or []
    if sib:
        L.append("")
        L.append(f"## Siblings in bucket '{rd['topic_bucket'] if rd else ''}' (compare against)")
        for s in sib:
            L.append(f"  [{s['score']}] arXiv:{s['id']}  {s['title'][:70]}  | {s['format']}")

    tr = r.get("bucket_trend") or {}
    if tr:
        L.append("")
        L.append(f"## Bucket trend (papers/day, non-gated)")
        L.append("  " + "  ".join(f"{d.split('-',1)[1]}:{n}" for d, n in tr.items()))

    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve an arXiv id or paper name into radar context.")
    ap.add_argument("query", help="arXiv id (2607.01127) or paper name / title fragment")
    ap.add_argument("--siblings", type=int, default=6, help="how many same-bucket papers to list (default 6)")
    ap.add_argument("--trend-days", type=int, default=30, help="trend window in days (default 30)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    args = ap.parse_args()

    report = build_report(args.query, siblings=args.siblings, trend_days=args.trend_days)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_text(report))


if __name__ == "__main__":
    main()

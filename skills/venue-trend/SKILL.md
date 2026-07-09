---
name: venue-trend
description: Use when the user names a whole conference/venue (e.g. "MLSys 2026", "分析下 NeurIPS 2026 的推理优化论文", "做一份 ICML venue 趋势报告", "venue trend report for <conf>") and wants its accepted papers fetched, filtered to LLM inference deployment optimization, grouped by subfield, and turned into a trend report with a macro synthesis. NOT for a single paper (use paper-interpret) and NOT for the daily rolling digest (that's scripts/daily.sh / paper-triage).
---

# venue-trend

One-shot **conference-level** trend analysis: given a venue, fetch its entire
accepted-paper set, classify each paper into an LLM-inference-deployment-
optimization subfield, and produce a report that leads with a macro synthesis
(the few root drivers behind the distribution) followed by tight per-subfield
sections. This skill is an **orchestrator** — the fetch/score/group stages are
plain CLI in this repo, per-subfield trend analysis runs as a Workflow, and the
macro synthesis delegates to the global `ljg-rank` skill.

Distinct from the daily pipeline: this pulls a venue's *full* history in one
pass, not a rolling window. Run it manually per conference, not on a cron.

## When to invoke

- User names a venue + year and wants the whole thing analyzed: "分析 MLSys 2026
  的 LLM 推理优化论文", "venue trend report for NeurIPS 2026", "做一份 ICML 2026
  推理部署方向的趋势报告".
- Trigger phrases: "venue trend", "会议趋势", "整个会议的论文", "趋势报告",
  "/venue-trend".

Do **not** invoke for:
- A single paper (arXiv id / name) → `paper-interpret`.
- Accept/reject over the daily digest → `paper-triage`.
- Fetching/re-scoring the daily rolling window → `scripts/daily.sh`.

## Repo location & prerequisites

Runs from the `llm-paper-radar` repo root (needs `scripts/venue_report.sh`,
`sources/openreview_venue.py`, `pipeline/venue_filter.py`,
`pipeline/venue_group.py`, `workflows/venue_trend_report.js`). If invoked
elsewhere, ask the user to `cd` in first.

Before running, source credentials so the fetch and the LLM judge work:

```bash
set -a && source .env && set +a
```

`.env` must have `ANTHROPIC_API_KEY` (real key, not the `sk-ant-xxx` placeholder)
and `OPENREVIEW_EMAIL`/`OPENREVIEW_PASSWORD`. **Anonymous OpenReview requests get
403-challenge-walled** (since ~2026-06-30) — without the account creds the fetch
silently returns 0 papers. If the fetch comes back empty or all-403, the missing
creds are the first thing to check, not the venue string.

## Step 1 — fetch → score → group

```bash
./scripts/venue_report.sh "<venue>"    # e.g. "MLSys.org/2026/Conference"
```

The venue string is the OpenReview invitation prefix (`<Host>/<Year>/Conference`).
This chains three stages and writes `data/scored/<slug>-grouped.json` (slug =
`<conf>-<year>`, e.g. `mlsys-2026`):

1. `sources.openreview_venue` — paginate the venue's `/-/Submission` notes,
   keep only accepted ones (`content.venueid.value == "<venue>"`), write
   `data/raw/<slug>/accepted.json`. Retries 403/429 with backoff; **refuses to
   write a partial result** — if it exits non-zero (retries exhausted, or fewer
   than ~20 accepted papers), do not proceed; wait and re-run (the per-page
   cache under `data/raw/<slug>/openreview_pages/` resumes from where it failed).
2. `pipeline.venue_filter` — per-paper LLM classification via
   `prompts/inference_relevance.md` → `hard_gate` + `subfield` + `reason`.
   A failed judge call is recorded as a diagnosable `hard_gate=true` entry, not
   silently dropped.
3. `pipeline.venue_group` — drop hard-gated papers, group the rest by subfield.

**New-venue caveat (first run only).** `_is_accepted` assumes MLSys's convention
(accepted papers get `venueid == "<venue>"`, rejects get a `.../Rejected_Submission`
suffix). Other venues *may* differ. On a venue you haven't run before, before
trusting the count, inspect a page of raw notes:

```bash
python3 -c "import json; d=json.load(open('data/raw/<slug>/openreview_pages/page-0000.json')); print(set(n['content'].get('venueid',{}).get('value') for n in d[:20]))"
```

Confirm one value equals the venue string exactly. If it doesn't (some venues
publish decisions under a separate invitation, or use a display `venue` string
instead), adjust `_is_accepted` in `sources/openreview_venue.py`, re-run its
tests, then re-run the fetch. Don't silently accept a low count.

## Step 2 — per-subfield trend analysis (Workflow)

Build the Workflow input from the grouped JSON (truncate abstracts to ~500 chars
to keep the payload small — the trend agents only need the gist):

```bash
python3 -c "
import json
g = json.load(open('data/scored/<slug>-grouped.json'))
groups = [{'subfield': k, 'papers': [{'title': p['title'], 'abstract': p['abstract'][:500], 'url': p['url']} for p in v]} for k, v in g.items()]
open('data/scored/<slug>-groups-arg.json','w').write(json.dumps(groups))
"
```

Then call the `Workflow` tool with `scriptPath:
"workflows/venue_trend_report.js"` and `args: {"title": "<Conf Year>", "groups":
<contents of that file>}`. It runs one agent per subfield in parallel (each
returns core problems / representative papers / method commonalities) and
returns a `{ report }` markdown string. A subfield whose agent fails is marked
"(analysis failed)" in-place, not dropped.

Keep the workflow's **subfield distribution table** and its per-subfield paper
lists — but you will re-tighten the prose in Step 4, so the workflow's verbose
`Core problems` / `Method commonalities` paragraphs are raw material, not the
final text.

## Step 3 — macro synthesis (ljg-rank)

Invoke the global **`ljg-rank`** skill (`Skill` tool, `skill: "ljg-rank"`) with
a one-paragraph domain description: venue + year, total in-scope count, and the
subfield distribution (name + count each). Point it at the grouped JSON / the
draft report for paper-level detail, and state the goal explicitly: find the
**independent root generators** behind the distribution — not a restatement of
the buckets, but the few forces that, if changed, would collapse or reshape the
whole picture.

`ljg-rank` writes its own full analysis (Chinese prose + ASCII diagrams) to
`~/Documents/notes/<timestamp>--<domain>的秩__rank.org` and reports that path.
That file is the source of truth — reference it by path, don't copy it wholesale
and don't hand-edit it.

## Step 4 — assemble the report

Write `venue-reports/<slug>.md` (per-conference reports live in `venue-reports/`,
NOT `digests/` — that's per-day). Structure, in order:

1. `# <Conf> <Year> — LLM Inference Deployment Optimization Trend Report`
2. `## Macro synthesis` — a 3-5 paragraph **English** condensation of the
   `ljg-rank` org file: the root generators, one example paper each, whether
   they compound or stay independent, and — if the analysis singles out a
   subfield as a "multiplicative overflow" / not reducible to one generator
   (as `compiler_kernel_fusion` was for MLSys 2026) — call that out explicitly.
   End with an italic pointer to the org file path.
3. `## Subfield distribution` — the count table (fold all single-paper
   subfields into one `other (N singletons)` row).
4. One `## <subfield> (N)` section per multi-paper subfield: **2-3 sentences**
   of sharp take (what the cluster is fighting + how approaches split
   internally), then the paper list with a one-line contribution tag per paper.
   Do NOT reproduce the workflow's `Core problems` / `Method commonalities` /
   `Representative papers` blocks verbatim — those are bloated templated output;
   distill them. Every paper keeps its OpenReview link.
5. `## other (N singletons)` — one line per single-paper direction: link +
   *italic subfield label* + one-clause contribution.

Target: tight enough to read in one sitting, every paper still linked. (The
MLSys 2026 report landed at ~130 lines for 57 papers — use it as the reference
shape: `venue-reports/mlsys-2026.md`.)

## Step 5 — commit

```bash
git add venue-reports/<slug>.md
git commit -m "docs: add <Conf> <Year> LLM inference deployment trend report"
```

The `~/Documents/notes/*.org` file is **not** committed — it lives outside the
repo (where `ljg-rank` always writes) and is referenced by path only. The
`data/raw/` and `data/scored/` intermediates are gitignored; don't force-add them.

## Don't

- **Don't skip Step 3.** A report with only per-subfield sections says what's in
  each bucket but not why the buckets look the way they do this year — the macro
  synthesis is the point, not a bonus.
- **Don't proceed on an incomplete fetch.** A 403-storm or a below-floor paper
  count means re-run later, not "analyze what we got" — a report built on a
  partial pull reads as complete and misleads.
- **Don't ship the workflow's raw verbose prose.** Always do the Step 4 tighten
  pass; the per-subfield agent output is dense templated text, not final copy.
- **Don't hardcode a new venue's accept-detection without checking** the actual
  `venueid` values first (Step 1 caveat).

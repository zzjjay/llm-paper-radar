---
name: venue-trend
description: Use when the user names a whole conference/venue (e.g. "MLSys 2026", "分析下 NeurIPS 2026 的推理优化论文", "做一份 ICML venue 趋势报告", "venue trend report for <conf>") and wants its accepted papers fetched, filtered to LLM inference deployment optimization, grouped by subfield, and turned into a trend report with a macro synthesis. NOT for a single paper (use paper-interpret) and NOT for the daily rolling digest (that's scripts/daily.sh / paper-triage).
---

# venue-trend

One-shot **conference-level** trend analysis: given a venue, fetch its entire
accepted-paper set, filter to LLM inference deployment optimization, then — with
the top model reading every abstract in one context — produce a report organized
around the real research concerns (what physical fact / workload shift / hardware
change the papers are collectively responding to), not a bucket-by-bucket
listing. The fetch/score/group stages are plain CLI in this repo; **the analysis
is done by you (the orchestrating model) reading the abstracts directly, not
delegated to sub-agents** — that delegation is exactly what made the first
version shallow. `ljg-rank` is available as an optional second lens.

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

## Step 2 — read every abstract yourself (the analysis, done by you)

**This is the core step, and you (the orchestrating model) do it directly — do
not fan it out to sub-agents.** The earlier design dispatched one weak-model
agent per subfield via a Workflow; each agent was blind to the others, so the
result was disjointed, templated, and shallow — bucket-by-bucket restatement,
not genuine cross-paper insight. The whole point of running the top model is to
hold all the papers in one context and see the connections between them. So read
them yourself.

Dump the in-scope papers with **full** abstracts (not truncated — you need the
mechanisms and numbers), grouped for convenience:

```bash
python3 -c "
import json
from collections import defaultdict
d=json.load(open('data/scored/<slug>.json'))
ins=[p for p in d if not (p.get('relevance_breakdown') or {}).get('hard_gate')]
g=defaultdict(list)
for p in ins: g[(p.get('relevance_breakdown') or {}).get('subfield','?')].append(p)
out=[]
for sf in sorted(g, key=lambda k:-len(g[k])):
    out.append(f'\n{\"=\"*60}\n{sf} ({len(g[sf])})\n{\"=\"*60}')
    for p in g[sf]: out.append(f'\n### {p[\"title\"]}\n{p[\"url\"]}\n{p[\"abstract\"]}')
open('/tmp/<slug>_abstracts.txt','w').write('\n'.join(out))
print(len(ins),'papers')
"
```

`Read` that file and actually read it. As you go, look for what no per-bucket
pass can see:

- **The shared physical fact** several clusters are all reacting to (for MLSys
  2026 it was "decode went memory-bound, and the KV cache is why").
- **Cross-bucket threads** — the same object or pressure showing up in multiple
  subfields (the KV cache appeared in kv_cache, long_context, scheduling, and
  spec-decoding papers alike).
- **Workload or hardware shifts** driving the distribution (reasoning models →
  long decode; agents/RAG → shared context; new GPU gen → rebalanced bottleneck).
- **Debates and reality-checks** — papers auditing whether a hyped technique
  actually holds up (e.g. a "does disaggregation / spec-decoding really help at
  production scale?" measurement paper is worth more than ten me-too systems).
- **What shifted and what's conspicuously absent** vs. prior years.

If the in-scope set is too large to hold in one context (rough rule: more than
~120 papers / ~200 KB of abstracts), fall back to the Workflow at
`workflows/venue_trend_report.js` for a first-pass per-subfield summary, then
still do the synthesis yourself over those summaries. For a normal venue
(dozens of in-scope papers) read the abstracts directly — it's strictly better.

## Step 3 — (optional) ljg-rank as a complementary root-cause pass

Your Step-2 read already organizes around root concerns. `ljg-rank` is an
optional **second lens**, not the primary analysis: invoke the global
**`ljg-rank`** skill with a one-paragraph domain description (venue, in-scope
count, subfield distribution) to get an independent Chinese root-cause
decomposition with ASCII diagrams. It writes to
`~/Documents/notes/<timestamp>--<domain>的秩__rank.org` and reports the path;
reference that path from the report, don't copy it in or hand-edit it. Use it to
sanity-check your own synthesis (do the roots agree?) or skip it if your Step-2
analysis already stands on its own.

## Step 4 — write the report

Write `venue-reports/<slug>.md` (per-conference reports live in `venue-reports/`,
NOT `digests/` — that's per-day). **Organize by research concern, not by the
mechanical subfield buckets** — the buckets cut across the real story, so leading
with them reproduces the disjointed structure this skill exists to avoid.
Structure:

1. `# <Conf> <Year> — LLM Inference Deployment Optimization Trend Report` + a
   one-line note on what was analyzed (N in-scope of M total accepts).
2. **The thesis** — a short opening (2-4 short paragraphs) stating the one or two
   forces the whole set is responding to, and any field-wide constraint you
   noticed (for MLSys 2026: frozen weights + vLLM/SGLang target → everything is
   training-free drop-in co-design). This is the "one thing to take away."
3. **Sections by concern** — one `##` per real thread you found in Step 2, each a
   few grounded paragraphs that *name specific papers, mechanisms, and numbers*
   (not "several papers do X" — say which, and what result). Fold papers into the
   thread they actually belong to; a paper can be mentioned in more than one.
   Every paper gets an inline OpenReview link on first mention.
4. **What shifted / what's absent** — a closing section on the year-over-year
   movement and notable gaps (a real "we didn't see much X this year" is a
   finding).
5. **Subfield distribution (reference)** — the count table at the *end*, marked
   as the mechanical classification, subordinate to the analysis.

Ground every claim in the abstracts. Cite numbers the papers report (speedups,
%s, model sizes). Prefer "X does Y, getting Z" over adjectives. Verify every
in-scope paper is linked somewhere:

```bash
python3 -c "
import json,re
d=json.load(open('data/scored/<slug>.json'))
ids={p['url'].split('id=')[-1] for p in d if not (p.get('relevance_breakdown') or {}).get('hard_gate')}
linked=set(re.findall(r'forum\?id=([A-Za-z0-9]+)', open('venue-reports/<slug>.md').read()))
print('missing:', ids-linked)
"
```

Fold any missing paper into its rightful thread — don't leave it out (an unlinked
in-scope paper is a coverage hole). Reference shape:
`venue-reports/mlsys-2026.md` (~95 lines, all 57 papers linked, organized by
seven research concerns rather than eight buckets).

## Step 5 — commit

```bash
git add venue-reports/<slug>.md
git commit -m "docs: add <Conf> <Year> LLM inference deployment trend report"
```

The `~/Documents/notes/*.org` file (if you ran Step 3) is **not** committed — it
lives outside the repo and is referenced by path only. The `data/raw/` and
`data/scored/` intermediates are gitignored; don't force-add them.

## Don't

- **Don't fan the analysis out to sub-agents.** Read the abstracts yourself with
  the top model. Delegated per-bucket summaries are what produced the shallow
  patchwork this skill was rebuilt to avoid (see Step 2).
- **Don't organize the report by the subfield buckets.** Organize by the research
  concerns you actually found; the buckets are a reference table at the end.
- **Don't write adjective-driven prose.** Every claim names a paper + mechanism +
  number from its abstract. "Several papers improve throughput" is a non-finding.
- **Don't proceed on an incomplete fetch.** A 403-storm or a below-floor paper
  count means re-run later, not "analyze what we got" — a report built on a
  partial pull reads as complete and misleads.
- **Don't leave an in-scope paper unlinked** (run the coverage check in Step 4).
- **Don't hardcode a new venue's accept-detection without checking** the actual
  `venueid` values first (Step 1 caveat).

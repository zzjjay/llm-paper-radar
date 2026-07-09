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
version shallow.

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

## Step 3 — write the report

Write `venue-reports/<slug>.md` (per-conference reports live in `venue-reports/`,
NOT `digests/` — that's per-day). **Organize by research concern, not by the
mechanical subfield buckets** — the buckets cut across the real story, so leading
with them reproduces the disjointed structure this skill exists to avoid.
Structure:

1. Title + a one-line note on what was analyzed (N in-scope of M total accepts)
   plus the two standing caveats compressed to one italic line: figures are
   authors' self-reported bests; one venue + auto-selected set (point to Method).
2. `## Takeaway` — the thesis in 2-3 tight paragraphs: the one or two forces the
   whole set responds to, and the field-wide constraint (for MLSys 2026: frozen
   weights + vLLM/SGLang target → training-free drop-in co-design).
3. **Numbered sections by concern** (`## 1.`, `## 2.`, …) — one per real thread
   from Step 2, each a tight paragraph or two that *names specific papers,
   mechanisms, and numbers* (not "several papers do X" — say which and what
   result). A paper can appear in more than one thread. Inline OpenReview link on
   first mention.
4. `## N. Maturity: practicality vs research` — a readiness axis
   cutting across the concerns: deployed-at-scale (named production systems) /
   industrial-grade-with-real-tooling / research-prototype. The most useful cut
   for a reader deciding what to adopt vs. watch.
5. `## What shifted, and what's conspicuously absent` — but **separate grounded
   from conjectural claims** (see rigor rules). Tag each.
6. `## Method & caveats` — how the set was selected + its biases, at the end.
7. **Subfield distribution** — the count table last, marked as the mechanical
   classification, subordinate to the analysis.

### Rigor rules (these are the ones a shallow report violates)

- **Ground vs. conjecture.** You have *this year's* abstracts, not prior years.
  Same-year claims ("KV is the largest single object here") are grounded; any
  year-over-year claim ("X faded", "barely existed two years ago", "more common
  than before") has no data behind it — mark it `[conjecture]`. External
  knowledge is allowed *if labeled*.
- **Venue confound.** This is one systems venue with an auto-selected set, so
  "what the field values" claims are really "what this venue accepted." When a
  trend is plausibly a submission-routing artifact (e.g. "pure-algorithm work
  faded" — it goes to NeurIPS/ICML, not a systems venue), tag it
  `[venue artifact]`, don't present it as a field shift.
- **Numbers are self-reported ceilings.** Every speedup/reduction is the authors'
  own best case, usually "up to X" on a favorable config or microbenchmark. Say
  this once up top; where a figure is production-trace vs microbenchmark, note it.
- **Verify quantifiers by counting — never eyeball "most / more than half."** A
  quick script settles it (the MLSys 2026 draft claimed "more than half the
  papers are about the KV cache"; an actual count was ~15-20 of 57, i.e. the
  largest single object but *not* a majority). Count, then write the number.
- **Analytical attention ∝ cluster size.** The biggest bucket deserves the
  deepest treatment, not a hand-wave. (The kernel/compiler cluster — the largest
  — was first dismissed as "overflow"; it needed its own full section.)
- **A thesis must cover the whole set, or admit what it doesn't.** Don't force a
  single-object narrative that only explains half. If there are two independent
  forces, say two, and state plainly what each does *not* touch.
- **Flag non-core papers.** Borderline in-scope items (visual-gen, multimodal,
  diffusion-LM — not pure text-LLM inference) get folded in but must be flagged,
  not silently counted as LLM-serving.
- **State causation precisely.** "Reasoning models *amplified* the memory-bound
  decode regime" — not "created" it (autoregressive decode was already
  memory-bound at small batch). Overclaimed causation is a tell of a shallow read.
- **Density: state each caveat once, then stop.** Don't repeat the self-reported
  or venue caveat per paragraph; don't re-link a paper already linked above;
  merge bullet lists into dense prose. Target roughly the shape below, not double.

Cite numbers the papers report. Prefer "X does Y, getting Z" over adjectives.
Verify every in-scope paper is linked somewhere:

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
in-scope paper is a coverage hole). Reference shape: `venue-reports/mlsys-2026.md`
(~65 lines, dense, all 57 papers linked, organized by research concern with a
Takeaway + numbered sections + Maturity + What-shifted + Method structure).

## Step 4 — commit

```bash
git add venue-reports/<slug>.md
git commit -m "docs: add <Conf> <Year> LLM inference deployment trend report"
```

The `data/raw/` and `data/scored/` intermediates are gitignored; don't
force-add them.

## Don't

- **Don't fan the analysis out to sub-agents.** Read the abstracts yourself with
  the top model. Delegated per-bucket summaries are what produced the shallow
  patchwork this skill was rebuilt to avoid (see Step 2).
- **Don't organize the report by the subfield buckets.** Organize by the research
  concerns you actually found; the buckets are a reference table at the end.
- **Don't write adjective-driven prose.** Every claim names a paper + mechanism +
  number from its abstract. "Several papers improve throughput" is a non-finding.
- **Don't state a year-over-year trend as fact** — you have no prior-year data;
  tag it `[conjecture]`. Same for venue-shaped trends → `[venue artifact]`.
- **Don't write "most / more than half" without counting.** Verify quantifiers
  against the data first.
- **Don't launder "up to X" peak numbers as results.** Flag them as self-reported
  ceilings; note microbenchmark vs production-trace.
- **Don't under-serve the largest cluster** or force a one-object thesis that only
  covers half the set.
- **Don't proceed on an incomplete fetch.** A 403-storm or a below-floor paper
  count means re-run later, not "analyze what we got" — a report built on a
  partial pull reads as complete and misleads.
- **Don't leave an in-scope paper unlinked** (run the coverage check in Step 3).
- **Don't hardcode a new venue's accept-detection without checking** the actual
  `venueid` values first (Step 1 caveat).

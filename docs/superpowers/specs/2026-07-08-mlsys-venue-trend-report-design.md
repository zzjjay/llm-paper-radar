# MLSys Conference Paper Trend Report — Design Spec

**Date**: 2026-07-08
**Author**: zhaolin (with Claude)
**Status**: Historical design record — partly superseded by `skills/venue-trend/SKILL.md`

> The fetch → score → group design below is accurate and current. The analysis
> stage evolved: it is no longer a per-subfield Workflow plus an `ljg-rank` macro
> pass (§2 steps 4-6, and the ljg-rank mentions in Goal / Success Criteria) —
> the top model now reads every in-scope abstract inline and writes the report
> organized by research concern. See `skills/venue-trend/SKILL.md` for the
> living flow. This file is kept for provenance.

## 1. Goal & Non-Goals

### Goal

Given an academic conference (**first target: MLSys 2026**), do a one-shot fetch of all its **accepted** papers, use an LLM to identify which ones fall under **LLM inference deployment optimization** (broader than `llm-paper-radar`'s existing quantization/compression scope — covers KV cache optimization, quantization, speculative decoding, scheduling & batching, MoE inference, long-context/prefill-decode disaggregation, multi-GPU/heterogeneous deployment, compiler & kernel fusion, etc.), group them by subfield, generate a research-trend summary per subfield, distill a macro-level synthesis across all subfields via the `ljg-rank` skill, and produce a final Markdown report.

This is a **new capability** inside the `llm-paper-radar` repo, fully independent of the existing daily rolling-window pipeline (`scripts/daily.sh`) — it does not modify that pipeline's behavior.

### Non-Goals (v1)

- No cross-conference generalization — hardcode support for MLSys 2026 only; field names / decision-parsing logic can be hardcoded. Generalizing to other venues is a later iteration.
- No full-text reading — scoring and trend summaries use title + abstract only.
- No year-over-year trend comparison (no historical baseline data) — only a "this edition" cross-sectional summary.
- No strict enumerated taxonomy for subfields — seed the prompt with an initial anchor list, let the LLM propose new subfield names under "other", and decide later (manually) whether to lock them in.

### Success Criteria

- Fully fetches all of MLSys 2026's accepted papers; a paper count below a rough floor (e.g. <20) is treated as an incomplete fetch and errors out rather than silently producing a report.
- Every paper in the scoring stage has a diagnosable hard_gate / subfield / reason; LLM call failures never silently drop a paper (reuses `pipeline/filter.py`'s "judge unavailable" recording pattern).
- The final report includes: a macro synthesis section (root drivers behind the subfield distribution, via `ljg-rank`) at the top, per-subfield paper-count distribution, a trend summary per subfield (core problems, representative papers), and the full paper list.

---

## 2. Architecture & Data Flow

New files only (nothing in the existing daily/weekly pipeline is modified):

```
sources/openreview_venue.py     # Batch-fetch all submissions + decisions for a venue, filter to accepted
prompts/inference_relevance.md  # LLM scoring prompt: is this LLM inference deployment optimization? + subfield
pipeline/venue_filter.py        # Scoring stage (reuses pipeline/filter.py's LLM-call framework/style)
pipeline/venue_group.py         # Group by subfield and tally (pure code)
scripts/venue_report.sh         # CLI entry point
workflows/venue_trend_report.js # Workflow script: parallel trend analysis + synthesis report
```

### Data flow

```
1. Fetch (sources/openreview_venue.py)
   fetch_venue(venue="MLSys.org/2026/Conference")
   → paginate through all /-/Submission notes
   → join with decision info (exact field name to be confirmed once the
     OpenReview API is reachable; the 403s observed so far look
     intermittent, not an API schema change)
   → keep only accepted papers, reuse the existing Paper model
   → write to data/raw/mlsys-2026/accepted.json
   → incrementally persist each raw page during pagination
     (data/raw/mlsys-2026/openreview_pages/) so a mid-run failure
     can resume instead of restarting from scratch

2. Score (pipeline/venue_filter.py)
   New prompt: prompts/inference_relevance.md
   - hard_gate: is this "LLM inference deployment optimization"?
   - subfield: free text, prompt seeds an anchor list:
     KV cache optimization, quantization, speculative decoding,
     scheduling & batching, MoE inference, long-context/PD disaggregation,
     multi-GPU/heterogeneous deployment, compiler & kernel fusion,
     other (must name the new subfield)
   → data/scored/mlsys-2026.json
   (Per-paper LLM failure: reuse filter.py's judge-unavailable pattern —
    hard_gate=True + diagnosable reason, never silently drop the paper)

3. Group (pipeline/venue_group.py, pure code)
   Group by subfield, tally counts, list papers per group
   → data/scored/mlsys-2026-grouped.json

4-5. Trend analysis + synthesis report (Workflow: workflows/venue_trend_report.js)
   pipeline(subfield_groups,
     group => agent("summarize this subfield's core problems /
                     representative papers / methodological commonalities",
                     {schema: TREND_SCHEMA}))
   → subfields are independent, run in parallel
   → a synthesis agent rolls up all subfield summaries + the count distribution
   → writes venue-reports/mlsys-2026.md (per-conference reports live in their
     own top-level directory, not digests/, since they're not per-day)

6. Macro synthesis (ljg-rank skill, manual invocation after step 5)
   Given the subfield distribution + full report, find the independent
   root drivers behind why the distribution looks the way it does — not
   a restatement of the subfields, but the few generators that would
   collapse the whole picture if changed. Condensed into a "Macro
   synthesis" section prepended to venue-reports/mlsys-2026.md; the
   skill's own full writeup (Chinese, with ASCII diagrams) is saved
   outside the repo under ~/Documents/notes/ and referenced by path.
```

---

## 3. Error Handling

- **Fetch stage (step 1)**:
  - Treat 403 challenges the same as the existing 429 handling — exponential backoff retry (e.g. 5s/10s/20s/40s, with generous retry budget, since the observed pattern is that the challenge clears on its own rather than being a permanent block).
  - After retries are exhausted, **do not write a result that looks "done" but is empty/partial** — exit non-zero and tell the user to re-run manually later.
  - If the accepted-paper count comes in below a rough floor (e.g. 20), treat the fetch as incomplete and error out rather than proceeding to analysis. This matches the repo's `AGENTS.md` principle of "don't silently skip expensive pipeline steps" — the same principle applies to "don't silently accept an incomplete fetch."
- **Scoring stage (step 2)**: a single paper's failure follows `pipeline/filter.py`'s existing pattern and does not abort the whole run.
- **Trend analysis stage (step 4)**: if a subfield's agent fails, that subfield is marked "analysis failed" in the report but its paper list still appears; other subfields are unaffected.

---

## 4. Testing

- **Fetch stage**: network-dependent, no unit tests — verify decision-field parsing with one manual run.
- **Scoring/grouping stage**: pure logic + prompt — build a handful of fixture papers (clearly relevant / clearly irrelevant / borderline) and run `venue_filter.py` against them, asserting the expected hard_gate and subfield outputs.
- **Trend analysis/report stage**: no automated tests — manually check the report for readability and whether the classification makes sense.

---

## 5. Known Risks

- The OpenReview API has shown intermittent 403s (Cloudflare challenge) since 2026-06-30. Retries mitigate this, but if a challenge window outlasts the retry budget, the fetch stage fails outright and needs a manual re-run.
- MLSys's decision-field format on OpenReview hasn't been verified yet (the API was 403'ing throughout this design session) — the first implementation step is to probe that structure and adjust `openreview_venue.py`'s parsing accordingly.

---
name: paper-triage
description: Use when the user asks to triage / curate / accept-reject / ingest papers from today's or recent llm-paper-radar digests. Trigger phrases include "triage papers", "triage today's digest", "看今天的 paper digest", "/paper-triage", "/triage-papers", "scout compression papers", or any reference to walking through the radar's daily/weekly digest interactively. The skill does NOT fetch or score papers — that's the cron-driven daily.sh. It only handles the human-in-the-loop accept/reject step on top of the already-rendered digest, calling scripts/seed_add.py and scripts/seed_reject.py on behalf of the user and optionally handing accepted papers to quark-wiki.
---

# paper-triage

Human-in-the-loop curation layer on top of the `llm-paper-radar` cron pipeline.

The radar (`scripts/daily.sh`) fetches → dedupes → filters → summarizes →
renders digests autonomously. This skill is the **interactive review
step** that runs **after** the cron-rendered digest exists: walk through
surfaced papers with the user, accept the good ones (add to `seeds.yaml`
and/or `quark-wiki`), reject the noise (log + optionally tighten the
prefilter blacklist).

## When to invoke

- User says "triage papers", "triage today's digest", "看今天 radar 的 paper",
  "/paper-triage", "/triage-papers", "review today's digest",
  "scout compression papers", etc.
- After a `daily.sh` run completed and the user wants to process the output.
- Any time the user references the radar's `digests/YYYY-MM-DD.md` and asks
  to act on individual papers.

Do **not** invoke for:
- Fetching/re-running the pipeline → that's `scripts/daily.sh` directly.
- Editing the rubric (`prompts/relevance.md`) → user does that by hand
  after reviewing `data/curation/rejected.jsonl`.
- Generic compression questions / ingest of a paper not in any digest →
  use `quark-wiki` directly.

## Repo location

This skill expects to be run inside the `llm-paper-radar` repo root (where
`config.yaml`, `seeds.yaml`, `scripts/`, `data/`, `digests/` all live). If
the user invokes the skill from outside, ask them to `cd` first.

## Tools the skill drives

Everything is deterministic CLI; no LLM calls from this skill itself
(the LLM-judge already ran in the radar's filter stage).

| Tool | Purpose | Side effects |
|---|---|---|
| `python scripts/seed_add.py --arxiv-id X [--name Y] [--note Z]` | Add to `seeds.yaml` under auto-detected bucket | Appends to `seeds.yaml` + `data/curation/accepted.jsonl` |
| `python scripts/seed_reject.py --arxiv-id X --reason "..." [--add-blacklist "p1,p2"]` | Log rejection, optionally tighten prefilter | Appends to `data/curation/rejected.jsonl`; optionally edits `config.yaml` |
| `Skill` tool with `quark-wiki` | Ingest a paper into the personal knowledge base | Writes a markdown page under `knowledge/` in quark-agent-assets |

## Workflow

### 1. Find the digest

Default to today's: `digests/$(date +%Y-%m-%d).md`. If the user names a
date, use that. If neither exists, run `ls digests/ | tail -5` and ask
which one they meant.

### 2. Parse surfaced papers

Each detail page block looks like:

```markdown
<a id="p-2605-19561"></a>
### 1. TORQ: Two-Level Orthogonal Rotation for MXFP4 Quantization (9/10)
**arxiv** · `2605.19561` · 2026-05-19
👥 Zukang Xu, Xing Hu, Dawei Yang · 🏷 cs.LG, cs.AI
🔗 [arXiv](https://arxiv.org/abs/2605.19561) · [PDF](...)
📡 Sources: arxiv
🧪 ptq · MXFP4 PTQ with two-level orthogonal rotation · Qwen3-32B · ...
#### 🧭 为什么推送这篇
**PTQ（训练后量化）** · topic 5/5 · practicality 4/5 · composite 9/10 · <reason>
#### 摘要
<chinese summary>
```

Extract per-paper: arxiv id, title, bucket (from "为什么推送这篇" line),
composite score, and the 摘要 + reason snippet.

For a fast pass, prefer reading the compact README rows instead — but
remember README pools papers across the windowed digest (could span 7
days), so always confirm the date with the user if it matters.

### 3. Decision loop

For each paper, use `AskUserQuestion`. **The `AskUserQuestion` tool caps
options at 4**, so use exactly these four:

| Label | Action |
|---|---|
| **Accept → seed + wiki** | Run both `seed_add.py` and the quark-wiki ingest. Default suggestion when the paper is clearly on-topic and the user has time. |
| **Accept → seed only** | Just `seed_add.py`. Use when the paper is a good citation-graph neighbor but the user doesn't want a full wiki page yet. |
| **Reject** | Ask for a one-phrase reason, then run `seed_reject.py`. Optionally ask "any blacklist phrases to add?" — only suggest this if the reason is "off-topic" or similar, never for "uninteresting" rejects. |
| **Defer** | Skip with no log. Sometimes the user wants to think about it. Don't pester. |

(If the user genuinely wants "wiki only — don't seed", they pick **Defer**
here and invoke `quark-wiki` directly afterward. Surfacing that as a
fifth option is not worth burning the 4-option limit — wiki-only is rare
and the workaround is one line.)

**Question text MUST include the full paper title.** Don't abbreviate
to a short name + tagline; users routinely ask "what's the full name?"
when the title is hidden, which wastes a round-trip. Include arXiv id
and bucket too. Example shape:

```
"OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs
and Beyond (arXiv:2605.19660, kv_cache, 10/10) — 22 HF upvotes,
3.0x decode speedup, open-source CUDA kernels. 怎么处理？"
```

Batch the questions when there are several papers — `AskUserQuestion`
accepts up to 4 questions in one call. For digests with more than 4
papers, paginate (4 papers at a time).

### 4. Execute decisions

Run the tools in order: rejects first (cheap, no LLM), then accepts.
On any single failure, report and continue with the rest — don't abort
the whole batch.

For each accept that includes "send to wiki", invoke the `quark-wiki`
skill via the `Skill` tool with the paper's arxiv id, title, abstract,
and the radar's reason snippet. Pass the bucket as a tag/category hint
so quark-wiki can file it appropriately.

### 5. Summarize at the end

After the batch, print a 2-3 line summary:
- N accepted (M with wiki ingest)
- N rejected (K added blacklist phrases)
- N deferred

Then ask: "Commit & push these changes?" — if yes, stage only the files
modified by the scripts (`seeds.yaml`, `data/curation/*.jsonl`,
optionally `config.yaml`, and any new files under `knowledge/`),
commit with a message like `curate: 2026-05-21 — +3 seeds, +2 rejects`,
and push. **Never auto-commit without asking.**

## Bucket gotchas

- `low_bits` takes priority over `ptq`/`qat` for **any** paper with weight
  bit-width ≤ 2. If `seed_add.py`'s auto-detection puts a 2-bit paper in
  `ptq` (because the LLM judged that way), the user can override with
  `--bucket low_bits`.
- The 6 valid buckets are: `ptq`, `low_bits`, `qat`, `kv_cache`,
  `pruning_distill`, `diffusion`. Anything else is rejected by
  `seed_add.py`.

## Handling Haiku hard-gate during seed_add

If a user wants to accept a paper that **was not in any recent `data/scored/*.json`**
(e.g. they want to seed an old paper from a different domain), `seed_add.py`
will call Haiku fresh. If Haiku hard-gates it, the script refuses to add
the paper. Two options:

1. Ask the user to confirm the bucket and re-run with `--bucket <name>`.
2. Suggest using `quark-wiki` instead — the paper is interesting but doesn't
   belong as a citation-graph seed.

## Quick reference

```bash
# Add seed (auto-bucket)
python scripts/seed_add.py --arxiv-id 2605.19561
python scripts/seed_add.py --name TORQ                      # fuzzy on data/summarized/
python scripts/seed_add.py --arxiv-id 2605.19561 --note "MXFP4 rotation"

# Reject
python scripts/seed_reject.py --arxiv-id 2605.99999 --reason "actually a survey"
python scripts/seed_reject.py --arxiv-id 2605.99999 --reason "diffusion-only" \
    --add-blacklist "stable diffusion,FID metric"

# Inspect curation history
tail -10 data/curation/accepted.jsonl
tail -10 data/curation/rejected.jsonl
```

## Don't

- **Don't auto-extract** blacklist phrases from a rejected paper's
  title/abstract. Patterns must be explicitly chosen by the user — easy to
  ban good keywords by accident.
- **Don't modify `prompts/relevance.md`** from this skill. Few-shot
  anchors affect the entire daily filter; a poorly chosen change can tank
  recall for a week. Surface "rubric drift" as a separate recommendation
  if `data/curation/rejected.jsonl` shows systematic miscalls (e.g. "20
  rejects of pruning_distill papers tagged as ptq" → suggest the user
  read the rubric, but don't edit it).
- **Don't auto-push.** Commit on user confirmation only. Cron pushes
  digests; this skill should never push under the user's git identity
  without explicit go-ahead.

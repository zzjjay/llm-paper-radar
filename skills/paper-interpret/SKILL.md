---
name: paper-interpret
description: Use when the user gives a single paper (arXiv id, arXiv URL, or paper name/acronym) and wants it interpreted — 翻译成中文 / 解读 / 讲原理 / 溯源 / paper-river / "帮我看这篇 paper" / "interpret this paper" / "这篇讲什么". Resolves the paper against the llm-paper-radar archive first (reusing its bilingual summary, scoring, triage history, and any existing lineage analysis), then produces the interpretation from whichever angles the user wants. NOT for triaging/accepting/rejecting a digest (use paper-triage) and NOT for fetching/re-scoring the pipeline (that's scripts/daily.sh).
---

# paper-interpret

Interpret one paper on demand. This skill is an **orchestrator**: it does the
radar-specific work itself (resolve → reuse local data → compare against the
archive) and delegates the generic reading work to the already-installed
`ljg-*` skills instead of re-implementing translation / storytelling / lineage.

## When to invoke

- User gives an arXiv id (`2607.01127`), an arXiv URL, or a name/acronym
  (`LogbQuant`, `解读 GPTQ-intrinsic LoRA`) and asks to understand it.
- Trigger phrases: "解读这篇 paper", "翻译成中文", "讲讲这篇的算法原理",
  "paper-river / 倒读 / 溯源这篇", "帮我看下 arXiv:XXXX", "interpret this paper".

Do **not** invoke for:
- Walking a daily digest accept/reject by accept → `paper-triage`.
- Fetching or re-scoring papers → `scripts/daily.sh` (cron pipeline).
- A batch of papers where the user wants analysis + visual cards → `ljg-paper-flow`.

## Repo location

Runs from the `llm-paper-radar` repo root (where `scripts/resolve_paper.py`,
`data/summarized/`, `seeds.yaml`, `paper-river/` live). If invoked elsewhere,
ask the user to `cd` into the repo first. The generic sub-skills (`ljg-read`,
`ljg-paper`, `ljg-paper-river`) are global and work from anywhere.

## Step 1 — resolve the paper (always do this first)

Run the resolver. It is read-only and never re-scores anything:

```bash
uv run python scripts/resolve_paper.py "<arxiv-id-or-name>"
```

Read its output. It tells you, in one shot:

- **Resolution** — matched local (`radar-local`), fetched live (`arxiv-live`),
  ambiguous (a candidate list — ask the user which id), or not found.
- **Paper** — title, authors, categories, abstract, links, code.
- **Radar scoring** (local only) — composite score, `topic_bucket`,
  `topic_relevance`, `practicality`, `format_or_method`, largest model tested,
  accuracy / inference-perf / calibration / peak-memory, the scorer's reason,
  and a ready-made **bilingual summary + highlights + related methods**.
- **Triage status** — accepted seed / rejected (with the human reason) / untouched.
- **Paper-river** — path to an existing lineage `.org`, or "none".
- **Siblings in bucket** + **bucket trend** — for novelty / timing judgment.

Handle the edge cases:
- **Ambiguous** → show the candidates, ask the user to pick an id, re-run.
- **Not found + is an id** → the resolver already tried a live fetch; if that
  also failed, fall back to `WebFetch` on `https://arxiv.org/abs/<id>`.
- **Not found + is a name** → ask the user for the arXiv id (names the radar
  never surfaced can't be looked up by title).

Do not re-grep `data/`, `seeds.yaml`, or `paper-river/` yourself — the resolver
already gathered all of it.

## Step 2 — pick angles

If the user named an angle ("翻译" / "讲原理" / "溯源"), do that. If they just
said "解读这篇", offer the menu (default to A+B when they don't choose):

- **A. 中文精读 / 翻译** — faithful Chinese interpretation of the abstract and,
  if the user wants the full text, hand off to **`ljg-read`** (伴读 + 英译中).
  If a radar `summary_zh` exists, use it as the spine and expand, don't redo it.
- **B. 算法背景 + 原理** — problem → why prior methods fall short → this paper's
  mechanism → result. For a story-shaped retelling hand off to **`ljg-paper`**
  (seven-beat spine). Ground the "prior methods" part in the resolver's
  `related_methods` and `siblings`.
- **C. Paper-river 溯源 (倒读法)** — if the resolver reports an existing
  `paper-river/*.org`, **read and present that** rather than regenerating. If
  none exists and the user wants it, hand off to **`ljg-paper-river`** on the
  arXiv URL, but **override its default output path**: save the result into this
  repo as `paper-river/<acronym>-<arxiv-id>.org` (dot form, same convention as
  the cron `gen_paper_river.sh`), NOT the skill's default `~/Documents/notes/`.
  Use the paper's short acronym for `<acronym>`. (Generating fresh is ~5-10 min
  of web research — tell the user.)
- **D. Radar-native analysis** (this skill's unique value — see below).
- **E. 全文中文详解** — when the user wants the *whole* paper in Chinese, NOT
  the selective 伴读 of (A). Rules:
  - *Copyright*: arXiv papers are usually under arXiv's non-exclusive license
    (check the license on the abstract page — only CC-BY/CC0 permit a verbatim
    full translation). Under the default license, do **not** produce a
    clause-by-clause translation of the whole paper. Produce a **section-by-section
    Chinese explanation in your own words** (复述) instead — full coverage of
    every section + appendix, with tables/numbers/formulas reproduced verbatim
    (facts and LaTeX aren't the copyrightable expression).
  - *Format = Markdown* (`translation_zh.md`, NOT `.org`): GitHub renders LaTeX
    math in Markdown (`$...$` inline, `$$...$$` on its own line) and native
    `| … |` tables, so formulas and tables display correctly when browsing the
    repo; `.org` renders neither. Do **not** use HTML — GitHub shows `.html` as
    source, not rendered. Open the file with a line noting it's a paraphrased
    section digest, not a verbatim translation.
    - *GitHub math is a restricted MathJax subset* — tell the subagent (and
      check afterwards) to avoid macros GitHub rejects: `\operatorname` (use
      `\arg\min`, `\arg\max`, or a bare `\min`/`\max`), `\lVert`/`\rVert` (use
      `\|`), and sizing prefixes `\big`/`\bigl`/`\bigr`/`\Big…` before a
      delimiter (drop them, or use `\left…\right`). Also **never nest `$…$`
      inside a `$$…$$` block** — e.g. `\text{$b$-bit}` — the inner `$` closes the
      outer math span early and orphans whatever follows (a `\left` then errors);
      write `b\text{-bit}` instead. Likewise **use `\lbrace`/`\rbrace`, never
      `\{`/`\}`** inside math — GitHub's markdown strips the backslash, turning
      `\left\{` into `\left{` (an invalid delimiter → the `\left` error). After
      writing, **run
      `uv run python scripts/check_math.py <file>`** — it flags exactly these
      GitHub failure modes (denylisted macros, stray/unbalanced `$`, brace and
      `\left`/`\right` mismatches) with `file:line`. Fix every hit until it
      reports clean, or the equations show a red "macro not allowed /
      unrecognized delimiter" error on GitHub.
  - *Figures*: the arXiv HTML usually has NO embedded figure images (they're
    vector), so grab them from the PDF: `curl -sL https://arxiv.org/pdf/<id> -o
    /tmp/p.pdf`, find each figure's page (`pdftotext -f N -l N … | grep 'Figure K:'`),
    render+crop with `pdftocairo -png -r 150 -x <x> -y <y> -W <w> -H <h> -f N -l N`,
    Read the crop to verify the box, save under
    `interpretations/<acro>-<id>/images/`, and reference it relatively at the
    right spot (`![图 K：<caption>](images/<name>.png)`). Skip if the paper has
    no meaningful figures.
  - *Mechanics*: this is a long output (~30k-token papers). Delegate to a
    **Sonnet subagent** (translation needs language, not Opus-level reasoning —
    cheaper, and it isolates the full text from the main context). Fetch the
    full text yourself first via `curl https://arxiv.org/html/<id>vN` (the
    pipeline fetchers only grab metadata, never the body — nothing to reuse),
    strip tags to a temp text file, and hand that path to the subagent. Tell
    the subagent to **write the file incrementally** (Write the front matter +
    first sections, then Edit-append each remaining section/appendix), because a
    single huge streamed Write tends to stall mid-response. Note: SKILL guidance
    is soft — if a run still stalls, resume the subagent and have it append the
    rest; the partial file on disk is the checkpoint.

## Step 3 — radar-native analysis (the part no generic skill can do)

Only this repo has the scored archive. When relevant, add:

1. **同类对比 / novelty** — compare against the `siblings` in the same
   `topic_bucket`. Is the core idea new, or a re-skin of an existing method?
   Name the closest siblings by id and say what's actually different.
2. **趋势定位 / timing** — from `bucket_trend`, is this direction heating up or
   cooling? Does that change whether it's worth deep-reading now?
3. **工程可落地性** — reuse the radar's `practicality` axis + `format_or_method`
   / largest-model-tested / inference-perf / calibration-cost / peak-memory to
   judge deployment difficulty (e.g. on AMD / production inference), not a vague
   academic take. Call out `unknown` fields as open questions to check in the PDF.
4. **是否已被 triage** — if accepted (seed) or rejected, surface the prior human
   verdict and reason so the user doesn't relitigate it. If rejected, lead with
   why the radar's curator passed on it.

## Output location

Everything the skill produces is archived **in this repo** — override each
sub-skill's default `~/Documents/notes/` output path and Write to the repo
instead. `<acronym>-<arxiv-id>` uses dot form; `<acronym>` = the paper's short
name.

- **Paper-river 溯源 (C)** → `paper-river/<acronym>-<arxiv-id>.org`, plus an
  `_en.org` English sibling (translate the zh original, same as the cron). Reuse
  an existing file instead of regenerating.
- **The other angles share one per-paper folder** `interpretations/<acronym>-<arxiv-id>/`
  (create it if missing). Inside it:
  - **`README.md` — the hub / 总入口** (always write/update this; it is what
    GitHub auto-renders when someone opens the folder — NOT `radar.org` or any
    other name). It holds:
    - the **radar-native analysis (D)** itself (same-bucket novelty, trend,
      practicality, triage verdict), written by this skill in Markdown;
    - a **navigation section** linking every angle file that exists in the
      folder (`paper.org`, `reading.org`, `translation_zh.org`), plus the
      paper-river lineage (`../../paper-river/<acronym>-<id>.org` + `_en.org`)
      and the radar data source (`../../data/summarized/<date>.json`);
    - the paper's abs / PDF / code links.
    Also echo the key D points in chat.
  - **中文精读 / 伴读 (A)** → `reading.org` (pass ljg-read this exact path to Write).
  - **原理故事 (B)** → `paper.org` (same path override for ljg-paper; if it
    extracts an overview image, put it in the same folder).
  - **全文中文详解 (E)** → `translation_zh.md` (Markdown so GitHub renders math +
    tables; figures cropped from the PDF into `images/`; per the rules in step 2E).

  Whenever you add or regenerate an angle file, update `README.md`'s navigation
  section so the hub always reflects what's actually in the folder.

Tell the user each path you saved to.

## Output

- Lead with a 1-2 line "what this paper is" in Chinese (match the repo's zh
  digest voice), then the requested angles.
- Always cite `arXiv:<id>` + link so the user can verify.
- When you reused radar data, say so ("radar 已有摘要/评分，以下在其基础上展开")
  so it's clear what's archived vs. freshly generated.
- Keep it tight — this is an interpretation, not a report. No score-the-paper,
  no praise sandwich.

## What this skill must NOT do

- Never run `scripts/daily.sh`, `pipeline.*`, or anything that fetches/re-scores.
- Never write to `seeds.yaml` / `rejected.jsonl` (that's `paper-triage`).
- Don't regenerate a paper-river that already exists on disk.

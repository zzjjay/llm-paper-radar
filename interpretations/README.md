# interpretations

On-demand single-paper interpretations from the `paper-interpret` skill
(`skills/paper-interpret/`). One folder per paper, named `<acronym>-<arxiv-id>/`.

## Layout

```
<acronym>-<arxiv-id>/
├── README.md          hub — always present
├── reading.org        伴读 (ljg-read)          — optional
├── paper.org          seven-beat story (ljg-paper) — optional
├── translation_zh.md  full Chinese digest (Sonnet) — optional
└── images/            figures cropped from the PDF
```

| File | Angle | Producer |
|------|-------|----------|
| `README.md` | Hub: radar-native analysis (novelty / trend / practicality / triage) + nav to the files below, the `../paper-river/` lineage, and the `data/summarized/` source | paper-interpret |
| `reading.org` | Chinese close-reading, selective, bilingual on key passages | ljg-read |
| `paper.org` | Background + mechanism, told as a story | ljg-paper |
| `translation_zh.md` | Every section + appendix in Chinese, paraphrased (not a verbatim translation) | Sonnet subagent |

Only `README.md` is guaranteed; the rest are generated per requested angle, and
the hub's nav lists only what exists.

## Conventions

- **`README.md` is the hub** — GitHub auto-renders it on folder open; keep its
  nav in sync when angle files are added.
- **Format** — angle files are `.org` (the `ljg-*` skills emit org); the digest
  is `.md` so GitHub renders its math and tables.
- **Math must be Markdown-inert** — inline `` $`…`$ ``, display ` ```math ` fences.
  Plain `$…$` / `$$…$$` get mangled by GitHub's Markdown pass (eaten underscores,
  collapsed `\\`, dropped `\{`). `scripts/check_math.py` lints for this.
- **Lineage lives elsewhere** — 倒读法 analyses are in `../paper-river/`
  (`<acronym>-<id>.org` + `_en.org`), from the cron pipeline.
- Everything here is committed with the repo, unlike the sub-skills' default
  `~/Documents/notes/`.

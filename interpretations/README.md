# interpretations

On-demand single-paper interpretations produced by the `paper-interpret` skill
(see `skills/paper-interpret/`). Each paper gets its own folder named with the
radar's `<acronym>-<arxiv-id>` convention, holding one file per angle:

```
interpretations/
└── <acronym>-<arxiv-id>/
    ├── README.md          # hub: radar-native analysis + nav to the files below and to lineage/data source
    │                      #      (GitHub renders this by default when the folder is opened)  (paper-interpret itself)
    ├── reading.org        # Chinese close-reading / 伴读 (selective, bilingual on skeleton passages)  (ljg-read)
    ├── paper.org          # background + mechanism (seven-beat story)  (ljg-paper)
    ├── translation_zh.md  # full section-by-section Chinese digest (all sections + appendices, paraphrased);
    │                      #   Markdown so GitHub renders formulas/tables  (Sonnet subagent)
    └── images/            # figures cropped from the PDF (referenced by translation_zh.md)
```

Angle files default to `.org` (伴读 / story, since the `ljg-*` skills emit org).
The full digest uses `.md` because GitHub only renders LaTeX math and native
tables in Markdown — but note that even in Markdown you must use the
Markdown-inert math forms (inline `` $`…`$ ``, display ` ```math ` fenced blocks);
plain `$…$` / `$$…$$` get mangled by GitHub's Markdown pass. `scripts/check_math.py`
lints for this.

`README.md` is the paper's hub and is always present; the other files are only
generated when their angle is requested (the hub's nav lists only what exists).

Deep-lineage "倒读法" analyses live in `../paper-river/` instead
(`<acronym>-<id>.org` + `_en.org`), matching the cron pipeline's archive.

These files are written on demand and committed with the repo, unlike the
sub-skills' default `~/Documents/notes/` Denote location.

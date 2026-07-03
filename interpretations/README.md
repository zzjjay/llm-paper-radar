# interpretations/

On-demand single-paper interpretations produced by the `paper-interpret` skill
(see `skills/paper-interpret/`). One paper can have several angle files, all
sharing the radar's `<acronym>-<arxiv-id>` prefix:

| suffix | angle | producer |
|---|---|---|
| `__reading.org` | 中文精读 / 翻译 | `ljg-read` (伴读 + 英译中) |
| `__paper.org` | 算法背景 + 原理（七拍故事） | `ljg-paper` |
| `__radar.org` | radar-native 分析（同类对比 / 趋势 / 落地性 / triage 判定） | `paper-interpret` itself |

Deep-lineage "倒读法" analyses live in `../paper-river/` instead (zh `.org` +
`_en.org`), matching the cron pipeline's archive.

These files are written on demand and committed with the repo, unlike the
sub-skills' default `~/Documents/notes/` Denote location.

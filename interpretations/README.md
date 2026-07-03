# interpretations/

On-demand single-paper interpretations produced by the `paper-interpret` skill
(see `skills/paper-interpret/`). Each paper gets its own folder named with the
radar's `<acronym>-<arxiv-id>` convention, holding one file per angle:

```
interpretations/
└── <acronym>-<arxiv-id>/
    ├── reading.org        # 中文精读 / 伴读（选择性，骨架段中英并列）  (ljg-read)
    ├── paper.org          # 算法背景 + 原理（七拍故事）  (ljg-paper; overview image, if any, sits here too)
    ├── radar.org          # radar-native 分析（同类对比 / 趋势 / 落地性 / triage 判定）  (paper-interpret itself)
    └── translation_zh.org # 逐节中文详解（全节+附录，复述式，非逐句译文）  (Sonnet subagent)
```

Not every paper has all three files — only the angles actually requested.

Deep-lineage "倒读法" analyses live in `../paper-river/` instead (`<acronym>-<id>.org`
+ `_en.org`), matching the cron pipeline's archive.

These files are written on demand and committed with the repo, unlike the
sub-skills' default `~/Documents/notes/` Denote location.

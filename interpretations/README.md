# interpretations

On-demand single-paper interpretations produced by the `paper-interpret` skill
(see `skills/paper-interpret/`). Each paper gets its own folder named with the
radar's `<acronym>-<arxiv-id>` convention, holding one file per angle:

```
interpretations/
└── <acronym>-<arxiv-id>/
    ├── README.md          # 总入口：radar-native 分析 + 指向下列各文件与溯源/数据源的导航
    │                      #        （GitHub 打开文件夹时默认渲染这个）  (paper-interpret itself)
    ├── reading.org        # 中文精读 / 伴读（选择性，骨架段中英并列）  (ljg-read)
    ├── paper.org          # 算法背景 + 原理（七拍故事）  (ljg-paper; overview image, if any, sits here too)
    └── translation_zh.org # 逐节中文详解（全节+附录，复述式，非逐句译文）  (Sonnet subagent)
```

`README.md` 是这篇 paper 的总入口，始终存在；其余 `.org` 只在对应角度被请求时才生成
（README.md 的导航区只列实际存在的文件）。

Deep-lineage "倒读法" analyses live in `../paper-river/` instead (`<acronym>-<id>.org`
+ `_en.org`), matching the cron pipeline's archive.

These files are written on demand and committed with the repo, unlike the
sub-skills' default `~/Documents/notes/` Denote location.

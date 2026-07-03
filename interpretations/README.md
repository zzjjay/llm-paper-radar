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
    ├── paper.org          # 算法背景 + 原理（七拍故事）  (ljg-paper)
    ├── translation_zh.md  # 逐节中文详解（全节+附录，复述式）；Markdown 让 GitHub 渲染公式/表格  (Sonnet subagent)
    └── images/            # 从 PDF 裁出的图（translation_zh.md 引用）
```

角度详解文件默认用 `.org`（伴读/故事，因 ljg-* skill 产出 org）；全文详解用 `.md`,
因为 GitHub 只对 Markdown 渲染 LaTeX 公式（`$...$` / `$$...$$`）和原生表格，`.org` 不渲染。

`README.md` 是这篇 paper 的总入口，始终存在；其余 `.org` 只在对应角度被请求时才生成
（README.md 的导航区只列实际存在的文件）。

Deep-lineage "倒读法" analyses live in `../paper-river/` instead (`<acronym>-<id>.org`
+ `_en.org`), matching the cron pipeline's archive.

These files are written on demand and committed with the repo, unlike the
sub-skills' default `~/Documents/notes/` Denote location.

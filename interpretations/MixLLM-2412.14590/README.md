# MixLLM — arXiv:2412.14590

> 只多花约 10% 的比特，把 4-bit 量化的精度拉到接近 8-bit。做法：在**输出通道之间**做混合精度，
> 用一个**全局**（跨层）重要性判据挑出该保精度的通道，再配一套能吃 int8 Tensor Core 的 kernel。

本文件是这篇 paper 的**解读总入口**。radar 视角在下方，各角度详读见导航。

## 论文

- **标题**：MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design
- **链接**：[abs](https://arxiv.org/abs/2412.14590) · [PDF](https://arxiv.org/pdf/2412.14590v2) · [code](https://github.com/microsoft/MixLLM)
- **作者**：Zhen Zheng, Xiaonan Song, Chuanjie Liu (Microsoft) · 2024-12（会议状态未确证：PDF 页脚标 "under review, MLSys"，AAAI'25 无确凿证据）

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版，给外行也能复述 |
| 中文伴读 | [reading.org](reading.org) | 选择性精读，骨架段中英并列 + 碰撞 |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文翻译（覆盖全节 + 附录；为版权计以复述方式呈现，非逐字直译） |

## Radar 视角（有限）

- **未收录**：论文发表于 2024-12，早于 radar 存档最早日期（2026-05），因此库里**无评分、无同类对比、无趋势数据**——"和同 bucket 论文比新在哪"这类 radar 独有分析这次给不了。
- **归类**：若纳入，属 `ptq`（混合精度 W4.4A8 训练后量化）。
- **落地性**（据论文）：偏高——Microsoft 出品、代码开源、集成 vLLM、A100/H100 真实 kernel + 加速数据，是明确面向部署的算法-系统协同工作。
- **triage 状态**：未 accept 也未 reject。

## 核心速览

- **配置 W4.4A8**：约 10% 高重要性输出通道走 8-bit（对称），90% 走 4-bit（非对称），激活统一 8-bit；均 group-wise，group size 128。平均 4.4 bit/权重。
- **全局重要性**：按每个输出通道对**模型最终损失**的影响排序（Taylor 展开 + 用 Fisher 信息矩阵近似 Hessian），一次算完（single-pass）。不同层因此分到不同数量的高精度通道——这是它区别于"层内启发式"的关键。
- **系统协同**：two-step dequantization 让混精配置也能用 int8 Tensor Core；range-dependent fast I2F 把昂贵的整数→浮点转换简化成一次浮点减法（靠 group size 128 保证点积范围可控）；软件流水重叠访存/反量化/MatMul。
- **结果**：多 10% 比特下，Llama-3.1-70B 的 perplexity 增量从 SOTA 约 0.5 压到 0.2 以内；三个主流模型上 MMLU-Pro 损失从 1.92 降到 0.99；A100/H100 上 SOTA 系统效率。

## 溯源

倒读法脉络：[../../paper-river/MixLLM-2412.14590.org](../../paper-river/MixLLM-2412.14590.org)
——从 MixLLM 倒推 6 层至源头：OBS(1993) → OBQ → GPTQ → LLM.int8() → SpQR/SqueezeLLM/OWQ
→ AWQ → Atom/SliM-LLM → MixLLM（二阶补偿量化 + outlier/salience 分离 这条谱系）。

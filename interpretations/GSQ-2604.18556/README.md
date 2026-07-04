# GSQ — arXiv:2604.18556

> 把「低比特量化只能靠复杂的 vector/trellis 方法冲精度」这个共识掀翻——GSQ 证明一个
> 精心优化的**纯 scalar** 量化器，在 2-3 bit 就能追平 QTIP 那条前沿，且能直接塞进现有
> GGUF/scalar 部署格式。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling
- **链接**：[abs](https://arxiv.org/abs/2604.18556) · [PDF](https://arxiv.org/pdf/2604.18556v2) · [code](https://github.com/IST-DASLab/GSQ)
- **作者**：Alireza Dadgarnia, Soroush Tabesh, Mahdi Nikdan, Michael Helcig, Eldar Kurtic, Maximilian Kleinegger, Dan Alistarh (IST-DASLab)
- **发表**：2026-04-20 · cs.CL / cs.LG

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版《差距不在格式，在优化》，给外行也能复述 |
| 中文伴读 | [reading.org](reading.org) | 选择性精读，骨架段中英并列 + 碰撞 + 复盘 |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文翻译（覆盖全节 + 附录；为版权计以复述方式呈现，非逐字直译；GitHub 渲染公式/表格/图 1） |
| 溯源倒读 | [../../paper-river/GSQ-2604.18556.org](../../paper-river/GSQ-2604.18556.org) （中）· [_en](../../paper-river/GSQ-2604.18556_en.org)（英） | 倒读法脉络 OBD→OBS→OBQ→GPTQ→GSQ |

数据来源：[data/summarized/2026-05-23.json](../../data/summarized/2026-05-23.json)（radar 打分记录）。

## Radar 记录

- composite=**9**  ·  bucket=**low_bits**  ·  topic_relevance=5  ·  practicality=4  ·  hard_gate=no
- format/method：W2/W3 scalar PTQ + Gumbel-Softmax grid optimization
- largest model tested：Kimi-K2.5（trillion-scale MoE）

## 同类对比 / novelty

同 `low_bits` 桶的邻居基本都在 2-bit 附近，但走不同路线：

| 论文 | 路线 |
|---|---|
| [UniSVQ (2606.10520)](https://arxiv.org/abs/2606.10520) | 2-bit 统一 scalar-vector |
| [CAT-Q (2606.26650)](https://arxiv.org/abs/2606.26650) | 1.58-bit ternary PTQ |
| [LC-QAT (2606.10531)](https://arxiv.org/abs/2606.10531) | 2-bit QAT + vector |
| [CodeGEMM (2512.17970)](https://arxiv.org/abs/2512.17970) | 2-bit codebook + 专用 GEMM kernel |
| [Recover-LoRA (2606.04238)](https://arxiv.org/abs/2606.04238) | W2 混合精度 + LoRA KD |
| [Qift (2606.02823)](https://arxiv.org/abs/2606.02823) | W2A4KV4，Hadamard rotation PTQ |

GSQ 的差异化很清晰：**别人靠更复杂的格式（vector/codebook）绕过 scalar 的精度 gap，
GSQ 靠优化把 gap 补回来**，产物仍是纯对称 scalar grid。是「新思路」不是「新瓶装旧酒」。

## 工程可落地性（practicality=4，桶内偏高）

- 部署路径最干净：复用现有 scalar / GGUF K-Quant kernel，无需自定义 CUDA kernel。
- 能直接改进公开 GGUF K-Quant checkpoint，且投影回同一部署格式。
- 规模验证到 trillion MoE（Kimi-K2.5），vector 方法在该规模基本用不了。
- **待确认（radar 标 unknown）**：calibration 成本（Gumbel-Softmax 联合优化明显重于 AWQ/GPTQ,
  未披露时长）、峰值显存（未披露）。复现前需先确认这两项工程成本。

## 趋势定位

`low_bits` 方向近月零星更新（每天 1-2 篇），不是爆发赛道但持续有产出。GSQ（4 月）是该方向
较早且被打到高分（composite 9）的一篇，算标杆之一。

## Triage 建议

尚未 triage。建议 accept 进 `low_bits` seeds——Alistarh 组、代码开源、部署实用，是 scalar
路线在 2-3 bit 的代表作。

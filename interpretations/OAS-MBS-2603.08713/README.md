# OAS-MBS — arXiv:2603.08713

> MXFP4 硬件省但精度输 NVFP4 约 10%——不换硬件、不改块大小，只改"scale 怎么选 +
> 再加一层粗尺子"，把差距压到 1% 以内，GEMM 开销仅 6.2%。

本文件是这篇 paper 的**解读总入口**。radar-native 分析在下方，各角度详读见导航。

## 论文

- **标题**：Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction
- **链接**：[abs](https://arxiv.org/abs/2603.08713) · [PDF](https://arxiv.org/pdf/2603.08713v1)
- **作者**：Jatin Chhugani, Geonhwa Jeong, Bor-Yiing Su, Yunjie Pan, Hanmei Yang, Aayush Ankit, Jiecao Yu, Summer Deng, Yunqing Chen, Nadathur Satish, Changkyu Kim
- **发表**：2026-01-30 · cs.AR / cs.AI / cs.LG / cs.PF

## 解读导航

| 角度 | 文件 | 内容 |
|---|---|---|
| 原理故事 | [paper.org](paper.org) | 七拍故事版《怪尺子，不如怪量法》——精度差距是软件用法问题还是硬件问题 |
| 中文伴读 | [reading.org](reading.org) | 选择性精读，骨架段三层翻译 + 碰撞提问（归档模式，Agent 代答） |
| 中文翻译 | [translation_zh.md](translation_zh.md) | 全文中文翻译（覆盖全部 7 节 + 附录 A-E；版权原因采用 section-by-section 复述，非逐字翻译；论文全部 8 张图均已裁剪自 PDF 并嵌入） |
| 溯源倒读 | [../../paper-river/OAS-MBS-2603.08713.org](../../paper-river/OAS-MBS-2603.08713.org)（中）· [_en](../../paper-river/OAS-MBS-2603.08713_en.org)（英） | 倒读法脉络 LLM.int8→SmoothQuant→GPTQ→块浮点/OCP MX→NVFP4→本文 OAS+MBS |

数据来源：[data/summarized/2026-01-30.json](../../data/summarized/2026-01-30.json)（radar 打分记录）。

## Radar 记录

- composite=**10**  ·  bucket=**ptq**  ·  topic_relevance=5  ·  practicality=5  ·  hard_gate=no
- format/method：MXFP4 PTQ（OAS + MBS，纯软件，无需硬件改动）
- largest model tested：unknown（论文未明确报告最大规模；已知覆盖 Llama3.1-8B、Qwen3-8B、
  Llama4-Maverick、DeepSeek-R1）
- accuracy：MXFP4 vs NVFP4 精度差从约 10% 缩减至平均低于 1%
- inference perf：GEMM overhead 平均 6.2%，保留 MX 硬件效率优势（tensor core 面积节省 12%）
- calibration cost：软件层面调整，无需硬件改动，校准成本低
- peak memory：unknown（论文未披露）

## 同类对比 / novelty

`ptq` 桶内近邻大多做的是「换量化格式/校准策略」，OAS-MBS 走的是「不换格式，修 scale 选法」：

| 论文 | 路线 |
|---|---|
| [ReSET (2606.13233)](https://arxiv.org/abs/2606.13233) | NVFP4 PTQ + step-aware 温度缩放 |
| [MatGPTQ (2602.03537)](https://arxiv.org/abs/2602.03537) | Matryoshka PTQ，多精度 W4/W8，bit-slicing |
| [Mix-Quant (2605.20315)](https://arxiv.org/abs/2605.20315) | NVFP4 prefill + BF16 decode（phase-aware） |
| [AAAC (2605.08692)](https://arxiv.org/abs/2605.08692) | W4A16，激活感知自适应 codebook |
| [Coverage-Based Calibration (2604.24008)](https://arxiv.org/abs/2604.24008) | W4A16 INT4，加权集合覆盖选校准样本 |
| [多尺度校准 (2602.07465)](https://arxiv.org/abs/2602.07465) | Hessian-based 多尺度序列长度校准 |

差异清晰：邻居们大多假设 NVFP4/自定义格式已经是精度上限，围绕它做校准或混合精度调优；
本文反过来问"MXFP4 差的 10% 是硬件能力问题还是软件用法问题"，答案是纯软件可补——
不是新瓶装旧酒，是对既有 OCP MX 标准的一次"用法层面"翻案。

## 工程可落地性（practicality=5，桶内最高分）

- 无需硬件改动：block size、scale 格式（E8M0）全部照旧，只改"怎么挑这个 2 的幂"+ 加一层
  macro scale，直接对标已有 OCP MX 硬件路径。
- GEMM 开销均值 6.2%，远低于同类方案 MXPlus 的 ~54%；但**形状敏感**——小矩阵维度（512）
  下开销可达 14.84%，小 batch/边缘部署场景需要单独评估。
- MBS-Dynamic 依赖查找表 + 在线搜索最优 scale，只在权重侧用（可离线预计算）；激活侧走
  Static 版本控制在线开销。
- **待确认（radar 标 unknown）**：largest model tested 与 peak memory 论文未披露，复现前
  需要翻 PDF 或补实验确认。
- 未做任何 QAT/微调，是纯 direct-cast 下的结果——若叠加 QAT，差距可能进一步缩小，也可能
  改变现有结论，论文自己列为未来工作。

## 趋势定位

`ptq` 桶持续高频出稿（近期 2-2/天，07-01/07-05 降到 1/天），是最卷的桶之一。本文 composite=10
是桶内顶格分，且是少见的「面向硬件标准之争」的工程性投稿（多数邻居是纯算法/校准侧微调），
值得在这个方向优先深读。

## Triage 建议

尚未 triage。composite=10、topic_relevance/practicality 双满分、且直接可用于生产推理配方，
建议 accept 进 `ptq` seeds——这是难得的"零硬件成本、直接压缩行业标准精度差距"的落地型工作。

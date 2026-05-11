# LLM 推理优化日报 · 2026-05-11

> 📅 抓取窗口: 2026-05-11 (UTC daily window)
> 📊 共扫描 93 篇 → 通过过滤 0 篇 (阈值 ≥7)

> 这是 [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar) 自动生成的最新一日 digest。
> 历史: [INDEX.md](INDEX.md) · 配置: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6

## 🔥 Top 5 (Full Detail)

### 1. SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting (9/10) 🔁
**hf_daily** · `2605.07243` · 2026-05-08
👥 Weijie Shi, Qiang Xu, Fan Deng... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.07243) · [PDF](https://arxiv.org/pdf/2605.07243.pdf)
📡 来源: hf_daily (👍 2, 💬 2)

#### 中文摘要
Speculative decoding 的现有方案存在两难困境：自回归 drafter（如 EAGLE-3）保持路径依赖但每层需调用一次 drafter，延迟较高；并行 drafter 减少调用次数但忽略位置间依赖导致验证拒绝率高。SpecBlock 提出块迭代式 drafting：每次 drafter forward 生成 K 个有依赖关系的 token（一个 block），通过层内 hidden state shift 和块间状态继承维持路径依赖。结合协同训练的 rank head 实现动态树结构分配，以及基于 bandit 的在线自适应更新，在降低 44-52% drafting 开销的同时，speedup 超越 EAGLE-3 达 8-19%。

- 🎯 块迭代 drafting：每次 forward 生成 K 个依赖 token，通过层内 hidden state shift 和块间状态继承双机制保持路径依赖
- 📊 drafting 开销仅为 EAGLE-3 的 44-52%，mean speedup 提升 8-13%；加入 cost-aware bandit 自适应后提升达 11-19%
- 💡 valid-prefix mask 在训练时屏蔽早期位置错误导致的后续位置 loss，避免 drafter 学习到推理阶段不存在的 prefix
- 🔧 cost-aware bandit 利用验证器的免费反馈，仅在预期吞吐增益超过更新开销时才更新 drafter，实现无额外成本的部署自适应

#### English Summary
Speculative decoding faces a dilemma between autoregressive drafters (e.g., EAGLE-3), which preserve path dependence but incur one drafter call per tree depth, and parallel drafters, which reduce calls but ignore inter-position dependencies leading to high rejection rates. SpecBlock introduces a block-iterative drafter that generates K dependent tokens per forward pass (a block), maintaining path dependence via intra-block layer-wise hidden state shifts and inter-block state inheritance. A co-trained rank head replaces fixed top-k tree expansion with dynamic per-position branching, and a valid-prefix mask prevents the drafter from learning on invalid prefixes during training. A cost-aware bandit at deployment leverages free verifier feedback to update the drafter only when expected throughput gain exceeds update cost, extending gains further.

- 🎯 Block-iterative drafting generates K dependent tokens per forward, with layer-wise hidden state shift within blocks and state inheritance across blocks to maintain path dependence
- 📊 Achieves 8-13% higher mean speedup over EAGLE-3 at only 44-52% of its drafting cost; cost-aware bandit adaptation extends the lead to 11-19%
- 💡 Valid-prefix mask drops loss at later positions once an earlier one is incorrect, preventing the drafter from training on prefixes that never appear at inference
- 🔧 Cost-aware bandit uses free verifier feedback to selectively update the drafter only when expected throughput gain exceeds update overhead, enabling zero-cost deployment adaptation

---

### 2. Shallow Prefill, Deep Decoding: Efficient Long-Context Inference via Layer-Asymmetric KV Visibility (9/10) 🔁
**hf_daily** · `2605.06105` · 2026-05-07
👥 Jungsuk Oh, Hyeseo Jeon, Hyunjune Ji... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.06105) · [PDF](https://arxiv.org/pdf/2605.06105.pdf)
📡 来源: hf_daily (👍 1, 💬 1)

#### 中文摘要
长上下文推理中，prefill 阶段生成的 KV cache 在所有层持久化并在 decode 阶段反复访问，代价高昂。SPEED（Shallow Prefill, dEEp Decode）提出一种非对称 KV 可见性策略：prompt token 的 KV states 仅在浅层（低层）物化，而 decode 阶段的 token 保持全层深度；仅保留一个 BoS anchor token 作为全层锚点。在 Llama-3.1-8B 指令微调实验中，使用 75% 层数处理 prefill token 时，128K 上下文下 TTFT 降低 33%、TPOT 降低 22%、active KV memory 减少 25%，OLMES benchmark 平均分从 51.4 仅降至 51.2。

- 🎯 方法：prompt token KV states 仅在前 75% 层物化，decode token 保持全深度，配合单个 BoS anchor 维持质量
- 📊 结果：128K 上下文下 TTFT -33%、TPOT -22%、active KV memory -25%，benchmark 分数 51.2 vs 基线 51.4
- 💡 创新：首次从 decode 可见性集合中直接移除上层 prefill token KV，而非压缩或廉价化其存储
- ⚠️ 局限：目前仅在 Llama-3.1-8B 单模型上验证，anchor 机制对不同任务的普适性有待进一步探究

#### English Summary
Long-context inference is expensive because prefill-stage KV states are cached at every layer and repeatedly attended to during autoregressive decode. SPEED (Shallow Prefill, dEEp Decode) introduces a phase-asymmetric KV-visibility policy: non-anchor prompt tokens materialize KV states only in lower layers, while decode-phase tokens remain full-depth, with a single BoS token serving as the full-depth anchor. In a Llama-3.1-8B instruction-tuning study at 128K context, using 75% of layers for prefill tokens reduces TTFT by 33%, TPOT by 22%, and active KV memory by 25%, with OLMES benchmark score dropping only from 51.4 to 51.2. Layer-wise analysis confirms that the retained shallow layers cover the key prompt-selection and representation-stabilization regions.

- 🎯 Method: Prompt token KV states materialized only in the bottom 75% of layers; decode tokens remain full-depth with a single BoS anchor to preserve quality
- 📊 Results: At 128K context, TTFT −33%, TPOT −22%, active KV memory −25%, with benchmark score 51.2 vs. full-depth baseline 51.4
- 💡 Innovation: Eliminates upper-layer prefill KV from the decode visibility set entirely, rather than compressing or cheapening their storage
- ⚠️ Limitation: Validated on Llama-3.1-8B only; generalizability of the BoS anchor mechanism across models and tasks needs further study

---

### 3. UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification (8/10) 🔁
**hf_daily** · `2605.06221` · 2026-05-07
👥 Qihang Fan, Huaibo Huang, Zhiying Wu... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.06221) · [PDF](https://arxiv.org/pdf/2605.06221.pdf)
📡 来源: hf_daily (👍 18, 💬 1)

#### 中文摘要
现有长文本 prefill 加速方法（稀疏 attention）仅对全 attention 模型有效，迁移到线性/滑窗混合架构时性能严重退化，且与 continuous batching 不兼容。UniPrefill 在 token 级别直接对计算进行块级动态稀疏化，适配几乎任意模型架构。系统层面将其实现为 continuous batching 算子，并扩展 vLLM 调度策略以原生支持 prefill-decode 协同处理与 tensor parallel，TTFT 最高提速 2.1×，并发请求增多时加速效果更显著。

- 🎯 方法：块级动态稀疏化在 token 级别直接加速计算，兼容全 attention、线性/全 attention 混合、滑窗/全 attention 混合等多种架构
- 📊 结果：TTFT 最高提速 2.1×，随并发请求数增加加速比进一步提升
- 🔧 工程：实现为 continuous batching 算子并扩展 vLLM 调度器，原生支持 prefill-decode 协同处理与 tensor parallel
- 💡 创新：首个同时解决架构通用性与 continuous batching 兼容性的 prefill 加速框架

#### English Summary
Existing sparse attention-based prefill acceleration methods achieve peak speedup only on full-attention models and suffer significant degradation when transferred to emerging hybrid architectures (e.g., linear/full or sliding-window/full attention hybrids), while also being incompatible with continuous batching. UniPrefill addresses this by applying block-wise dynamic sparsification directly at the token level, making it applicable to virtually any model architecture. On the system side, it is implemented as a continuous batching operator with extensions to vLLM's scheduler for native prefill-decode co-processing and tensor parallelism support. UniPrefill achieves up to 2.1× TTFT speedup, with gains amplifying as the number of concurrent requests increases.

- 🎯 Method: Block-wise dynamic sparsification at token level, compatible with full-attention, linear/full-attention hybrid, and sliding-window/full-attention hybrid architectures
- 📊 Result: Up to 2.1× TTFT speedup, with acceleration becoming more pronounced as concurrent request count grows
- 🔧 Engineering: Implemented as a continuous batching operator with vLLM scheduler extensions for native prefill-decode co-processing and tensor parallel support
- 💡 Innovation: First prefill acceleration framework that simultaneously addresses architecture generality and continuous batching compatibility

---

### 4. MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference (8/10) 🔁
**hf_daily** · `2605.07363` · 2026-05-08
👥 Ruijie Zhou, Fanxu Meng, Yufei Xu... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.07363) · [PDF](https://arxiv.org/pdf/2605.07363.pdf)
📡 来源: hf_daily (👍 12, 💬 1)

#### 中文摘要
*(摘要生成失败)*

#### English Summary
*(Summary generation failed)*

---

### 5. Fast Byte Latent Transformer (7/10) 🔁
**hf_daily** · `2605.08044` · 2026-05-08
👥 Julie Kallini, Artidoro Pagnoni, Tomasz Limisiewicz... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.08044) · [PDF](https://arxiv.org/pdf/2605.08044.pdf)
📡 来源: hf_daily (👍 5)

#### 中文摘要
字节级语言模型（BLT）在不依赖子词词表的情况下可匹配token级模型性能，但逐字节自回归生成速度极慢。本文提出三种加速方案：BLT-D引入块级扩散目标（block-wise diffusion），实现每步并行生成多字节；BLT-S借鉴投机解码思想，让本地解码器越过patch边界起草字节后用完整模型验证；BLT-DV在扩散生成后加入自回归验证步骤以提升质量。三种方法均可将内存带宽开销相比原始BLT降低50%以上，系统性解决了字节级LM的实用化瓶颈。

- 🎯 BLT-D以块级diffusion辅助目标训练，每解码步并行生成多字节，大幅减少前向传播次数
- 📊 所有方法estimated内存带宽开销均比原始BLT低50%以上
- 💡 BLT-S将speculative decoding迁移至字节级模型：本地解码器越界起草，完整模型单次前向验证
- 🔧 BLT-DV在diffusion生成后附加自回归验证，平衡速度与生成质量

#### English Summary
Byte-level LMs like BLT match subword-tokenizer-based models but suffer from slow byte-by-byte autoregressive generation. This work introduces three acceleration techniques for BLT: BLT-D trains with an auxiliary block-wise diffusion objective to generate multiple bytes in parallel per decoding step; BLT-S adapts speculative decoding so that BLT's local decoder drafts beyond normal patch boundaries and a single full-model forward pass verifies them; and BLT-DV combines diffusion-based parallel generation with an autoregressive verification step to recover quality. All three methods achieve an estimated memory-bandwidth cost more than 50% lower than standard BLT, substantially lowering barriers to practical deployment of byte-level LMs.

- 🎯 BLT-D trains with a block-wise diffusion auxiliary objective, enabling parallel multi-byte generation per decoding step and reducing the total number of forward passes
- 📊 All proposed methods achieve estimated memory-bandwidth cost >50% lower than the original BLT on generation tasks
- 💡 BLT-S transfers speculative decoding to the byte-level setting: local decoder drafts past patch boundaries, verified by a single full-model forward pass
- 🔧 BLT-DV augments diffusion-based generation with an autoregressive verification step to trade some speed for higher generation quality

---

## 📚 完整列表 (按分数降序)

| # | Title | Score | Sources | Code | Date |
|---|-------|-------|---------|------|------|
| 1 | [SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting](https://arxiv.org/abs/2605.07243) | 9 | hf_daily | — | 05-08 |
| 2 | [Shallow Prefill, Deep Decoding: Efficient Long-Context Inference via Layer-Asymmetric KV Visibility](https://arxiv.org/abs/2605.06105) | 9 | hf_daily | — | 05-07 |
| 3 | [UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification](https://arxiv.org/abs/2605.06221) | 8 | hf_daily | — | 05-07 |
| 4 | [MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference](https://arxiv.org/abs/2605.07363) | 8 | hf_daily | — | 05-08 |
| 5 | [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044) | 7 | hf_daily | — | 05-08 |


## 🔁 Revisited

- [SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting](https://arxiv.org/abs/2605.07243) — score 9
- [Shallow Prefill, Deep Decoding: Efficient Long-Context Inference via Layer-Asymmetric KV Visibility](https://arxiv.org/abs/2605.06105) — score 9
- [UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification](https://arxiv.org/abs/2605.06221) — score 8
- [MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference](https://arxiv.org/abs/2605.07363) — score 8
- [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044) — score 7
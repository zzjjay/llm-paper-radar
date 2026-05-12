# LLM Inference Optimization Daily · 2026-05-11

> 📅 Window: 2026-05-11 (UTC daily)
> 📊 Scanned 93 papers → passed filter 5 (threshold ≥7)

> Auto-generated daily digest from [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar).
> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6

## 🔥 Top 5 (Full Detail)

### 1. SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting (9/10) 🔁
**hf_daily** · `2605.07243` · 2026-05-08
👥 Weijie Shi, Qiang Xu, Fan Deng... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.07243) · [PDF](https://arxiv.org/pdf/2605.07243.pdf)
📡 Sources: hf_daily (👍 2, 💬 2)

#### Summary
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
📡 Sources: hf_daily (👍 1, 💬 1)

#### Summary
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
📡 Sources: hf_daily (👍 18, 💬 1)

#### Summary
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
📡 Sources: hf_daily (👍 12, 💬 1)

#### Summary
*(Summary generation failed)*

---

### 5. Fast Byte Latent Transformer (7/10) 🔁
**hf_daily** · `2605.08044` · 2026-05-08
👥 Julie Kallini, Artidoro Pagnoni, Tomasz Limisiewicz... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.08044) · [PDF](https://arxiv.org/pdf/2605.08044.pdf)
📡 Sources: hf_daily (👍 5)

#### Summary
Byte-level LMs like BLT match subword-tokenizer-based models but suffer from slow byte-by-byte autoregressive generation. This work introduces three acceleration techniques for BLT: BLT-D trains with an auxiliary block-wise diffusion objective to generate multiple bytes in parallel per decoding step; BLT-S adapts speculative decoding so that BLT's local decoder drafts beyond normal patch boundaries and a single full-model forward pass verifies them; and BLT-DV combines diffusion-based parallel generation with an autoregressive verification step to recover quality. All three methods achieve an estimated memory-bandwidth cost more than 50% lower than standard BLT, substantially lowering barriers to practical deployment of byte-level LMs.

- 🎯 BLT-D trains with a block-wise diffusion auxiliary objective, enabling parallel multi-byte generation per decoding step and reducing the total number of forward passes
- 📊 All proposed methods achieve estimated memory-bandwidth cost >50% lower than the original BLT on generation tasks
- 💡 BLT-S transfers speculative decoding to the byte-level setting: local decoder drafts past patch boundaries, verified by a single full-model forward pass
- 🔧 BLT-DV augments diffusion-based generation with an autoregressive verification step to trade some speed for higher generation quality

---

## 📚 Full List (by score, descending)

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
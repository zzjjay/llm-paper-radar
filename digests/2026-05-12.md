# LLM Inference Optimization Daily · 2026-05-12

> 📅 Window: 2026-05-12 (UTC daily)
> 📊 Scanned 97 papers → passed filter 3 → highlighted 3 (threshold ≥7)

> Auto-generated daily digest from [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar).
> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6

## 🔥 Highlights by topic

### Pruning / sparsity (top 1 of cap 3)

### 1. SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training (9/10) 🔁
**hf_daily** · `2605.08738` · 2026-05-09
👥 Shengkun Tang, Zekun Wang, Bo Zheng... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.08738) · [PDF](https://arxiv.org/pdf/2605.08738.pdf)
📡 Sources: hf_daily (👍 8, 💬 1)
🧪 pruning+distillation · Structured expert pruning + knowledge distillation + multi-token prediction · Qwen3-Next-80A3B → 23A2B · cal: Large-scale continual pretraining required (progressive pruning schedules over training run) · perf: Unknown

#### Summary
Systematic study of structured pruning and knowledge distillation (KD) for compressing MoE LLMs during large-scale continued pretraining, addressing initialization, expert compression, and training strategy choices. Key findings: pruned initialization consistently beats training from scratch; one-shot expert compression methods converge similarly after sufficient training, motivating a partial-preservation expert merging strategy; combining LM loss with KD and multi-token prediction (MTP) distillation outperforms KD alone; progressive pruning schedules outperform one-shot compression. Applied to Qwen3-Next-80A3B, producing a 23A2B model with competitive performance.

- 🎯 Method: Compresses Qwen3-Next-80A3B → 23A2B (~3.5× active-parameter reduction) via pruning + KD + MTP distillation during continued pretraining.
- 📊 Result: Pruned initialization consistently outperforms training target architecture from scratch under equal training budget across depth, width, and expert compression.
- 💡 Innovation: Partial-preservation expert merging + MTP distillation yields consistent downstream gains; progressive pruning schedules outperform one-shot compression.
- 📊 Result: Different one-shot expert compression methods converge to similar final performance after large-scale continual pretraining, simplifying compression design choices.

---

### KV cache compression (top 1 of cap 3)

### 2. Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction (9/10) 🔁
**hf_daily** · `2605.09649` · 2026-05-10
👥 Ngoc Bui, Hieu Trung Nguyen, Arman Cohan... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.09649) · [PDF](https://arxiv.org/pdf/2605.09649.pdf)
📡 Sources: hf_daily (👍 10, 💬 1)
🧪 kv_cache · learnable retention gates with global eviction policy · cal: Unknown; appears to require lightweight gate training · perf: Reduces KV memory footprint; end-to-end speedup not explicitly stated

#### Summary
KV cache eviction in long-context inference typically degrades vs. full-cache attention, but this work argues selective eviction can improve generation by reducing attention dilution from irrelevant tokens. The method introduces learnable retention gates that assign utility scores to KV entries, with a shared final scoring projection to calibrate scores across all layers and heads, enabling a single global eviction policy where tokens from different layers, heads, and modalities compete for cache capacity. Theoretical analysis justifies geometric retention as a query-agnostic proxy for future utility, and empirical results show the method matches or surpasses full-cache inference across long-context language, vision-language, and multi-turn dialogue benchmarks while substantially reducing KV memory.

- 🎯 Method: Lightweight learnable retention gates + shared cross-layer/head scoring projection for global KV cache eviction under unified memory budget.
- 💡 Innovation: Reframes KV eviction not as compression approximation but as attention dilution reduction — selective eviction can outperform full-cache inference.
- 📊 Result: Substantially reduced KV memory while matching or surpassing full-cache inference on long-context LM, vision-language, and multi-turn dialogue benchmarks.
- 💡 Innovation: Single global eviction policy allows tokens across different layers, heads, and modalities to compete directly for cache capacity.

---

### Speculative decoding (top 1 of cap 3)

### 3. SlimSpec: Low-Rank Draft LM-Head for Accelerated Speculative Decoding (7/10) 🔁
**hf_daily** · `2605.10453` · 2026-05-11
👥 Anton Plaksin, Sergei Krutikov, Sergei Skvortsov... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.10453) · [PDF](https://arxiv.org/pdf/2605.10453.pdf)
📡 Sources: hf_daily (👍 7, 💬 1)
🧪 speculative_decoding · low-rank LM-head parameterization · cal: minimal training pipeline adjustments required · perf: 4-5x speedup over standard LM-head; 8-9% end-to-end gain vs prior vocab-truncation methods

#### Summary
Speculative decoding bottlenecks in the drafter's LM-head projection to large vocabularies are typically addressed via vocabulary truncation, which adds complexity. SlimSpec applies low-rank parameterization to the EAGLE-3 drafter's LM-head, compressing the inner representation dimension while preserving full vocabulary coverage. Evaluated across three target models and multiple benchmarks in both latency- and throughput-bound regimes, SlimSpec achieves 4-5× acceleration over the standard LM-head and up to 8-9% end-to-end speedup improvement over existing methods with minimal training and inference pipeline changes.

- 🎯 Method: Low-rank factorization of the drafter LM-head in EAGLE-3 to compress inner representation dimension while retaining full vocabulary support.
- 📊 Result: 4-5× acceleration over standard LM-head architecture with competitive acceptance length across three target models.
- 📊 Result: Up to 8-9% end-to-end speedup improvement over existing vocabulary truncation methods.
- 💡 Innovation: Avoids vocabulary curation, complex inference-time logic, or training setup modifications required by prior truncation approaches.

---

## 📚 Full List (by score, descending)

| # | Title | Score | Topic | Pract | Bucket | Sources | Code | Date |
|---|-------|-------|-------|-------|--------|---------|------|------|
| 1 | [Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction](https://arxiv.org/abs/2605.09649) | 9 | 5 | 4 | KV cache compression | hf_daily | — | 05-10 |
| 2 | [SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training](https://arxiv.org/abs/2605.08738) | 9 | 5 | 4 | Pruning / sparsity | hf_daily | — | 05-09 |
| 3 | [SlimSpec: Low-Rank Draft LM-Head for Accelerated Speculative Decoding](https://arxiv.org/abs/2605.10453) | 7 | 3 | 4 | Speculative decoding | hf_daily | — | 05-11 |


## 🔁 Revisited

- [SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training](https://arxiv.org/abs/2605.08738) — score 9
- [Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction](https://arxiv.org/abs/2605.09649) — score 9
- [SlimSpec: Low-Rank Draft LM-Head for Accelerated Speculative Decoding](https://arxiv.org/abs/2605.10453) — score 7
# LLM Inference Optimization Daily · 2026-05-11

> 📅 Window: 2026-05-11 (UTC daily)
> 📊 Scanned 93 papers → passed filter 4 → highlighted 4 (threshold ≥7)

> Auto-generated daily digest from [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar).
> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6

## 🔥 Highlights by topic

### Knowledge distillation (top 2 of cap 3)

### 1. UniSD: Towards a Unified Self-Distillation Framework for Large Language Models (7/10) 🔁
**hf_daily** · `2605.06597` · 2026-05-07
👥 Yiqiao Jin, Yiyang Wang, Lucheng Fu... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.06597) · [PDF](https://arxiv.org/pdf/2605.06597.pdf)
📡 Sources: hf_daily (👍 10, 💬 1)
🧪 distillation · Self-distillation with multi-teacher agreement, EMA stabilization, token-level contrastive learning, feature matching, divergence clipping

#### Summary
UniSD addresses the instability and unreliability of self-distillation (SD) for LLMs, where self-generated trajectories produce noisy supervision signals. The framework integrates five complementary mechanisms—multi-teacher agreement, EMA teacher stabilization, token-level contrastive learning, feature matching, and divergence clipping—into a unified pipeline to systematically study how these components interact and when SD outperforms static imitation. Evaluated across six benchmarks and six models from three families, the full pipeline (UniSDfull) achieves +5.4 points over the base model and +2.8 points over the strongest baseline.

- 🎯 Method: Unifies multi-teacher agreement, EMA stabilization, token-level contrastive learning, feature matching, and divergence clipping into one SD framework.
- 📊 Result: UniSDfull improves +5.4 points over base model and +2.8 points over strongest baseline across six benchmarks.
- 💡 Innovation: Systematic ablation reveals which SD components drive gains and how they interact across tasks and model families.
- ⚠️ Limitation: Requires no stronger external teacher, but relies on self-generated trajectories whose correctness remains task-dependent.

---

### 2. Trajectory as the Teacher: Few-Step Discrete Flow Matching via Energy-Navigated Distillation (7/10) 🔁
**hf_daily** · `2605.07924` · 2026-05-08
👥 Amin Karimi Monsefi, Dominic Culver, Nikhil Bhendawade... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.07924) · [PDF](https://arxiv.org/pdf/2605.07924.pdf)
📡 Sources: hf_daily (💬 1)
🧪 distillation · trajectory-shaped discrete flow matching distillation with energy guidance · 170M · cal: training-only shaping; inference cost unchanged · perf: 128x faster inference (8 vs 1024 steps)

#### Summary
Discrete flow matching for text generation requires many forward passes, and trajectory distillation is limited not by student capacity but by noisy teacher trajectories built from blind stochastic jumps. TS-DFM introduces an energy-based compass that evaluates candidate continuations at each midpoint during training to select coherent trajectories, while leaving inference cost unchanged. On 170M-parameter language modeling, the distilled 8-step student achieves 32% lower perplexity than the 1,024-step teacher at 128× speedup, outperforming discrete-generation baselines trained on 6× more data or using 5× larger models.

- 🎯 Method: Energy-navigated trajectory shaping at training midpoints guides discrete flow matching distillation without adding inference cost.
- 📊 Result: 8-step student achieves 32% lower perplexity than the 1,024-step teacher at 128× inference speedup on 170M-parameter LM.
- 📊 Result: Best perplexity among discrete-generation baselines, outperforming methods with 6× more data or 5× larger models.
- 💡 Innovation: Reframes distillation bottleneck as trajectory quality rather than student capacity, using lightweight energy evaluator only during training.

---

### KV cache compression (top 1 of cap 3)

### 3. Shallow Prefill, Deep Decoding: Efficient Long-Context Inference via Layer-Asymmetric KV Visibility (8/10) 🔁
**hf_daily** · `2605.06105` · 2026-05-07
👥 Jungsuk Oh, Hyeseo Jeon, Hyunjune Ji... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.06105) · [PDF](https://arxiv.org/pdf/2605.06105.pdf)
📡 Sources: hf_daily (👍 1, 💬 1)
🧪 kv_cache · Layer-asymmetric KV visibility, shallow prefill deep decode · Llama-3.1-8B · cal: No calibration; layer-wise diagnostic study provided · perf: 33% TTFT improvement, 22% TPOT improvement, 25% active KV memory reduction

#### Summary
Long-context LLM inference is bottlenecked by prefill KV cache creation and repeated decode-phase attention over all cached prompt tokens. SPEED introduces a phase-asymmetric KV-visibility policy where non-anchor prompt tokens only materialize KV states in lower layers (shallow prefill), while decode-phase tokens remain full-depth. A minimal BoS anchor token preserves full-depth visibility. On Llama-3.1-8B at 128K context with 75% layer depth for prefill tokens, SPEED achieves 51.2 vs 51.4 OLMES average score baseline while delivering 33% TTFT improvement, 22% TPOT reduction, and 25% active KV memory reduction.

- 🎯 Method: Prefill tokens restricted to lower 75% of layers for KV cache; decode tokens remain full-depth; BoS token serves as full-depth anchor.
- 📊 Result: 33% TTFT speedup, 22% TPOT reduction, 25% KV memory savings at 128K context with <0.2 point OLMES accuracy drop.
- 💡 Innovation: Phase-asymmetric KV visibility—removes prefill tokens from upper-layer decode attention entirely, rather than compressing or approximating their KV states.
- ⚠️ Limitation: Evaluated only on Llama-3.1-8B instruction-tuning; upper-layer prompt KV removal may impact tasks requiring deep cross-token reasoning.

---

### Diffusion compression (top 1 of cap 3)

### 4. Normalizing Trajectory Models (7/10) 🔁
**hf_daily** · `2605.08078` · 2026-05-08
👥 Jiatao Gu, Tianrong Chen, Ying Shen... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.08078) · [PDF](https://arxiv.org/pdf/2605.08078.pdf)
📡 Sources: hf_daily (👍 8, 💬 1)
🧪 diffusion_compression · normalizing flow trajectory model, step distillation · cal: trainable from scratch or from pretrained flow-matching models · perf: 4-step generation; speedup vs standard diffusion not quantified end-to-end

#### Summary
Diffusion models' Gaussian denoising assumption breaks down in few-step generation regimes, and existing solutions (distillation, consistency training, adversarial objectives) abandon the likelihood framework. NTM replaces each reverse step with an expressive conditional normalizing flow, enabling exact likelihood training across the full trajectory. Architecturally, NTM uses shallow invertible blocks per step combined with a deep parallel predictor across the trajectory, supporting training from scratch or initialization from pretrained flow-matching models. The exact trajectory likelihood enables self-distillation via a lightweight denoiser, achieving competitive text-to-image generation in 4 sampling steps while preserving exact likelihoods.

- 🎯 Method: Each reverse diffusion step modeled as a conditional normalizing flow with exact likelihood, enabling end-to-end training across the trajectory.
- 💡 Innovation: Self-distillation from NTM's own score function trains a lightweight denoiser producing high-quality samples in 4 steps without external teacher.
- 📊 Result: Matches or outperforms strong text-to-image baselines in 4 sampling steps while uniquely retaining exact trajectory likelihood.
- 🔧 Engineering: Shallow invertible blocks per step + deep parallel trajectory predictor; initializable from pretrained flow-matching models.

---

## 📚 Full List (by score, descending)

| # | Title | Score | Topic | Pract | Bucket | Sources | Code | Date |
|---|-------|-------|-------|-------|--------|---------|------|------|
| 1 | [Shallow Prefill, Deep Decoding: Efficient Long-Context Inference via Layer-Asymmetric KV Visibility](https://arxiv.org/abs/2605.06105) | 8 | 4 | 4 | KV cache compression | hf_daily | — | 05-07 |
| 2 | [UniSD: Towards a Unified Self-Distillation Framework for Large Language Models](https://arxiv.org/abs/2605.06597) | 7 | 4 | 3 | Knowledge distillation | hf_daily | — | 05-07 |
| 3 | [Normalizing Trajectory Models](https://arxiv.org/abs/2605.08078) | 7 | 4 | 3 | Diffusion compression | hf_daily | — | 05-08 |
| 4 | [Trajectory as the Teacher: Few-Step Discrete Flow Matching via Energy-Navigated Distillation](https://arxiv.org/abs/2605.07924) | 7 | 4 | 3 | Knowledge distillation | hf_daily | — | 05-08 |


## 🔁 Revisited

- [UniSD: Towards a Unified Self-Distillation Framework for Large Language Models](https://arxiv.org/abs/2605.06597) — score 7
- [Trajectory as the Teacher: Few-Step Discrete Flow Matching via Energy-Navigated Distillation](https://arxiv.org/abs/2605.07924) — score 7
- [Shallow Prefill, Deep Decoding: Efficient Long-Context Inference via Layer-Asymmetric KV Visibility](https://arxiv.org/abs/2605.06105) — score 8
- [Normalizing Trajectory Models](https://arxiv.org/abs/2605.08078) — score 7
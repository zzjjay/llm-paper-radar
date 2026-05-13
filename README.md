# LLM Inference Optimization Daily · 2026-05-12

> 📅 Window: 2026-05-12 (UTC daily)
> 📊 Scanned 518 papers → passed filter 14 → highlighted 12 (threshold ≥7)

> Auto-generated daily digest from [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar).
> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6

## 🔥 Highlights by topic

### PTQ (post-training quantization) (top 2 of cap 3)

### 1. ConQuR: Corner Aligned Activation Quantization via Optimized Rotations for LLMs (10/10)
**arxiv** · `2605.10793` · 2026-05-11
👥 Chayne Thrash, Ali Abbasi, Soheil Kolouri · 🏷 cs.LG
🔗 [arXiv](https://arxiv.org/abs/2605.10793) · [PDF](https://arxiv.org/pdf/2605.10793v1)
📡 Sources: arxiv
🧪 ptq · W4A4 rotation-based quantization via Procrustes · Llama-3-70B · cal: Lightweight online calibration; closed-form Procrustes updates, no gradient optimization. · perf: Unknown; weight-activation quantization should enable 4x+ memory reduction.

#### Summary
Activation outliers in LLMs cause large quantization errors, and existing rotation-based methods (e.g., QuaRot, SpinQuant) either require end-to-end rotation training or offline activation storage. ConQuR learns orthogonal rotations post-training by solving the orthogonal Procrustes problem to align normalized activations with hypercube corners, distributing activation energy evenly across dimensions. An online calibration procedure updates rotations incrementally as samples are processed, eliminating disk storage of activations and allowing adaptation to quantized distributions. Evaluated on Llama-2 and Llama-3 (3B–70B), ConQuR achieves competitive or better perplexity and commonsense reasoning accuracy versus prior rotation-based methods without costly training or large activation corpora.

- 🎯 Method: Closed-form orthogonal Procrustes updates align activations to hypercube corners, avoiding gradient-based optimization over the orthogonal group.
- 💡 Innovation: Online calibration updates rotations sample-by-sample, eliminating need for stored activation corpora and adapting to quantized distributions.
- 📊 Result: Competitive or improved perplexity and commonsense reasoning on Llama-2 and Llama-3 models ranging 3B–70B parameters.
- ⚠️ Limitation: Performance claims are competitive rather than strictly superior to end-to-end trained rotation methods like SpinQuant.

---

### 2. ADMM-Q: An Improved Hessian-based Weight Quantizer for Post-Training Quantization of Large Language Models (9/10)
**arxiv** · `2605.11222` · 2026-05-11
👥 Ryan Lucas, Mehdi Makni, Xiang Meng... · 🏷 cs.LG
🔗 [arXiv](https://arxiv.org/abs/2605.11222) · [PDF](https://arxiv.org/pdf/2605.11222v1)
📡 Sources: arxiv
🧪 ptq · ADMM-based weight quantization, W2/W3/W4, composable with SmoothQuant/SpinQuant · Qwen3-8B

#### Summary
ADMM-Q addresses degraded model utility in PTQ weight quantization at sub-4-bit regimes by replacing GPTQ-style solvers with a combinatorial ADMM-based layer-wise reconstruction optimizer. The algorithm uses operator-splitting to continuously update weights while gradually enforcing quantization constraints, augmented with penalty scheduling, preconditioning, and local search post-processing. It is modular and composable with existing pipelines (SmoothQuant, SpinQuant, range clipping, rotations), achieving large perplexity reductions on Qwen3-8B: W3A16 (12.85→10.06), W4A8 SmoothQuant (9.29→8.68), and W2A4KV4 SpinQuant (66.11→19.42).

- 🎯 Method: ADMM-based combinatorial optimizer for layer-wise PTQ weight quantization with convergence guarantees, drop-in replacement for GPTQ.
- 📊 W2A4KV4 SpinQuant on Qwen3-8B: WikiText-2 perplexity drops from 66.11 → 19.42 vs. GPTQ baseline.
- 📊 W3A16 weight-only quantization: perplexity 12.85 → 10.06; W4A8 SmoothQuant: 9.29 → 8.68 on Qwen3-8B.
- 💡 Fully composable with existing techniques: range clipping, learned/random rotations, activation scaling, SmoothQuant, SpinQuant.

---

### QAT / low-bit pretraining (top 2 of cap 2)

### 3. Pretraining large language models with MXFP4 (8/10)
**arxiv** · `2605.09825` · 2026-05-11
👥 Musa Cim, Poovaiah Palangappa, Miro Hodak... · 🏷 cs.LG, cs.AI
🔗 [arXiv](https://arxiv.org/abs/2605.09825) · [PDF](https://arxiv.org/pdf/2605.09825v1)
📡 Sources: arxiv
🧪 qat · MXFP4 full-pipeline quantization · Llama-3.1-8B · cal: full pretraining on C4 dataset

#### Summary
Full-pipeline FP4 pretraining of LLMs frequently diverges, and this work systematically isolates the root cause by progressively enabling MXFP4 quantization across Fprop, Dgrad, and Wgrad stages during Llama 3.1-8B pretraining on C4. The key finding is that Wgrad quantization—not Fprop or Dgrad—is the primary driver of convergence degradation, caused by structured micro-scaling errors along sensitive gradient paths rather than insufficient stochasticity. Deterministic Hadamard rotations (not stochastic rounding or randomized Hadamard rotations) consistently stabilize training, and experiments run on hardware with native MXFP4 support (AMD Instinct MI355X GPUs) avoid software emulation artifacts.

- 🎯 Method: Deterministic Hadamard rotations applied to Wgrad restore stable MXFP4 training; stochastic rounding and randomized Hadamard rotations both fail.
- 📊 Result: FP4 in Fprop+Dgrad alone causes only modest additional token requirements; Wgrad quantization is the dominant source of divergence.
- 💡 Innovation: Instability traced to structured micro-scaling errors in gradient paths, not stochasticity—reframing the design space for FP4 training stabilization.
- 🔧 Engineering: Experiments use native MXFP4 hardware support on AMD Instinct MI355X GPUs, eliminating software emulation bias.

---

### 4. BCJR-QAT: A Differentiable Relaxation of Trellis-Coded Weight Quantization (7/10)
**arxiv** · `2605.10655` · 2026-05-11
👥 Venugopalan Iyengar · 🏷 cs.LG
🔗 [arXiv](https://arxiv.org/abs/2605.10655) · [PDF](https://arxiv.org/pdf/2605.10655v1)
📡 Sources: arxiv
🧪 qat · 2-bit trellis-coded QAT via BCJR relaxation · Llama-3.2-1B · cal: Unknown; full QAT with forward-KL distillation likely multi-hour+. · perf: Not reported; BCJR kernel 6.57× speedup but calibration/training speedup unclear.

#### Summary
Trellis-coded quantization (QTIP) achieves state-of-the-art 2-bit PTQ for LLMs, but QAT on a trellis is blocked by the non-differentiable Viterbi argmax. BCJR-QAT replaces the Viterbi argmax with the BCJR forward-backward sum-product algorithm at temperature T, yielding a differentiable soft codeword as a Boltzmann expectation over trellis paths that recovers the hard QTIP code as T→0. A fused Triton kernel makes BCJR tractable on consumer GPUs (6.57× speedup, fp32 parity), and end-to-end forward-KL distillation on Llama-3.2-1B at 2 bpw beats QTIP-PTQ by −0.084 PPL on WikiText-2, with super-additive gains when applied across multiple layers.

- 🎯 Method: Replaces non-differentiable Viterbi argmax in trellis-coded QAT with differentiable BCJR sum-product at temperature T, recovering QTIP hard code as T→0
- 🔧 Engineering: Fused Triton kernel achieves 6.57× speedup over naive BCJR with fp32 numerical parity on a single consumer GPU
- 📊 Result: Single-layer BCJR-QAT beats QTIP-PTQ by −0.084 PPL on WikiText-2 at 2 bpw for Llama-3.2-1B; multi-layer gains are super-additive
- 💡 Innovation: Drift-budget theory quantifies when BCJR-QAT can escape the QTIP-PTQ Voronoi basin; high-T phase must be skipped to avoid overshoot

---

### KV cache compression (top 1 of cap 2)

### 5. Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction (9/10) 🔁
**hf_daily** · `2605.09649` · 2026-05-10
👥 Ngoc Bui, Hieu Trung Nguyen, Arman Cohan... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.09649) · [PDF](https://arxiv.org/pdf/2605.09649.pdf)
📡 Sources: hf_daily (👍 10, 💬 1)
🧪 kv_cache · learnable retention gates with global scoring projection · cal: Lightweight retention gates; calibration cost unknown · perf: Reduced KV memory footprint; end-to-end speedup not explicitly reported

#### Summary
KV cache eviction in long-context inference typically degrades performance versus full-cache attention, but this work argues eviction can actually improve generation by removing irrelevant tokens that dilute attention from useful evidence. The method introduces learnable retention gates that assign utility scores to KV entries, with a shared final scoring projection to calibrate scores globally across all layers, heads, and modalities under a unified memory budget. A single global eviction policy lets tokens compete directly for cache capacity, supported by theoretical analysis linking attention dilution to retention of irrelevant tokens and geometric retention as a query-agnostic utility proxy. Across long-context language, vision-language reasoning, and multi-turn dialogue benchmarks, the method matches or surpasses full-cache inference while substantially reducing KV memory.

- 🎯 Method: Learnable retention gates + shared cross-layer/head scoring projection enable a single global KV eviction policy under a unified memory budget.
- 💡 Innovation: Theoretically shows selective eviction reduces attention dilution, reframing KV compression as a reasoning improvement mechanism, not just approximation.
- 📊 Result: Matches or surpasses full-cache inference on long-context language, vision-language, and multi-turn dialogue benchmarks with substantially reduced KV memory.
- 🔧 Engineering: Global eviction allows tokens from different layers, heads, and modalities to compete directly for cache capacity, avoiding per-layer budget fragmentation.

---

### Speculative decoding (top 2 of cap 2)

### 6. SlimSpec: Low-Rank Draft LM-Head for Accelerated Speculative Decoding (7/10) 🔁
**arxiv** · `2605.10453` · 2026-05-11
👥 Anton Plaksin, Sergei Krutikov, Sergei Skvortsov... · 🏷 cs.LG, cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.10453) · [PDF](https://arxiv.org/pdf/2605.10453.pdf)
📡 Sources: hf_daily (👍 7, 💬 1), arxiv
🧪 speculative_decoding · low-rank LM-head decomposition · perf: 4-5x speedup over standard LM-head; 8-9% better end-to-end vs vocabulary truncation.

#### Summary
Speculative decoding drafters face a computational bottleneck at the LM-head due to projection onto large vocabularies. SlimSpec addresses this by applying low-rank parameterization to the drafter's LM-head, compressing the inner representation rather than truncating the output vocabulary, thereby preserving full vocabulary support without requiring special curation or complex inference logic. Evaluated with EAGLE-3 across three target models in both latency- and throughput-bound regimes, SlimSpec achieves 4-5× acceleration over the standard LM-head while surpassing vocabulary truncation baselines by up to 8-9% end-to-end speedup.

- 🎯 Method: Low-rank factorization of the EAGLE-3 drafter's LM-head compresses inner representation while preserving full vocabulary support.
- 📊 Result: 4-5× LM-head acceleration over standard architecture; up to 8-9% end-to-end speedup improvement over existing vocabulary truncation methods.
- 💡 Innovation: Avoids vocabulary truncation complexity (curation, dynamic inference logic, training modifications) with minimal pipeline changes.
- ⚠️ Limitation: Evaluation scoped to EAGLE-3 drafter; generalization to other speculative decoding frameworks not demonstrated.

---

### 7. CATS: Cascaded Adaptive Tree Speculation for Memory-Limited LLM Inference Acceleration (7/10)
**arxiv** · `2605.11186` · 2026-05-11
👥 Yuning Han, Yangchenchen Jin, Dylan Zhao... · 🏷 cs.LG, cs.AI
🔗 [arXiv](https://arxiv.org/abs/2605.11186) · [PDF](https://arxiv.org/pdf/2605.11186v1)
📡 Sources: arxiv
🧪 speculative_decoding · self-speculative decoding with cascaded verification and parameter offloading · cal: none reported (self-speculative, no separate draft model) · perf: up to 5.08x wall-clock speedup on edge devices; 1.45x vs SOTA

#### Summary
Auto-regressive LLM decoding is memory-bandwidth-bound, and existing speculative decoding methods assume sufficient HBM to co-reside target and draft models—an assumption violated on edge devices with limited DRAM. CATS is a self-speculative decoding framework that performs cascaded verification and correction using intermediate layers of the target model itself, structured around the device's memory budget and parameter offloading patterns, requiring no additional model memory beyond the target model. On real edge devices across five benchmarks, CATS achieves up to 5.08x wall-clock speedup with no quality degradation, outperforming the current SOTA by up to 1.45x.

- 🎯 Method: Self-speculative cascaded tree decoding that uses the target model's own intermediate layers as draft, with no extra memory overhead.
- 📊 Result: Up to 5.08x wall-clock speedup on real edge devices with zero generation quality degradation.
- 📊 Result: Outperforms SOTA speculative decoding methods by up to 1.45x under edge memory constraints.
- 💡 Innovation: Cascaded verification adapts to device memory budget and parameter offloading patterns, enabling speculative decoding on DRAM-limited edge platforms.

---

### Knowledge distillation (top 2 of cap 2)

### 8. SOMA: Efficient Multi-turn LLM Serving via Small Language Model (8/10)
**arxiv** · `2605.11317` · 2026-05-11
👥 Xueqi Cheng, Qiong Wu, Zhengyi Zhou... · 🏷 cs.CL, cs.AI
🔗 [arXiv](https://arxiv.org/abs/2605.11317) · [PDF](https://arxiv.org/pdf/2605.11317v1)
📡 Sources: arxiv
🧪 distillation · localized knowledge distillation + LoRA fine-tuning · cal: soft-prompt mining + LoRA fine-tuning on early turns; cost unknown · perf: reduced latency/memory in multi-turn dialogue; gate enables rollback

#### Summary
Multi-turn LLM serving incurs high cost because full dialogue history is concatenated each turn and routed to large proprietary models. SOMA exploits early conversation turns to estimate a local response manifold, then adapts a small surrogate model via localized LoRA fine-tuning to handle subsequent turns. Soft prompts are learned to maximize semantic divergence between large and small model responses (surfacing misaligned directions), with anti-degeneration control and distillation into prompt-free LoRA inference; a gating mechanism enables one-time switching with rollback on quality drift.

- 🎯 Method: Early turns used to mine hard cases via soft prompt optimization maximizing semantic divergence, then distilled into localized LoRA adapters for the surrogate small model.
- 💡 Innovation: One-time switch gate from large to small model with drift-triggered rollback, eliminating per-turn large model inference after warm-up phase.
- 🔧 Engineering: Surrogate runs without prompts at inference (prompts only used during local fine-tuning), reducing latency, memory, and API costs for multi-turn serving.
- ⚠️ Limitation: Abstract lacks concrete speedup/compression ratios or accuracy delta numbers; quantitative results are only described as 'extensive experiments show effectiveness'.

---

### 9. ReAD: Reinforcement-Guided Capability Distillation for Large Language Models (7/10)
**arxiv** · `2605.11290` · 2026-05-11
👥 Xueqi Cheng, Xugui Zhou, Tyler Derr... · 🏷 cs.CL, cs.AI
🔗 [arXiv](https://arxiv.org/abs/2605.11290) · [PDF](https://arxiv.org/pdf/2605.11290v1)
📡 Sources: arxiv
🧪 distillation · reinforcement-guided capability distillation with contextual bandit budget allocation

#### Summary
Capability distillation compresses LLMs while preserving task-relevant abilities, but existing methods treat capabilities as independent targets, ignoring cross-capability interference. ReAD addresses this by modeling capability interdependence: it infers task-essential capabilities, generates capability-targeted supervision dynamically, and uses an uncertainty-aware contextual bandit to adaptively allocate a fixed token distillation budget based on expected utility gains. Experiments show ReAD improves downstream task utility under the same token budget while reducing harmful cross-capability spillover compared to strong baselines.

- 🎯 Method: Uncertainty-aware contextual bandit adaptively allocates fixed distillation token budget across capabilities based on expected utility gains
- 💡 Innovation: Explicitly models cross-capability transfer during distillation, identifying that budget allocation induces systematic, budget-dependent capability interference
- 📊 Result: Outperforms strong baselines in downstream utility under equal token budget while reducing harmful spillover and wasted distillation effort
- ⚠️ Limitation: Evaluation scope limited to fixed token budget regime; absolute compression ratios and speedup numbers not reported in abstract

---

### Pruning / sparsity (top 1 of cap 2)

### 10. SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training (9/10) 🔁
**hf_daily** · `2605.08738` · 2026-05-09
👥 Shengkun Tang, Zekun Wang, Bo Zheng... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.08738) · [PDF](https://arxiv.org/pdf/2605.08738.pdf)
📡 Sources: hf_daily (👍 8, 💬 1)
🧪 pruning · MoE expert pruning + knowledge distillation + multi-token prediction distillation · Qwen3-Next-80A3B (compressed to 23A2B) · cal: large-scale continual pretraining at compression scale

#### Summary
Addresses the question of how structured pruning and knowledge distillation should be applied during large-scale MoE model pretraining. Systematically studies depth/width/expert compression on MoE models, finding that pruning a pretrained MoE consistently outperforms training the target architecture from scratch under equal compute budgets. Introduces a partial-preservation expert merging strategy, combines KD with LM loss, proposes multi-token prediction (MTP) distillation, and uses progressive pruning schedules—compressing Qwen3-Next-80A3B to a 23A2B model with competitive performance.

- 🎯 Method: Partial-preservation expert merging + MTP distillation + progressive pruning to compress Qwen3-Next-80A3B → 23A2B (~71% active parameter reduction)
- 📊 Result: Pruned MoE initialization consistently outperforms training target architecture from scratch under same token/compute budget across all compression axes
- 💡 Innovation: MTP distillation and combining KD with LM loss yield consistent gains, especially on knowledge-intensive benchmarks
- 📊 Result: Progressive pruning schedules outperform one-shot compression given same training tokens, indicating gradual architecture transitions improve optimization

---

### Other (top 2 of cap 2)

### 11. Compute Where it Counts: Self Optimizing Language Models (7/10)
**arxiv** · `2605.10875` · 2026-05-11
👥 Yash Akhauri, Mohamed S. Abdelfattah · 🏷 cs.LG, cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.10875) · [PDF](https://arxiv.org/pdf/2605.10875v1)
📡 Sources: arxiv
🧪 other · dynamic budget allocation with per-token sparse attention, structured MLP pruning, activation quantization · cal: Policy training on teacher-forced episodes; cost not specified · perf: Improved quality-efficiency pareto-front vs static allocation; end-to-end latency unknown

#### Summary
Static compression methods apply uniform compute budgets across all tokens during LLM inference, ignoring token-difficulty heterogeneity. SOL (Self-Optimizing Language Models) addresses this with a lightweight policy network that reads LLM hidden states and dynamically selects per-token efficiency actions—jointly controlling attention sparsity, MLP activation pruning, and quantization bit-width—while keeping base model weights frozen. The policy is trained via group-relative policy optimization (GRPO) on teacher-forced episodes, sampling multiple 'counterfactual' compute schedules over the same token path and optimizing a reward balancing LM quality against a target budget constraint. SOL improves MMLU accuracy by up to 7.3% over uniform budget allocation at matched compute, dominating the quality-efficiency Pareto frontier across all tested model variants and compute regimes.

- 🎯 Method: Lightweight policy network trained with GRPO dynamically assigns per-token attention sparsity, MLP pruning, and quantization bit-width without modifying base model weights.
- 📊 Result: SOL improves MMLU accuracy by up to 7.3% over uniform budget allocation strategies at matched compute budget.
- 💡 Innovation: Counterfactual schedule sampling in teacher-forced episodes enables credit assignment across discrete efficiency actions without altering the token generation path.
- 📊 Result: SOL achieves a superior quality-efficiency Pareto front over both static compression and random schedule search baselines across all experiments.

---

### 12. Reinforce Adjoint Matching: Scaling RL Post-Training of Diffusion and Flow-Matching Models (7/10)
**arxiv** · `2605.10759` · 2026-05-11
👥 Andreas Bergmeister, Stefanie Jegelka, Nikolas Nüsken... · 🏷 cs.LG, cs.CV
🔗 [arXiv](https://arxiv.org/abs/2605.10759) · [PDF](https://arxiv.org/pdf/2605.10759v1)
📡 Sources: arxiv
🧪 other · RL post-training via adjoint-matching consistency loss · Stable Diffusion 3.5M · cal: RL post-training required; cost reduction vs baselines (50x steps) but absolute cost unclear. · perf: No end-to-end inference speedup or latency reported.

#### Summary
RL post-training of diffusion/flow-matching models typically requires expensive SDE rollouts, reward gradients, or surrogate losses that break pretraining's regression structure. Reinforce Adjoint Matching (RAM) derives a consistency loss by combining KL-regularized reward maximization with the adjoint-matching optimality condition and a REINFORCE identity: the optimal policy tilts the clean-endpoint distribution toward high-reward samples while leaving the noising law unchanged, enabling a simple regress-against-corrected-target update with no SDE rollouts or backward adjoint sweeps. On Stable Diffusion 3.5M, RAM achieves highest reward on composability, text rendering, and human preference benchmarks, reaching Flow-GRPO's peak reward in up to 50× fewer training steps.

- 🎯 Method: RAM corrects pretraining regression targets with reward via REINFORCE identity—no SDE rollouts, adjoint sweeps, or reward gradients needed.
- 📊 Matches Flow-GRPO peak reward in up to 50× fewer training steps on Stable Diffusion 3.5M.
- 📊 Achieves highest reward on composability, text rendering, and human preference among evaluated methods.
- 💡 Optimal KL-regularized policy tilts clean-endpoint distribution toward high-reward samples, preserving noising law and pretraining's regression structure.

---

## 📚 Full List (by score, descending)

| # | Title | Score | Topic | Pract | Bucket | Sources | Code | Date |
|---|-------|-------|-------|-------|--------|---------|------|------|
| 1 | [ConQuR: Corner Aligned Activation Quantization via Optimized Rotations for LLMs](https://arxiv.org/abs/2605.10793) | 10 | 5 | 5 | PTQ (post-training quantization) | arxiv | — | 05-11 |
| 2 | [Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction](https://arxiv.org/abs/2605.09649) | 9 | 5 | 4 | KV cache compression | hf_daily | — | 05-10 |
| 3 | [SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training](https://arxiv.org/abs/2605.08738) | 9 | 5 | 4 | Pruning / sparsity | hf_daily | — | 05-09 |
| 4 | [ADMM-Q: An Improved Hessian-based Weight Quantizer for Post-Training Quantization of Large Language Models](https://arxiv.org/abs/2605.11222) | 9 | 5 | 4 | PTQ (post-training quantization) | arxiv | — | 05-11 |
| 5 | [SOMA: Efficient Multi-turn LLM Serving via Small Language Model](https://arxiv.org/abs/2605.11317) | 8 | 4 | 4 | Knowledge distillation | arxiv | — | 05-11 |
| 6 | [Pretraining large language models with MXFP4](https://arxiv.org/abs/2605.09825) | 8 | 5 | 3 | QAT / low-bit pretraining | arxiv | — | 05-11 |
| 7 | [SlimSpec: Low-Rank Draft LM-Head for Accelerated Speculative Decoding](https://arxiv.org/abs/2605.10453) | 7 | 3 | 4 | Speculative decoding | arxiv+hf_daily | — | 05-11 |
| 8 | [ReAD: Reinforcement-Guided Capability Distillation for Large Language Models](https://arxiv.org/abs/2605.11290) | 7 | 4 | 3 | Knowledge distillation | arxiv | — | 05-11 |
| 9 | [Curriculum Learning-Guided Progressive Distillation in Large Language Models](https://arxiv.org/abs/2605.11260) | 7 | 4 | 3 | Knowledge distillation | arxiv | — | 05-11 |
| 10 | [CATS: Cascaded Adaptive Tree Speculation for Memory-Limited LLM Inference Acceleration](https://arxiv.org/abs/2605.11186) | 7 | 3 | 4 | Speculative decoding | arxiv | — | 05-11 |
| 11 | [Compute Where it Counts: Self Optimizing Language Models](https://arxiv.org/abs/2605.10875) | 7 | 4 | 3 | Other | arxiv | — | 05-11 |
| 12 | [Reinforce Adjoint Matching: Scaling RL Post-Training of Diffusion and Flow-Matching Models](https://arxiv.org/abs/2605.10759) | 7 | 4 | 3 | Other | arxiv | — | 05-11 |
| 13 | [BCJR-QAT: A Differentiable Relaxation of Trellis-Coded Weight Quantization](https://arxiv.org/abs/2605.10655) | 7 | 4 | 3 | QAT / low-bit pretraining | arxiv | — | 05-11 |
| 14 | [Evolving Knowledge Distillation for Lightweight Neural Machine Translation](https://arxiv.org/abs/2605.09924) | 7 | 4 | 3 | Knowledge distillation | arxiv | — | 05-11 |


## 🔁 Revisited

- [Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction](https://arxiv.org/abs/2605.09649) — score 9
- [SlimSpec: Low-Rank Draft LM-Head for Accelerated Speculative Decoding](https://arxiv.org/abs/2605.10453) — score 7
- [SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training](https://arxiv.org/abs/2605.08738) — score 9
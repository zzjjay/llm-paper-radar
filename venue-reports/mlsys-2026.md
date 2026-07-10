# MLSys 2026 — LLM Inference Deployment Optimization Trend Report

*57 of ~135 accepts whose primary contribution is LLM serving. All figures are authors' self-reported bests ("up to X", often microbenchmark) — ceilings, not expectations. One systems venue, auto-selected set — see [Paper distribution](#paper-distribution).*

## Takeaway

Two independent forces drive almost everything, and their even split is itself the finding: **no single 2026 technique dominates**, so a team optimizing LLM inference cannot buy one fix — quantization, eviction, disaggregation, and new hardware are all live simultaneously, and picking one depends on the caller's own traffic, not on a converged best practice. **Workload:** reasoning models moved the cost center from prefill to memory-bound decode, and agents/RAG/multi-turn broke the "requests are independent" assumption servers were designed around. **Hardware:** new NVIDIA generations widened the gap between compute and memory/interconnect, and heterogeneity (AMD, GH200, client GPUs, phones) turned from an edge case into a design requirement — driving the largest bucket, kernels/compilers, none of it about KV.

They collide at the **KV cache** (~15–20 of 57, the most-contested single object but not a majority) — and the telling part isn't that it's contested, it's that **four incompatible answers compete for the same object with no unification**: quantize it, evict it, avoid touching it via computation reuse, or replicate/transport it as durable state (all four detailed in §1). That is a real gap in the field's state of the art, not an artifact of this survey — a practitioner has to bet on their own workload's redundancy pattern rather than follow a settled recipe.

One reading of the constraint that shapes everything (frozen weights + an existing engine): the algorithmic low-hanging fruit is largely gone, so what's publishable now is system-level orchestration around a fixed model and a fixed engine — which is also why standalone quantization nearly vanished here (§5) despite not vanishing as a research topic elsewhere (a venue effect, not a field one). And of the two root forces, only one compounds on its own: hardware asymmetry is cyclical — vendors will eventually respond to the exact kernel-authoring pain this set documents — while the workload force doesn't self-correct: better reasoning models produce more tokens per request, and every new agent framework shares more context across requests. **[conjecture]:** three years out, the workload force is the one still setting the agenda, regardless of whose GPU wins.

## 1. Decode went memory-bound, and the KV cache is the battleground

Each token re-reads the growing KV cache, so long decode is IO-bound ([MAC-Attention](https://openreview.net/forum?id=b6HBRCejb7), [SpecGen](https://openreview.net/forum?id=yeqrwcWjPu)) — amplified, not created, by reasoning-length outputs. Four stances: **shrink** — [Kitty](https://openreview.net/forum?id=r3mQiuYKIN) 2-bit KV via sensitivity-ranked Key channels (~8× memory); **skip** — [FlexiCache](https://openreview.net/forum?id=GgX6dPJx9M) offloads temporally-stable heads' cold pages (−70% GPU mem), [OPKV](https://openreview.net/forum?id=EB5bgzv4qA) recallable sparsity over paged KV, [BLASST](https://openreview.net/forum?id=6INSBXTQ4x) thresholds out whole attention blocks; **reuse compute** — [MAC-Attention](https://openreview.net/forum?id=b6HBRCejb7) reuses attention for similar queries (constant per-token cost on a hit); **treat as durable state** (the notable shift — KV as replicated, transferable, fault-tolerant data) — [GhostServe](https://openreview.net/forum?id=xKjYiUgeOK) erasure-codes it, [RaidServe](https://openreview.net/forum?id=5pl9fdbEkq) cyclic placement + backup, [fabric-lib](https://openreview.net/forum?id=SjVa05wEiY) (Perplexity, prod) portable RDMA KV transport.

## 2. The workload changed: long reasoning outputs and shared context

**Long reasoning:** [SkipKV](https://openreview.net/forum?id=0EsV9SIm8p) evicts KV at sentence not token granularity — token-level eviction makes CoT models regenerate to recover dropped content; [Locality-Aware Beam Scheduling](https://openreview.net/forum?id=dTo8jAXm9K) targets test-time beam search where KV, not weights, bottlenecks consumer GPUs. **Shared context** (reuse over recompute): [ContextPilot](https://openreview.net/forum?id=RnKvDy1jv2) cross-user/turn context index, [FlashAgents](https://openreview.net/forum?id=m14PPUfgEc) inter-agent token streaming + radix prefix cache, [BatchLLM](https://openreview.net/forum?id=IuVHde07l6) global prefix sharing for offline batch (LRU drops reusable prefixes), [Stream2LLM](https://openreview.net/forum?id=FuRo7Ur5Ib)/[TeleRAG](https://openreview.net/forum?id=YsOyCpMUYD) hide RAG retrieval behind compute. Sharpest: [Span Queries](https://openreview.net/forum?id=qcGGSXpFcM) unifies chat/RAG/agentic as one commutativity-annotated call tree and optimizes cache locality generically (492-line vLLM change).

## 3. Prefill and decode want different machines — disaggregation matures past the hype

PD disaggregation is the assumed baseline now; this year is refinement + reality-check. [Beyond the Buzz](https://openreview.net/forum?id=NqC5tcBsa0) (100Ks of design points on Dynamo) finds it wins only for prefill-heavy traffic + large models + dynamic rate matching — not free. [PLA-Serve](https://openreview.net/forum?id=dzjCkSEDyG) disaggregates *within* prefill by prompt length; [TriInfer](https://openreview.net/forum?id=nNovi8fvGN) adds a third encode stage for multimodal (not text-LLM).

## 4. MoE broke the dense-serving playbook

Dense-model optimizations backfire on MoE. [Demystifying the MoE Serving Tax](https://openreview.net/forum?id=lELxqcgrsN): MoE serves 2–3× worse than FLOP-equivalent dense, and load imbalance hurts prefill but helps decode. Three fix layers: [Layered Prefill](https://openreview.net/forum?id=yyDbI3HXco) reschedules by layer-group (chunked prefill reloads expert weights, +39% traffic); [CRAFT](https://openreview.net/forum?id=zdRvzU9ZCe) cost-aware expert replication (current schemes over-replicate); [FarSkip-Collective](https://openreview.net/forum?id=ruOpvLzsGV) adds architecture skip-connections (self-distilled to <1% loss, ≤109B) to overlap all-to-all — the one paper that retrains, raising the adoption bar.

## 5. Speculative decoding: attack the same bottleneck from the compute side

Fill idle decode-time compute by guess-and-verify. [PRISM](https://openreview.net/forum?id=cvU2HuuxEf) decouples drafter capacity from cost across steps; [SpecGen](https://openreview.net/forum?id=yeqrwcWjPu) self-speculates via sparse attention (no separate drafter). Diffusion is entering as the non-AR drafter: [SpecDiff-2](https://openreview.net/forum?id=o42VU86ZsV) calibrates a diffusion drafter to an AR verifier, [TiDAR](https://openreview.net/forum?id=onfxEjoE4L) fuses both in one pass, [CDLM](https://openreview.net/forum?id=eB8yjR6alL) makes diffusion LMs cache-compatible. Essential caveat: [Performance or Illusion?](https://openreview.net/forum?id=fzkqtezFEi) shows measured ≪ theoretical SD speedup at production batch sizes.

## 6. Hardware moved; the software layer scrambles to keep up

Largest cluster (16/57), second story as big as KV. **Rebalanced on-chip bottleneck:** [FlashAttention-4](https://openreview.net/forum?id=mN5RtvuYl3) rebuilds the pipeline for Blackwell's asymmetric scaling (async MMA, SW-emulated exp); [TokenWeave](https://openreview.net/forum?id=rh2Ylffkq6) fuses AllReduce–RMSNorm in 2–8 SMs via NVSHARP; [SuperInfer](https://openreview.net/forum?id=RuslSHdIHa) schedules for GH200 NVLink-C2C; [ScaleSearch](https://openreview.net/forum?id=innqECyZPK) fixes NVFP4's default scale. **Heterogeneity:** [HipKittens](https://openreview.net/forum?id=xxSSrndQrI) makes AMD a real target (shipped in AITER); [ParallelKittens](https://openreview.net/forum?id=Cv5e5uRXFb) packages multi-GPU overlap into 8 primitives (interconnect < compute); [db-SP](https://openreview.net/forum?id=XgKteNxNe0) load-balances block-sparse sequence parallelism (on DiT, generalizes). **The response — raise the abstraction** so nobody hand-writes per (chip × attention-variant × shape): DSLs [Wave](https://openreview.net/forum?id=gcXV1E8HRH)/[Flashlight](https://openreview.net/forum?id=lboOMA8XWr), dynamic-megakernel compiler [Event Tensor](https://openreview.net/forum?id=PJqFhAbUHa), schedule-decoupled overlap [DynaFlow](https://openreview.net/forum?id=i0yqC9954S), and [FlashInfer-Bench](https://openreview.net/forum?id=IyryZno8Hh) — the generate→bench→hot-swap loop for LLM-written kernels.

## 7. The objective function broadened: energy, edge, cost, reliability

**Energy** (a real cross-cutting thread): [BEAM](https://openreview.net/forum?id=BfNBXM8CCT) spends SLO slack on energy (−51%), [CORE](https://openreview.net/forum?id=PSyHQ8kVUT) coordinates phone CPU/GPU/mem frequencies, [Layered Prefill](https://openreview.net/forum?id=yyDbI3HXco) −22%/token. **Edge/client:** [IntAttention](https://openreview.net/forum?id=CPCRITwAaP) integer softmax on Arm, [VRAM-Constrained xLM](https://openreview.net/forum?id=VKqQYg6JPb) CPU-GPU sharding for NVIDIA's client SDK. **Cost/ops/reliability:** [BOute](https://openreview.net/forum?id=ZVQb92umqX) Bayesian routing+placement, [OptiKIT](https://openreview.net/forum?id=om4H7AI2hc) optimization for non-experts, [MorphServe](https://openreview.net/forum?id=PDu13oOl4G) quantization as a runtime knob, [HELIOS](https://openreview.net/forum?id=CV52m9NJFK) depth as a runtime knob, [FaaScale](https://openreview.net/forum?id=jgL8LuOVyT) serverless cold-scale, [AIRS](https://openreview.net/forum?id=g1RWik4Gy1) Google's rating pipeline, [Breaking the Ice](https://openreview.net/forum?id=eoEobeKTNZ) vLLM cold-start profiling. Two one-offs: [Attribution-based Sparse Activation](https://openreview.net/forum?id=gJFigZeb5D) (skip low-attribution FFN neurons) and [Shannonic](https://openreview.net/forum?id=NhMxI0GbB8) (lossless tensor compression for transfer).

Notably, **standalone weight quantization has narrowed to a niche** — only [MixLLM](https://openreview.net/forum?id=VBbMRQ4VOc) (mixed-precision by output-feature importance), Kitty (KV), and ScaleSearch (BFP scale) appear as theses; otherwise quant is a runtime knob (MorphServe). Read this as a venue-routing effect, though — pure quant algorithms target NeurIPS/ICML, not a systems venue — not a field-wide decline.

## 8. Maturity: practicality vs research

- **Deployed at scale (experience reports):** [Meta deployment config](https://openreview.net/forum?id=gEbKQeIdxB) (Llama, ~1B MAU), [LinkedIn SLM](https://openreview.net/forum?id=re82zZczHj) (millions QPS), fabric-lib (Perplexity), HipKittens (AMD AITER), AIRS (Google), OptiKIT — trustworthy but workload-specific.
- **Industrial, real tooling:** FlashAttention-4, TokenWeave, BatchLLM, Beyond the Buzz / FlashInfer-Bench (on Dynamo/SGLang/vLLM).
- **Research prototypes (the majority):** most KV/spec-decoding/MoE algorithm papers — open code, benchmark-validated; "up to X" least safe to extrapolate ([Performance or Illusion?](https://openreview.net/forum?id=fzkqtezFEi)).

Ideas densest in KV + spec decoding; production-proven work concentrates in scheduling, kernels, communication.

## Paper distribution

| Subfield | # | Subfield | # |
|---|---|---|---|
| compiler_kernel_fusion | 11 | multi_gpu_heterogeneous | 5 |
| scheduling_batching | 8 | moe_inference | 4 |
| long_context_pd_disaggregation | 7 | quantization | 2 |
| kv_cache | 7 | other (8 singletons) | 8 |
| speculative_decoding | 5 | | |

*Mechanical classifier labels; the analysis above is organized by research concern instead. Selection: 57 auto-filtered from ~135 accepts by an LLM classifier (`pipeline/venue_filter.py`), hard-gating non-inference-deployment work. Two biases — a few kept papers aren't pure text-LLM (db-SP, TriInfer, diffusion-LM), and MLSys pre-selects for systems work, so read this as "what MLSys accepted," not "what the field did."*

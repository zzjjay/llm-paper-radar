# MLSys 2026 — LLM Inference Deployment Optimization Trend Report

## Macro synthesis

The 57 in-scope papers look scattered across 8+ named subfields, but underneath them are only two independent root drivers.

**1. Hardware's internal resources don't scale at the same rate.** Matmul throughput grows faster than memory bandwidth, memory capacity, and cross-GPU interconnect — a physical fact of chip scaling, not a software problem. FlashAttention-4 states this most directly: on Blackwell, tensor-core throughput doubled while shared-memory bandwidth and the exponential units didn't, so the whole pipeline had to be redesigned. HipKittens hits the same wall from a different angle — porting kernel primitives from NVIDIA to AMD requires rethinking the primitives, not translating code. Quantization work (MixLLM, block-floating-point scale search) is the same response at the numeric-representation level: since memory is scarcer than compute, shrink the numbers.

**2. Inference workloads are not uniform, but serving systems have long treated them as if they were.** Within one request, prefill is compute-bound and decode is bandwidth-bound (PLA-Serve, TriInfer, "Beyond the Buzz"). Within one context, some tokens/heads matter more than others (FlexiCache, SkipKV, Kitty). Within one decode step, GPU compute sits idle because only one token comes out — speculative decoding (PRISM, SpecDiff-2, TiDAR) fills it. Within one MoE model, experts aren't equally busy (CRAFT, FarSkip-Collective). The pattern repeats at every granularity — token, head, expert, phase, request — and each subfield is the same move at a different scale: find the part of the work that doesn't deserve equal treatment, and skip, cache, or relocate it.

The two drivers are independent — hardware could freeze and workloads would still be phase-heterogeneous; a perfectly uniform workload would still need new kernels for a new chip — but they compound: mobile DVFS scheduling is both firing at once (independent CPU/GPU/memory frequency scaling × prefill/decode having different power profiles).

**compiler_kernel_fusion isn't inside either driver — it's the overflow from multiplying them.** Hardware generations keep turning over *and* workloads keep fragmenting into finer granularities, so the (hardware × workload-slice) space has outgrown hand-tuning kernel-by-kernel. FlashInfer-Bench, Wave, Event Tensor and that cluster respond not to driver 1 or 2 but to their product growing faster than human kernel-writing capacity.

*(Fuller root-rank writeup with ASCII diagrams, in Chinese: `~/Documents/notes/20260709T005837--MLSys2026推理优化的秩__rank.org`)*

## Subfield distribution

| Subfield | # Papers |
|---|---|
| compiler_kernel_fusion | 11 |
| scheduling_batching | 8 |
| long_context_pd_disaggregation | 7 |
| kv_cache | 7 |
| speculative_decoding | 5 |
| multi_gpu_heterogeneous | 5 |
| moe_inference | 4 |
| quantization | 2 |
| other (8 singletons) | 8 |

## compiler_kernel_fusion (11)

Hand-writing kernels can't keep up with new hardware × new attention variants × dynamic shapes. Most papers raise the abstraction so expert performance doesn't need per-combo hand-tuning (DSLs/compilers/tile primitives); a second cluster leans on async/overlap as the speedup lever; a few change the attention math itself; and FlashInfer-Bench builds the generate→benchmark→deploy loop instead of a kernel.

- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://openreview.net/forum?id=mN5RtvuYl3) — async-MMA pipeline redesign for Blackwell
- [HipKittens: Fast and Furious AMD Kernels](https://openreview.net/forum?id=xxSSrndQrI) — tile primitives ported to AMD CDNA
- [Wave: A Symbolic Python DSL And Compiler for High-Performance Machine Learning](https://openreview.net/forum?id=gcXV1E8HRH) — Python-embedded kernel DSL
- [Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants](https://openreview.net/forum?id=lboOMA8XWr) — auto-fused kernels, no static templates
- [Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel](https://openreview.net/forum?id=PJqFhAbUHa) — IR for data-dependent megakernels
- [ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels](https://openreview.net/forum?id=Cv5e5uRXFb) — overlap primitives for multi-GPU
- [TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference](https://openreview.net/forum?id=rh2Ylffkq6) — TP compute/comms overlap
- [DynaFlow: Transparent and Flexible Intra-Device Parallelism via Programmable Operator Scheduling](https://openreview.net/forum?id=i0yqC9954S) — overlap heterogeneous operators
- [BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding](https://openreview.net/forum?id=6INSBXTQ4x) — training-free block-sparse attention
- [IntAttention: A Fully Integer Attention Pipeline for Efficient Edge Inference](https://openreview.net/forum?id=CPCRITwAaP) — integer-only softmax path
- [FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems](https://openreview.net/forum?id=IyryZno8Hh) — generate→bench→deploy loop for AI-written kernels

## scheduling_batching (8)

Serving as resource allocation under SLO/throughput constraints, without touching the model — the lever is exploiting slack or heterogeneity. Papers split by what they optimize for: throughput, energy, latency-SLO, config search, or scaling speed.

- [BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching](https://openreview.net/forum?id=IuVHde07l6) — offline throughput via global prefix reuse
- [SuperInfer: SLO-Aware Rotary Scheduling and Memory Management for LLM Inference on Superchips](https://openreview.net/forum?id=RuslSHdIHa) — anti head-of-line-blocking on superchips
- [MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing](https://openreview.net/forum?id=PDu13oOl4G) — adapt precision/KV to workload swings
- [BEAM: Joint Resource-Power Optimization for Energy-Efficient LLM Inference under SLO constraints](https://openreview.net/forum?id=BfNBXM8CCT) — spend latency slack on energy
- [HELIOS: Adaptive Model And Early-Exit Selection for Efficient LLM Inference Serving](https://openreview.net/forum?id=CV52m9NJFK) — runtime model + early-exit selection
- [FaaScale: Unlocking Fast LLM Scaling for Serverless Inference](https://openreview.net/forum?id=jgL8LuOVyT) — fast elastic scaling via multicast weight transfer
- [Optimizing Deployment Configurations for LLM Inference](https://openreview.net/forum?id=gEbKQeIdxB) — Meta's HW/parallelism/runtime config search at scale
- [AIRS: Scaling Live Inference in Resource Constrained Environments](https://openreview.net/forum?id=g1RWik4Gy1) — live serving under tight resource limits

## long_context_pd_disaggregation (7)

All treat prefill as a separable phase: split it onto its own hardware, overlap it in time, or delete redundant prefill via reuse. "Beyond the Buzz" is the measurement study rather than a system.

- [PLA-Serve: A Prefill-Length-Aware LLM Serving System](https://openreview.net/forum?id=dzjCkSEDyG) — length-aware prefill/decode disaggregation
- [TriInfer: Hybrid EPD Disaggregation for Efficient Multimodal Large Language Model Inference](https://openreview.net/forum?id=nNovi8fvGN) — adds a third encode stage for multimodal
- [Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token](https://openreview.net/forum?id=FuRo7Ur5Ib) — stream retrieved context into prefill
- [FlashAgents: Accelerating Multi-Agent LLM Systems via Streaming Prefill Overlap](https://openreview.net/forum?id=m14PPUfgEc) — stream tokens between agents
- [ContextPilot: Fast Long-Context Inference via Context Reuse](https://openreview.net/forum?id=RnKvDy1jv2) — context index to cut redundant prefill
- [Using Span Queries to Optimize Cache and Attention Locality](https://openreview.net/forum?id=qcGGSXpFcM) — generalized cache-locality abstraction
- [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation](https://openreview.net/forum?id=NqC5tcBsa0) — empirical study across the design space

## kv_cache (7)

Every method exploits non-uniform importance in the cache — across tokens, heads/channels, queries/beams, or time — and almost all are training-free plug-ins over paged attention. They split by goal: memory footprint, throughput, reliability, or accuracy-under-compression.

- [Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost](https://openreview.net/forum?id=r3mQiuYKIN) — channel-wise precision, ~8x memory cut
- [FlexiCache: Leveraging Temporal Stability of Attention Heads for Efficient KV Cache Management](https://openreview.net/forum?id=GgX6dPJx9M) — offload stable heads, keep unstable on GPU
- [OPKV: A High-Throughput Plugin-Driven Framework for Recallable Sparsity in Paged KV Cache Systems](https://openreview.net/forum?id=EB5bgzv4qA) — recallable sparsity in paged KV
- [SkipKV: Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models](https://openreview.net/forum?id=0EsV9SIm8p) — sentence-level eviction for long CoT
- [MAC-Attention: a Match-Amend-Complete scheme for fast and accurate attention computation](https://openreview.net/forum?id=b6HBRCejb7) — reuse attention for similar queries
- [Locality-Aware Beam Scheduling for Efficient Test-Time Compute with a Consumer-grade GPU](https://openreview.net/forum?id=dTo8jAXm9K) — beam/token locality to cut KV transfers
- [GhostServe: A Lightweight Checkpointing System in the Shadow for Fault-Tolerant LLM Serving](https://openreview.net/forum?id=xKjYiUgeOK) — erasure-coded KV for fast fault recovery

## speculative_decoding (5)

Same draft-then-verify skeleton; the difference is where the cheap draft comes from — a restructured AR drafter, non-AR diffusion drafters, or self-speculation with sparse attention. "Performance or Illusion?" is the reality-check that measures whether any of it survives production batching.

- [PRISM: Parametrically Refactor Inference for Speculative Decoding Draft Models](https://openreview.net/forum?id=cvU2HuuxEf) — decouple draft params across steps
- [SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding](https://openreview.net/forum?id=o42VU86ZsV) — diffusion drafter calibrated to AR verifier
- [TiDAR: Think in Diffusion, Talk in Autoregression](https://openreview.net/forum?id=onfxEjoE4L) — draft (diffusion) + verify (AR) in one pass
- [Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding](https://openreview.net/forum?id=yeqrwcWjPu) — self-speculation, no separate drafter
- [Speculative Decoding: Performance or Illusion?](https://openreview.net/forum?id=fzkqtezFEi) — production-engine benchmark of SD variants

## multi_gpu_heterogeneous (5)

Squeeze throughput and resilience out of imperfect, non-uniform clusters via load/resource balancing — over flaky GPUs, mixed GPU types, sparse-attention shards, or NIC vendors. Abstraction layers range from comms/fault-tolerance infra up to Bayesian orchestration.

- [RaidServe: High-performance Resilient Serving](https://openreview.net/forum?id=5pl9fdbEkq) — TP serving under irregular GPU availability
- [BOute: Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via Multi-Objective Bayesian Optimization](https://openreview.net/forum?id=ZVQb92umqX) — Bayesian routing over heterogeneous GPUs
- [db-SP: Accelerating Sparse Attention for Visual Generative Models with Dual-Balanced Sequence Parallelism](https://openreview.net/forum?id=XgKteNxNe0) — balance sparse-attention load across GPUs
- [fabric-lib: RDMA Point-to-Point Communication for LLM Systems](https://openreview.net/forum?id=SjVa05wEiY) — vendor-portable RDMA p2p
- [Efficient, VRAM-Constrained xLM Inference on Clients](https://openreview.net/forum?id=VKqQYg6JPb) — adapt placement to CPU/GPU client conditions

## moe_inference (4)

All blame expert parallelism and fix it at the systems layer, not the model — architecture skip-connections for overlap, hot-expert replication for imbalance, or layer-granular scheduling. "Demystifying" is the measurement (2-3x tax) that frames the other three.

- [FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models](https://openreview.net/forum?id=ruOpvLzsGV) — skip connections for compute/comms overlap
- [CRAFT: Fine-Grained Cost-Aware Expert Replication For Efficient Mixture-of-Experts Serving](https://openreview.net/forum?id=zdRvzU9ZCe) — replicate hot experts to fix load imbalance
- [From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill](https://openreview.net/forum?id=yyDbI3HXco) — layer-granular stall-free scheduling
- [Demystifying the Mixture of Experts Serving Tax](https://openreview.net/forum?id=lELxqcgrsN) — measures and categorizes the MoE serving tax

## quantization (2)

Both reject one-size-fits-all quant params and search a discrete space guided by an error metric, staying GPU-kernel-friendly — MixLLM allocates precision across output features; ScaleSearch picks the best per-block BFP scale.

- [MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design](https://openreview.net/forum?id=VBbMRQ4VOc)
- [Search Your Block Floating Point Scales!](https://openreview.net/forum?id=innqECyZPK)

## other (8 singletons)

One paper each — directions that surfaced but didn't cluster this year. Several are early "map the problem" studies (cold-start, DVFS) rather than mature solution spaces.

- [Attribution-based Sparse Activation in Large Language Models](https://openreview.net/forum?id=gJFigZeb5D) — *neuron sparse activation*: per-input neuron skipping via attribution, training-free
- [Shannonic: Efficient Entropy-Optimal Compression for ML Workloads](https://openreview.net/forum?id=NhMxI0GbB8) — *tensor lossless compression*: near-entropy ANS coding for tensors
- [Meeting SLOs, Slashing Hours: Automated Enterprise LLM Optimization with OptiKIT](https://openreview.net/forum?id=om4H7AI2hc) — *automated inference optimization*: auto-search deployment configs
- [Breaking the Ice: Analyzing Cold Start Latency in vLLM](https://openreview.net/forum?id=eoEobeKTNZ) — *cold start latency*: profiling study of vLLM startup
- [CDLM: Consistency Diffusion Language Models for Faster Sampling](https://openreview.net/forum?id=eB8yjR6alL) — *diffusion LM inference*: consistency distillation to cut sampling steps
- [TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval](https://openreview.net/forum?id=YsOyCpMUYD) — *RAG inference*: lookahead prefetch to overlap retrieval with decode
- [Scaling Up Large Language Models Serving Systems for Semantic Job Search](https://openreview.net/forum?id=re82zZczHj) — *production serving*: SLM compression + serving lessons (LinkedIn)
- [Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE](https://openreview.net/forum?id=PSyHQ8kVUT) — *mobile DVFS*: coordinated CPU/GPU/memory frequency scaling

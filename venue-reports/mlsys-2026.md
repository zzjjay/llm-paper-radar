# MLSys 2026 — LLM Inference Deployment Optimization Trend Report

*Analysis of the 57 accepted papers (of ~135 total accepts) whose primary contribution is optimizing how LLMs are served. Grounded in the mechanisms each abstract reports. Every speedup/reduction figure below is the authors' own self-reported best — usually "up to X", often a favorable config or microbenchmark; treat them as ceilings. One systems venue, auto-selected set — see [Method & caveats](#method--caveats) before reading any distribution claim as a field trend.*

## The one thing to take away

Two independent shifts sit under almost every paper, and the set divides roughly evenly between them.

**The workload changed shape.** Reasoning models emit thousands of chain-of-thought tokens, so within a request the cost center moved from reading the prompt (prefill, compute-bound) to writing the answer (decode, memory-bound). And agents, RAG, and multi-turn chat made requests stop being independent — they share large spans of context (system prompts, retrieved docs, history, agent templates). Systems built for short, independent, single-turn chat are being retrofitted for long, interdependent traffic.

**The hardware changed.** New NVIDIA generations rebalanced the on-chip bottleneck (Blackwell doubled tensor-core throughput but not the memory bandwidth / exponential units around it), interconnect fell further behind compute, and heterogeneity — AMD, GH200 superchips, mixed fleets, client GPUs, phones — became something you design for. This force drives the single largest bucket (kernels/compilers) and the multi-GPU work, none of it about the KV cache.

The two meet at the **KV cache** — the most-contested single object (~15–20 of the 57), though *not* a majority and untouched by the hardware half. The workload makes it grow (long decode) and makes it reusable (shared context); one hardware fact — memory bandwidth scarcer than compute — makes *reading* it the decode bottleneck. So there is no honest "it's all about X": there are two forces, and the KV cache is the busiest place one of them lands.

One constraint shapes the whole set: weights are frozen (no retraining a released model) and the target is an existing engine (vLLM / SGLang / TensorRT-LLM), so nearly everything is **training-free, drop-in, algorithm-system co-design**. (Partly the venue talking — a pure-algorithm method would be submitted elsewhere.)

## 1. Decode went memory-bound, and the KV cache is the battleground

The recurring physical fact, stated almost verbatim across papers: during decode each new token re-reads the entire, growing KV cache, so long generation is IO-bound, not compute-bound ([MAC-Attention](https://openreview.net/forum?id=b6HBRCejb7), [SpecGen](https://openreview.net/forum?id=yeqrwcWjPu)). (Autoregressive decode was already memory-bound at small batch; the workload shift *amplified* this into the dominant regime, didn't create it.) The responses escalate in four stances:

- **Shrink it.** [Kitty](https://openreview.net/forum?id=r3mQiuYKIN) pushes KV to 2-bit — where accuracy normally breaks — by keeping a small fraction of sensitivity-ranked *Key channels* at higher precision (~8× KV-memory cut, 2.1–4.1× throughput). Quantization survived here only as a KV-specific tool.
- **Skip it.** Cached tokens don't matter equally: [FlexiCache](https://openreview.net/forum?id=GgX6dPJx9M) finds *some attention heads are temporally stable* and offloads their cold pages to host memory (−70% GPU footprint); [OPKV](https://openreview.net/forum?id=EB5bgzv4qA) makes offload-and-recall sparsity a clean plugin over paged KV; [BLASST](https://openreview.net/forum?id=6INSBXTQ4x) skips whole attention blocks under one calibrated softmax threshold.
- **Reuse computation over it.** [MAC-Attention](https://openreview.net/forum?id=b6HBRCejb7) reuses prior attention results for semantically similar recent queries (match–amend–complete), making per-token cost constant on a cache hit — keep everything, recompute less (attention-phase microbenchmark gains; end-to-end is smaller).
- **Treat it as durable, distributed state** — the most conceptually notable shift, KV graduating from "a buffer" to state needing placement, transfer, replication, fault tolerance: [GhostServe](https://openreview.net/forum?id=xKjYiUgeOK) erasure-codes KV so a GPU failure doesn't force recomputation; [RaidServe](https://openreview.net/forum?id=5pl9fdbEkq) does cyclic KV placement + proactive backup for resilient TP serving; [fabric-lib](https://openreview.net/forum?id=SjVa05wEiY) (Perplexity, in production) builds portable RDMA transport because KV transfer for disaggregation outgrew simple collectives.

## 2. The workload changed: long reasoning outputs and shared context

**Reasoning models made outputs long,** amplifying the decode/KV bottleneck. [SkipKV](https://openreview.net/forum?id=0EsV9SIm8p) finds token-level KV eviction actively fails on chain-of-thought (unstable scoring, and it makes the model generate *more* to re-derive evicted content), so it evicts at sentence granularity and steers toward conciseness. [SpecGen](https://openreview.net/forum?id=yeqrwcWjPu) is framed around reasoning shifting the bottleneck to memory. [Locality-Aware Beam Scheduling](https://openreview.net/forum?id=dTo8jAXm9K) targets test-time-compute beam search, where concurrent decode paths make KV, not weights, the bottleneck on consumer GPUs.

**Agents, RAG, and multi-turn made context shared,** so the win is detecting and reusing overlap instead of recomputing prefill. [ContextPilot](https://openreview.net/forum?id=RnKvDy1jv2) indexes overlapping context across users/turns; [FlashAgents](https://openreview.net/forum?id=m14PPUfgEc) streams tokens between agents and dedupes shared templates via a radix-tree prefix cache; [BatchLLM](https://openreview.net/forum?id=IuVHde07l6) makes *global* prefix sharing explicit for offline batch (up to 10.8× over vLLM/SGLang on microbenchmarks + one industry workload) because LRU eviction drops about-to-be-reused prefixes; [Stream2LLM](https://openreview.net/forum?id=FuRo7Ur5Ib) and [TeleRAG](https://openreview.net/forum?id=YsOyCpMUYD) hide RAG retrieval latency behind compute. The sharpest framing is [Span Queries](https://openreview.net/forum?id=qcGGSXpFcM): chat, RAG, inference-time scaling, and agentic workloads are all special cases of one structure (an expression tree of inference calls with commutativity constraints), so cache locality can be optimized generically — 10–20× TTFT in 492 lines of vLLM changes.

## 3. Prefill and decode want different machines — disaggregation matures past the hype

Prefill (compute-bound, whole prompt) and decode (memory-bound, one token at a time) have opposite profiles, so splitting them onto separate instances (PD disaggregation) is now the assumed baseline others build on. This year's work is refinements and reality-checks — a maturing subfield:

- [Beyond the Buzz](https://openreview.net/forum?id=NqC5tcBsa0) is the sobering measurement study (hundreds of thousands of design points on NVIDIA Dynamo): disaggregation wins only for prefill-heavy traffic and larger models, and only with dynamic rate matching and elastic scaling. It isn't free.
- [PLA-Serve](https://openreview.net/forum?id=dzjCkSEDyG) notes prefill itself isn't uniform (short vs long prompts) and disaggregates *within* prefill by length.
- [TriInfer](https://openreview.net/forum?id=nNovi8fvGN) extends two phases to three (encode–prefill–decode) for multimodal, giving the vision encoder its own pool (3.7× throughput) — a multimodal, not text-LLM, system.

## 4. MoE broke the dense-serving playbook

MoE is now mainstream for frontier models, and a cluster is cleaning up the fallout: optimizations designed for dense models backfire on MoE. [Demystifying the MoE Serving Tax](https://openreview.net/forum?id=lELxqcgrsN) quantifies it — MoE serves 2–3× worse than a FLOP-equivalent dense model — and finds a counterintuitive twist: expert load imbalance *hurts prefill but can help decode* (fewer experts activated). The fixes hit different layers: [Layered Prefill](https://openreview.net/forum?id=yyDbI3HXco) shows standard chunked prefill inflates MoE memory traffic ~39% via redundant expert-weight reloads and re-schedules by *layer groups* instead of token chunks; [CRAFT](https://openreview.net/forum?id=zdRvzU9ZCe) does cost-aware expert replication and shows current schemes *over-replicate*; [FarSkip-Collective](https://openreview.net/forum?id=ruOpvLzsGV) goes furthest — modifying the architecture (skip connections, recovered by self-distillation to within 1% accuracy up to 109B) to overlap the all-to-all expert communication that blocks compute (32.6% TTFT on DeepSeek-V3). FarSkip is the one place this cluster breaks the frozen-weights constraint — it retrains, which raises the adoption bar.

## 5. Speculative decoding: attack the same bottleneck from the compute side

If decode is memory-bound, the GPU's compute sits idle — so spend it guessing tokens ahead and verifying in parallel (the compute-side complement to §1). The draft-quality-vs-cost tension drives the designs: [PRISM](https://openreview.net/forum?id=cvU2HuuxEf) decouples draft-model capacity from draft cost by refactoring parameters across predictive steps; [SpecGen](https://openreview.net/forum?id=yeqrwcWjPu) avoids a separate drafter via self-speculation with a sparse-attention "draft." Notably, **diffusion is entering as the non-autoregressive drafter**: [SpecDiff-2](https://openreview.net/forum?id=o42VU86ZsV) calibrates a discrete-diffusion drafter to an AR verifier, [TiDAR](https://openreview.net/forum?id=onfxEjoE4L) fuses diffusion drafting and AR sampling in one forward pass, and standalone [CDLM](https://openreview.net/forum?id=eB8yjR6alL) makes diffusion LMs practical via consistency distillation + KV-cache compatibility. The essential paper is the reality-check: [Speculative Decoding: Performance or Illusion?](https://openreview.net/forum?id=fzkqtezFEi) benchmarks SD variants in *production* vLLM and finds large gaps between measured and theoretical speedup at realistic batch sizes — the standing caveat against taking the other four's peak numbers at face value.

## 6. Hardware moved; the software layer scrambles to keep up

The largest cluster in the set (kernels/compilers 11 + multi-GPU 5 = 16 of 57), and the second story as big as the KV cache. Two hardware pressures multiplied the surface a kernel author must cover, and the shared response is to raise the abstraction so nobody hand-writes per (chip × attention-variant × dynamic-shape).

**Pressure A — new generations rebalanced the on-chip bottleneck.** [FlashAttention-4](https://openreview.net/forum?id=mN5RtvuYl3) is the cleanest case: Blackwell's asymmetric scaling breaks the Hopper-era kernel's assumptions, so FA-4 redesigns the pipeline around fully-async MMA and software-emulates the exponential to cut non-matmul work. [TokenWeave](https://openreview.net/forum?id=rh2Ylffkq6) (Microsoft) exploits Hopper/Blackwell NVSHARP/Multimem for a fused AllReduce–RMSNorm in 2–8 SMs; [SuperInfer](https://openreview.net/forum?id=RuslSHdIHa) co-designs scheduling for GH200's NVLink-C2C; [ScaleSearch](https://openreview.net/forum?id=innqECyZPK) exists only because GPUs added native NVFP4/microscaling-BFP and the default max-magnitude scale wastes accuracy.

**Pressure B — heterogeneity went mainstream.** [HipKittens](https://openreview.net/forum?id=xxSSrndQrI) makes AMD a real target (tile primitives rivaling hand-tuned assembly on CDNA3/4, productionalized in AMD's AITER); [ParallelKittens](https://openreview.net/forum?id=Cv5e5uRXFb) names the cause outright — interconnect improving slower than compute — and packages multi-GPU overlap into 8 primitives; [db-SP](https://openreview.net/forum?id=XgKteNxNe0) shows interconnect-bound sequence parallelism needs sparsity-aware load balancing once attention is block-sparse (shown on diffusion transformers; not a text-LLM result, but the imbalance analysis generalizes).

**The response — raise the kernel-authoring abstraction.** [Wave](https://openreview.net/forum?id=gcXV1E8HRH) and [Flashlight](https://openreview.net/forum?id=lboOMA8XWr) are Python/compiler-native kernel DSLs; [Event Tensor](https://openreview.net/forum?id=PJqFhAbUHa) compiles *dynamic* megakernels (data-dependent shapes static megakernels can't handle); [DynaFlow](https://openreview.net/forum?id=i0yqC9954S) decouples model definition from execution schedule for intra-device overlap. The most forward-looking is [FlashInfer-Bench](https://openreview.net/forum?id=IyryZno8Hh): now that LLMs can *write* GPU kernels, it builds the generate→benchmark→hot-swap-into-vLLM/SGLang loop to make that safe and continuous — the plausible next stage of who writes the kernel.

## 7. The objective function broadened: energy, edge, cost, reliability

The field is visibly widening past "tokens/sec on an H100."

- **Energy as a first-class objective** (a distinct cross-cutting thread, not a one-off): [BEAM](https://openreview.net/forum?id=BfNBXM8CCT) spends post-SLO latency slack on energy by co-tuning GPU frequency + batch shape (up to −51% GPU energy); [CORE](https://openreview.net/forum?id=PSyHQ8kVUT) coordinates CPU/GPU/memory frequencies for phone LLMs; [Layered Prefill](https://openreview.net/forum?id=yyDbI3HXco) reports −22% per-token energy as a side benefit.
- **The edge/client as a real target:** [IntAttention](https://openreview.net/forum?id=CPCRITwAaP) removes the FP-softmax detour dominating INT8 attention on Arm CPUs; [Efficient VRAM-Constrained xLM Inference](https://openreview.net/forum?id=VKqQYg6JPb) does CPU-GPU pipelined sharding for NVIDIA's client SDK; [Locality-Aware Beam Scheduling](https://openreview.net/forum?id=dTo8jAXm9K) targets consumer GPUs.
- **Cost, operability, reliability:** [BOute](https://openreview.net/forum?id=ZVQb92umqX) co-optimizes query routing + heterogeneous-GPU deployment via Bayesian optimization; [OptiKIT](https://openreview.net/forum?id=om4H7AI2hc) democratizes optimization for non-expert teams; [MorphServe](https://openreview.net/forum?id=PDu13oOl4G) treats quantization as a *runtime knob*, hot-swapping layers under load and reusing freed memory for KV; [HELIOS](https://openreview.net/forum?id=CV52m9NJFK) treats depth as the runtime knob (multiple early-exit models, load only likely-used layers); [FaaScale](https://openreview.net/forum?id=jgL8LuOVyT) tackles serverless cold-scale; [AIRS](https://openreview.net/forum?id=g1RWik4Gy1) is Google's production LLM-rating pipeline under a fixed TPU budget; [Breaking the Ice](https://openreview.net/forum?id=eoEobeKTNZ) profiles vLLM cold-start (CPU-bound, predictable).

Two orthogonal one-offs: [Attribution-based Sparse Activation](https://openreview.net/forum?id=gJFigZeb5D) (skip low-attribution FFN neurons per input, training-free — the KV-sparsity instinct applied to the MLP) and [Shannonic](https://openreview.net/forum?id=NhMxI0GbB8) (near-Shannon-limit *lossless* tensor compression for bandwidth-bound transfer).

## 8. Maturity: what's shipping vs what's a prototype

Cutting across all seven concerns is a readiness axis — the one most useful for deciding what to adopt vs. watch:

- **Deployed at scale (named production systems):** [Optimizing Deployment Configurations](https://openreview.net/forum?id=gEbKQeIdxB) (Meta, Llama at ~1B MAU) and [Semantic Job Search](https://openreview.net/forum?id=re82zZczHj) (LinkedIn, millions QPS) are experience reports, not prototypes; [fabric-lib](https://openreview.net/forum?id=SjVa05wEiY) runs three production systems at Perplexity; [HipKittens](https://openreview.net/forum?id=xxSSrndQrI) ships in AMD's AITER; [AIRS](https://openreview.net/forum?id=g1RWik4Gy1) is a live Google pipeline; [OptiKIT](https://openreview.net/forum?id=om4H7AI2hc) reports in-production gains. Safest to trust, hardest to generalize (numbers are workload-specific).
- **Industrial-grade, integrated with real tooling** (open-sourced, evaluated on real hardware/traces): [FlashAttention-4](https://openreview.net/forum?id=mN5RtvuYl3), [TokenWeave](https://openreview.net/forum?id=rh2Ylffkq6), [BatchLLM](https://openreview.net/forum?id=IuVHde07l6), [Beyond the Buzz](https://openreview.net/forum?id=NqC5tcBsa0) / [FlashInfer-Bench](https://openreview.net/forum?id=IyryZno8Hh) (on Dynamo / SGLang+vLLM).
- **Research prototypes** (the majority): often open code, but validated on benchmarks/traces, not deployment — most of the KV, spec-decoding, and MoE algorithm papers. Their "up to X" figures are the least safe to extrapolate; [Performance or Illusion?](https://openreview.net/forum?id=fzkqtezFEi) (measured ≪ theoretical at production batch sizes) is the standing warning.

The takeaway: the *ideas* are densest in KV-cache and speculative decoding; the *production-proven* work concentrates in scheduling/deployment-config, kernels, and communication — the boring-but-shipped end.

## What shifted, and what's conspicuously absent

Claims about **this year's set** are grounded in the 57 abstracts; claims about **change over time** have no prior-year data behind them and are marked **[conjecture]**.

- **The KV cache is the largest single object of study** (~15–20 of 57). *(this set)*
- **Weight quantization is nearly absent as a standalone thesis** — only mixed-precision weight quant ([MixLLM](https://openreview.net/forum?id=VBbMRQ4VOc): bits by output-feature importance, Llama-3.1-70B MMLU-Pro loss 1.92→0.99 at +10% bits), KV-quant ([Kitty](https://openreview.net/forum?id=r3mQiuYKIN)), and format scale-search ([ScaleSearch](https://openreview.net/forum?id=innqECyZPK)) survive; quant otherwise appears as a *runtime knob* ([MorphServe](https://openreview.net/forum?id=PDu13oOl4G)). *(this set)* — but **[venue artifact]**: pure weight-quant algorithms target NeurIPS/ICML/ICLR, so their absence at a systems venue says where they're submitted, not that the field moved on.
- **Systems co-design is table stakes** — nearly every paper ships an engine integration. *(this set)*, but again **[venue artifact]** more than a trend.
- **Reality-check / measurement papers are a visible presence** — [Beyond the Buzz](https://openreview.net/forum?id=NqC5tcBsa0), [Performance or Illusion?](https://openreview.net/forum?id=fzkqtezFEi), [Demystifying the MoE Tax](https://openreview.net/forum?id=lELxqcgrsN) all audit hyped techniques at scale. *(this set)* that they exist; **[conjecture]** that they're more common than before.
- **Reasoning + agentic workloads are the driver** behind the KV/prefix/long-decode work — the papers cite CoT length and agent/RAG context sharing as motivation *(this set)*. That this workload barely existed two years ago and will dominate next year is **[conjecture]**.

## Method & caveats

The 57 were filtered from ~135 MLSys 2026 accepts by an LLM classifier (Claude Sonnet, `pipeline/venue_filter.py` against `prompts/inference_relevance.md`), which hard-gates papers whose *primary* contribution isn't LLM-inference-deployment (training-only, model releases, pure ASIC/FPGA, eval-only, non-LLM). Two biases behind every distribution claim above: (1) it's an automated rubric — borderline cases were kept, so a handful (db-SP visual-gen, TriInfer multimodal, the diffusion-LM papers) aren't pure text-LLM; (2) MLSys pre-selects for systems/co-design work before the classifier runs, so this is "what MLSys 2026 accepted," not "what the field did."

### Subfield distribution (mechanical labels)

| Subfield | # |
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

These are the classifier's buckets; the analysis is organized by research concern instead, since the buckets cut across the real story — the KV cache spans four of them, and the hardware force drives both compiler_kernel_fusion and multi_gpu_heterogeneous.

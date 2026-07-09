# ASPLOS 2026 — LLM Inference Deployment Optimization Trend Report

*29 LLM-inference papers hand-selected from the 167 ASPLOS 2026 accepts (title-filtered, then read from abstracts sourced via Semantic Scholar — ASPLOS isn't on OpenReview, so the standard fetcher doesn't apply; see [Method](#paper-distribution)). All figures are authors' self-reported bests ("up to X"), and — important here — many hardware results are from simulation, not silicon (flagged in §8). **Scope note:** unlike the [MLSys 2026 report](mlsys-2026.md), which hard-gates custom-silicon papers, this one includes them — at an architecture venue, algorithm-hardware co-design is the core contribution, and excluding it would gut the picture.*

## Takeaway

ASPLOS 2026 shares MLSys's physical driver almost exactly — decode is memory-bound, the KV cache is the villain ([PAT](https://doi.org/10.1145/3779212.3790200): "decode attention is memory-bound due to massive KV cache loading"). What differs is the *answer*. MLSys stays on commodity GPUs behind vLLM/SGLang and reaches for software (eviction, quant-as-runtime-knob, scheduling); **ASPLOS reaches for hardware**. Roughly a third of the in-scope set is custom silicon or new substrates — ReRAM/HBM PIM ([REPA](https://doi.org/10.1145/3779212.3790212), [STARC](https://doi.org/10.1145/3779212.3790226)), wafer-scale SRAM compute-in-memory ([Ouroboros](https://doi.org/10.1145/3779212.3790197)), near-storage SmartSSDs ([HILOS](https://doi.org/10.1145/3779212.3790119)), FPGA+GPU ([DFVG](https://doi.org/10.1145/3779212.3790153)), and MoE accelerators ([EARTH](https://doi.org/10.1145/3779212.3790155)) — exactly the category the MLSys rubric throws out.

Two more structural differences, both grounded in this set vs. the MLSys set:
- **Utilization via intra-GPU multiplexing, not cross-node disaggregation.** Where MLSys's serving cluster matured *disaggregation*, ASPLOS's leans the other way — pack prefill and decode onto the *same* GPU ([MuxWise](https://doi.org/10.1145/3779212.3790236), [Bullet](https://doi.org/10.1145/3779212.3790135)).
- **Quantization is alive and well** — as number-format + kernel + datapath co-design ([Tilus](https://doi.org/10.1145/3760250.3762219), [M2XFP](https://doi.org/10.1145/3779212.3790185), [Mugi](https://doi.org/10.1145/3779212.3790189)) — whereas at MLSys standalone quant had nearly vanished. This is a venue effect: number formats and kernels *are* architecture contributions.

The enabling constraint also flips: MLSys work is bound by frozen weights + an existing engine ("must plug into vLLM"); much of ASPLOS proposes new hardware, new formats, or new kernels, so the bar is "what a new chip/format/kernel could do," not "deployable tomorrow." The cost of that freedom shows up in §8 (a lot of it is simulated).

## 1. Memory-bound decode and the KV cache — same villain, hardware answers

The KV cache is the dominant object here too, attacked below the software line. On-GPU/kernel: [PAT](https://doi.org/10.1145/3779212.3790200) is a prefix-aware decode-attention kernel that packs queries sharing a prefix (−53.5% attention latency); [TPLA](https://doi.org/10.1145/3779212.3790237) partitions DeepSeek-style latent attention across tensor-parallel devices to shrink per-device KV (1.8–1.9×); [I/O Analysis is All You Need](https://doi.org/10.1145/3779212.3790174) derives an I/O-optimal long-sequence attention accelerator from first principles. Below the memory line: [REPA](https://doi.org/10.1145/3779212.3790212) (ReRAM PIM) and [STARC](https://doi.org/10.1145/3779212.3790226) (HBM-PIM with sparsity-aware KV mapping) put KV offload/compute *in memory*; [HILOS](https://doi.org/10.1145/3779212.3790119) pushes attention into near-storage SmartSSDs for offline long-context (up to 7.86×); [Ouroboros](https://doi.org/10.1145/3779212.3790197) keeps KV entirely on-wafer in SRAM CIM. [SpeContext](https://doi.org/10.1145/3779212.3790224) targets the reasoning-length KV explosion with a distilled retrieval head + async prefetch — the same reasoning-workload driver MLSys flagged.

## 2. GPU utilization under the prefill/decode mismatch — multiplex, don't (only) disaggregate

The prefill(compute-bound)/decode(memory-bound) mismatch is the recurring framing, but the ASPLOS response is to raise single-GPU utilization by running both at once. [MuxWise](https://doi.org/10.1145/3779212.3790236) does intra-GPU prefill-decode multiplexing (2.2× throughput under SLO); [Bullet](https://doi.org/10.1145/3779212.3790135) does spatial-temporal orchestration of concurrent prefill+decode; [QoServe](https://doi.org/10.1145/3779212.3790206) co-schedules interactive+batch on shared infra with fine-grained QoS; [Shift Parallelism](https://doi.org/10.1145/3779212.3790219) dynamically switches tensor↔sequence parallelism to track traffic; [BlendServe](https://doi.org/10.1145/3779212.3790133) reorders offline batches to overlap compute- and memory-heavy requests; [XY-Serve](https://doi.org/10.1145/3760250.3762228) smooths workload variability into fixed-tile meta-kernels on Ascend NPUs.

## 3. Speculative decoding — pushed into hardware and disaggregation

Same guess-and-verify idea as MLSys, but ASPLOS splits it across devices and chips. [SwiftSpec](https://doi.org/10.1145/3779212.3790246) (ByteDance) disaggregates draft and target across GPUs with fused CUDA kernels for single-request low latency (347 tok/s on Llama-3-70B); [DFVG](https://doi.org/10.1145/3779212.3790153) runs draft-on-FPGA, verify-on-GPU to exploit the two stages' complementary resource profiles (3.26× + 5.8× energy); [SpeContext](https://doi.org/10.1145/3779212.3790224) applies "speculative context sparsity" to long-context reasoning; [EARTH](https://doi.org/10.1145/3779212.3790155) folds speculative expert prefetch into a MoE accelerator.

## 4. MoE inference — an offloading/edge problem, not datacenter expert-parallelism

Where MLSys framed MoE as a datacenter expert-parallelism "serving tax," ASPLOS frames it as fitting MoE into constrained memory. [MoE-APEX](https://doi.org/10.1145/3779212.3790187) does adaptive-precision expert offloading on edge devices (replaces cache-miss experts with low-precision variants, 1.3–9.75× on llama.cpp); [EARTH](https://doi.org/10.1145/3779212.3790155) co-designs entropy-encoded expert storage + speculative prefetch in hardware; [oFFN](https://doi.org/10.1145/3779212.3790194) exploits ReLU-fied FFN activation sparsity (outlier- and hot/cold-neuron-aware) for up to 5.46× FFN speedup.

## 5. Low-precision — number formats, kernels, and datapaths (the quant story MLSys had lost)

A healthy cluster, and distinctly architectural. [M2XFP](https://doi.org/10.1145/3779212.3790185) adds minimal metadata to microscaling MXFP4 to recover accuracy (−70.6% accuracy loss vs MXFP4, beats NVFP4) via algorithm-hardware co-design; [Tilus](https://doi.org/10.1145/3760250.3762219) (NVIDIA) is a tile-level GPGPU DSL for arbitrary 1–8-bit types, beating Triton/Marlin low-precision kernels; [Mugi](https://doi.org/10.1145/3779212.3790189) builds value-level-parallelism datapaths for low-precision LLM GEMM *and* nonlinear ops (softmax); [ZipServ](https://doi.org/10.1145/3779212.3790250) does lossless weight compression with a fused decompress-into-Tensor-Core GEMM (−30% model size, 1.22× e2e).

## 6. The ASPLOS signature: algorithm–hardware co-design and beyond-NVIDIA heterogeneity

The distinguishing thread. Beyond the KV/quant papers already named, the substrate diversity is the point: ReRAM PIM ([REPA](https://doi.org/10.1145/3779212.3790212)), HBM-PIM ([STARC](https://doi.org/10.1145/3779212.3790226)), wafer-scale SRAM CIM ([Ouroboros](https://doi.org/10.1145/3779212.3790197)), near-storage SmartSSD ([HILOS](https://doi.org/10.1145/3779212.3790119)), FPGA ([DFVG](https://doi.org/10.1145/3779212.3790153)), custom MoE/low-precision datapaths ([EARTH](https://doi.org/10.1145/3779212.3790155), [Mugi](https://doi.org/10.1145/3779212.3790189)), Ascend NPUs ([XY-Serve](https://doi.org/10.1145/3760250.3762228)), and AMD GPUs — [MSCCL++](https://doi.org/10.1145/3779212.3790188) (Microsoft), a portable GPU-communication layer for AI inference, is in production on Azure and adopted by AMD's RCCL. Where MLSys's heterogeneity was mostly "AMD + new NVIDIA gens," ASPLOS spans PIM/CIM/near-storage/FPGA/NPU — it treats the memory wall as a reason to change the hardware, not just the kernel.

## 7. Edge and on-device deployment as a first-class target

A pronounced edge emphasis, largely absent from MLSys's datacenter focus. [Neuralink](https://doi.org/10.1145/3676642.3736114) optimizes neuron placement in phone flash under IOPS limits (1.49×); [FastTTS](https://doi.org/10.1145/3779212.3790161) makes test-time-scaling reasoning viable on a single 24 GB consumer GPU (matching cloud accuracy); [MoE-APEX](https://doi.org/10.1145/3779212.3790187) and [SpeContext](https://doi.org/10.1145/3779212.3790224) both target memory-constrained edge. Reasoning/agentic workloads drive several of these — the same shift MLSys identified, here pushed to the device.

Three in-scope papers serve generative models beyond text LLMs, using the same toolkit (KV reuse, sequence parallelism, caching): [BAT](https://doi.org/10.1145/3779212.3790131) (generative recommenders, bipartite-attention prefix reuse), [MoDM](https://doi.org/10.1145/3760250.3762220) (image diffusion, cache-and-mixture serving), [TetriServe](https://doi.org/10.1145/3779212.3790233) (DiT, step-level sequence parallelism).

## 8. Maturity: practicality vs research

- **In production / shipped:** [MSCCL++](https://doi.org/10.1145/3779212.3790188) (Azure production + adopted by AMD RCCL), [SwiftSpec](https://doi.org/10.1145/3779212.3790246) (ByteDance), [XY-Serve](https://doi.org/10.1145/3760250.3762228) (Ascend-native), [Tilus](https://doi.org/10.1145/3760250.3762219) (NVIDIA, open-source). Trustworthy, but often hardware-specific.
- **Real-GPU prototypes** (open code, measured on commodity GPUs): the serving/kernel software — MuxWise, Bullet, QoServe, PAT, BlendServe, TPLA, SwiftSpec, FastTTS, oFFN, ZipServ, etc.
- **Simulation-only** — the crucial ASPLOS caveat: the PIM/CIM/accelerator results are from simulators or synthesized designs, not deployed silicon ([REPA](https://doi.org/10.1145/3779212.3790212) "GPU-PIM prototype", [STARC](https://doi.org/10.1145/3779212.3790226) "simulated HBM-PIM", [Ouroboros](https://doi.org/10.1145/3779212.3790197), [EARTH](https://doi.org/10.1145/3779212.3790155), [M2XFP](https://doi.org/10.1145/3779212.3790185), [Mugi](https://doi.org/10.1145/3779212.3790189)). Their "4×/9×" numbers are projections against modeled baselines — a different evidentiary tier than the on-GPU papers, and the reason ASPLOS's headline speedups shouldn't be compared one-to-one with MLSys's measured ones.

Takeaway: the ideas with the boldest ceilings are the hardware ones; the results you can bank on today are the GPU-software ones.

## Paper distribution

By theme (a paper can span more than one; there is no OpenReview classifier here — grouping is manual):

| Theme | # | Theme | # |
|---|---|---|---|
| Serving / GPU-utilization / scheduling | 6 | Low-precision (format/kernel/datapath) | 4 |
| KV cache & attention | 8 | Edge / on-device | 4 |
| Speculative decoding | 4 | Non-text-LLM generative serving | 3 |
| MoE inference | 3 | (custom-hardware / PIM / CIM / near-storage / FPGA, cross-cutting) | ~9 |

*Selection & caveats: ASPLOS isn't on OpenReview, so these 29 were hand-picked from the official program's 167 accepted titles (LLM-inference by title), then read from Semantic-Scholar abstracts (29/29 found). Biases: (1) title-based pre-filter can miss a paper whose LLM-inference angle isn't in its title; (2) ASPLOS is an architecture venue, so this is "what ASPLOS accepted" — heavy on hardware co-design — not "what LLM-inference research looks like" overall; (3) scope was deliberately widened vs. the MLSys report to include custom-silicon work. Cross-venue comparisons to MLSys 2026 are grounded in both reports, but reflect venue selection, not a field-wide trend.*

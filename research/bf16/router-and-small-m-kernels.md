# Router and small-m kernels

This file covers the FP32 router, the small-batch dense layers (m up to 32), split-K reduction options, launch gaps, and what we learned about Triton's limits on these shapes.

Labels: "verified" means we measured it or read the primary source, "third-party" means another team reported it, "inference" is our own reading.

## The router

The router multiplies a 256 x 8192 FP32 weight matrix, 8.39 MB, whether you read it as FP32 or as a BF16 hi + lo pair. At the practical read floor that is about 3.5 µs at any m up to 256. Compute at m = 256 with the hi + lo pair is about 2.1 GFLOP, roughly 2 µs at peak, so it does not move the floor.

SGLang's own router runs two or three kernels (`dot_kernel`, `elementwise`, `reduce_1Block`) and takes 10 to 26 µs against a 1.75 µs byte floor. That gap is why even router-only fixes score above 1.0x.

Router time in µs by m, from different sources:

|m|nathenfernandes v18|our v3 (Triton)|thethings|vjeux file, our scored run|SGLang|
|-|-|-|-|-|-|
|1|3.94|5.73|5.43|3.84|10.1|
|8|4.34|5.90|5.84|5.19|26.1|
|16|4.23|5.97|5.90|5.25|14.7|
|32|5.04|7.72|6.14|5.43|15.5|
|128|6.19|10.53|9.86|6.23|16.0|
|256|8.44|12.38|10.48|7.20|17.5|

(SGLang's time in the scored run, job 6c7a0061944e4339, was 10.62, 26.16, 15.01, 15.12, 16.34 and 17.55 µs.)

### How far is the submitted file from its bound

From the judge profile of the scored run. The bound is the 8.4 MB FP32 stream plus x at 4.1 TB/s, plus 1.4 µs fixed cost (both our assumptions). "Gain" is the whole-score gain if that shape reached the bound.

|m|Kernel|GPU µs|Ours µs|SGLang µs|Floor µs|Bound µs|Gap µs|Gain|
|-|-|-|-|-|-|-|-|-|
|1|rgemv_kernel<1,1> (FP32 weights, plain GEMV)|2.4|3.84|10.62|1.75|3.45|0.39|0.19%|
|8|sk2_kernel|3.8|5.19|26.16|1.78|3.48|1.71|0.74%|
|16|sk2_kernel|4.0|5.25|15.01|1.81|3.52|1.73|0.61%|
|32|sk2_kernel|4.8|5.43|15.12|1.86|3.58|1.85|0.60%|
|128|rk_kernel (TMA, split-K, 12 K-splits)|4.7|6.23|16.34|2.21|3.99|2.24|0.54%|
|256|rk_kernel (TMA, split-K, 8 K-splits)|5.9|7.20|17.55|2.68|4.54|2.66|0.52%|

The router carries a weight of 0.104 in the bf16 score in our estimate. All six router shapes together are worth at most 3.2% of the score, about 1.6% at half capture. That is below what one judge check can resolve, so we dropped router work as a priority. The m = 128 router grid is 4 x 2 x 12 = 96 CTAs on 132 SMs; another public file moved to 16 K-splits (grid 128) after a sweep. That is a one-constant change worth about 0.1% at most.

The router candidate we added to the submitted file is a Triton partial kernel plus a reduce kernel, the approach described in an X post by @anshulcreates as far as we can tell. It is absent from the upstream file, and it lost the prepare-time comparison at every m. The numbers below are from the submitted file's own two runs (jobs 6c7a0061944e4339 and a35a24f311274d15, in that order). They are raw graph-replay µs and include the fixed replay cost, so only the ordering is meaningful.

|m|Triton router|Best native kernel|
|-|-|-|
|1|9.28 / 9.02|rgemv 7.10 / 6.78|
|8|9.50 / 9.31|sk2 8.48 / 8.06|
|16|9.54 / 9.28|sk2 8.48 / 8.16|
|32|10.98 / 10.59|sk2 8.86 / 8.67|
|128|14.30 / 13.86|rk 9.57 / 9.18|
|256|15.78 / 15.49|rk 10.56 / 10.91|

The two later runs of the rs128 variant (jobs 635004eab6f9489a and c440acdd631e4328) show the same ordering with the same kernels.

Router-only Triton work gets a modest result everywhere we saw it. A PostTrain write-up titled "Open Kernel narrow split-K router" (Oct 23) reports 1.03 to 1.05x. @anshulcreates reports 10.3 and 26.3 µs down to 5.1 and 11.9 µs with a total score of 1.055x (claims, not reproduced). The router-only tier on the board sits at 1.15 to 1.25x.

### Router designs we know of

- SGLang's `dsv3_router_gemm` (https://github.com/sgl-project/sglang/pull/7627, derived from an NVIDIA TensorRT kernel). One CTA per expert row, 256 CTAs of 128 threads. Each thread loads 16 bytes (8 BF16) per step, 7 steps for K = 7168, keeps a per-token FP32 accumulator, reduces with XOR shuffles inside a warp and then through shared memory across the 4 warps. No atomics, no memset, deterministic, one launch, templated on token count 1 to 16, with PDL on SM90 and later. It needs BF16 inputs, FP32 output and M <= 16, and K = 7168 is hard-coded for DeepSeek-V3, so it does not fit our K = 8192 FP32 weights. A design reference only. The PR's file view has no benchmark numbers. The PR threads (this one and https://github.com/flashinfer-ai/flashinfer/pull/2019) report 1.36 to 1.82x on B200 for it, and a summary on the PR thread claims 1.86 to 2.5x, which we did not check.
- thethings, m <= 32: the same tensor-core GEMV as their dense layers, with the hi and lo tiles feeding one accumulator.
- thethings, m = 128/256: a Triton split-K GEMM with TMA loads. The epilogue does a relaxed `atomic_add` of each CTA's FP32 partial into a zeroed buffer that lives for the whole run; a ticket counter picks the last CTA, which copies the tile into the fresh output and re-zeroes the buffer. This replaced a serial scheme where the last CTA reads all S partials, and took router m=128 from 12.4 to 9.3 to 10.2 µs. The float atomic order varies, so the result is not bitwise deterministic. The 1e-4 tolerance absorbs it, and the rules do not ask for determinism.
- nathenfernandes says the router keeps FP32 weights or uses the permitted BF16 hi/lo representation. Nothing more is public.
- The vjeux file: a plain FP32 GEMV (`rgemv`) at m = 1, a split-K kernel (`sk2`) at m = 8/16/32, and a TMA split-K kernel (`rk`) at m = 128/256, presumably on the BF16 hi/lo pair (we did not check).

A CUDA-core FP32 router needs about 0.5 µs of math at m = 8, 1.1 µs at m = 16 and 2.2 µs at m = 32 at perfect efficiency (132 SMs x 128 lanes x 1.83 GHz is about 31 T FMA/s; inference). So CUDA cores can only compete at m <= 8. On the bf16 board x is BF16, so tensor cores work at every m (x times hi plus x times lo, FP32 accumulate). At m = 128/256, FP32 FMA would take about 19 µs at m = 256, so the hi + lo pair on tensor cores is the only sensible route there. The 2.4 µs `rgemv` at m = 1 shows a plain FP32 GEMV streaming the 8.4 MB at about 3.5 TB/s.

A TF32 router is not allowed. See lossless-weight-packing-and-formats.md for the rule and the error estimate.

## Dense layers at m <= 32

### What cuBLAS does at m = 1

Its main kernels are already close to HBM bandwidth. For qkv_gate at one token the main kernel takes 9.8 µs for 37.7 MB, about 3.85 TB/s of the 4.8 peak (verified from the profile of the unmodified starter, job 76c338ff8cc64f7b, 0.908x). The waste is elsewhere: a separate split-K reduce kernel (1.5 to 1.7 µs) plus launch gaps, 14.2 µs measured against 11.5 µs of GPU-busy time.

Where SGLang's time goes at m = 1, against the practical read floor of 1.7 µs + bytes / 4.66 TB/s (our analysis from published floors and our profiles, so an inference):

|Shape|SGLang µs|Read floor µs|Split|
|-|-|-|-|
|qkv_gate|14.2|9.8|1.7 launch and tail, 8.1 bytes, 4.4 cuBLAS inefficiency (1.6 to 1.7 of it is its `splitKreduce_kernel`)|
|o|6.8|5.3|1.7, 3.6, 1.5|
|pooler|21.1|16.1|1.7, 14.4, 5.0 (2.6 to 3.2 TB/s effective)|
|unpooler slice|7.1|3.5|1.7, 1.8, 3.6|
|router|10.1|3.5 (FP32, 8.4 MB)|1.7, 1.8, 6.6 (three kernels)|

So about 1.7 µs per shape is launch, ramp and tail for everyone. Everything above the floor is kernel design.

### What the submitted file picks at m <= 32

From the prepare-time comparison logged by the scored run and the check run of the submitted file (jobs 6c7a0061944e4339 and a35a24f311274d15, in that order; raw replay µs, so they include the fixed replay cost). At m = 1 the winner is a CUDA GEMV the log calls `cgemv`; at m = 8, 16 and 32 it is `sk2`. Every pick beats the default cuBLAS call in this comparison, by ratios from about 1.1x (o at m = 32) to 1.5x (qkv_gate at m = 1).

|Layer, m|Pick|Pick µs|Default cuBLAS µs|
|-|-|-|-|
|qkv_gate, 1|cgemv|12.54 / 12.13|17.31 / 18.21|
|o, 1|cgemv|8.64 / 8.32|10.14 / 9.82|
|pooler, 1|cgemv|17.25 / 17.02|24.70 / 24.74|
|unpooler slice, 1|cgemv|6.98 / 6.62|10.24 / 9.95|
|qkv_gate, 8|sk2|13.09 / 12.77|17.70 / 17.34|
|o, 8|sk2|9.02 / 8.74|10.02 / 9.60|
|pooler, 8|sk2|18.05 / 17.82|25.18 / 24.74|
|unpooler slice, 8|sk2|7.94 / 7.62|10.59 / 10.14|
|qkv_gate, 16|sk2|13.25 / 12.74|17.47 / 17.50|
|o, 16|sk2|9.12 / 8.77|10.11 / 9.79|
|pooler, 16|sk2|18.18 / 17.73|25.34 / 24.83|
|unpooler, 16|sk2|17.54 / 17.63|22.34 / 21.86|
|qkv_gate, 32|sk2|14.27 / 13.95|17.86 / 17.60|
|o, 32|sk2|9.34 / 8.99|10.18 / 9.89|
|pooler, 32|sk2|19.26 / 18.37|25.57 / 25.15|
|unpooler, 32|sk2|18.75 / 18.27|22.50 / 22.18|

### The CUDA GEMV design the 1.5x entries share

This is our reading of nathenfernandes's list and the thethings report (third-party), plus our own sketch of how the pieces fit (inference):

- Weights are repacked once in `prepare()` into the A-fragment order of `mma.m16n8k16`, so each lane streams its own 16 bytes straight from global memory with no shared-memory staging.
- One CTA owns a block of 16 output rows. Warps split K inside the CTA and reduce through shared memory in a fixed order. In-CTA K-split won; cross-CTA split-K lost to its reduction tail (+1 to 3 µs).
- Each warp keeps several 16-byte loads in flight and software-pipelines: it issues the next batch of weight loads before the current batch's MMA. That alone was worth about 2 µs on qkv_gate at m = 1.
- x is tiny and stays in L1/L2.

thethings reaches 17.1 µs for the pooler at m = 1 (about 3.9 TB/s, within about 5% of a pure read kernel) against 21.1 for SGLang. Their Triton-only GEMV plateaued near 3 TB/s, about where cuBLAS sits. Code generation is touchy: a runtime stride in the inner address arithmetic cost them up to 60%, and changing the loop shifted register allocation by 25% until they retuned the `minBlocks` launch bound. Sweep warps, K-split, loads in flight and pipeline mode per shape, on the real GPU.

### Why our Triton kernels topped out

Our measured Triton results, taken from GPU kernel time, were about 3.3 TB/s on the pooler and 4.1 TB/s on o (4.1 µs for 16.8 MB; the scored time for o at m = 1 is 6.4 µs, which includes call overhead). That matches the public ceiling. The outside evidence we collected points the same way:

|Source|What it says|
|-|-|
|PyTorch, "Accelerating Llama3 FP8 with Triton" (https://pytorch.org/blog/accelerating-llama3/)|Triton SplitK with atomic reduction beat cuBLAS up to 1.71x (FP16) at small M, but reached only about 50% of peak DRAM bandwidth (1.65 of 3.35 TB/s at M = 16). Good GEMMs reach 70 to 80%.|
|PyTorch, Hopper TMA unit (https://pytorch.org/blog/hopper-tma-unit/)|Triton with TMA raised global-memory throughput from 910 GB/s to 1.45 TB/s at M = 128 (59%). In 2024 Triton moved TMA descriptors with extra host-to-device copy kernels, which hurt latency; newer APIs may differ (unverified).|
|Decompose-K in Triton (https://shreyansh26.github.io/post/2026-06-21_decompose-k-triton/)|Two-stage split-K (FP32 partials, then a small reducer) on BF16 H100: parity to +2% against Inductor's custom path. A naive reducer was 0.92x.|
|IBM split-K paper (https://arxiv.org/html/2402.00025v2)|About 124% average speedup over a data-parallel Triton baseline on H100 for skinny M = 1 to 16. Not measured against cuBLAS.|
|AutoMegaKernel preprint (https://arxiv.org/pdf/2606.09682.pdf)|A hand-written BF16 GEMV reached about 63% of peak bandwidth while cuBLAS reached about 90%, on L4, L40S and A10G rather than H100.|
|"Memory-Bound but Not Bandwidth-Limited" (https://arxiv.org/pdf/2605.30571)|Batch-1 decode on H100 reaches a falling fraction of peak bandwidth as peak rises; launch overhead is a large part, and CUDA graphs hide only some of it.|
|Cohere megakernels (https://cohere.com/blog/megakernels)|Decode is GEMV-dominated and set by streaming weights from HBM. Cites a fused Llama-1B kernel at 78% of H100 bandwidth at batch 1. We did not open the cited work.|
|mistral.rs GEMV (https://docs.rs/crate/mistralrs-quant/latest/source/src/gemv/mod.rs)|A production BF16 GEMV for batch 1 to 8 using vectorized bf16x2 loads and `__ldg`. No bandwidth numbers.|

Our reading: custom BF16 GEMVs often land at 60 to 80% of peak bandwidth and cuBLAS is already near 80%, so the gain comes from removing launches and the separate reduce, and from reading fewer bytes. thethings reports a Triton ceiling of about 3 TB/s, and our own v3 landed near it on the pooler.

### Bytes in flight

The GTC 2010 talk on better performance at lower occupancy (https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf, slides 1 to 40) gives the rule: needed parallelism is latency times throughput, and for memory that means keeping latency x bandwidth bytes in flight. Threads stall on use, not on the load, so more independent and wider loads (float4) stand in for occupancy: 87% of pin bandwidth at 8% occupancy with 8 float4 per thread.

What that means for the H200 is our estimate. If DRAM latency is 600 to 800 ns (an assumption, we read no source for it), the chip needs 3 to 4 MB in flight, roughly 25 to 30 KB per SM. Our early small-m Triton config (BM16, BN16, BK64, 3 stages, 128 programs, about one per SM) has hi + lo tiles of 4 KB per stage, so about 8 to 12 KB in flight per SM, which is probably too little. The RTX 3050 Ti we test on has about 224 GB/s over 20 SMs and needs only 6 to 9 KB per SM, so every config saturates it and a local sweep cannot see this. Configs with more bytes in flight (BN 32, BK 128, 4 stages, 16 KB of weights per stage) were within noise of the best locally, so they cost nothing there and should help on the H200. v3 later used configs of this kind (BK 128, 4 to 5 stages, BN 32 on the pooler), and its pooler at decode 1 still ran at cuBLAS parity on the judge (21.2 against 21.1 µs, job a58c6e3b903744ea), so the estimate did not pay off there.

### What our Triton kernels compile to

We compiled v3's partial kernel for sm_90 offline (Triton 3.9 locally, the judge has 3.8):

|Layer|Registers|Spills|Shared memory|16-byte async copies|MMA|
|-|-|-|-|-|-|
|unpooler|42|0|60 KB|70|HMMA.16816 (mma.sync)|
|o|54|0|60 KB|110|HMMA|
|pooler|32|0|48 KB|30|HMMA|
|qkv_gate|42|0|60 KB|40|HMMA|

No TMA and no wgmma: Triton emitted Ampere-style `cp.async` plus mma.sync for these BM = 16 tiles. There are no spills. With one CTA per SM (128 to 144 programs over 132 SMs), 48 to 60 KB is in flight per SM, and the levers are more stages, more CTAs per SM, or TMA through tensor descriptors. One pitfall: an offline compile without the pointer divisibility hint (`tt.divisibility` 16) emits scalar 16-bit loads, which is not what the real launch does.

## Split-K reduction, memset and launch gaps

- No memset or `zero_()` in `kernel_fn`. A PyTorch forum thread (https://discuss.pytorch.org/t/slowdown-in-cuda-graph-execution-using-cudamemsetasync-vs-a-fill-kernel/211124) reports a 4-byte memset node taking 1.7 µs on H100 SXM5 followed by a gap of almost 13 µs in a real profile; standalone, 1000 ops took 1490 µs with memset against 1045 µs with a fill kernel. The cause is unconfirmed. Any workspace or counter should be created once in `prepare()` and left clean by the kernel itself.
- Atomic split-K into the output needs a zeroed output on every call, which means a memset node. GemLite's split-K design implies the same (inference; its page does not say how it zeroes). That makes atomic split-K a poor fit.
- Do launch gaps between our own kernels count? Yes. The scored time is the whole graph replay less about 4 µs, and thethings measured about 0.65 µs before a kernel's first CTA and about 0.9 µs after its last. v3's split-K shapes paid `_reduce` (1.1 µs) plus that gap, which a single-kernel design removes. Folding the reduce into one launch is worth about 1.5 to 2 µs per split shape, +0.03 to 0.04x overall, and moot once a CUDA GEMV replaces those kernels.
- Last-block reduction in Triton is covered in its own subsection below.
- A forum report (not our measurement) says a float `red.add` into a self-zeroing workspace failed about 1 in 10 to 20 thousand replays. We have no details, so read that as a warning. A soak of 2,000 replays would catch such a bug only 10 to 18% of the time; testing it properly takes 30,000 or more replays on a GPU. A deterministic last-CTA reduction or a second small reduce kernel is the safer choice.
- Semaphore split-K with spinning on other CTAs (see the Tilus tutorial, https://nvidia.github.io/tilus/stable/tutorials/matmul-ampere/matmul/matmul_v5.html) can deadlock if the CTAs are not all co-resident. A last-arriver counter avoids spinning.
- Persistent stream-K GEMV with a TMA ring cost about 1.3 µs of handshake and lost to the simple GEMV (13.0 against 12.2 µs, thethings). Thread-block clusters with DSMEM reduction took the router at m = 1 from 6.0 to 5.3 µs but made m = 32 worse, about +0.5% overall. Clusters of 4 place only 120 CTAs on an H200, and a remote `mbarrier.arrive.release.cluster` costs about 1 µs.
- PDL between our own kernels is worth at most 0.5 to 1 µs per extra kernel. The Triton docs expose `gdc_wait()` and `gdc_launch_dependents()` (on main; presence in 3.8 unverified) and a `launch_pdl` launch option (spelling inferred from the driver code). Our v3 check profile for pooler at m = 16, which runs on cuBLAS, shows its split-K kernel and reduce overlapping inside the judge's captured graph, so overlap survives capture, but the profile does not say whether PDL or a second stream did it. Prefer one kernel.

### Last-block reduction in Triton

The scheme: every K-split CTA writes its FP32 partial, one thread does `atomic_add(counter, 1)`, and the CTA that sees the final count sums the partials. We went back and forth on whether this is safe in Triton 3.8.

What Triton exposes (verified in its docs): `atomic_add` with `scope` in cta, gpu or sys and `sem` in acquire, release, acq_rel or relaxed, defaulting to `acq_rel` and `gpu`. `tl.debug_barrier()` is a block-level `bar.sync` and nothing more. There is no device-scope fence primitive.

Our first argument was that a single shared counter only synchronizes the last block with whichever block wrote it most recently, the usual multi-producer flaw, so we shelved the idea. That argument was wrong. The CUDA programming guide's memory model says an acquire synchronizes with a release when it reads the released value or a value written by any side effect in the release sequence headed by it, and later read-modify-write atomics on the same object continue that sequence. So a chain of `acq_rel` RMWs at gpu scope on one counter does synchronize the last block with all S producers, provided each block puts a block barrier between its threads' partial stores and the atomic, and the last block puts another barrier after its acquire. The cuda-samples `threadFenceReduction` needs `__threadfence()` only because its `atomicInc` is relaxed.

Still unverified: whether Triton 3.8 lowers the scalar atomic to a single `atom.acq_rel.gpu` issued by one thread after the barrier. That can be read from the PTX offline. Also, a pass on the RTX 3050 Ti (sm_86) cannot prove race freedom on the H200, which has a different SM count, L2 and CTA residency. We parked the idea, since it is worth only 0.03 to 0.04x. Forum threads on the fence requirement: https://forums.developer.nvidia.com/t/question-related-threadfence/40941 and https://forums.developer.nvidia.com/t/is-threadfence-useful-at-all/244413. The Triton barrier page is https://triton-lang.org/main/python-api/generated/triton.language.debug_barrier.html.

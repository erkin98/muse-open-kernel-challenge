# Techniques considered and verdicts

This file lists every technique we looked at for muse_bf16_proj, with what the evidence says and where it ended up.

Verdict words: "in the file" means it runs in the file we submitted, which is the public vjeux solution plus a Triton router candidate (see judge-results.md). "Tried, lost" means we built or measured it and it did not win. "Not tried" means we read about it and did not build it. "Not allowed" means the rules forbid it. "Parked" means we decided not to spend time on it. Evidence is from our own measurements unless the row says third-party. Estimated savings are our arithmetic, in µs per shape per load, and the +x figures are whole-score effects relative to our v3 (1.139x).

## Dense layers at small m

|Technique|Evidence|Verdict|
|-|-|-|
|CUDA GEMV with weights repacked into MMA fragment order, 16 bytes per lane, software-pipelined, in-CTA K-split|Third-party: pooler at m=1 in 17.1 µs (3.9 TB/s). Our estimate against v3: pooler -4.1, qkv_gate -2.8, unpooler slice -2.1, unpooler m=16 -0.7, o -0.5 µs, about +0.12x. Applying nathenfernandes-level pooler kernels alone to v3 would give 1.273x|In the file (vjeux's kernels). Our own clean-room CUDA version reached about 1.25x overall|
|Triton GEMV and split-K kernels (our v3)|About 3.3 TB/s on pooler, 4.1 on o (kernel time); thethings reports the same ~3 TB/s Triton ceiling|1.139x; superseded|
|Software pipelining of weight loads (issue the next batch before the current MMA)|Third-party: about 2 µs on qkv_gate at m=1|Used by the public 1.5x entries; we did not check it in the file|
|More bytes in flight per SM (BN 32, BK 128, 4 stages)|Within noise on the RTX 3050 Ti, which cannot show the effect. Little's law said it should matter on the H200 (inference)|Used in v3; its pooler at decode 1 still ran at cuBLAS parity on the judge (21.2 against 21.1 µs)|
|Autotune in `prepare()` with cold-L2 graph replays|The vjeux file does this per layer and m, behind an FP64 self-test|In the file|
|Dispatch by ranges of m with masked runtime-m kernels|Guards against overfitting the scored m values|Recommended; see judge-environment-and-validation.md|
|GemLite-style kernel families (GEMV at M=1, split-K GEMM at M=2 to 32, plain GEMM above, persistent split-K, per-GPU cached autotune; https://github.com/mobiusml/gemlite)|Quantized weights, so its dequantization does not transfer. Its atomic split-K needs a zeroed output each call. BF16 outputs were 5 to 7% slower at M=1 to 64 from an FP32 fallback in its atomic add|Design reference only|
|Two-stage split-K (FP32 partials plus a small reducer)|Parity to +2% against Inductor on H100 BF16; a naive reducer was 0.92x (https://shreyansh26.github.io/post/2026-06-21_decompose-k-triton/)|We kept this shape in v3 (`_partial` plus `_reduce`); costs about 1.1 µs plus a launch gap|
|m=128 pooler as a Triton tile of 64 weight rows x 128 tokens x BK128, split-K 2|Third-party: 23.3 against cuBLAS's 25.4 µs|Not tried|

## Lossless compression

|Technique|Evidence|Verdict|
|-|-|-|
|12-bit format: raw low byte, sign plus 3-bit exponent window, exact exception list|Entropy 10.46 bits per weight; third-party gains of 13 to 20% on big decode shapes; about +0.07 to 0.08x|In the file|
|Fixed 3-bit exponent code over only 8 exponents|About 1.3% escapes in our CPU measurement|Too many escapes to patch cheaply|
|Entropy or variable-length coding|Entropy is 1.5 bits under the fixed code, but decode cost in a GEMV loop is unknown|Not tried|
|Compressing the router's hi/lo pair|Third-party: slower, the router is too small and latency-bound|Tried by others, lost|
|Compression at m=32 on qkv_gate and o; unpooler slice|Third-party: slower (the slice by +2%). The submitted file does use the 12-bit path at m=32 on all four dense layers and for the unpooler slice at m=8, and it beats SGLang there (1.15 to 1.47x at decode 32)|Third-party claim not reproduced; the path is in the submitted file|
|12-bit weights unpacked inside a wgmma GEMM at m=128|alygg77 lost 2.4 to 4.1 µs on o and unpooler and 12 µs on pooler; our rs128 build took 1.6 to 2x the raw replay time of the existing picks|Tried, lost|

## Router

|Technique|Evidence|Verdict|
|-|-|-|
|One fused kernel at every m (plain FP32 GEMV at m=1, split-K at m=8/16/32, TMA split-K with BF16 hi/lo at m=128/256)|Our scored run: 3.84 to 7.20 µs against SGLang's 10.6 to 26.2|In the file|
|Our Triton router (partial plus reduce)|Lost the prepare-time comparison at all six m, for example 14.30 against 9.57 µs raw replay at m=128|Tried, lost|
|Triton split-K router at m=128/256 with relaxed `atomic_add` into a persistent zeroed buffer, ticket counter, last-CTA copy-out and re-zero|Third-party: m=128 from 12.4 to 9.3 to 10.2 µs. Not bitwise deterministic, within the 1e-4 tolerance|Not adopted|
|`dsv3_router_gemm` design (one CTA per expert row, 16-byte loads, warp shuffle then shared memory, PDL)|K=7168 BF16 is hard-coded, M <= 16|Design reference only|
|FP32 CUDA-core FMA router|Math time 0.5 µs at m=8, 1.1 at m=16, 2.2 at m=32; 19 µs at m=256|Only sensible for m <= 8|
|Cluster / DSMEM reduction|Third-party: m=1 6.0 to 5.3 µs, m=32 worse, about +0.5% overall|Skip|
|K-split 16 for the m=128 router (grid 128 instead of 96)|At most 0.1% of the score|Not done|
|Whole-score ceiling for all six router shapes|3.2%, about 1.6% at half capture|Router work deprioritized|
|TF32 router|Changes the weights' precision; error about 2.8e-4 relative L2 against a 1e-4 tolerance|Not allowed|

## Reduction, launch gaps and synchronization

|Technique|Evidence|Verdict|
|-|-|-|
|In-CTA K-split (warps split K, reduce through shared memory)|Third-party: beat cross-CTA split-K, whose reduction tail costs +1 to 3 µs|Used by the public 1.5x entries; we did not check it in the file|
|Semaphore split-K that spins on other CTAs|Deadlocks if the CTAs are not co-resident|Avoid|
|Last-block counter reduction in Triton|Memory-model argument says an `acq_rel` RMW chain plus block barriers suffices; Triton has no device fence and its lowering is unverified. Worth +0.03 to 0.04x|Parked|
|Atomic split-K into the output|Needs a zeroed output every call, so a memset node|Avoid|
|memset or `zero_()` in `kernel_fn`|A forum thread reports a 1.7 µs memset node followed by a ~13 µs gap on H100|Avoid; make workspaces in `prepare()`|
|Float `red.add` into a self-zeroing workspace|A forum report, not our measurement, of failures at about 1 in 10 to 20 thousand replays|Avoid|
|Persistent stream-K GEMV with a TMA ring|Third-party: 1.3 µs handshake, 13.0 against 12.2 µs|Skip|
|PDL between our own kernels|At most 0.5 to 1 µs per extra kernel; presence in Triton 3.8 unverified|Prefer one kernel|
|L2 prefetch (`cp.async.bulk.prefetch.L2`)|Third-party: never helped|Skip|
|TMA multicast of x across a 2-CTA cluster|Third-party: gave nothing|Skip|
|Warp specialization on its own|Third-party: no standalone win; comes with wgmma kernels|Not a lever|
|Inline PTX in general|NVIDIA's own write-up calls it a last resort, 7 to 14% on a GH200 epilogue (https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/)|Only where needed (`prmt`, `lop3`, fences)|

## Large m (128 and 256)

|Technique|Evidence|Verdict|
|-|-|-|
|TMA + wgmma kernels (gq, gqx) for qkv_gate and pooler at m=128|Beat cuBLAS in the file: qkv_gate 18.34 against 20.29, pooler 24.86 against 28.96 µs raw replay|In the file|
|CUTLASS kernel for pooler at m=256|31.14 against 35.58 µs raw replay|In the file|
|cuBLAS on a K-major copy of W|Picked for o at m=128 and unpooler at m=128 and 256 by small margins|In the file|
|cuBLASLt algorithm pinning|Every configuration is algorithm id 66 on our H200 and none beat the existing pick on any shape. Third-party: about -2% best case|Tried, lost|
|gemm90 split-K wgmma GEMM|14% to 107% slower than the existing pick on all eight shapes|Tried, lost|
|Six more gq/gqx instantiations (bigger token tiles, two consumer warpgroups, other clusters)|All slower, for example qkv_gate m=128 at 29.6 against 18.5 µs|Tried, lost|
|swap-AB register-A wgmma with 12-bit weights (rs128)|Pooler m=128 43 to 46.5 µs against 25 to 26|Tried, lost|
|Third-party wgmma + TMA and swap-AB kernels|Tied or lost to cuBLAS at m=128/256; ingest-bound at ~85 GB/s per SM|Others tried, lost|
|CUTLASS stream-K or split-K for qkv_gate at m=256|36 output tiles on 132 SMs is the structural problem; stream-K workspace setup may add a memset|Not tried|
|128 weight rows x 256 tokens tile plus split-K|Shapes are L2-traffic bound (~14 TB/s)|Not tried beyond gemm90|
|Triton TMA descriptors|Third-party: GEMV 12.6 against 13.9 µs, not enough; 2024 Triton added copy kernels for descriptors|Not tried|

## Libraries and DSLs

|Technique|Evidence|Verdict|
|-|-|-|
|DeepGEMM `bf16_gemm_nt`|Third-party: identical to cuBLAS, it dispatches there|No gain|
|CUTLASS 4.5 sm90 sweep of 12 configurations|Third-party: slower than cuBLAS at m=128/256|No gain (the one custom win is the m=256 pooler pick)|
|FlashInfer `mm_bf16` / TGV|SM100 and later only (verified in v0.7.0.post1 docs). Its SM90 GEMM is FP8|Not usable|
|quack-kernels 0.6.5|In the image; small-M performance unknown|Not tried|
|CuTe DSL 4.8.0, TileLang 0.1.12|No public evidence of a win on these shapes|Not tried|
|Gluon|Experimental|Skipped|
|Offline SASS inspection of our Triton kernels|Showed cp.async plus mma.sync, no TMA or wgmma, no spills (details in router-and-small-m-kernels.md)|Done|

## Not allowed

|Technique|Why|
|-|-|
|L2 persistence or `evict_last` to survive the cold-L2 flush|Games the measurement; thethings refused it too|
|TF32 for the router|Not the permitted hi + lo exception, and too inaccurate|
|FP8 for BF16 weights|Changes precision|
|Compressed, encoded or base64 source|Forbidden since the Nov 2 rule update|
|Starting processes (`subprocess`, `nvcc --version`)|Gets a submission rejected|

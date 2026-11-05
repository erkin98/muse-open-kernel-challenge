# Linear exp2, deeper pipelines and cold-code warming

Our first two judge checks on top of vjeux's public entry (a cheaper exp2 with deeper pipelines, then cold-code warming) are written up here, along with how we validated kernels without a GPU and the ideas we dropped before running them.

Baseline throughout: vjeux's entry run unchanged, sha256 cf04259b008238a7cdb3eb7583202c908746142863a454a5e7bfc466d7f88e6d, job c20df44ab71d4005, 2.605x (95% interval 2.565 to 2.644). A second run of the same bytes, job 705c7dd9f34446dd, gave 2.589x (2.550 to 2.637). Both runs are the "base" in the table below.

## Validation without a GPU

We had no local GPU that could run these sm_90a kernels, so every candidate went through a static route before it was allowed to use a judge run:

- NVRTC 13.2 (nvcc's front end, from the NVIDIA pip wheel) compiles the device code to PTX for compute_90a, instantiating all 20 `attn_kernel<L, FLAT, MODE>` variants.
- `ptxas -arch=sm_90a` (12.9) reports registers, spills, stack and wgmma warnings.
- `cuobjdump -sass` gives the instruction mix. We read the SASS around every changed region.
- A CPU model of the q_len 8 path, using the judge's own input generator and checks, covers every q_len 8 test shape plus scored-like verify shapes, so numerics are checked against the tolerance before any run.

One trap: a build with clang reported serialized wgmma in the flat q_len 8 kernel, and the NVRTC build does not. The serialization was a clang artifact, and nvcc-family output is the one to trust for what the judge compiles.

Mode map of the kernel, used below: mode 13 is d1s (32 CTAs, last-CTA finalize), 15 is v1s (33 CTAs, cooperative), 11 is d1f and v1f (252 CTAs, cooperative, 8 workspace copies), 1 is d16s and d32s, 3 is v16s and v32s (cooperative), and the flat kernel (mode 0) serves all batch 16 and 32 full cases.

## Check 1: linear-mantissa exp2 and deeper pipelines

Job 24a52be614264969, file sha256 bfd232073ef8f01ccdea03b7d2c1c81205b6237bc2ca73d207e93701c357fb53. Two unrelated changes went in together, deliberately, because they touch different shapes.

1. Cheaper exp2 for P. The new helper builds 2^x with a linear mantissa from one FFMA plus one LEA, with no MUFU. In the q_len 8 packing paths it runs first and sets a flag bit when the input falls outside the approximation's range; flagged lanes repack through the exact path. SASS per pair: FFMA, `LEA R, R, 0xff800000, 0x8`, `F2FP.SATFINITE.E4M3`. The rules explicitly allow an approximated exp2 for P in FP8, so this was within bounds.
2. Deeper cp.async pipelines for q_len 1: flat from 3 stages to 5, per-request from 3 to 6. Shared memory rose from 60 to 93 KB (flat) and 56 to 105 KB (per-request); two CTAs per SM still fit.

Static results: 0 spills in all 20 kernels, no wgmma warnings, per-request q_len 8 at 127 registers or fewer (the bound is 128).

CPU numerics, relative L2 error against the judge's bound, exact-E4M3 P versus linear-mantissa P. The approximation does not move the error materially:

|test|exact P|linear P|bound|
|-|-|-|-|
|t3|0.0760|0.0754|0.15|
|t6|0.0929|0.0939|0.15|
|t8|0.0836|0.0838|0.15|
|t9|0.0922|0.0924|0.15|
|t10|0.0780|0.0804|0.15|
|t13|0.0707|0.0700|0.15|
|t14|0.0717|0.0735|0.15|
|t16|0.0389|0.0388|0.106|
|verify-sliding-like|0.0693|0.0711|0.15|
|verify-full-coherent-like|0.0311|0.0321|0.090|

The split by shape: verify cases test the exp2 change (their stage counts are unchanged), decode 16 and 32 full test the flat 5-stage pipeline, decode 1 full and decode 32 sliding test the 6-stage per-request pipeline, and decode 1 and decode 16 sliding are close to controls.

Result: 2.550x (95% interval 2.512 to 2.588), 16/16 tests, judge time 1,477.0 us. Worse than the base. The deeper q_len 1 pipelines cost decode 16 full 86.6 to 90.6 us and decode 32 full 165.6 to 170.3 us, beyond the base file's two runs (86.6 and 87.0; 165.6 and 165.8). The linear exp2 gave nothing on verify full: v16f 101.0 to 101.1 us, v32f 184.1 to 183.6 us. Both changes were dropped and the next candidate started again from the base.

## Check 2: warming the finalize code before it is needed

Job a58eeeeaf93c4bed, file sha256 7e8dff25b2193b69b48e71b514319ba11b85803d39192e5446354438d11a7dcb, built from the base (not from check 1).

The idea comes from Kushagrabainsla's public notes. Every judge replay starts with a flushed L2, and the finalize runs once per launch after the split counter, so its instructions come from DRAM on the critical path. The fix is to make the finalize an out-of-line function. While one thread does the counter atomic (and, on cooperative grids, the spin), the other warps call the finalize once in a dry mode where every global load and store is predicated off in inline PTX, so the instructions are fetched into the instruction cache with no memory traffic. After the barrier the real call runs warm.

Design choices and why:

- A `__noinline__` callee keeps exactly one copy of the code that runs twice. A two-pass loop could be peeled by the compiler into two copies, which would warm nothing. SASS confirmed two calls to one target.
- Every global access in the dry pass is predicated in PTX rather than skipped by a branch, so loads cannot stall it and nothing is written, while control flow stays identical to the real pass.
- For q_len 8, the first build put the eight-copy path in a callee that needed 128 registers and spilled (52 to 200 bytes of spill, 56 to 176 bytes of stack in the kernels). We kept the callee lean, which limited warming to the one-copy cooperative path. Verify 1 full (8 copies) and the non-cooperative q_len 8 path kept their original code.
- The counter arrival moved from thread 0 to lane 0 of the last warp, so the low warps (which hold the finalize rows) were the ones doing the dry pass.

Prior evidence we relied on: Kushagrabainsla's merge going from 6,928 to 2,524 cycles, and a kernel going from 10.8 to 8.9 us; arashattar's report that 32 KB of straight-line code costs about 1 us cold and that code size moves timings by 0.2 to 3 us; NVIDIA's 2024-08-08 blog post on reducing instruction cache misses ("No Instruction" stalls). vjeux had tried a version of this in his history and left it off. That version still did its loads in the dry call and used a force-inlined finalize, so a second inlined copy would have warmed nothing; it did not rule out this design.

Static results: 0 spills and no stack in every kernel. The flat kernels and the q_len 8 non-cooperative and cluster kernels compiled to byte-identical machine code, which makes d16f, d32f, v16f and v32f in-run controls. q_len 1 per-request registers rose from 96-124 to 172-180 (still within 128 threads x 2 CTAs), q_len 8 cooperative 96-115 against 96-119. The real pass does the same loads and arithmetic in the same order as before, so outputs are bit-identical.

What we expected: more than 0.3 us off d1s and v1s together. Per microsecond saved, the score model gives +2.0% for d1s, +1.5% for v1s, +0.7% for d1f, +0.5% for v1f, and +0.2 to 0.45% for each of d16s, d32s, v16s and v32s.

Result: 2.576x (95% interval 2.537 to 2.614), 16/16, judge time 1,462.2 us. No effect; warming was dropped.

## The two checks per shape (microseconds, ours)

The base-run values for each shape, and the run-to-run difference between them, are in [judge-runs-and-noise.md](judge-runs-and-noise.md).

|shape|check 1|check 2|
|-|-|-|
|d1s|5.2|5.1|
|d1f|11.4|11.0|
|d16s|8.7|8.4|
|d16f|90.6|86.5|
|d32s|10.8|10.2|
|d32f|170.3|165.5|
|v1s|7.0|6.8|
|v1f|14.2|14.2|
|v16s|11.7|11.5|
|v16f|101.1|102.1|
|v32s|15.0|14.7|
|v32f|183.6|186.2|

Base run 1 is job c20df44ab71d4005 and base run 2 is job 705c7dd9f34446dd. We read only the batch-1 shapes and the four controls for check 2; the other sliding rows are listed for completeness. The controls ran 0 to about 1% above base run 1 (v16f and v32f about +1%, d16f and d32f unchanged), and d1s and v1s did not drop by 0.3 us, so there was no warming effect to find. The reading rule we set beforehand was a drop of at least 0.3 us on both d1s and v1s after dividing out the control offset.

Two explanations remain open: the finalize may simply not be cold enough on these shapes for warming to matter, or the dry pass may not overlap the counter window well enough. We could not tell them apart without a profiler, so we did not extend the idea.

## Ideas considered and not run

- Same-address `red.v4.f32` serialization (alygg77 measures about 17 ns per op) at d1s and v1s, where 32 or 33 CTAs share one workspace copy, would cost up to about 0.5 us of queueing. Four copies for d1s were tried later in a separate check and were slower (5.53 us against 4.94 to 5.17), see [ideas-tried-and-results.md](ideas-tried-and-results.md). vjeux's history shows the opposite move for v1s (several copies down to one) gaining, from 9.8 to 9.6 us in that build.
- Warming the tile body with a dry first pass during the K/V wait for modes 13 and 15. It would only be worth doing if the finalize warm-up had paid, and it is invasive (predicated cp.async wait, state reset).
- Growing the scratch buffer. alygg77 reports that 3 MB of extra scratch slowed unchanged shapes by 0.5 to 1.3 us, so we kept scratch fixed.
- Closed by other people's measurements, so no judge run spent: more pipeline stages or lookahead (our check 1 agrees), polynomial or FMA-pipe exp2 (alygg77 tried degree 5; arashattar's variants measured +24 to +105 us as reported; our linear exp2 gave nothing), full-verify pair balancing, producer-side V transpose, single-CTA and cluster merges, page sorting, finalize caps, wave balancing, folding Q preparation, and a producer-share probe. The same discussion puts full decode at 86 to 90% of the random 128-byte read ceiling (about 4.05 to 4.1 TB/s), which leaves little there without a new design.

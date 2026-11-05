# Large-m plan and results

This file covers our attempts to speed up the m = 128 and m = 256 verify loads on top of the vjeux file: why those shapes are the weak spot, the plan, five judge checks, and the results, which were all negative.

Labels: "verified" means we measured it or read the primary source, "third-party" means another team reported it, "inference" is our own reading. Times marked "raw replay" come from the prepare-time comparison inside the file; they include the fixed graph-replay cost and are only good for ordering candidates.

## Why verify m = 128 and 256 is the weak spot

Per-load ratios (SGLang's summed layer time over ours) for the scored run of the submitted file, job 6c7a0061944e4339:

|Load|decode 1|decode 16|decode 32|verify m=8|verify m=128|verify m=256|
|-|-|-|-|-|-|-|
|Ratio|1.730|1.563|1.486|1.912|1.225|1.169|

The geometric mean is 1.491, matching the judge. The two large verify loads are the drag. The o layer runs at cuBLAS speed there (0.99 to 1.00x), and qkv_gate, pooler and unpooler gain only 1.02 to 1.13x.

What lifting those two loads would be worth (our arithmetic):

|Change|Score|
|-|-|
|verify 128 and 256 loads to 1.35|1.553x|
|both to 1.45|1.590x (about rank 70 on the Nov 4 board)|
|both to 1.60|1.643x|
|decode 16 and 32 loads +8%|1.530x|

At these sizes the work is compute-heavy. Pooler at m = 256 is 17.2 GFLOP, about 17 µs at peak, against 27.5 µs for us and 30.8 for SGLang. nathenfernandes's published pooler times are 20.5 µs at m = 128 and 24.2 µs at m = 256, so that much is reachable by someone.

Two structural notes from what we measured:

- The CUTLASS pooler kernel at m = 256 moves about 384 MB through L2 in 26.6 µs, and cuBLAS qkv_gate at m = 256 about 250 MB in 19.5 µs. That is roughly 13 to 14 TB/s from L2 to the SMs (our plan's estimate, not derived from a counter), so these shapes are probably traffic-bound, and the lever is tiles with more reuse (128 weight rows x 256 tokens plus split-K).
- qkv_gate at m = 256 has only 36 output tiles on 132 SMs if the tile is 64 weight rows by 256 tokens (our assumption; cuBLAS's own kernel uses other tiles), which is a structural problem for any tile-per-CTA scheme.

What thethings found at m = 128/256 (third-party): cuBLAS's pooler at m = 128 uses 64x64 tiles, so every weight byte is fetched by two CTAs, about 2.6 TB/s effective. A Triton tile of 64 weight rows x 128 tokens x BK 128 with split-K 2 beat it, 23.3 against 25.4 µs. cuBLAS reaches about 340 TFLOP/s on the pooler at m = 128 and 590 at m = 256. The best of cuBLASLt's 8 heuristic algorithms was about 2% better summed over the four dense layers (pooler m=128 alone went from 25.0 to 23.0 µs). Their own TMA + wgmma kernels lost to cuBLAS; a swap-AB wgmma kernel (weights as the M = 64 side, tokens as N) ties or loses because its main loop is ingest-bound at about 85 GB/s per SM, as x is re-read from L2 for every row tile. DeepGEMM's `bf16_gemm_nt` was identical to cuBLAS (it dispatches there), and a CUTLASS 4.5 sweep of 12 sm90 configurations was slower than cuBLAS.

## What runs at m = 128 and 256 in the submitted file

From the prepare-time comparison logged by the submitted file's scored run and check run (jobs 6c7a0061944e4339 and a35a24f311274d15, in that order; raw replay µs). "torch" is the default cuBLAS call and "torchT" is cuBLAS on a K-major copy of W:

|Shape|Pick|Pick µs|Default cuBLAS µs|
|-|-|-|-|
|qkv_gate m=128|gqx (TMA + wgmma)|18.34 / 17.79|20.29 / 19.78|
|o m=128|torchT|10.62 / 10.40|10.78 / 10.43|
|pooler m=128|gq (TMA + wgmma)|24.86 / 24.64|28.96 / 28.16|
|unpooler m=128|torchT|22.78 / 22.18|24.10 / 23.39|
|router m=128|rk|9.57 / 9.18|19.68 / 19.39|
|o m=256|torch / torchT|12.61 / 12.16|12.61 / 12.26|
|pooler m=256|CUTLASS|31.14 / 30.75|35.58 / 34.50|
|unpooler m=256|torchT|27.52 / 27.52|28.19 / 27.81|
|router m=256|rk|10.56 / 10.91|20.77 / 20.26|

qkv_gate at m = 256 has no entry in the log, meaning no custom candidate in the submitted file, so it stays on cuBLAS (23.30 raw replay in the third check). So the only custom wins at the large sizes are the gq/gqx kernels at m = 128, the CUTLASS pooler at m = 256 and the router. Everything else is cuBLAS, which is why the ratios sit near 1.0 to 1.15.

## The plan

Step 1 was a cuBLASLt algorithm search:

- A C++-only `load_inline` extension against cublasLt (no nvcc), built right after the main extension and before the CUTLASS thread, so two builds never overlap. A failed build only drops the route.
- In `prepare()`, for qkv_gate, o, pooler and unpooler at m = 128 and 256, and for both weight layouts (row-major with TRANSA = T, and a K-major copy with TRANSA = N): take the heuristic's top-N plus the best configuration per algorithm id, deduplicated.
- Keep only configurations with split-K 1 or FP32 split-K reduction, reading the config attributes back after the query instead of trusting the preference mask.
- Capture each configuration into a CUDA graph and reject any whose graph has a node other than a kernel (memset, memcpy, allocation), or whose capture fails.
- Each survivor becomes a candidate that is checked against FP64 and then timed with the file's cold-L2 graph replays. A candidate has to beat the existing route by 2% before it replaces it, and it only applies at the m it was tuned for; other m fall back to the default call.
- Keep a change only if the weak shapes improve by more than about 3% and no other shape regresses.

A local glue test on the RTX 3050 Ti passed (17.8 s build with the host compiler, 20 to 30 configurations per shape, all kernel-only under capture, FP32 split-K configurations accurate to about 2e-4). Local timings meant nothing.

Step 2 depended on what step 1 showed: if cuBLASLt gave little on qkv_gate at m = 256, add TMA/wgmma table rows with a 256-token tile, or try a CUTLASS split-K or stream-K configuration.

## Checks and results

|Check|Job|Change|Score (95% interval)|Verify m=128 / m=256 ratio|
|-|-|-|-|-|
|baseline, scored run|6c7a0061944e4339|none|1.491x|1.225 / 1.169|
|baseline, check run|a35a24f311274d15|none|1.520x (1.497 to 1.543)|1.260 / 1.176|
|1|329e7f847e0d4b45|cuBLASLt configurations|1.497x (1.475 to 1.520)|1.231 / 1.152|
|2|31e944e4fdba45fb|check 1 plus gemm90 and torchT|1.484x (1.462 to 1.506)|1.230 / 1.153|
|3|1dd4d6bbe0214964|base file plus six new gq/gqx instantiations|1.492x (1.470 to 1.515)|1.241 / 1.162|
|4|c440acdd631e4328|rs128 variant|1.505x (1.483 to 1.528)|1.236 / 1.172|
|5|635004eab6f9489a|rs128 variant, second build|1.4915x (1.469 to 1.514)|1.245 / 1.169|

The last two ratios are our own sums from the judge's per-shape rows. No route changed in any of them, so all differences are noise. The same unchanged file scored 1.491x and 1.520x; alygg77 reports four runs of one file between 1.526x and 1.550x.

### Check 1: cuBLASLt configurations

On the H200 every cuBLASLt configuration is algorithm id 66, 17 unique per weight layout, with tile ids 17 to 28, 135, 184 and 208, cluster ids 3 to 9 and split-K 1 to 5. All were kernel-only under capture and accurate. None beat the existing pick except, marginally, unpooler at m = 256 (27.36 against 27.62 µs, inside the 2% margin). Best cuBLASLt against the pick, raw replay µs:

|Shape|cuBLASLt|Existing pick|
|-|-|-|
|qkv_gate m=128|19.81|gqx 18.05|
|o m=128|10.66|10.46|
|pooler m=128|26.82|gq 25.09|
|unpooler m=128|22.88|22.88|
|qkv_gate m=256|23.36|23.10|
|o m=256|12.54|12.42|
|pooler m=256|32.45|CUTLASS 31.49|
|unpooler m=256|27.36|27.62|

No configuration beat the existing pick on any shape, though against the plain cuBLAS call a few did (pooler m=128 26.82 against 29.02 µs, pooler m=256 32.45 against 35.30), and those shapes already run on gq and CUTLASS. A second run gave identical picks. Enumerating explicit tile x cluster x split-K combinations within the same algorithm id did not seem worth a check.

### Check 2: split-K wgmma GEMM and cuBLAS on a K-major copy

Check 1 plus "gemmx" (the file's existing split-K wgmma GEMM, gemm90, already compiled for every layer) as a candidate for the BF16 layers with the config's split count, and "torchT" for qkv_gate and pooler. gemmx lost to the existing pick by 14% to 107% on every shape:

|Shape|gemmx|Existing pick|
|-|-|-|
|qkv_gate m=128|22.14|gqx 18.34|
|o m=128|18.18|10.82|
|pooler m=128|28.61|gq 25.09|
|unpooler m=128|30.37|22.88|
|qkv_gate m=256|28.64|23.30|
|o m=256|26.27|12.70|
|pooler m=256|36.58|CUTLASS 31.10|
|unpooler m=256|41.57|27.68|

torchT for qkv_gate and pooler never won either (pooler m=256: 33.82 against 31.10). The cuBLASLt configurations changed no pick, a second independent sample.

### Check 3: more gq/gqx instantiations

Six new instantiations of the existing gq and gqx kernels, the only custom kernels that beat cuBLAS on this GPU: bigger token tiles, two consumer warpgroups, other cluster sizes. They covered qkv_gate at m = 128/256, pooler at m = 256, unpooler at m = 128 and o at m = 128, built as a separate extension. Before the check they passed a clang CUDA syntax pass (with a deliberately bad config as a negative control), clang PTX plus ptxas for sm_90a (0 spills, 90 to 188 registers) and a plumbing test with a stand-in extension. On the judge the extension built and all six passed the FP64 self-test, but none won, raw replay µs:

|Shape|New kernel|Existing pick|
|-|-|-|
|qkv_gate m=128|29.57|gqx 18.53|
|o m=128|12.90 / 13.09|torch 10.62|
|unpooler m=128|24.70 / 24.93|torchT 22.78|
|qkv_gate m=256|25.50|torch 23.30|
|pooler m=256|gq 33.98, gqx 34.53|CUTLASS 31.07|

### Checks 4 and 5: the rs128 design

This is the design from a public file header, described in where-the-score-comes-from.md: swap-AB wgmma with the weights as the M = 64 side and the 12-bit pack decoded into A registers, K split over a one-wave grid, BF16 partials reduced into a self-zeroing output. The live rules, as we read them on Nov 4, allow BF16 or FP16 split-K partials, which is what made this design possible. We built a version for pooler, unpooler and o at m = 128 that kept FP32 partials in the existing workspace and let the last CTA to arrive sum them in a fixed order.

It lost badly in the prepare-time comparison in both runs, raw replay µs:

|Shape|rs128 candidates|Best existing|
|-|-|-|
|o m=128|18.69 to 21.09|torchT 10.85|
|pooler m=128|43.23 to 46.53|gq 25.25 to 25.76|
|unpooler m=128|38.40 to 45.28|torchT 22.85 to 23.46|

The in-kernel phase counters from the second run show where the time went. Median cycles per CTA for the pooler variants: fill 3.4 to 4.2 thousand, main loop 47.8 to 50.9 thousand, escape correction 2.8 to 2.9 thousand, epilogue 8.5 to 8.8 thousand, with a whole CTA lasting 35.6 to 37.7 µs. Unpooler variants: main loop 40.8 to 46.9 thousand cycles, CTA 31.4 to 35.2 µs. The o variants took about 13 µs per CTA (main loop around 9.4 to 9.6 thousand cycles). So the unpacking and escape correction are not the main cost. The main loop itself is too slow: the pooler variants took 1.7 to 1.8x the raw replay time of the existing pick. Our earlier concern, that integer unpacking does not overlap HGMMA, may be part of it, but we did not isolate the cause. The self-test passed and nothing was disabled, so the failure is purely about speed.

## Outcome

Across the five checks the large-m shapes ran on the same kernels as the scored run, and the per-load ratios stayed within noise (verify m=128 between 1.225 and 1.260, verify m=256 between 1.152 and 1.176). The submitted file remains the best code we have for this problem.

Measured negatives, each in a full cold-L2 judge run:

1. cuBLASLt configuration choice: every configuration is algorithm id 66 and none beats the existing picks.
2. gemm90, a split-K wgmma GEMM, for the BF16 layers: 14% to 107% slower than the existing picks on all eight shapes.
3. Six more gq/gqx instantiations: all slower.
4. A swap-AB register-A wgmma kernel with 12-bit weights unpacked into registers (rs128): roughly 1.6x to 2x the raw replay time of the existing picks at m = 128.

What nobody here has tried:

- CUTLASS stream-K or split-K for qkv_gate at m = 256. Beware that CUTLASS stream-K workspace setup may add a memset per call, which the rules exclude.
- A tile with more reuse, 128 weight rows x 256 tokens plus split-K, since the shapes are L2-traffic bound. gemm90 is the closest thing we tried, and it lost.
- The other team's pooler numbers (20.5 and 24.2 µs) suggest 2 to 3.5 µs is reachable on the pooler. None of our routes found it.
- The one-constant K-split change for the m = 128 router (12 to 16 splits), worth 0.1% at best.

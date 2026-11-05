# Where the bf16_proj score comes from

This file covers the leaderboard shape for muse_bf16_proj, what each score level implies about bandwidth and fixed overhead per call, what separated our early 1.14x from the 1.5x entries, and what the public record says about how the 1.5x cluster formed.

A note on labels. "Verified" means we read the primary source or computed the number from our own saved data. "Third-party" means another team reported it and we did not reproduce it. "Inference" is our own reading or estimate.

## The problem in brief

The task is `out = x @ W.T` on one H200 SXM at tiny batch sizes. Four layers are BF16: qkv_gate (W is 2304 x 8192, 37.7 MB), o (8192 x 1024, 16.8 MB), pooler (4096 x 8192, 67.1 MB) and unpooler (8192 x 4096, 67.1 MB, though only the first 1024 rows are used when m is 8 or less). The router is 256 x 8192 in FP32, 8.4 MB.

The score is the geometric mean over six loads of SGLang's time divided by ours. The loads are decode at m = 1, 16, 32 and verify at m = 8, 128, 256. Each load is the sum of its five layers, timed as a CUDA graph replay with a cold L2, with about 4 µs of fixed replay cost subtracted from both sides. Since each load is a sum, the big layers dominate. The router has the largest per-layer ratio but is a small share of any load.

The task statement gives byte floors at 4.8 TB/s. At m = 1 they are 7.87 µs for qkv_gate, 3.50 for o, 13.99 for pooler, and 1.75 for the router and the unpooler slice.

## Leaderboard shape

Snapshot of the board taken just after midnight on Nov 4, 1,711 ranked entries (verified from the saved board file):

|Band|Entries|
|-|-|
|1.60x and up|42|
|1.50 to 1.60x|436|
|1.25 to 1.50x|400|
|1.15 to 1.25x|530|
|below 1.15x|303|

Rank 1 was 1.694x, rank 10 1.646x, rank 20 1.618x, rank 50 1.596x, rank 100 1.586x, rank 200 1.566x, rank 300 1.549x and rank 500 1.483x.

Screening does not break up the 1.55x cluster. Only 23 rows carried a `screening` field (12 screened, 10 pending, 1 under review), and 17 rows carried a `screened_speedup`, which is a team's best screened run. Those values were 1.53 to 1.63x. Rank 4 showed 1.671x with a screened run at 1.617x; rank 16 showed 1.624x with one at 1.620x. Rank 13 (1.637x, screened) was reached on Oct 26, four days after the track opened. So the roughly 1.55x recipe was found early and survives screening. Screened speedups of the top teams sat 0.3 to 4.6% under their displayed best (for example 1.646 against 1.571), so the top of the board is a bit optimistic.

## What a 1.5x entry has to look like

This is our arithmetic, not a measurement. Take each shape to cost `floor x 4.8 / BW + OH`, where BW is the effective bandwidth in TB/s and OH is the per-call time the judge's 4 µs subtraction leaves behind. The geomean over the six loads, using the SGLang times from our v3 check (job a58c6e3b903744ea):

|BW (TB/s) \ OH (µs)|0|0.5|1.0|1.5|2.0|
|-|-|-|-|-|-|
|3.6|1.52|1.45|1.38|1.32|1.27|
|4.0|1.69|1.60|1.52|1.45|1.38|
|4.1|1.73|1.64|1.56|1.48|1.41|
|4.3|1.82|1.71|1.62|1.54|1.47|
|4.6|1.94|1.83|1.72|1.63|1.55|
|4.8 (the floor)|2.03|1.90|1.79|1.69|1.60|

Read it this way: 4.0 to 4.3 TB/s effective and about 1 µs of overhead on all 30 shapes lands at 1.52 to 1.62x without any compression. In practice every public entry above 1.5x that we know of uses a 12-bit lossless weight format, and none of them reached the m = 128/256 speeds this table assumes. They make up the difference on the decode shapes by reading fewer bytes. thethings measured this directly: 1.26x with an uncompressed GEMV stack, 1.34x with compression.

One caution on the overhead term. A profile's per-kernel GPU time is a separate measurement, not "scored time minus a constant". The residual between scored and profiled time ranges from -0.5 µs (SGLang pooler at m = 1: 20.96 scored against 21.6 profiled, which looks like overlap between its two kernels) to +2.7 µs (SGLang qkv_gate at m = 1). It would be wrong to treat 2 µs as an irreducible fixed cost.

Practical read ceiling, from thethings (third-party): hand-written read kernels under a judge-like cold-L2 script took 5.3 µs for 16.8 MB, 9.9 µs for 37.7 MB and 16.1 µs for 67 MB, and a TMA bulk-copy streamer matched them. Our three-point fit is `t = 1.7 µs + bytes / 4.66 TB/s`, with residuals under 0.1 µs. That is about 97% of the 4.8 TB/s peak plus 1.7 µs of launch, ramp and tail per shape. The task's floors leave out that 1.7 µs, which is why the small shapes can never get near them. thethings estimates that every shape at its practical floor would give about 1.57x.

If that is right, anything above 1.6x needs three things at once: near the read ceiling everywhere, below it on the big decode shapes through compression, and well above cuBLAS at m = 128/256. One budget that does this lands at 1.66x: per-load ratios of 1.85 (decode 1), 1.70 (decode 16), 1.55 (decode 32), 2.30 (verify m=8), 1.40 (verify m=128) and 1.32 (verify m=256). That asks for more than the best published numbers at decode 32 and at both large verify loads.

## Where our Triton entry stood

Our v3 (check a58c6e3b903744ea, 1.139x, 95% interval 1.122 to 1.156) ran custom Triton kernels only for BF16 at m <= 8, plus a Triton router. At m = 16, 32, 128 and 256, every BF16 layer was still cuBLAS at about 1.00x. Those four loads are 20 of the 30 shapes.

Per-shape gaps against a "4.1 TB/s plus 1.0 µs" target (µs, SGLang time to target, and v3 time to target for the router):

|Load|qkv_gate|o|pooler|unpooler|router|
|-|-|-|-|-|-|
|decode 1|14.18 to 10.21|6.77 to 5.10|21.06 to 17.38|7.10 to 3.05|5.73 to 3.05|
|decode 16|14.08 to 10.28|6.66 to 5.17|21.76 to 17.46|18.62 to 17.46|5.97 to 3.12|
|decode 32|14.27 to 10.37|6.98 to 5.24|21.81 to 17.55|18.67 to 17.55|7.72 to 3.18|
|verify m=8|13.89 to 10.25|6.69 to 5.13|21.54 to 17.41|7.14 to 3.06|5.90 to 3.08|
|verify m=128|16.93 to 10.86|7.34 to 5.67|25.10 to 18.14|19.97 to 18.14|10.53 to 3.59|
|verify m=256|19.52 to 12.44|9.41 to 6.24|30.54 to 21.34|24.70 to 21.34|12.38 to 4.14|

There are two regimes. At m <= 32 the shapes are bandwidth-bound and the fix is a better streaming GEMV: deeper pipelines, no separate split-K reduce, all 132 SMs busy. At m = 256 the floors are compute-bound: qkv_gate is 9.66 GFLOP, 9.77 µs at 989 TFLOP/s against 7.87 µs for its bytes, and pooler and unpooler are 17.2 GFLOP each, 17.4 µs. There the lever is wgmma utilization. m = 128 sits between: bandwidth-bound, but needing roughly 60% tensor-core utilization at the same time.

The biggest single gaps were qkv_gate (about 3.5 to 7.3 µs per load) and pooler (about 3.8 to 9.4 µs). Both pay cuBLAS's split-K plus a separate `splitKreduce_kernel`, 1.6 to 3.0 µs in the profiles. The m = 128/256 router was another 6.9 to 8.2 µs.

Why does cuBLAS stream better than our Triton did? The nvjet kernel names encode tile and stages; `nvjet_sm90_tst_64x16_64x16_2x1` reads as a 64x16 tile, K-block 64, 16 stages, 2x1 cluster. That would be roughly 64 x 64 x 2 B x 16 = 128 KB of weights in flight per CTA, and cuBLAS's unpooler kernel at m = 16 reaches about 3.6 TB/s (67.1 MB in 18.5 µs). Our v3 pooler had 5 stages of 32 x 128 x 2 B, 40 KB in flight, and reached about 3.3 TB/s. Decoding the names this way is inference.

A staged estimate from our early stacking plan. It is our arithmetic on the per-case table of job a5ac82b328cd4734 (which reproduces 1.151x against the judge's 1.150x), with per-layer time taken as `max(bytes / BW, FLOPs / 660 TFLOP/s) + c`:

|Stack|Score|
|-|-|
|Measured at the time|1.151x|
|One launch where we launched two (router, unpooler slice, qkv m <= 8; 1.5 µs each)|1.201x|
|Every BF16 layer at every m at 4.0 TB/s, 2.0 µs per call|1.357x|
|The same at 4.3 TB/s, 1.0 µs per call|1.567x|
|Floor: 4.8 TB/s, 0 µs per call, 989 TFLOP/s|2.031x|

The reading we took: the 1.2x cluster on the board is a router-only fix like ours; the 1.55x cluster matches "every layer near 4.3 TB/s with about 1 µs per call"; the top entry needs about 4.5 TB/s and under 1 µs. That reading was incomplete, because the public 1.5x entries do not stream BF16 at 4.3 TB/s. They run a plain CUDA GEMV at about 4 TB/s on 12-bit weights, which read about 25% fewer bytes. The 1.567x row is better read as "an equivalent effective bandwidth" than as a plan.

## What separates 1.14x from 1.52x

nathenfernandes Fernandes published per-shape judge times for his v18 (1.5186x, 95% interval 1.496 to 1.541, job 66d8b007f6c641ee; his repository says it held rank 1 on Oct 24). We applied his per-shape speedups to the SGLang times from our v3 check and adopted one layer at a time. This is verified arithmetic on third-party numbers.

|Adopt his kernels for|Score|
|-|-|
|nothing (our v3)|1.139x|
|pooler only, all six loads|1.273x|
|plus qkv_gate|1.357x|
|plus the router|1.420x|
|plus unpooler|1.486x|
|plus o|1.520x|
|every m <= 32 shape, nothing at m = 128/256|1.433x|
|only the m = 128/256 shapes|1.208x|

The pooler is the biggest lever by a distance. It is 67 MB, it appears in every load, and v3 left it at cuBLAS speed. Everything at m <= 32 is worth about +0.29x; the two big verify loads are worth about +0.07x.

His v18 per shape, µs, SGLang time in his run then his time:

|Load|qkv_gate|o|pooler|unpooler|router|
|-|-|-|-|-|-|
|decode 1|14.10 to 9.34|7.04 to 5.10|21.25 to 13.29|6.78 to 3.72|10.46 to 3.94|
|decode 16|14.32 to 10.25|6.85 to 5.48|21.66 to 13.98|18.72 to 13.74|14.66 to 4.23|
|decode 32|14.62 to 12.16|6.91 to 5.81|21.84 to 16.13|18.78 to 15.60|15.07 to 5.04|
|verify m=8|14.56 to 9.69|6.94 to 5.50|21.30 to 13.61|7.01 to 4.28|26.30 to 4.34|
|verify m=128|16.83 to 13.99|7.42 to 6.89|25.09 to 20.49|19.98 to 19.37|16.29 to 6.19|
|verify m=256|19.50 to 17.09|9.34 to 8.98|30.96 to 24.19|24.29 to 23.32|17.71 to 8.44|

Per-load ratios are 1.68x, 1.60x, 1.41x, 2.03x, 1.28x and 1.24x in the order decode 1, 16, 32, verify 8, 128, 256. Four cells are under the 16-bit byte floor from the task statement: pooler at decode 1 (13.29 against 13.99), decode 16 (13.98 against 14.06) and verify m=8 (13.61 against 14.02), and unpooler at decode 16 (13.74 against 14.06). That can only happen by reading fewer bytes.

The four techniques behind the cluster, as the two public write-ups list them (details are in the other files in this folder):

1. A CUDA-core or mma.sync GEMV in CUDA C++ built with `load_inline` instead of Triton, for every shape with m <= 32, with weights repacked in `prepare()` into MMA fragment order.
2. Lossless 12-bit weights for the large decode shapes.
3. The router as one fused kernel at every m.
4. Better kernels at m = 128/256: Hopper wgmma with TMA where it was measured to win, and pinned cuBLASLt algorithms.

## Public code and our own entry

To be plain about it: the bf16 file we submitted is the public vjeux solution (github.com/vjeux/okc-2026, Apache-2.0, commit f35a41e of Oct 28) plus a Triton router candidate of our own. A line diff against the upstream file at that commit shows 104 added lines and none removed or changed. They are a 24-line attribution and licence notice at the top, the Triton imports, and the router candidate (two Triton kernels, launch code, and a few lines that register it as a route). The candidate lost our prepare-time timing comparison at every m (see router-and-small-m-kernels.md), so the kernels that actually run are vjeux's. Our 1.49x scored run is therefore a measurement of that public base on the judge, and the reader should not take it as evidence of original kernel work. Our own from-scratch tracks topped out at about 1.14 to 1.15x in Triton (1.139x for v3, 1.150x for an earlier router-only run) and at about 1.25x in CUDA (see judge-results.md).

Why we think public code seeded the cluster, with the evidence we saw (the pages exist and say this; the causal link is inference):

- vjeux/okc-2026 describes five complete submissions, one per problem, each in native CUDA built through `load_inline`. Its bf16 file uses losslessly compressed BF16 weights. Initial commit Oct 27, a refactor (f35a41e) on Oct 28. vjeux's own row on our saved board was 1.5014x (rank 471, best run Oct 26).
- alygg77/okc-muse-spark-kernels (Apache-2.0) reports 1.5502x, rank 294 on our Nov 4 board; the README puts it at 75 of 1,401 on Oct 29. Its README says all five files start from vjeux's public kernels and that most of the speed comes from that base. For bf16 it keeps vjeux's 12-bit format and adds custom WGMMA o-projection kernels, pooler tuning and nathenfernandes's small-m routes; it credits its cuBLASLt plan pins to code posted by Matsvei on a second public code-sharing site. The README also reports judge noise: one unchanged file scored 1.526x, 1.541x, 1.544x and 1.550x in four runs on one day.
- NathenFernandes/muse-bf16-proj (MIT) reports 1.5186x, rank 416.
- A GitHub search for "muse_bf16_proj" finds 4 repositories and 922 code results. Many entrants keep their submission files public.
- Timing, from the saved board: of the 436 entries in 1.50 to 1.60x, only 46 have their best run dated Oct 24 to 26. The rest are dated after vjeux's release, with 19, 23, 36, 62, 39, 62, 75 and 74 per day from Oct 27 to Nov 3. That fits public code seeding the cluster, but board growth alone could produce the same shape, so it is a correlation.

Whether the organizers' review treats derivative entries differently from original ones is unknown to us. We could not read their screening document.

## Outside claims we collected

|Claim|Source|What is public|What our file has|
|-|-|-|-|
|bf16 1.569x|@AdiverseK on X, Oct 31|a five-point list: 12-bit lossless weights for m 1 to 32; own TMA/wgmma kernels, pinned cuBLASLt plans and an FP32 router path for m 128 and 256; a per-layer, per-batch auto-tuner timed under CUDA graphs|the 12-bit path, the vjeux base and the auto-tuner (cold-L2 graph timing); the TMA/wgmma kernels gave nothing beyond cuBLAS in our checks|
|bf16 1.595x, 12-bit pack|@NivasT49781 on X, Oct 30|nothing about mechanism. The same post names a MULDEC graft for MoE (2.382x); we found no other trace of that term|the 12-bit pack, same family|
|bf16 1.5502x|alygg77's repository|custom WGMMA o-projection kernels (6.96 and 9.26 µs at m = 128 and 256 against 7.30 and 9.26 for our file), pooler tuning, nathenfernandes's o m=8 and pooler m=8/16 routes (+0.60%), a deepcopy fix|none of these confirmed in our file|
|bf16 1.5742 (check), 1.5690 (scored)|the bf16 file in keell0renz's public repository (commit 3435290), header only|a component list with no per-cell numbers; the two scores reached us second-hand and we did not see them in the part of the header we read|see below|
|router 10.3 and 26.3 µs down to 5.1 and 11.9 µs, total 1.055x|@anshulcreates on X, Oct 24 and 27|numbers and the idea (stream the 8 MB once, reuse across tokens)|our router is already faster (router-and-small-m-kernels.md)|

Other X posts we saw (all claims): @max_grev at 1.5233x (rank 8 on Oct 25) and 1.5406x on Oct 29, saying "It doesn't seem like I can do much better than 1.5406x"; @KazimAIZJU at 1.292x and 1.329x; @ashwin3005 at 1.234x; @utsav_khairnar at 1.15x with a Triton kernel that does the matmul in one launch instead of 2 to 3 and a router about 2x faster. They sort into the two clusters: Triton router fixes land at 1.15 to 1.25x and full CUDA stacks at 1.50 to 1.55x.

Where could the extra 5% over 1.49x come from? The header of the 1.57x file lists components our entry lacks. Our decomposition, from per-shape bounds (an estimate, not a measurement), gives the whole-score effect if our shapes reached the bound for that component:

|Component|Shapes|At the bound|At 50%|At 25%|
|-|-|-|-|-|
|swap-AB register-A wgmma with the 12-bit pack decoded into A registers, K split over a one-wave grid, BF16 partials reduced into a self-zeroed output|qkv_gate, o, pooler, unpooler at m = 128|6.0%|2.7%|1.3%|
|wgmma split-K router with self-zeroing|router at m = 128, 256|1.1%|0.5%|0.3%|
|flag-zeroed split-K router at m = 8, 16, 32 (and a register GEMV for unpooler at m <= 8)|router m = 8/16/32; unpooler m = 8|2.0% (+0.65% unpooler)|1.0%|0.5%|
|K-split 16 for the m = 128 router|router m = 128|0.1%|||
|m = 256 tiling or pins (CUTLASS split-K, a 128x256 tile)|four BF16 layers at m = 256|2.1%|1.0%|0.5%|

For the first component the bound uses 12-bit bytes against a 989 TFLOP/s x 0.75 roof, with gaps of 6.3, 2.3, 7.8 and 4.4 µs on qkv_gate, o, pooler and unpooler out of the 70 µs verify m=128 load. At m = 256 the shapes are compute-bound, so 12 bits do not help there.

The first three rows at half capture add up to about 4.2%, within a point of the +5.2 to +5.6% claimed, so the claim is consistent with its parts. But only the first row is large enough to matter alone. The same header says each (layer, m) cell is timed against the older path and the older path is kept if the new one is not faster, so a 1.5742 check score does not prove the new kernel won any cell. We tried that first row; the result is in large-m-plan-and-results.md and it was a loss.

## Sources

- nathenfernandes Fernandes: https://github.com/NathenFernandes/muse-bf16-proj, per-shape results at https://raw.githubusercontent.com/NathenFernandes/muse-bf16-proj/solution/results/v18_shapes.csv and ranking history at https://raw.githubusercontent.com/NathenFernandes/muse-bf16-proj/solution/results/ranking-history.md
- thethings.ai: https://thethings.ai/p/okc/muse-bf16-proj-guide/03-strategy and https://thethings.ai/p/okc/muse-bf16-proj-report
- vjeux: https://github.com/vjeux/okc-2026
- alygg77: https://github.com/alygg77/okc-muse-spark-kernels
- GitHub search: https://github.com/search?q=muse_bf16_proj&type=repositories
- X search: https://x.com/search?q=muse_bf16_proj&f=live and https://x.com/search?q=openkernelchallenge%20bf16

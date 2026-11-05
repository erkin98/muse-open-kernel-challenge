# Where the MoE kernel's time goes

This file collects the diagnostic measurements we took on the W4A16 MoE problem: the per-launch timeline of each build, the fits that separate bytes from fixed cost, the small tuning arms, and how much noise an identical-code run carries.

## The problem and the builds

Six scored shapes: decode with 1, 16 and 32 tokens (d1, d16, d32) and verify with 8, 128 and 256 tokens (v1, v128, v256), expert-parallel over 8 GPUs, against SGLang. The score is the geometric mean of the speedups, so one shape moving 1% moves the score 0.167%.

The weights are uniform-random 4-bit codes, so the code bytes cannot be compressed. Only the scales can shrink, and they already use 6-bit offsets (scale bytes span 93 to 126 for gate/up and 94 to 126 for down). A touched expert streams about 27.5 MB in the wgmma build and 28.3 MB in the skinny builds.

Our base was a file we call v3 (sha256 prefix c0828b74). It has four builds, picked by token count M:

|build|M range|what it is|registers|
|-|-|-|-|
|tiny|1 to 4|skinny build, d1|149|
|tiny2|5 to 8|skinny build, v1|149|
|mid|9 to 16|skinny build, d16|153|
|wk6p|17 and up|wgmma build with a stream-K schedule over 256-row, 16-stage units, d32, v128, v256|96|

SASS sizes of the four, as the fingerprints we used to prove "unchanged code": tiny 5336 instructions, tiny2 5384, mid 5344, wk6p 22,920. An NVRTC 13.2 plus ptxas build reproduces the judge's own compile log exactly (registers and zero spills), which made static gating possible.

## Noise: what one judge run can resolve

Runs of the base code include c663b9dfc629461b (2.362x, CI 2.327-2.397), 03072c0ab2854c2e (2.344x, CI 2.309-2.379) and 36fb48423c7d4e62 (2.363x, CI 2.328-2.398, a diagnostic payload run). Paired speedup ranges across identical code:

|shape|range|
|-|-|
|d1|3.061-3.114|
|d16|1.955-2.001|
|d32|1.667-1.668 (four identical-code readings in all)|
|v1|2.189-2.279|
|v128|2.604-2.623|
|v256|2.866-2.898|

In microseconds the base's two board-config runs spread d1 5.6%, d16 2.0%, d32 3.1%, v1 11.7% (61.5 to 69.2 us over four runs), v128 2.4% and v256 2.0%. SGLang's time moves with the node and the routing, so the paired ratio is far more stable than our microseconds. d32 read the same to about 0.05% across four runs, so a 0.5% move there is visible. d16 needs about 3% and v1 about 5%.

A caveat we learned the expensive way: a run that carries a large diagnostic payload is not a clean baseline. The extra compiled modules and arms shifted even unchanged shapes. d32 ran v3 code in the two later diagnostic runs and read 1.651 and 1.654 instead of 1.667 to 1.668, and v256 read 2.85 and 2.852 against 2.866 to 2.898.

## The diagnostic run (job 36fb48423c7d4e62)

This run executed v3's scored code unchanged (score 2.363x, 5/5) plus a diagnostic block at the end of `prepare()`: instrumented twins of the four builds, controlled-routing sweeps, bandwidth probes and small tuning arms, all on synthetic inputs and freed afterwards. Prepare took 120 s in total (39 s for the normal part and 81 s of diagnostics). Reserved memory afterwards was 11,704 MiB against 11,700 to 11,716 for v3.

Before launch we wrote down when each of the two main levers would count as dead. For d32: the controlled-routing intercept of wk6p at M=32 minus the pure-stream intercept under 5 us, and the timeline spreads (A-done, B-done, barrier wait) each under 3 us. For d16: the mid intercept within 5 us of the pure stream, and the ring-depth arm moving under 1.5%. Neither test declared a lever dead. The d32 excess was 21 us against the 5 us bar and the tail 4 to 5 us against 3 us, and the d16 excess was 10.5 us, so both levers stayed alive.

### Fits: bytes against fixed cost

Controlled routing means we feed inputs that touch E experts (one n-tile each) and fit time t = a + b E. The pure stream is a TMA read of the same bytes with no compute.

|build, M|graph-replay fit (us)|eager fit (us)|intercept excess over the pure stream|
|-|-|-|-|
|tiny, M=1|13.8 + 6.41 E|14.1 + 6.47 E|+8.0|
|tiny2, M=8|14.8 + 6.46 E|15.2 + 6.42 E|+9.1|
|mid, M=16|16.6 + 6.56 E|16.6 + 6.58 E|+10.5|
|wk6p, M=16|27.7 + 6.14 E|27.7 + 6.14 E|+21.6|
|wk6p, M=32|27.2 + 6.17 E|27.3 + 6.16 E|+21.2|
|pure TMA stream|-|6.1 + 0.2234 per MB (4.48 TB/s)|0|

The wgmma build's slope, 6.14 us per expert at 27.5 MB, is exactly the pure-stream rate. The skinny builds sit 1.5% to 3.8% above it. So the per-expert stream is at the practical DRAM ceiling, and everything left is cost per launch. The wgmma build pays about 21 us over the pure stream and mid pays 10.5.

The pure stream itself: 17 MB reads at 1.69 TB/s, 34 MB at 2.46, 67 MB at 3.20, 134 MB at 3.77, 268 MB at 4.07 and 537 MB at 4.26. Ring depth 3, 4 and 5 stages of 36 KB gave 4.26, 4.28 and 4.27 TB/s at 537 MB, so the ring is not what limits the stream.

### Timelines

Per-CTA stamps, medians in microseconds from the earliest CTA start: routing finished, gather finished and the join, phase A done, phase B done, grid barrier release, and the final pass after the barrier. "Wait" is barrier release minus the CTA's own B-done. Each row is one sampled routing, so rows are not comparable across batch sizes: more routes means a longer phase A and B.

|build, M|routed|join|A done|B done|barrier release|wait med/max|final pass med/max|
|-|-|-|-|-|-|-|-|
|tiny, 1|2.3|-|12.3|19.2|20.2|1.3/2.6|0.3/0.8|
|tiny2, 8|2.3|-|46.6|72.7|74.8|2.6/3.8|0.5/0.8|
|mid, 16|2.8|-|47.6|74.0|76.5|2.8/4.4|0.3/0.8|
|wk6p, 16|1.8|3.6|51.7|75.8|78.8|2.6/5.1|1.0/4.6|
|wk6p, 24|1.8|4.1|84.0|124.2|127.5|3.1/4.4|1.0/4.4|
|wk6p, 32|1.8|4.4|93.4|137.0|141.6|3.8/5.9|1.0/5.4|
|wk6p, 48|1.8|4.4|69.6|103.2|113.2|10.8/12.8|1.0/4.9|
|wk6p, 128|2.3|6.1|125.4|181.8|186.1|4.9/9.0|4.1/5.1|
|wk6p, 256|3.1|7.4|127.5|186.6|192.0|5.4/10.5|6.4/8.2|

Reading the table. First, the wgmma build's join, where the first weight stage can start, lands 3.6 to 4.4 us in, about 2 us after routing, because the consumers gather the activations into shared memory before the producers join. Second, the median barrier wait on wk6p is 2.6 to 10.8 us, and with a hot expert (a count of 31 or 32 tokens, four n-tiles) A-done and B-done spread by 8 to 11 us between the first and last CTA, against 5 us at a count of 15. Third, the final pass after the barrier takes up to 4.4 to 5.4 us on wk6p at M=16 to 48, against 0.8 us on the skinny builds. Even CTAs with no tokens show about 1.0 us there. The timer quantum is 0.256 us.

The 1.0 us on idle CTAs looks like cold instruction fetch of the tail code, but we could not separate that from other effects. A later run measured the gap between a launch right after an L2 flush and one that runs behind another launch: tiny at M=1 -1.1 us, mid at M=16 +1.7, wk6p at M=16 +5.5, wk6p at M=32 +6.1, and the later "wk6m" variant at M=32 +5.4 (with 3, 12, 12, 16 and 16 touched experts). That gap includes warm L2 data, warm TLB and warm instruction cache together, so it does not isolate the code. We report it as measured and leave the interpretation open.

### Tuning arms in that run

|arm|result|verdict|
|-|-|-|
|FCI 4 instead of 3 on wk6p (the stream-K flush-cost constant)|0.1% to 0.5% faster at M 17 to 256 (for example 124.8 vs 124.9 us at M=24, 149.6 vs 150.3 at M=64)|Closed. Our own bar was 1.5% at two or more token counts|
|mid with one more weight-ring slot (NS=4)|1.2% slower at M=16, 0.3% slower at M=24|Closed|
|Grid 128, 124, 120, 112 CTAs instead of 132|mid: +1.1%, +4.2%, +6.1%, +11.7%. wk6p at M=32: +1.3%, +1.5%, +3.5%, +8.0%|Closed. Use all SMs|
|mid used above M=16|M=17 and 20 tie with wk6p (108.3 / 108.3, 114.7 / 114.6). M=24 128.8 vs 124.7, M=32 140.3 vs 126.9, M=48 154.1 vs 133.7, M=64 181.2 vs 150.4|The 16/17 boundary stays. At M=16 wk6p is 5.7% slower than mid (103.6 vs 98.0)|

## What the judge's per-repetition data say

Each judged shape reports 100 repetitions per run. Our time and SGLang's are almost perfectly correlated across repetitions (0.99 to 1.00 at d16 and d32), because both are driven by which experts that repetition's routing touches. Fitting time against a simulated distribution of touched experts (about 11.8 plus or minus 2.4 at d16 and 15.9 plus or minus 2.9 at d32) gave a marginal cost per extra expert of 5.5 to 5.7 us at d32 and 6.4 to 7.0 at d16, which is about the DRAM peak of 4.8 TB/s. d1 runs sit on plateaus near 25.6 to 26.5, 32 to 33 and 39 us, steps of 6.5 us per extra expert (28.3 MB over 6.5 us is 4.35 TB/s) above a fixed part of about 13 us.

At d32 the moment-fit intercept is about 40 us against 27 us in the controlled fit, because hot experts and route counts leak into it; at d1 the two agree (about 13 to 14 us). We use the controlled numbers. Real-byte rates, counting 28.3 MB per touched expert: d16 about 3.5 TB/s, d32 about 3.65, v1 about 3.9, v128 about 3.57, v256 about 3.28. The judge's speed-of-light floor counts only 75% of the touched experts, so the ratio of our time to that floor overstates the gap.

## What this left us with

The expert stream runs at 4.2 to 4.35 TB/s. What remains is fixed cost per launch: a start-up where the first weight stage waits about 4 us, imbalance when one expert is hot, the barrier wait, and a final pass that costs 4 to 5 us on the wgmma build against 0.8 us on the skinny builds. The variants we built against those targets are in dequant-and-scheduling-variants.md. None of them moved the score.

A task-generator model of the stream-K schedule (300 random routings) points the same way. Estimated consumer work per CTA is below estimated DRAM time per CTA (125 vs 178 us at v256, 100 vs 154 at v128, 66 vs 108 at d32), yet the observed times are 210, 177 and 124 us. The difference is events: ramp, unit ends, the A-to-B transition, the tail and the final pass. alygg77 and keell0renz measured the same kind of loss.

A reading we wrote down after the later variants failed: the fixed cost of wk6p (about 27 us intercept against 17 us for mid) is made of the start (join near 4.2 us: routing 1.8 us plus ranges about 2.2 us), the barrier wait (3 to 9 us) and the final pass. Moving work across the grid barrier just moves the critical path. A change has to remove work from the CTAs that finish last, not shift it.

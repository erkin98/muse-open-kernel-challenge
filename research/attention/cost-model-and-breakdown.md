# Cost model and per-shape breakdown

Where the attention score comes from, shape by shape: the weights, the bounds we compared ourselves to, and a two-parameter model of the heavy full-attention shapes.

The per-shape speedup table and the short "what did not help" list are in [notes/attention.md](../../notes/attention.md); this file goes underneath them.

All numbers come from the judge profile of our best file (check 4, job 8c9a79b3381c4707, 2.6211x).

## How the score is put together

Each of the six loads (decode and verify, at 1, 16 and 32 requests) is 46 sliding-window layers plus 16 full-attention layers. The score is the geometric mean over the six loads of SGLang's time divided by ours. We rebuilt that in a small script and it reproduces the judge: 2.6212 against 2.6211. A spot check by hand agrees too: for decode 1, 46 x 4.91 + 16 x 10.54 is 394 us, which is the per-load figure.

Because the score is a geometric mean, a 1% cut in one shape moves the score by that shape's share of the log-score in percent. Sliding shapes carry 32.5% of the log-score and full shapes 67.5%. The four full shapes at batch 16 and 32 together carry 53% (12.5 to 14.2% each), and the two batch-1 sliding shapes carry 9.5% each.

## Per-load view

|load|SGLang us|ours us|speedup|sliding share of our time|
|-|-|-|-|-|
|decode 1|1431|394|3.63x|57%|
|decode 16|4483|1778|2.52x|22%|
|decode 32|7610|3123|2.44x|15%|
|verify 1|1627|522|3.12x|57%|
|verify 16|4652|2144|2.17x|25%|
|verify 32|7848|3651|2.15x|19%|

At batch 16 and 32 the full layer is 75 to 85% of our time. At batch 1 the sliding layer is more than half, and that is where we are slowest relative to any plausible floor.

## Distance to a practical bound

For each shape we computed a bound and the gain the whole score would see if that shape alone reached it.

|shape|ours us|bound us|ours / bound|score gain if bound reached|
|-|-|-|-|-|
|decode 1 sliding|4.91|3.00|1.64|4.29%|
|decode 1 full|10.54|6.40|1.65|3.11%|
|decode 16 sliding|8.54|3.33|2.56|2.44%|
|decode 16 full|86.60|75.38|1.15|1.79%|
|decode 32 sliding|10.06|5.25|1.92|1.23%|
|decode 32 full|166.24|149.35|1.11|1.52%|
|verify 1 sliding|6.51|3.00|2.17|6.37%|
|verify 1 full|13.88|6.41|2.17|4.43%|
|verify 16 sliding|11.55|3.58|3.23|3.18%|
|verify 16 full|100.77|75.63|1.33|3.52%|
|verify 32 sliding|14.73|5.76|2.56|2.02%|
|verify 32 full|185.85|149.85|1.24|2.90%|

How the bound is built matters more than the table, so here it is. For the stream-bound shapes it is the bytes floor at 4.8 TB/s, divided by 0.854 (the fraction of peak we assume a random-row stream reaches, about 4.1 TB/s), plus a fixed 1.4 us. For decode 32 full that gives 126.3 us / 0.854 + 1.4 = 149.3 us. The batch-1 full shapes are chains (split-KV tail plus finalize), and we hold them at a practical chain bound of about 6.4 us instead of the stream formula. The sliding shapes start from a launch-plus-dependency-chain floor of about 3.0 us and the table shows it rising for the shapes that read more bytes (5.25 us for decode 32 sliding, 5.76 us for verify 32 sliding).

These are assumptions. The 4.1 TB/s random-row ceiling comes second hand from a community discussion, and we never re-measured it on the judge's H200 (we had no H200 of our own, and the judge only runs the problem's tests). If the true ceiling is near 4.5 TB/s, the four heavy full shapes hide up to 10% of headroom; if it is near 4.1, they hide about 2%. That single unknown decides whether the biggest block of the score has any room left.

Reading the table: the heavy full shapes are closest to their bound (1.11 to 1.33) but carry the most weight. The sliding chains are 1.6 to 3.2 times above a floor that is itself a guess, and verify 1 sliding (6.37%) and verify 1 full (4.43%) are the largest single prizes if the floor is real.

## Why every K and V read is a random row

One hypothesis was that the "coherent" distribution in the reference input means contiguous KV runs that bigger bulk copies could exploit. Reading `reference.py` (roughly lines 55 to 72 and 184 to 194) refutes that. The page table is a `torch.randperm` over all tokens, and the comment says the physical pages are scattered, so each K row and each V row is a separate 128-byte read at a random address. "Coherent" refers to a shared direction planted in the values of low-attention keys (`_coherent_tail`), which shapes the score distribution and says nothing about memory layout. Skipping or merging those tail keys would break the rule that every key is attended.

## A two-parameter model of the heavy full shapes

Batch 16 and batch 32 run the same code path with twice the bytes, so a two-point fit of time = fixed + slope x bytes-floor gives a marginal stream cost and a per-call fixed cost.

The fit gives a slope of 1.261 times the bytes floor for decode full (d16f, d32f) and 1.342 for verify full (v16f, v32f). Those convert to marginal rates of 3.81 TB/s (79% of the 4.8 TB/s peak) and 3.58 TB/s (75%), with fixed parts of 6.95 us and 15.69 us per call. The rate and fixed-part columns are also in [notes/attention.md](../../notes/attention.md); the slopes are the new part.

So the heavy full shapes are not 3.65 TB/s machines with a small overhead. They stream at 3.6 to 3.8 TB/s on the margin and also pay 7 us (decode) or 15.7 us (verify) per call. The batch-1 full shapes fall below the line (decode 1 full has its own intercept near 5.2 us, verify 1 full near 8.1 us at the same slope), so the fixed part depends on the launch plan.

Caveats we cannot remove: these are two-point fits, the 7 and 15.7 us include about 0.8 us of launch outside the kernel, and we have not decomposed the rest (pipeline fill, segment-list build, the tail of the equal-share grid, the split-KV merge, the finalize). Which of those dominates is unidentified without a profiler on the H200.

## Bytes in flight: an argument we made and then had to soften

Little's law says bytes in flight per SM = rate x latency / 132 SMs.

|rate|latency|needed per SM|
|-|-|-|
|4.1 TB/s|0.8 to 1.0 us|24 to 30 KB|
|4.5 TB/s|0.8 to 1.0 us|27 to 33 KB|
|4.5 TB/s|1.5 us (loaded)|50 KB|

The flat full kernels use 64-key tiles, 8 KB per stage for K and 8 KB for V (16 KB per stage), three stages for one query token and four for eight, with up to two CTAs per SM. That is 32 to 64 KB per CTA in flight and up to twice that per SM, at or above what the law asks. Together with the measured loss from deeper pipelines (see [ideas-tried-and-results.md](ideas-tried-and-results.md)), this made "move more bytes per SM" look like the wrong lever, and we wrote it up that way.

Then the concurrency experiments came back. Halving the CTAs in flight (one CTA per SM instead of two) made the four heavy full shapes 12 to 19% slower, and a similar halving on the sliding layer cost 12 to 20% on two shapes. So the stream is still sensitive to concurrency at one CTA per SM, and the Little's-law argument only holds as a statement about the current two-CTA configuration. Our current position: more depth hurts, less concurrency hurts, and the setting we have looks close to the best the register budget allows. Whether the rest of the gap to the ceiling is DRAM-side (random 128-byte rows) or sits in the fixed part is open.

## What shaving the fixed part would be worth

Moving only the modelled parameters, whole-score effect:

|change|effect|
|-|-|
|decode fixed part 7 us down by 1 / 2 / 3.5 us|+0.24% / +0.48% / +0.84%|
|verify fixed part 15.7 us down by 2 / 4 / 8 us|+0.40% / +0.80% / +1.63%|
|verify marginal rate raised to the decode rate (3.58 to 3.81 TB/s)|+1.44%|
|decode marginal rate to 1.20 x floor (4.0 TB/s) / 1.12 x floor (4.3 TB/s)|+1.26% / +3.03%|

The best-supported targets are the verify call's fixed 15.7 us and its lower marginal rate (about +2.4% if half of each were removed), ahead of gather depth. Candidate causes worth a profiler if one is ever available: the FP32 atomic merge of the 8-token partials (about 32 KB per split), q-tile preparation for eight tokens, the finalize slice, and the tail of the equal-share grid. None of this has been measured. It is a model, and the gains in the table are what the model says, not what any kernel delivered.

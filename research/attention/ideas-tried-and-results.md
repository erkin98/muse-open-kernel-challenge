# Ideas tried and what they returned

This file lists every attention idea we measured or ruled out on paper, with its result, followed by our ranking of the headroom that is left.

The base is vjeux's public entry (Apache-2.0), which scored 2.605x when we ran it unchanged. Our best file, check 4 (job 8c9a79b3381c4707), reached 2.6211x. Almost all of that comes from the base; our own contribution is the two small changes marked "kept" in the table below. Per-shape numbers for the best file are in [cost-model-and-breakdown.md](cost-model-and-breakdown.md), and a short list of failed ideas is in [notes/attention.md](../../notes/attention.md).

## Constraints that closed some ideas before we tried them

Taken from the challenge rules and organizer guidance: the kernel must be CUDA-graph capturable, nothing may be memset in the timed call, nothing may be cached from the input data, the KV cache stays FP8, queries may not be coarser than E4M3, every key must be attended, the source has to be readable, and tuning to exact sizes is discouraged. The rules do allow storing P in FP8 E4M3 at any power-of-two scale, approximating exp2 (a polynomial or bit manipulation), and approximate math accurate to BF16 precision.

These are why we never tried skipping or averaging the low-attention tail keys, and why we kept per-shape overrides to a minimum.

## The list

|idea|where it applies|result|effect on score|status|
|-|-|-|-|-|
|Redesign the heavy full gather to put more bytes in flight per SM|d16f, d32f, v16f, v32f (53% of the score)|not built; arithmetic and later experiments make it doubtful (see cost-model file)|up to +6% only if the random-row ceiling is near 4.5 TB/s, else about +1%|open, new design|
|Shorten the sliding chain (clusters, one-tile, cooperative barrier, cold-code warm-up)|six sliding shapes (32.5%)|clusters, a one-tile split and a cooperative grid barrier lost in another team's judged attempts; cold-code warm-up lost in our check 2; a latency-chain study ended negative|0|closed|
|Bit-trick exp2 producing e4m3 P codes directly|v16f, v32f|job 24a52be614264969: v16f 101.0 to 101.1 us, v32f 184.1 to 183.6 us, inside noise|0|measured, dropped|
|Packed half2 softmax and FP16 QK^T at 8 tokens|v16f, v32f|SASS shows 5 instructions per pair, same as the current code|0|dropped, static|
|Four workspace copies for decode 1 sliding|d1s|job 66be38f978b54004: d1s 5.53 us against 4.94 to 5.17 for the base|negative|measured, dropped|
|128-byte bulk copy per KV row|heavy full shapes|copy-op rate loses elsewhere; needs a proxy fence and a replay-race soak we could not run|0|dropped|
|Exact-key tuned overrides|v32s, d1f|both come with the base file. The d1f entry gives decode 1 full the plan the file already uses for single-request 8-token calls, so it is a regime rule in effect. The v32s entry is an isolated key whose setting matches the file's own formula for 16 to 23 requests; we did not measure its neighbours|0|assessed, kept (two exact keys)|
|L2 prefetch-size hint on the K/V row copies|heavy full shapes|redundant by construction, see below|0|dropped|
|Flat q_len 8 edits (key range and segment list built in one warp, e4m3 query tile written straight from raw rows, finalize run twice so the first pass warms the code)|v16f, v32f, v1f|check 3, job 04aae1ba8df8488f, submitted as 0140d9facd2a410d|+0.3 to 0.4%|kept|
|BF16 vector reductions of split-KV partials|v1f only|check 4, job 8c9a79b3381c4707|+0.15%|kept|
|Deeper cp.async pipelines (5 stages flat, 6 per-request, from 3)|d16f and d32f (flat); d1f and d32s (per-request)|job 24a52be614264969: d16f 86.6 to 90.6 us, d32f 165.6 to 170.3 us, beyond the base file's own run-to-run range; the per-request rows stayed inside the spread|negative on the flat shapes|dropped|
|Cold-code warming of the per-request finalizes|batch-1 and small sliding shapes|job a58eeeeaf93c4bed: no effect, see its own file|0|dropped|

The two kept changes are both smaller than the run-to-run spread of the headline score (about 1%, see [judge-runs-and-noise.md](judge-runs-and-noise.md)), so on the headline alone they are indistinguishable from a lucky draw. We judged them from per-shape rows and byte-identical control kernels. The flat q_len 8 gain is for the bundle of three edits; we did not isolate which of them mattered. One oddity: warming the finalize did nothing for the batch-1 per-request shapes, yet one of the three flat q_len 8 edits runs the finalize twice to warm it. Either the flat path differs enough for warming to matter there, or the gain comes from the other two edits. We cannot say which.

## Concurrency experiments

We ran these as host-only changes (no kernel code touched), each in its own judge run, all off the board. The base for all of them was check 4, job 8c9a79b3381c4707 (2.6211x).

|run|change|score|rows that moved|verdict|
|-|-|-|-|-|
|job 704eb239130b4db7|flat full grid from 264 CTAs to 132|2.424x|d16f 96.9 us (+12%), d32f 187.6 (+13%), v16f 117.4 (+16%), v32f 220.5 (+19%); sliding rows unchanged|worse, concurrency matters|
|job 0eda2f0eb85b43f2|sliding grids use half the CTA target|2.556x|v16s 13.7 us (+20%), d32s 11.2 (+12%); others inside noise; full rows moved +1 to 2% and -1%, which were controls|worse|
|job 800c1940cf774991|q_len 8 sliding at 16 requests: 16 splits instead of 8|2.627x|v16s 11.8 us against 11.2 to 11.6 for the base: no gain; the headline difference is noise on unchanged shapes|no gain|

Halving the CTAs in flight hurts in both layers, by 12 to 19% on the full shapes and 12 to 20% on two sliding shapes. So the two-CTAs-per-SM setting is near the best the register budget allows, and this weakens the earlier claim that bytes in flight were already sufficient. At the same time, the earlier check with deeper pipelines lost 3 to 4 us on d16f and d32f. Both directions lose. We do not know why deeper stages cost time: the shared-memory footprint rose (flat q_len 1 from 60 to 93 KB, per-request from 56 to 105 KB) while two CTAs per SM still fit, so occupancy is not an obvious explanation.

The follow-up sweep, which varied split counts, workspace copies and finalize mode for every shape, is in [launch-plan-sweep.md](launch-plan-sweep.md).

## The L2 prefetch-size hint, in detail

One micro-lever we only analysed and never ran: adding an L2 size hint (a 128-byte prefetch hint) to the `cp.async` 16-byte pieces that load each K and V row. Evidence against, in order of weight:

- In the gather loop, eight consecutive producer lanes each load one of the eight 16-byte pieces of a single 128-byte row, so a warp instruction fetches four whole rows. The L2 already sees full-line requests, which makes a prefetch-size hint redundant by construction. This is an inference from the code; we did not measure it.
- Another public repository reports the same kind of hint on 64-byte half-row copies helping on an H100 (350 to 311 us) and doing nothing on an H200 (157.7 against 157.6 us). The judge is an H200.
- Best case, 2% on the four stream shapes, is about 1% on the score, below the 2.5 to 3% we would want before believing a change, and about the size of judge noise.

It would have cost one token in the copy helper plus a static gate, with no race risk, since it is a pure hint. We skipped it.

## Where the remaining headroom is, ranked by what it could do for the whole score

|rank|idea|best case|evidence|
|-|-|-|-|
|1|heavy full gather redesign|+6% if the H200 random-row ceiling is near 4.5 TB/s, about +1% if near 4.1|every tested lever failed or is noise; the ceiling is unmeasured|
|2|shorten the sliding chain by 10%|+3.4%|clusters, one-tile and a cooperative barrier lost in another team's attempts; nc=4 on d1s and warm-up lost in our checks|
|3|more trimming of the q_len 8 flat full shapes (v16f, v32f, 26% of the score)|about +0.5%|already banked: flat q8 variant and BF16 vector reductions; exp tricks changed nothing, which says MUFU is not the limiter|
|4|per-shape launch keys at the exact scored sizes|+0.3 to 1%|organizers warned about per-shape variants; a rules risk, and the sweep found nothing to gain anyway|
|5|more judge draws of slightly different bytes, keep the best|+0.5 to 0.8%|not a speed gain; it only shifts the number inside the noise|

None of these reaches rank 10 (+14.8%). Rank 100 (+2.8%) would need one real 3% gain, and nothing here delivers one. Idea 5 is listed only for completeness; we do not consider it legitimate progress.

A smaller note on what failed and why it is informative: the exp2 experiments (bit trick, linear mantissa) and the packed half2 analysis all point the same way, that the softmax math is not the limiter on the q_len 8 shapes. The time is in memory and in the fixed parts of the call.

# muse_attention_decode: where the time goes

Numbers are from the judge profile of the submitted file (job 8c9a79b3381c4707, 2.6211x). The score is the geometric mean over six loads of SGLang's time divided by ours; each load is 46 sliding-window layers plus 16 full-attention layers.

## Per shape

|shape|SGLang us|ours us|speedup|share of the score|
|-|-|-|-|-|
|decode 1 sliding|20.32|4.91|4.14x|9.5%|
|decode 1 full|31.04|10.54|2.94x|7.1%|
|decode 16 sliding|27.49|8.54|3.22x|3.7%|
|decode 16 full|201.15|86.60|2.32x|13.0%|
|decode 32 sliding|31.20|10.06|3.10x|2.5%|
|decode 32 full|385.95|166.24|2.32x|14.2%|
|verify 1 sliding|22.72|6.51|3.49x|9.6%|
|verify 1 full|36.35|13.88|2.62x|7.1%|
|verify 16 sliding|29.31|11.55|2.54x|4.1%|
|verify 16 full|206.51|100.77|2.05x|12.5%|
|verify 32 sliding|34.50|14.73|2.34x|3.1%|
|verify 32 full|391.31|185.85|2.11x|13.6%|

Share is the change in the log of the score per unit change in that shape's log time. The full layer carries 67.5% of the score, the sliding layer 32.5%.

## Cost model of the heavy full shapes

Fitting time = fixed part + bytes / rate on batch 16 and batch 32 (the same code path with twice the bytes):

|family|marginal rate|share of the 4.8 TB/s peak|fixed part per call|
|-|-|-|-|
|decode full|3.81 TB/s|79%|6.95 us|
|verify full|3.58 TB/s|75%|15.69 us|

The KV cache is read as random 128-byte rows (the page table is a random permutation of all tokens), so the stream does not reach peak, and each call pays a fixed cost for pipeline fill, the split-KV merge and the finalize. Two-point fits; the fixed part was not decomposed further because we had no profiler on the H200.

## What changed relative to the vjeux base

- A flat 8-token (verify) kernel that builds its key range and segment list in one warp, writes the e4m3 query tile straight from the raw rows, and runs the grid-wide finalize twice so the first pass warms the code. About +0.3 to 0.4% on the score.
- The one-request 8-token cooperative kernel accumulates split-KV partials as BF16 pairs (half the bytes). About +0.15%, visible on verify 1 full only.

## What did not help

|idea|result|
|-|-|
|exp2 by a bit trick straight into e4m3 probability codes|no movement on verify 16 / 32 full|
|packed half2 softmax with FP16 QK^T at 8 tokens|same instruction count per pair in the SASS|
|4 workspace copies for decode 1 sliding|slower (5.53 us against 4.94 to 5.17)|
|128-byte bulk copy per KV row|copy-op rate loses; needs a replay-race soak we could not run|
|L2 prefetch-size hint on the K/V row copies|redundant: 8 lanes already fetch a whole row per instruction|
|deeper cp.async pipelines|3 to 4 us slower on decode 16 / 32 full|
|one CTA per SM on the full layer (132 instead of 264)|12 to 19% slower on the four heavy full shapes|
|half as many splits on the sliding layer|verify 16 sliding +20%, decode 32 sliding +12%|
|a sweep of split counts, workspace copies and finalize mode on every shape (8 files)|no setting beat the existing plan beyond noise; cooperative finalize is essential on verify 1 full (+71% without it)|
|clusters, one-tile, cooperative barrier (public attempts, and our own checks)|lost or noise|

Reading one file's rows per shape works because the 12 scored shapes are timed independently: a change that touches one shape leaves the others where they were.

## What rank 10 would take

Rank 10 was 3.008x (about +15% over 2.621x). Spread evenly that is 13% less time on every shape. The full shapes alone would need decode 32 full at 116 us, below the 126 us bytes floor at 4.8 TB/s; the sliding shapes alone would need decode 1 sliding at 2.3 us, below any plausible launch-plus-chain floor. So both layers have to improve at once: roughly 9% off every full shape and 20% off every sliding shape.

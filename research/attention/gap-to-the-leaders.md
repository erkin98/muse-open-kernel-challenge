# What the gap to the leaders requires

Arithmetic only: how much time each rank target would take off our 2.6211x, and which combinations of shape improvements could get there.

The leaderboard values come from the Nov 4 midday reading (rank 1 at 3.339x) and were still the same on the morning of Nov 5, where rank 10 was still 3.008x. Other teams keep moving, so every target is probably a little higher by the time you read this. The shape weights and bounds are the ones in [cost-model-and-breakdown.md](cost-model-and-breakdown.md).

## Rank targets

|target|score|gain over 2.6211x|uniform cut on every shape|cut on full shapes only|cut on sliding shapes only|
|-|-|-|-|-|-|
|rank 100|2.694x|+2.8%|2.7%|4.0%|8.1%|
|rank 50|2.784x|+6.2%|5.8%|8.5%|16.9%|
|rank 10|3.008x|+14.8%|12.9%|18.4%|34.5%|
|rank 1|3.339x|+27.4%|21.5%|30.1%|52.5%|

Mixes of a cut on every full shape and a cut on every sliding shape:

|full cut|sliding cut|score|gain|
|-|-|-|-|
|2%|0%|2.657x|+1.4%|
|0%|10%|2.710x|+3.4%, just past rank 100|
|5%|5%|2.759x|+5.3%|
|3%|10%|2.767x|+5.6%|
|10%|10%|2.912x|+11.1%|
|5%|15%|2.857x|+9.0%|
|9%|20%|2.999x|+14.4%, about rank 10|

Rank 10 therefore means roughly 9% off every full shape and 20% off every sliding shape at once. Rank 100 means roughly 8 to 10% off the sliding shapes or 4% off the full ones. Rank 100 sounds close, but our sliding shapes are already chains of a few microseconds, and the single-lever experiments we ran (see [ideas-tried-and-results.md](ideas-tried-and-results.md) and [launch-plan-sweep.md](launch-plan-sweep.md)) gave nothing near 10% on them.

## What rank 1 would need

The gap is ln(3.339 / 2.6211) = 0.242, which is a uniform 21.5% time cut on every shape. Taking each layer type alone:

- Sliding shapes alone would need a log cut of 0.745, which is every sliding row at about 0.47 of today's time. That puts decode 1 sliding at 2.33 us, below the 3.0 us launch-plus-chain floor. Not possible if the floor is real.
- Full shapes alone would need a log cut of 0.359, which is about 0.70 of today's time. That puts decode 32 full at 116 us, below the 126 us bytes floor at 4.8 TB/s. Not possible.

So both layers have to improve. The batch-1 full shapes are chains (split-KV tail plus finalize, practical bound 6.4 us), so in the scenarios below they are held at that bound and the bytes-floor factor only applies to the four batch 16 and 32 full shapes.

|scenario|score|gain|
|-|-|-|
|heavy full at 1.05 x bytes floor (about 4.57 TB/s), sliding unchanged|3.305x|+26.1%|
|heavy full at 1.10 x bytes floor, sliding unchanged|3.230x|+23.2%|
|heavy full at 1.05 x bytes floor, sliding -25%|3.690x|+40.8%|
|heavy full at 1.10 x bytes floor, sliding -35%|3.778x|+44.1%|

Rank 1 is not reached by the full shapes alone even at 1.05 times the floor (3.305x against 3.339x). Adding a sliding cut closes it. The only reading we found that fits the numbers is that a 3.3x entry runs the four heavy full shapes within roughly 5 to 10% of the bytes floor and also shortens the sliding chains by something like 10 to 25%. We cannot verify that; it is a consistency argument.

A correction worth recording: the first version of this arithmetic let the batch-1 full shapes drop to 4.5 us as well and printed 3.405x for the full-only case. That overstated it, because those shapes sit on a chain bound we do not expect anyone to beat. The 3.305x above is the corrected figure.

Our own heavy full shapes sit at 1.11 to 1.33 of the practical bound and 1.32 to 1.59 of the bytes floor (decode 32 full is 166 us against a 126 us floor).

## Bounds-reached scenarios, for scale

|scenario|score|gain|
|-|-|-|
|every shape at its assumed bound|3.956x|+50.9%|
|full shapes at bound, sliding as is|3.107x|+18.5%|
|sliding at bound, full as is|3.174x|+21.1%|
|only the two q_len 8 full shapes (verify 16, verify 32) at bound|2.792x|+6.5%|
|only decode 16 and 32 full at bound|2.709x|+3.3%|
|full as is, sliding -35%|2.965x|+13.1%|

Even with every shape at the bound we assumed, the result is 3.956x. In log terms the rank 1 gap (0.242) is about 59% of the whole distance from us to that bound (0.412). If our bounds are about right, a 3.34x entry has closed a large part of the headroom on both layers.

## Hypotheses we tested on paper

|hypothesis|test|result|
|-|-|-|
|the gap is on the sliding chains alone|arithmetic above|refuted (needs decode 1 sliding below the floor)|
|the gap is on the full shapes alone|arithmetic above|refuted (needs decode 32 full below the bytes floor)|
|the leader has both: heavy full shapes within 5 to 10% of the floor and sliding chains 10 to 25% shorter|capped scenarios|consistent, unverifiable|
|"coherent" input means contiguous KV runs that bulk copies could exploit|reading `reference.py`|refuted: the page table is a random permutation, see the cost-model file|

## What this means in practice

Nothing in the arithmetic points to a single lever. Reaching rank 100 needs one real 3% gain; rank 10 needs gains on every shape of both layers; rank 1 needs the heavy full shapes near the bytes floor and the chains shortened as well. The unmeasured quantity that decides how far the full shapes can go is the random-row read ceiling of the judge's H200. We could not measure it.

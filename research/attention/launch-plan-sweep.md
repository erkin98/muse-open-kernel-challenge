# Launch-plan sweep over all twelve shapes

One round of eight judge runs tested different launch-plan settings (split counts, workspace copies, finalize mode) on all twelve shapes at once, and this file has the per-shape results.

## Why it was possible

The judge times the twelve scored shapes independently, so one file can carry a different launch-plan variant for each shape. The kernel code is untouched; the variants only change host-side planning (how many splits, how many workspace copies, whether the finalize is cooperative or last-CTA). Eight files therefore gave eight variants per shape for eight judge runs. All eight passed 16/16 correctness tests, including the edge cases, so none of the settings broke the kernel.

Scores of the eight files, all off the board: sweep0 2.459x, sweep1 2.507x, sweep2 2.486x, sweep3 2.479x, sweep4 2.403x, sweep5 2.333x, sweep6 2.627x, sweep7 2.611x. The headline numbers are not the interesting part. Each file mixes good and bad variants across shapes, so we read the per-shape rows.

Sweep file 6 carried the unchanged plan. The judge reported it as a duplicate of job 1c1107b8a97d4e45, an earlier check of the same bytes as the submitted file, so its 2.627x is a re-read of that earlier result and not a fresh draw. It counts once, as that earlier draw.

## The reference spread per shape

Rows pooled from runs of identical code (the board run of job 8c9a79b3381c4707, two extra draws of the same kernel, the unchanged shapes inside the three concurrency experiments, and sweep file 6), in microseconds. Because file 6 is a duplicate (see above), the pool is a little smaller than it looks. The range is the lowest and highest value seen.

|shape|reference us|range|
|-|-|-|
|d1s|4.9|4.6 to 5.1|
|d1f|10.8|10.5 to 11.1|
|d16s|8.2|8.0 to 8.5|
|d16f|86.7||
|d32s|10.1|9.7 to 10.4|
|d32f|166.9||
|v1s|6.6|6.4 to 6.9|
|v1f|13.8|13.5 to 14.2|
|v16s|11.4|11.2 to 11.6|
|v16f|100.4||
|v32s|14.6|14.4 to 14.8|
|v32f|184.6||

Unchanged sliding rows move by up to 5% between runs, so only larger differences count. See [judge-runs-and-noise.md](judge-runs-and-noise.md).

## Results per shape

Variant means the changed setting, followed by the row relative to the reference.

|shape|worse than reference|inside the spread|better than reference|
|-|-|-|-|
|d1s|16 splits +16%, 24 +18%, 28 +20%, 20 +27%, 12 +35%, 8 +49%|32 (the base)|none|
|d16s|32 splits +29%, 6 +10%|8, 10, 12, 24|none|
|d32s|16 splits +38%, 6 +11%, 5 +12%|7, 10, 12|none|
|v1s|every variant from 12 to 30 splits, +14% to +36%|33 (the base)|none|
|v16s|9, 10, 11, 14 and 16 splits, +4% to +13%|12|none|
|v32s|2 splits +50%, 3 +19%, 5 +13%, 6 +9%|7, 8|none (the base, 4 splits, is the optimum)|
|d1f|cooperative finalize off +6%, four workspace copies +10%, 200 splits +10%|220, 240, 264 splits|none|
|v1f|cooperative finalize off +71%, 200 splits +15%, 224 +7%, 240 +9%|252 (the base), four workspace copies|264 splits at 13.5 us (-2.2%), at the low edge of the base spread, a single sample|
|d16f, d32f||cooperative finalize on: no change (86.8 and 166.6 us)||
|v16f, v32f||last-CTA finalize instead of cooperative: v16f +1.3%, v32f -0.1%||

## What we took from it

The launch plan the kernel already ships is at or within noise of the best setting we tried on every shape. Nothing is a candidate for submission. The only sub-noise hint is v1f at 264 splits, which would be worth at most +0.16% on the score and is one sample at the bottom of the spread.

The cooperative finalize turned out to matter very unevenly:

- On verify 1 full the cooperative finalize is essential: turning it off costs 71%.
- On decode 1 full it is worth about 6%, and on the large full shapes it makes no difference either way.

The penalties on the sliding shapes are steep: d1s at 8 splits is 49% slower, v32s at 2 splits is 50% slower, d32s at 16 splits is 38% slower, d16s at 32 splits is 29% slower, and v1s loses 14% or more with any other split count we tried.

## Paper analysis of the sliding partitions

Before running the sweep we checked on paper whether a different partition of the sliding window could shorten the critical path. From the tile counts, the window is 32 tiles of 64 keys for decode and 33 tiles for 8-token verify (the extra tile holds the last 7 keys). That is an inference from the partition counts, not a number we read in the source.

- d1s: 32 splits of one tile each. v1s: 33 splits of one tile each (the last tile with 7 keys). d16s: 16 splits of two tiles. d32s: 8 splits of four tiles. All balanced.
- v16s uses 8 splits holding [5,5,5,5,5,5,3,0] tiles, and v32s uses 4 splits holding [9,9,9,6]. The critical path is ceil(33/8) = 5 tiles and ceil(33/4) = 9 tiles whichever way the tiles are dealt out, so re-partitioning cannot shorten it at the same split count.

That argument is consistent with the sweep: at v16s every split count from 9 to 16 was worse or equal, and at v32s the base value of 4 splits was the optimum.

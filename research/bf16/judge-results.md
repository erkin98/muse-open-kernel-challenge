# Judge results for muse_bf16_proj

This file lists every judged run we have a record of for muse_bf16_proj, with per-shape times for the runs of the file we submitted, the byte floors per shape, and a note on how noisy the numbers are.

## What the entries are

The bf16 file we submitted is the public vjeux solution (github.com/vjeux/okc-2026, Apache-2.0, commit f35a41e, fetched Nov 4) plus a Triton router candidate that lost our prepare-time comparison at every m, so the kernels that run are vjeux's. A line diff against the upstream file at that commit shows 104 added lines and none removed or changed: a 24-line attribution and licence notice at the top, the Triton imports, and the router candidate (two Triton kernels, launch code, and a few lines that register it as a route). The scores from the sixth row down are therefore measurements of that public base, not of kernels we wrote. The first rows are the unmodified starter and our own from-scratch tracks.

## Runs

|Job|What it was|Score (95% interval)|
|-|-|-|
|76c338ff8cc64f7b|the unmodified starter|0.908x|
|a5ac82b328cd4734|early Triton entry, router fixed|1.150x from the judge, 1.151x when we recomputed it from the per-case table|
|a58c6e3b903744ea|v3: Triton, custom kernels only for BF16 at m <= 8 plus a router|1.139x (1.122 to 1.156)|
|e80520a1edb0468b|clean-room CUDA track, best at the time|1.252x|
|6c7a0061944e4339|submitted vjeux-based file, scored run|1.491x, rank 341 at our last look on Nov 4|
|a35a24f311274d15|same file, check run (sha256 prefix 1a2bf6ba407a)|1.520x (1.497 to 1.543)|
|329e7f847e0d4b45|check 1 of the large-m work (sha256 prefix bbd32492ebbf)|1.497x (1.475 to 1.520)|
|31e944e4fdba45fb|check 2 (sha256 prefix 8a5d95afa292)|1.484x (1.462 to 1.506)|
|1dd4d6bbe0214964|check 3 (sha256 prefix bf00be7e5fad)|1.492x (1.470 to 1.515)|
|c440acdd631e4328|rs128 variant|1.505x (1.483 to 1.528)|
|635004eab6f9489a|rs128 variant, second build|1.4915x (1.469 to 1.514)|

All of the vjeux-based runs passed 9/9 tests. The reported times were 3,177.9 µs (a35a24f311274d15), 3,226.5 (329e7f847e0d4b45), 3,255.5 (31e944e4fdba45fb), 3,237.1 (1dd4d6bbe0214964), 3,209.0 (c440acdd631e4328) and 3,239.0 (635004eab6f9489a). The clean-room CUDA track's later best was recorded as 1.258x after five checks; we did not keep the job id for that one, so the 1.252x row is the one we can point to.

How the rows relate: a5ac82b328cd4734 and a58c6e3b903744ea are both early Triton versions and we did not keep a note of how they differ. The large-m checks all changed only the large-m routes and are explained in large-m-plan-and-results.md.

## Reading the case labels

The judge labels loads as "decode 1/16/32" and "verify 1/16/32". The number looks like a batch of requests, and verify seems to carry 8 tokens each, so verify 1, 16 and 32 are m = 8, 128 and 256 (our reading). The judge profile spells this out ("verify 16 m=128"), and the SGLang times match the m = 8, 128, 256 columns in nathenfernandes's published table. In the table below the verify rows use the m values.

## Per-shape times for the submitted file

Times in µs. "SGLang" and "a35a" are from the check run a35a24f311274d15; the next three columns are our time in the three large-m checks. SGLang's own time drifts between runs by a few tenths of a µs (decode 1 qkv_gate was 14.0 to 14.3, verify m=8 router 26.1 to 26.4), so compare our times, not speedups, across runs. The last column is the speedup in a35a24f311274d15.

|Load, layer|SGLang|a35a|329e|31e9|1dd4|Speedup|
|-|-|-|-|-|-|-|
|decode 1, qkv_gate|14.3|8.9|8.9|9.1|8.9|1.609x|
|decode 1, o|6.7|4.8|4.9|5.1|5.3|1.388x|
|decode 1, pooler|21.3|13.1|12.9|13.6|13.5|1.629x|
|decode 1, unpooler|6.8|3.3|3.7|3.8|3.5|2.060x|
|decode 1, router|10.4|3.6|3.5|3.8|4.0|2.922x|
|decode 16, qkv_gate|13.9|9.3|9.4|9.4|9.9|1.492x|
|decode 16, o|6.6|5.2|5.4|5.6|5.6|1.277x|
|decode 16, pooler|21.2|13.7|13.9|14.3|14.5|1.547x|
|decode 16, unpooler|18.5|13.8|13.8|13.7|14.0|1.341x|
|decode 16, router|14.9|5.0|5.0|5.1|5.0|2.961x|
|decode 32, qkv_gate|14.6|10.8|10.9|11.0|10.8|1.350x|
|decode 32, o|7.1|6.1|5.9|5.8|5.9|1.161x|
|decode 32, pooler|21.8|14.8|15.0|14.7|15.2|1.473x|
|decode 32, unpooler|18.9|15.2|14.6|14.3|14.9|1.241x|
|decode 32, router|15.5|5.5|5.5|5.4|5.5|2.802x|
|verify m=8, qkv_gate|14.2|9.4|9.6|9.8|9.4|1.512x|
|verify m=8, o|6.7|5.6|5.8|5.7|5.1|1.209x|
|verify m=8, pooler|21.5|13.9|14.3|14.5|14.2|1.544x|
|verify m=8, unpooler|7.0|4.3|4.7|4.4|4.2|1.632x|
|verify m=8, router|26.1|4.8|5.3|5.1|5.1|5.437x|
|verify m=128, qkv_gate|17.1|14.6|14.9|14.7|14.8|1.168x|
|verify m=128, o|7.3|7.1|7.5|7.4|7.3|1.036x|
|verify m=128, pooler|25.4|22.0|22.2|21.8|21.9|1.151x|
|verify m=128, unpooler|20.3|19.1|19.1|19.0|19.0|1.065x|
|verify m=128, router|16.6|6.0|6.4|5.8|5.9|2.759x|
|verify m=256, qkv_gate|19.7|19.6|19.9|19.8|19.7|1.004x|
|verify m=256, o|9.5|9.2|9.1|9.4|9.0|1.035x|
|verify m=256, pooler|31.2|27.6|27.7|27.8|27.6|1.130x|
|verify m=256, unpooler|24.3|23.7|24.2|24.1|24.2|1.026x|
|verify m=256, router|17.8|7.1|7.0|7.2|7.1|2.509x|

What the table says: the decode loads and verify m=8 are where the 12-bit path pays, with speedups of 1.16x to 2.1x on the dense layers and 2.8x to 5.4x on the router. At verify m=128 and m=256 the dense layers sit at 1.00x to 1.17x. The router is the exception, 2.5x to 2.8x even at the large sizes. Verify m=8 router is the single biggest ratio only because SGLang takes 26 µs there.

## Byte floors per shape

From the "speed of light" column of the judge profile, in µs. Floors at m = 256 are compute-bound for the larger layers.

|Layer|m=1|m=16|m=128|m=256|
|-|-|-|-|-|
|qkv_gate|7.87|7.93|8.42|9.77|
|o|3.50|3.56|3.99|4.48|
|pooler|13.99|14.06|14.64|17.37|
|unpooler|1.75 (slice)|14.06|14.64|17.37|
|router|1.75|1.81|2.21|2.68|

The router floor is 1.78 µs at m = 8 and 1.86 at m = 32. Pooler at verify m=8 has a floor of 14.02 µs.

## Per-load ratios for the scored run

Decode 1: 1.731. Decode 16: 1.564. Decode 32: 1.487. Verify m=8: 1.916. Verify m=128: 1.225. Verify m=256: 1.169. Geometric mean 1.492, which matches the judge's 1.491. The same file's check run, a35a24f311274d15, had verify m=128 at 1.260 and verify m=256 at 1.176.

## How noisy is a score

The judge's 95% intervals on these runs are about plus or minus 0.023 (1.475 to 1.520 around 1.497, for instance). The same unchanged file scored 1.491x as a scored run and 1.520x as a check, a 1.9% spread. alygg77's README reports one unchanged file at 1.526x, 1.541x, 1.544x and 1.550x in four runs on one day. Final ranks come from the organizers' median reruns, so resubmitting the same code in the hope of a lucky draw does not help.

Per-shape noise is plus or minus 1 to 2%. We only keep a custom path where it beats SGLang by 3% or more, and thethings reports using the same rule.

On the other side of the board, screening and reruns cost the top teams 0.3 to 4.6% against their best displayed run (see where-the-score-comes-from.md), so the displayed scores near the top are optimistic.

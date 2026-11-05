# Headroom model for all four boards

This file compares the measured per-shape times of our four board entries against estimated practical bounds, to show where a kernel could still get faster and what one shape is worth to the score.

The numbers are a snapshot of the Nov 4 board entries (attention 2.6045x, bf16 1.4914x, moe 2.3293x, decode_layer 2.0960x). They are not final scores. Everything is computed from the judge's existing per-shape profiles; no extra judge runs were spent on it.

Read this part first. The bounds rest on a handful of constants, and only the first one (the floor definition) is given. The rest are our assumptions, and two of them (the practical read rate and the latency floor) matter a lot. The model reproduces all four reported scores to four digits, which shows that the shape weights and loads are right. It says nothing about whether the bounds are right. A sensitivity table at the end shows how far the ceilings move when two of the assumptions change.

## Reading the columns

- `ours` and `sol` are the judge's numbers for the board run, in µs. `sol` is the judge's floor: bytes at 4.8 TB/s or FLOPs at 989 TFLOP/s, whichever takes longer, counted in uncompressed bytes (from each problem's TASK.md).
- `GPU` is the profile's kernel time (for decode_layer it is busy time or the sum over many kernels). `ovh` is `ours` minus `GPU`, the per-call overhead outside the kernel.
- `bound` is our practical bound for the shape: bytes divided by ETA (or FLOPs by ETA_C), plus FIX, and never below LF. All constants are in the assumptions table.
- `gap` is `ours` minus `bound`. `gain@100%` is the exact whole-score gain (geomean over the six loads) if that one shape reached its bound. `gain@25%` is the same if it closed a quarter of its gap. The 25% figure is a planning number, not a prediction.
- `class` is stream when bytes or math limit the shape, and chain when the latency of a dependent chain (launch, load, reduce, finalize) limits it. For chain shapes the bound is only the assumed floor LF.
- A load's time is the sum of weight times shape time over its shapes. The score is the geomean over the six loads of SGLang time divided by our time.

## Assumptions

Every number that is not read from a judge file.

|id|value|status and source|
|-|-|-|
|A1|4.8 TB/s, 989 TFLOP/s|Given. This is the floor definition in each TASK.md.|
|A2|ETA 0.854 (a 4.1 TB/s practical read rate)|Assumption. The evidence is a figure of 4.05 to 4.1 TB/s for random 128-byte row reads, quoted second hand in a community thread and never re-measured by us. The best stream we have demonstrated is the bf16 unpooler at m=16: 3.98 TB/s (12.7 µs kernel for 50.6 MB).|
|A3|ETA_C 0.75 (tensor-core efficiency)|Assumption backed by the FA3 paper abstract: FP16 attention reaches 740 of 989 TFLOP/s, 75% (https://arxiv.org/abs/2407.08608). Our own best is the bf16 unpooler at m=256 at 73% of the roof (23.83 vs 17.37 µs). Used only for the bf16 m=256 shapes.|
|A4|FIX 1.4 µs added to every byte or math bound|Assumption fitted on the profiles. The best bf16 m=1 kernel (pooler) sits 1.39 µs above bytes/ETA, the median per-call overhead over attention, bf16 and moe is 0.81 µs, and the smallest kernel in the attention and bf16 profiles is 2.4 µs.|
|A5|LF 3.0 µs, the least a call can take|Assumption. The three smallest `ours` values in the bf16 profile are 3.50 (unpooler m=1), 3.84 (router m=1) and 4.51 (unpooler m=8). Attention chains are longer than any of these (decode 1 sliding is launch, qkv load, K/V load, reduce, finalize = 4.94 µs), so 3.0 is a floor attention probably cannot reach.|
|A6|bf16 weights at m <= 32 are 12 bits|Read in our bf16 board file (sha256 1a2bf6ba): lines 36 to 47 and 116 to 121 show a raw low byte plus a 4-bit exponent-window code, with escapes under 1/200. Bytes are 0.75 of the TASK.md floor. The router stays FP32 (8.4 MB). For m = 128 and 256 the file reads 16-bit weights.|
|A7|3.9 T exp results per second|FA3 paper section 3 and its footnote: 16 results per SM per clock at 132 SMs and 1830 MHz, taken from the CUDA guide. Used only to label the attention verify shapes.|
|A8|256 KV bytes per context token|2 KV heads x 64 dims x 1 B (FP8) x (K and V). Check: 80,000 tokens is 20.48 MB, which is 4.27 µs, equal to the decode 1 full floor in TASK.md.|
|A9|Layer sizes for the bf16 math bound|From the m=1 floors: qkv_gate 18.9 M parameters, o 8.4 M, pooler 33.6 M, unpooler 33.7 M, router 2.1 M (FP32).|

## Findings

### F1. Where the room is

|board|all-shapes ceiling|only stream shapes|only chain shapes|if every gap closes by a quarter|
|-|-|-|-|-|
|attention|52%|9.9%|36% (depends on A5)|9.0%|
|bf16|15%|14.0%|0.9%|3.4%|
|moe|56%|56%|0|9.8%|
|decode_layer|56%|56%|0|9.8%|

A ceiling assumes every shape of that class hits its bound, which no kernel does. What we have seen so far is much smaller. The two attention variants we put through the judge scored 2.550x and 2.576x, both under the 2.605x board entry. The first moe variant came out +0.6% raw and +1.4% against its same-build controls. So the 25% column is optimistic for any single idea.

### F2. What a judge run can detect

Two judge runs of the same attention file (c20df44ab71d4005 and 705c7dd9f34446dd) differ per shape as follows:

|shape|difference|
|-|-|
|decode 1 sliding|+0.23 µs (+4.7%)|
|decode 1 full|-0.61 µs (-5.4%)|
|verify 1 sliding|-0.21 µs (-3.1%)|
|verify 1 full|+0.16 µs|
|decode 16 sliding|+0.17 µs|
|decode 16 full|+0.40 µs (+0.5%)|
|decode 32 sliding|+0.25 µs|
|decode 32 full|+0.24 µs (+0.1%)|
|verify 16 sliding|+0.24 µs|
|verify 16 full|+0.60 µs (+0.6%)|
|verify 32 sliding|-0.20 µs|
|verify 32 full|+1.53 µs (+0.8%)|

Two consequences. A change under about 0.5 µs on a 5 to 15 µs shape cannot be seen in one run, and neither can a change under about 1% on a 100 to 190 µs shape. And the headline score has a 95% interval of at least plus or minus 1.5%. With that much noise, a tuning idea has to promise a whole-score gain of roughly 2.5 to 3% to justify a judge run, and then needs a second run to repeat the result.

### F3. Attention

- The heavy full shapes are close to the assumed bound. Decode 16 full is 1.15 times its bound and decode 32 full is 1.11 times (86 to 90% of the quoted ceiling). Their single-shape ceilings are 1.8% and 1.5%. There is no room there without a new design.
- Verify 16 full (1.34 of bound, gap 25 µs) and verify 32 full (1.23, gap 34 µs) have the largest absolute gaps. They are not about bytes alone. 152 M and 304 M exp results need 39 and 78 µs of special-function time at the 3.9 T/s rate (A7), on top of FP8 wgmma and the 128-byte row stream, and all of it has to overlap inside one SM. Ceilings are 3.6% and 2.8%. The cheap levers have already been tried and dropped: an FMA-emulated or linear exp2, deeper pipeline stages, pair balancing and a V transpose. In our first attention judge run, a linear-mantissa exp2 changed verify 16 full and verify 32 full by nothing (101.14 vs 101.04 µs and 183.63 vs 184.09 µs), and the deeper pipelines slowed decode 16 full and decode 32 full by 3 to 4 µs (90.56 vs 86.58 and 170.30 vs 165.55).
- The batch-1 shapes (decode 1 sliding 4.94, decode 1 full 11.27, verify 1 sliding 6.80, verify 1 full 14.31 µs) carry the largest single-shape ceilings (4.2, 3.6, 6.7 and 4.5%). But their bound is just the assumed floor, their time is a dependent chain, and the two runs we already spent on them showed nothing (cold-code warming: 5.1 vs 4.94 µs on decode 1 sliding, 6.8 vs 6.8 on verify 1 sliding).
- The sliding shapes at batch 16 and 32 (ours over bound of 2.0 to 3.2) are chains of 7 to 15 µs. We have no first-principles bound for them. The board file already carries two overrides keyed to exact scored sizes (verify 32 sliding and decode 1 full). We stopped adding more, because the challenge bans tuning that only works at exact scored sizes. Any new setting would have to cover a contiguous range.
- The largest ours-over-bound ratios are verify 16 sliding 3.18, verify 32 sliding 2.58, decode 16 sliding 2.51, verify 1 sliding 2.27 and verify 1 full 2.23. These are latency chains, so the ratio alone is not evidence of headroom.

### F4. bf16

This is the board where another team's results show that more exists. Public posts report 1.569x and 1.595x against our 1.491x, so a gain of +5 to +7% is reachable.

- Our m <= 32 kernels sit at 1.0 to 1.5 times bytes/ETA + FIX. Already at the bound: pooler at m=1, 8 and 16 (gaps 0 to 0.37 µs) and unpooler at m=16 (0.26 µs). Open gaps: qkv_gate (1.6 to 2.2 µs at m=8, 16 and 32), o (0.7 to 1.5 µs), pooler m=32 (1.3 µs), unpooler m=8 (1.5 µs: 4.51 µs against 3.50 µs at m=1 for the same 1024-row slice) and unpooler m=32 (1.2 µs), and the router (1.7 to 1.85 µs at m=8, 16 and 32, but only 0.4 µs at m=1, where a plain FP32 GEMV already streams the 8.4 MB in 2.4 µs).
- At m=128 and 256: qkv_gate has gaps of 4.0 and 5.3 µs, pooler m=128 has 3.75, the router has 2.2 and 2.7, o has 1.2 and 1.9, and the unpooler is at its bound at both sizes.
- By layer, the sum of single-shape ceilings over the six m values is: qkv_gate 4.2%, router 3.2%, o 2.3%, pooler 2.2%, unpooler 1.5%.
- Largest ours-over-bound ratios: router m=256 1.59, m=128 1.56, m=32 1.52, m=16 1.49, m=8 1.49; qkv_gate m=256 1.37 and m=128 1.35; o m=32 1.33.
- No single shape or layer reaches +2.5% alone. The router (five shapes, one kernel family) and qkv_gate (six shapes) are the only layer-wide levers near that size, and only if half the gap is captured.

### F5. moe and decode_layer

The analysis for these two boards is thin, because the model has fewer separate causes to offer.

- moe: ours over `sol` is 1.74 to 2.20 in all six loads. Most of that is already understood: about 6 µs of bytes per distinct expert, plus fences and a tail. A 1% gain on the moe kernel is worth about 1% on the moe board and about 0.46% on decode_layer.
- decode_layer: ours over `sol` is 1.49 to 2.35. The biggest ratios are the batch-32 and batch-16 verify sliding shapes (2.35 and 2.01), where the moe kernel is 52 to 55% of the load.

### F6. What follows from the model

- Attention parameter search is not worth a judge run on this evidence. No single tunable or shape offers an expected whole-score gain of 2.5% or more that one run could see (F2, F3). The heavy shapes are at 1.11 to 1.34 of bound with the cheap levers already used, and the chain shapes are below the resolution of a run. The one candidate that stays on the table is a packed half2 softmax with FP16-accumulated QK^T on the q_len 8 flat kernels (verify 16 full and verify 32 full). The estimate for it is a realistic 0 to 1% whole-score gain, under the bar. It would be a large rewrite of the score-to-P path in two kernels, and a figure quoted from another team puts barrier and data-wait shares at about 12.5% each, which suggests those two shapes are not purely issue-bound. It would need two static checks first: the SASS instruction count per pair and a numerics model.
- bf16: a static study of the router found gaps of 0.4 to 2.7 µs per shape, which is a 3.2% ceiling for the router (1.6% at half capture). Our earlier work at m=128 and 256 (three judge runs) was all negative. The one lever above 5% is a design from another public repo (chenwainuo's rs128): a swap-AB, register-A wgmma kernel at m=128 that decodes the 12-bit pack straight into registers. Its ceiling in our model is 6.0% (2.7% at half capture). It is a hard kernel to write, and a public attempt by alygg77 to put 12-bit weights inside the m=128 GEMM had lost, so we did not pursue it. See `ideas-from-keell0renz-repo.md` for how that design is described in the public source.

## Tables

### Board summary

The model reproduces the judge's score. Gains are exact geomean effects. Ceiling means every shape of that class reaches its bound (not reachable; it shows where the room is). 25% means one quarter of each gap.

|board|judge run|reported|model|ceiling: stream shapes|ceiling: chain shapes|ceiling: all|25% of all gaps|
|-|-|-|-|-|-|-|-|
|attention (board file sha256 cf04259b)|job c20df44ab71d4005|2.6045x|2.6042x|9.9%|36.3%|52.3%|9.0%|
|bf16 (board file sha256 1a2bf6ba)|job 6c7a0061944e4339|1.4914x|1.4915x|14.0%|0.9%|15.1%|3.4%|
|moe|job 41f596feaa60434e|2.3293x|2.3294x|55.5%|0.0%|55.5%|9.8%|
|decode_layer|job 9b0d0973f11a4758|2.0960x|2.0959x|55.8%|0.0%|55.8%|9.8%|

The judge's reported intervals are 2.565 to 2.644 for attention and 1.469 to 1.514 for bf16.

Per-call overhead (`ours` minus profile GPU time, over the single-kernel boards attention, bf16 and moe): median 0.81 µs, 10th percentile -4.01, 90th percentile 1.77. The smallest kernel GPU time in the attention and bf16 profiles is 2.4 µs. The decode_layer `GPU` column is a sum or busy time over many kernels, so its `ovh` is not comparable.

### attention (job c20df44ab71d4005)

|shape|w|ours|GPU|ovh|sol|ours/sol|bound|ours/bound|gap|gain@100%|gain@25%|class and reason|
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|decode 1 sliding|46|4.94|4.2|0.7|0.11|44.91|3.00|1.65|1.94|4.20%|0.94%|chain: latency chain; bytes only 0.11 µs|
|decode 1 full|16|11.27|9.9|1.4|4.27|2.64|6.40|1.76|4.87|3.60%|0.82%|chain: split-KV tail + finalize; bytes 4.3 µs, MUFU 0.3 µs|
|decode 16 sliding|46|8.37|6.8|1.6|1.65|5.07|3.33|2.51|5.04|2.37%|0.56%|chain: latency chain; bytes only 1.65 µs|
|decode 16 full|16|86.58|85.8|0.8|63.18|1.37|75.38|1.15|11.20|1.79%|0.43%|stream: bandwidth, random 128-B rows|
|decode 32 sliding|46|10.00|8.9|1.1|3.29|3.04|5.25|1.90|4.75|1.22%|0.30%|chain: latency chain; bytes only 3.29 µs|
|decode 32 full|16|165.55|164.8|0.8|126.35|1.31|149.35|1.11|16.20|1.46%|0.35%|stream: bandwidth, random 128-B rows|
|verify 1 sliding|46|6.80|6.0|0.8|0.12|56.67|3.00|2.27|3.80|6.71%|1.41%|chain: latency chain; bytes only 0.12 µs|
|verify 1 full|16|14.31|13.5|0.8|4.28|3.34|6.41|2.23|7.90|4.53%|1.01%|chain: split-KV tail + finalize; bytes 4.3 µs, MUFU 2.6 µs|
|verify 16 sliding|46|11.39|10.5|0.9|1.86|6.12|3.58|3.18|7.81|3.11%|0.72%|chain: latency chain; bytes only 1.86 µs|
|verify 16 full|16|101.04|101.5|-0.5|63.39|1.59|75.63|1.34|25.41|3.57%|0.81%|stream: bytes + exp + FP8 mma must overlap; MUFU alone 39 µs|
|verify 32 sliding|46|14.85|14.4|0.4|3.72|3.99|5.76|2.58|9.09|2.06%|0.49%|chain: latency chain; bytes only 3.72 µs|
|verify 32 full|16|184.09|184.2|-0.1|126.78|1.45|149.85|1.23|34.24|2.77%|0.64%|stream: bytes + exp + FP8 mma must overlap; MUFU alone 78 µs|

Top shapes by single-shape score gain at 100% of the gap: verify 1 sliding 6.71%, verify 1 full 4.53%, decode 1 sliding 4.20%, decode 1 full 3.60%, verify 16 full 3.57%, verify 16 sliding 3.11%.

Per-load time, sum of w x ours in µs: decode 1 408, decode 16 1770, decode 32 3109, verify 1 542, verify 16 2141, verify 32 3629.

### bf16 (job 6c7a0061944e4339)

|shape|w|ours|GPU|ovh|sol|ours/sol|bound|ours/bound|gap|gain@100%|gain@25%|class and reason|
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|decode 1 qkv_gate m1|62|8.79|6.7|2.1|7.87|1.12|8.31|1.06|0.48|0.23%|0.06%|stream: 12-bit stream|
|decode 1 o m1|62|5.22|4.2|1.0|3.50|1.49|4.47|1.17|0.75|0.36%|0.09%|stream: 12-bit stream|
|decode 1 pooler m1|62|13.68|13.0|0.7|13.99|0.98|13.69|1.00|0.00|0.00%|0.00%|stream: 12-bit stream|
|decode 1 unpooler m1|62|3.50|2.6|0.9|1.75|2.00|3.00|1.17|0.50|0.24%|0.06%|chain: 12-bit stream; latency floor binds|
|decode 1 router m1|62|3.84|2.4|1.4|1.75|2.19|3.45|1.11|0.39|0.19%|0.05%|stream: 8 MB FP32 stream|
|decode 16 qkv_gate m16|62|9.97|8.2|1.8|7.93|1.26|8.36|1.19|1.61|0.56%|0.14%|stream: 12-bit stream|
|decode 16 o m16|62|5.25|4.5|0.8|3.56|1.47|4.53|1.16|0.72|0.25%|0.06%|stream: 12-bit stream|
|decode 16 pooler m16|62|14.12|14.0|0.1|14.06|1.00|13.75|1.03|0.37|0.13%|0.03%|stream: 12-bit stream|
|decode 16 unpooler m16|62|14.01|12.7|1.3|14.06|1.00|13.75|1.02|0.26|0.09%|0.02%|stream: 12-bit stream|
|decode 16 router m16|62|5.25|4.0|1.2|1.81|2.90|3.52|1.49|1.73|0.61%|0.15%|stream: 8 MB FP32 stream|
|decode 32 qkv_gate m32|62|10.58|9.9|0.7|8.00|1.32|8.43|1.26|2.15|0.70%|0.17%|stream: 12-bit stream|
|decode 32 o m32|62|6.11|4.8|1.3|3.62|1.69|4.58|1.33|1.53|0.50%|0.12%|stream: 12-bit stream|
|decode 32 pooler m32|62|15.11|15.2|-0.1|14.14|1.07|13.82|1.09|1.29|0.42%|0.10%|stream: 12-bit stream|
|decode 32 unpooler m32|62|15.03|14.0|1.0|14.14|1.06|13.82|1.09|1.21|0.39%|0.10%|stream: 12-bit stream|
|decode 32 router m32|62|5.43|4.8|0.6|1.86|2.92|3.58|1.52|1.85|0.60%|0.15%|stream: 8 MB FP32 stream|
|verify 1 qkv_gate m8|62|9.93|8.3|1.6|7.90|1.26|8.34|1.19|1.59|0.69%|0.17%|stream: 12-bit stream|
|verify 1 o m8|62|5.74|4.2|1.5|3.53|1.63|4.50|1.28|1.24|0.54%|0.13%|stream: 12-bit stream|
|verify 1 pooler m8|62|13.96|13.3|0.7|14.02|1.00|13.71|1.02|0.25|0.11%|0.03%|stream: 12-bit stream|
|verify 1 unpooler m8|62|4.51|3.7|0.8|1.76|2.56|3.00|1.50|1.51|0.65%|0.16%|chain: 12-bit stream; latency floor binds|
|verify 1 router m8|62|5.19|3.8|1.4|1.78|2.92|3.48|1.49|1.71|0.74%|0.18%|stream: 8 MB FP32 stream|
|verify 16 qkv_gate m128|62|15.24|14.0|1.2|8.42|1.81|11.26|1.35|3.98|0.98%|0.24%|stream: bytes of 16-bit weights + wgmma ramp|
|verify 16 o m128|62|7.30|5.4|1.9|3.99|1.83|6.07|1.20|1.23|0.30%|0.07%|stream: bytes of 16-bit weights + wgmma ramp|
|verify 16 pooler m128|62|22.29|20.0|2.3|14.64|1.52|18.54|1.20|3.75|0.92%|0.22%|stream: bytes of 16-bit weights + wgmma ramp|
|verify 16 unpooler m128|62|18.91|18.8|0.1|14.64|1.29|18.54|1.02|0.37|0.09%|0.02%|stream: bytes of 16-bit weights + wgmma ramp|
|verify 16 router m128|62|6.23|4.7|1.5|2.21|2.82|3.99|1.56|2.24|0.54%|0.13%|stream: bytes of 16-bit weights + wgmma ramp|
|verify 32 qkv_gate m256|62|19.75|20.0|-0.2|9.77|2.02|14.43|1.37|5.32|1.05%|0.26%|stream: tensor-core bound (the floor is the 989 TFLOP/s roof)|
|verify 32 o m256|62|9.26|8.1|1.2|4.48|2.07|7.37|1.26|1.89|0.36%|0.09%|stream: tensor-core bound (the floor is the 989 TFLOP/s roof)|
|verify 32 pooler m256|62|27.47|26.8|0.7|17.37|1.58|24.56|1.12|2.91|0.57%|0.14%|stream: tensor-core bound (the floor is the 989 TFLOP/s roof)|
|verify 32 unpooler m256|62|23.83|23.8|0.0|17.37|1.37|24.56|0.97|0.00|0.00%|0.00%|stream: tensor-core bound (the floor is the 989 TFLOP/s roof)|
|verify 32 router m256|62|7.20|5.9|1.3|2.68|2.69|4.54|1.59|2.66|0.52%|0.13%|stream: bytes of 16-bit weights + wgmma ramp|

Top shapes by single-shape score gain at 100% of the gap: verify 32 qkv_gate m256 1.05%, verify 16 qkv_gate m128 0.98%, verify 16 pooler m128 0.92%, verify 1 router m8 0.74%, decode 32 qkv_gate m32 0.70%, verify 1 qkv_gate m8 0.69%.

Per-load time, sum of w x ours in µs: decode 1 2172, decode 16 3013, decode 32 3240, verify 1 2438, verify 16 4338, verify 32 5426.

### moe (job 41f596feaa60434e)

|shape|w|ours|GPU|ovh|sol|ours/sol|bound|ours/bound|gap|gain@100%|gain@25%|class|
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|decode 1 m1|60|25.09|29.1|-4.0|11.42|2.20|14.77|1.70|10.32|9.23%|1.82%|stream|
|decode 16 m16|60|93.99|100.4|-6.4|52.52|1.79|62.90|1.49|31.09|6.92%|1.45%|stream|
|decode 32 m32|60|129.18|136.6|-7.4|70.76|1.83|84.26|1.53|44.92|7.38%|1.53%|stream|
|verify 1 m8|60|65.05|69.8|-4.8|37.32|1.74|45.10|1.44|19.95|6.29%|1.34%|stream|
|verify 16 m128|60|180.09|208.3|-28.2|101.34|1.78|120.07|1.50|60.02|6.99%|1.46%|stream|
|verify 32 m256|60|216.19|193.3|22.9|108.94|1.98|128.96|1.68|87.23|8.99%|1.79%|stream|

Top shapes by single-shape score gain at 100% of the gap: decode 1 m1 9.23%, verify 32 m256 8.99%, decode 32 m32 7.38%, verify 16 m128 6.99%, decode 16 m16 6.92%, verify 1 m8 6.29%.

Per-load time, sum of w x ours in µs: decode 1 1505, decode 16 5639, decode 32 7751, verify 1 3903, verify 16 10805, verify 32 12971.

### decode_layer (job 9b0d0973f11a4758)

This run was later withdrawn from the board (the submitted decode_layer entry now is job 8aa6ec880e394012, 2.083x). The numbers below are kept as analysis of that profile.

|shape|w|ours|GPU|ovh|sol|ours/sol|bound|ours/bound|gap|gain@100%|gain@25%|class|
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|decode 1 sliding|46|77.53|83.5|-6.0|40.02|1.94|48.26|1.61|29.27|5.52%|1.20%|stream|
|decode 1 full|16|82.49|86.2|-3.7|44.18|1.87|53.13|1.55|29.36|1.70%|0.41%|stream|
|decode 16 sliding|46|135.13|144.6|-9.5|76.72|1.76|91.24|1.48|43.89|4.01%|0.90%|stream|
|decode 16 full|16|212.62|230.6|-18.0|137.46|1.55|162.36|1.31|50.26|1.47%|0.35%|stream|
|decode 32 sliding|46|157.66|176.2|-18.5|84.51|1.87|100.36|1.57|57.30|4.14%|0.93%|stream|
|decode 32 full|16|309.22|328.7|-19.5|208.10|1.49|245.08|1.26|64.14|1.48%|0.35%|stream|
|verify 1 sliding|46|108.99|122.8|-13.8|62.29|1.75|74.34|1.47|34.65|4.48%|1.00%|stream|
|verify 1 full|16|117.39|138.8|-21.4|66.44|1.77|79.20|1.48|38.19|1.56%|0.37%|stream|
|verify 16 sliding|46|255.00|263.6|-8.6|126.61|2.01|149.66|1.70|105.34|5.65%|1.22%|stream|
|verify 16 full|16|344.45|354.7|-10.2|188.68|1.83|222.34|1.55|122.11|2.02%|0.48%|stream|
|verify 32 sliding|46|338.21|319.4|18.8|143.81|2.35|169.80|1.99|168.41|6.90%|1.44%|stream|
|verify 32 full|16|495.90|522.3|-26.4|267.18|1.86|314.26|1.58|181.64|2.23%|0.52%|stream|

Top shapes by single-shape score gain at 100% of the gap: verify 32 sliding 6.90%, verify 16 sliding 5.65%, decode 1 sliding 5.52%, verify 1 sliding 4.48%, decode 32 sliding 4.14%, decode 16 sliding 4.01%.

Per-load time, sum of w x ours in µs: decode 1 4886, decode 16 9618, decode 32 12200, verify 1 6892, verify 16 17241, verify 32 23492.

### Sensitivity of the ceilings to ETA and LF (attention and bf16)

|setting|attention stream|attention chain|bf16 stream|bf16 chain|
|-|-|-|-|-|
|ETA 0.80, LF 3|6.3%|35.0%|11.4%|0.0%|
|ETA 0.854, LF 3 (base)|9.9%|36.3%|14.0%|0.9%|
|ETA 0.90, LF 3|12.9%|37.4%|18.0%|0.9%|
|ETA 0.854, LF 2.5|9.9%|41.0%|15.2%|0.0%|
|ETA 0.854, LF 3.5|9.9%|32.1%|12.8%|1.4%|

Moving ETA by about 0.05 changes the stream ceilings by roughly 3 to 4 points, and moving LF by 0.5 µs changes the attention chain ceiling by 4 to 5 points. We have not measured either constant on this hardware, so the chain ceiling in particular is better read as somewhere between 32 and 41% than as 36.3%.

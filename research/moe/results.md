# muse_moe_w4a16: judged runs

Every MoE run we kept notes on, from the vjeux baseline at 2.332x to our submitted file at 2.362x, is listed here with per-shape speedups and our reading of them.

## Shapes

The problem scores six loads: decode with 1, 16 and 32 tokens, and verify with 1, 16 and 32 requests of 8 tokens each (8, 128 and 256 tokens). The kernel picks a build by token count: a tiny build up to 4 tokens, tiny2 up to 8, mid up to 16, and a wgmma build (WGS) from 17 up.

Our changes touch only the wgmma build. Decode 32, verify 16 (128 tokens) and verify 32 (256 tokens) can move. Decode 1, decode 16 and verify 1 (8 tokens) run code that did not change, so we treat them as controls.

## Runs

|run|job|score|95% interval|time us|
|-|-|-|-|-|
|vjeux file, check|c7c028cd24304193|2.332x|2.297-2.367x|5,766.5|
|vjeux file, board entry|41f596feaa60434e|2.329x|2.294-2.364x|5,773.6|
|our first build (check 1)|2eaae7ab93824649|2.3431x|2.3080-2.3782x|5,739.6|
|our frozen-dispatch build (v3, check)|03072c0ab2854c2e|2.3442x|2.3091-2.3794x|5,736.9|
|v3, submitted|c663b9dfc629461b|2.362x|not recorded|not recorded|

The first four passed 5/5 tests; the submitted run's test count is not recorded. Relative L2 error in check 1 was 2.385e-3 to 2.563e-3. The board entry was rank 397 of 1,387 when we started. The board run, check 1 and the v3 check all ran on cluster nyc3b, each on a different node. The cluster of the first vjeux check is not recorded.

The submitted file is v3 byte for byte (sha256 prefix c0828b74, from the repo's results table). The submit was a fresh judge draw of the same bytes and landed at 2.362x, at the top of the 2.33 to 2.36x range we had forecast from the two checks.

## Speedup per shape

|shape|M|vjeux check c7c028cd|vjeux board 41f596fe|check 1 2eaae7ab|v3 03072c0a|
|-|-|-|-|-|-|
|decode 1|1|3.125|3.083|3.067|3.114|
|decode 16|16|1.975|1.976|1.959|1.955|
|decode 32|32|1.620|1.625|1.667|1.668|
|verify 1|8|2.213|2.221|2.199|2.189|
|verify 16|128|2.578|2.557|2.609|2.604|
|verify 32|256|2.820|2.841|2.880|2.866|

Build used by each shape in check 1 and v3: decode 1 tiny, verify 1 tiny2, decode 16 mid (all unchanged from vjeux); decode 32, verify 16 and verify 32 wk6p (the new wgmma build).

Absolute times in microseconds, SGLang then ours:

|shape|vjeux check|v3|
|-|-|-|
|decode 1|77.3 / 24.7|77.3 / 24.8|
|decode 16|189.0 / 95.7|189.3 / 96.8|
|decode 32|213.2 / 131.6|213.5 / 128.0|
|verify 1 (8 tokens)|151.2 / 68.3|151.4 / 69.2|
|verify 16 (128 tokens)|460.5 / 178.6|473.3 / 181.8|
|verify 32 (256 tokens)|596.5 / 211.5|600.9 / 209.7|

SGLang's own time moves between runs too (verify 16: 460.5 us in one, 473.3 us in another), which is part of why a single run's per-shape ratio is soft. Kernel time per shape in check 1's profile, in the order above: 30.3, 81.0, 104.5, 56.6, 166.4 and 217.9 us.

## How we read check 1 and v3

Check 1 and v3 run the same device code for the wgmma build with the same dispatch, and they agree (2.3431x and 2.3442x).

Against the board run of the vjeux file, the three control shapes came out lower in both checks. Their geometric-mean ratio was 0.992 for check 1 and 0.995 for v3. The three changed shapes came out higher:

|shape|check 1 vs board|v3 vs board|
|-|-|-|
|decode 32|+2.6%|+2.6%|
|verify 16|+2.0%|+1.8%|
|verify 32|+1.4%|+0.9%|

The geometric mean of the changed shapes is +2.0% (check 1) and +1.8% (v3) over the board run. Dividing by the control ratio gives about +2.8% and +2.3%. Check 1 also timed the new build against the old one inside prepare, on the judge's GPU, and measured 2.2 to 2.5% for the same three shapes (ratios 0.9753, 0.9800 and 0.9753 against the original build). The in-process timing and the judge table agree on size, and the movement sits on the shapes we changed. We think the gain is real. It is also small: about +0.6% on the score, well inside the interval overlap.

Before the v3 check we expected 2.352x, taking the board run's control values and check 1's heavy shapes. The result was 2.3442x because the controls did not come back to the board's values.

### The memory theory, and why it was wrong

Check 1 held both vjeux's WGS weight copy and our 6-bit copy, about 840 MiB extra. alygg77 had reported that extra prepare-time memory slows unchanged shapes, so low controls looked explained. v3 drops the second copy and holds 24 MiB less than the board entry (840 MiB of 6-bit copy where vjeux allocates 864 MiB). Its controls were still low: decode 16 and verify 1 stayed 0.9 to 1.4% below the board in both checks, while decode 1 came back (+1.0% in v3, -0.5% in check 1).

That is evidence against a memory explanation. It rests on two runs, so it is inference. The simpler reading is that the board run's decode 16 and verify 1 were favorable draws.

We also briefly misread the judge's `prepare_pad` field as our own extra memory. It is a random layout pad the judge adds on every run: alygg77 reports pads from 0.85 to 30.5 MB in 512-byte steps (H200_MEASURED_FACTS section 5; they do not say MB or MiB, and check 1's 32.3 MB sits a little above that range), and all of ours are multiples of 512 (inference: we rely on that statement). The values:

|run|prepare_pad B|baseline_prepare_pad B|prepare s|
|-|-|-|-|
|board entry|4,228,608|21,607,424|137.3|
|check 1|32,270,848|29,744,640|85|
|v3|10,886,656|25,366,016|41.8|

The "+2.5 MB" we first noticed in check 1 was the difference between two random draws.

## Prepare time

The board entry spent 137.3 s in prepare (limit 300 s). Check 1 took 85 s because it compiled seven builds and ran a gate and an A/B. v3 compiles four builds and takes 42 s:

|build|compile s|registers|local B|shared B|
|-|-|-|-|-|
|tiny2|34|149|0|214,784|
|mid|35|153|0|214,784|
|tiny|38|149|0|214,784|
|wk6p|42|96|0|231,168|

State tensors were 3,589 MiB, with 11,700 MiB reserved.

## Verdict

To call it a clear win we wanted a score above the old entry's upper bound, 2.364x. 2.344x does not clear that, and the controls did not recover. We forecast that a fresh draw would land between 2.33 and 2.36x. The submit scored 2.362x. With single-run noise around 1%, we would not read anything into the last digit.

The scale bytes of the judge's weights differ per run: 93..126 and 94..126 in check 1, 94..126 and 95..126 in v3. All sit inside the 6-bit range the build needs.

## Files and checksums

|role|sha256|
|-|-|
|vjeux's file, the board entry|126d0c6c...a06fb8 (first and last digits only; the middle was not recorded)|
|check 1: three candidate builds plus the prepare-time A/B|37fd0831aa386e3cb645d46f88c14bb66185c6fba7b846dbaeead1065f0ea2c1 (207,594 bytes)|
|v3: frozen dispatch, the submitted file|c0828b748ead01558c77ac36b84180431cc2307ab88f0048132e0630cc24d511 (198,445 bytes)|

The device source in v3 is byte-identical to check 1's. The three embedded source blocks (CUDA_SRC, CPP_SRC, BW_SRC) have sha256 prefixes 40f78ec6347c23f8, d5a77b0fe9c65574 and 2008496bd3fdb5f7 in both files, which is why check 1's compile log (96 registers, 0 B local) applies to v3.

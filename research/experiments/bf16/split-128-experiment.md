# Twelve-bit weights inside the m=128 GEMM (the rs128 experiment)

This file records two judged checks of a register-sourced wgmma kernel that reads our 12-bit weight packing for the 128-token bf16 projections, what the phase timers showed, and why we stopped after the second check.

Short version: the kernel was correct on the judge and about twice as slow as the dense kernels it was meant to beat. A prologue fix we expected to matter moved the totals by 2 to 4 us. The phase timers then put the loss in the main loop, and we could not say why it was slow. We closed the line without a win.

## Where this sits

The bf16 entry (problem muse_bf16_proj) stores each projection matrix in a lossless 12-bit packing: 8 raw bits per weight plus a 4-bit exponent code. Weights whose exponent falls outside the code window are "escapes" and live in a small side table. The packing cuts HBM traffic by 25%, and it is why the skinny kernels (m up to 32) are fast.

Above m = 32 the entry uses dense paths: a custom TMA GEMM (gq, with a variant gqx for qkv_gate) or cuBLAS through torch (torchT is the transposed formulation). The verify-16 load is m = 128 (16 requests, 8 tokens each; inference from the arithmetic, the tune log labels it m=128). On the board run (job 6c7a0061944e4339, file sha256 prefix 1a2bf6ba, score 1.491) the m=128 picks were:

|Layer|Pick|Raw tune time|Verify-16 cell, ours|
|-|-|-|-|
|qkv_gate|gqx|18.34 us|15.24 us|
|o|torchT|10.62 us|7.30 us|
|pooler|gq|24.86 us|22.29 us|
|unpooler|torchT|22.78 us|18.91 us|

Raw tune times are the autotuner's graph-replay numbers and include about 4.9 us of empty-graph cost, so kernel time is that much lower.

The entry already contained a kernel with the 12-bit design we wanted, called `wide`, switched off for bf16 by a constant because it never beat cuBLAS for m between 33 and 256. No timing of it was recorded anywhere we could find. We reopened the idea because a static look at its SASS suggested a structural reason for the loss.

## Why `wide` loses (static reading, inference)

We built `wide` with NVRTC 13.2 for compute_90a and ptxas 12.9 for sm_90a. It uses 105 registers, spills nothing, and ptxas does not print a wgmma serialization warning. So the loss is not a compiler accident.

In the main loop, each k16 step does this: wait for the wgmma issued two steps back, load the code words and low bytes from shared memory, run about five dependent integer ops to rebuild the bf16 A fragments, then issue the next HGMMA. That chain is roughly 60 to 70 clocks. One HGMMA m64n128k16 takes 64 clocks on a sub-partition. With one warpgroup per SM, nothing else fills the tensor pipe while the decode runs. The overlap window is one k-step wide and the decode chain is as long as the window. We never measured this directly. It is our reading of the SASS.

## The rs128 design

The idea was to give the decode a much wider window and a second warpgroup to hide behind:

- One CTA covers 128 weight rows, as two consumer warpgroups of 64 rows each, plus one producer warp that issues every copy (the x tile by TMA, the weight tiles by bulk copy).
- Work is organized in units of 64 k. A unit is four k16 wgmmas committed as one group. After the commit, `wait_group 1` retires the previous group, and the next unit is loaded and decoded into the register buffer the previous group used. The decode then has a window of 256 tensor clocks per warpgroup (512 with the other warpgroup interleaved) against roughly 100 to 150 clocks of shared-load plus decode latency. The four decode chains inside a unit are independent.
- One 16 KB x tile feeds all 128 rows (12 KB of packed weights), so x traffic per weight byte halves compared with 64-row CTAs.
- Weights come from the existing compressed stream, so no new copy of any matrix is made.
- K is split across CTAs (pooler 4 ways, unpooler 2, o 2). Partial sums go to the existing FP32 workspace, and the last CTA to arrive adds them in a fixed order, so the result is deterministic.
- Escapes are patched into the accumulators after the main loop, from a per-CTA table.
- Dispatch is limited to 97 <= m <= 128. The kernel is taken only if it beats the best existing path by more than 3% at m=128 and at m=97, in the entry's own cold-L2 graph timing. Otherwise every layer keeps its old pick.

That last rule matters for reading the scores below. If rs128 loses the tuning, the judged run executes exactly the board's kernels, and the headline score is the board's score plus run-to-run noise.

Static checks before the first run: NVRTC plus ptxas gave 127 registers, no spill and no serialization message. In SASS, each unit shows one ARRIVE, four HGMMA, one `DEPBAR.LE gsb0, 0x1`, and the next unit's loads and decode between that wait and the next ARRIVE. A CPU model of the stream-to-fragment path was bit-exact on synthetic weights with about 1200 forced escapes (relative error against an FP64 reference between 0 and 1.7e-17 at m = 128, 100 and 33). A stubbed test of the Python dispatch and self-test code passed 31 of 31.

## Check 1

Job c440acdd631e4328, file sha256 prefix c5c8c39d. Passed 9/9, 1.505x (95% interval 1.483 to 1.528). Every rs128 variant passed the FP64 self-test at m = 33 and 128 and the bitwise double-run at m = 100 and 128, so none was disabled.

The tuner's raw numbers at m=128, in us:

|Layer|rs128 variants|Best existing pick|
|-|-|-|
|pooler|46.53 / 44.16 / 46.37|gq 25.25|
|unpooler|43.14 / 41.57 / 45.28|torchT 22.85|
|o|20.48 / 21.09|torchT 10.85|

rs128 lost by about 2x everywhere, so no pick changed. The verify-16 cells matched the board run within noise (pooler 22.1, unpooler 19.0, o 7.2 us against 22.29, 18.91, 7.30). Four pipeline stages were slightly faster than seven.

Our first diagnosis was wrong. We blamed the consumer prologue: it stages the x values that the escapes need with a loop whose every iteration does two dependent global loads. The SASS confirmed that structure (the compiler unrolled by four: four loads of table entries, four dependent loads of x, four shared stores), and the arithmetic looked like it could account for 15 to 30 us on the pooler. The epilogue loop also paid one L2 round trip per iteration.

## Check 2

Version 2 removed those serial pieces. The escape table is copied with cp.async, the x values for the escapes are gathered with 4-byte cp.async after the first unit's wgmma is issued, the correction scan uses a ballot over 32 table entries per step, and the epilogue issues its split loads in batches. It also recorded per-phase clock64 stamps in a separate launch at the end of tuning.

Job 635004eab6f9489a, file sha256 prefix 18e74e37. Passed 9/9, 1.491x (95% interval 1.469 to 1.514), again no variant disabled and no pick changed. Verify-16 cells: pooler 22.3, unpooler 18.4, o 7.1 us.

Raw tune times at m=128, in us:

|Layer|rs128 variants|Best existing pick|
|-|-|-|
|pooler|44.61 / 43.23 / 44.54|gq 25.76|
|unpooler|39.36 / 38.40 / 41.82|torchT 23.46|
|o|18.69 / 18.75|torchT 10.85|

The prologue fix moved the totals by only 2 to 4 us (pooler variant 0: 46.5 to 44.6, unpooler variant 0: 43.1 to 39.4, o variant 0: 20.5 to 18.7). The check-1 diagnosis was mostly wrong.

The phase timers say where the time goes. Median cycles per CTA (the SM clock works out to about 1.77 GHz, from the sum of phases over the CTA time):

|Variant|Fill|Main loop|Corrections|Epilogue|CTA time|
|-|-|-|-|-|-|
|pooler/0 (16t_r192, 6 stages)|4222|50934|2836|8774|37.7 us|
|pooler/1 (same stream, 4 stages)|3415|47830|2876|8491|35.6 us|
|pooler/2 (32_r144, 6 stages)|3772|50549|2880|8661|37.1 us|
|unpooler/0 (32_r144, 7 stages)|4185|43252|2129|9439|33.2 us|
|unpooler/1 (same stream, 4 stages)|3149|40790|2094|9123|31.4 us|
|unpooler/2 (16t_r72, 7 stages)|3918|46913|2115|9293|35.2 us|
|o/0 (8h_r88, 7 stages)|3123|9360|904|9830|13.0 us|
|o/1 (32h_r100, 7 stages)|4120|9604|921|10103|13.7 us|

Reading it:

- The main loop takes 1170 to 1590 cycles per 64-k unit (pooler variant 1: 47830 over 32 units is about 1495; unpooler variant 1: 40790 over 32 is about 1275; o variant 0: 9360 over 8 is 1170). The tensor work per unit is about 512 cycles, and the HBM time per unit at 4 TB/s is about 700 cycles. We are at roughly twice the slower of the two.
- The epilogue costs 8.5K to 10K cycles, about 5 us. That covers the partial stores, two thread fences, the ticket, up to about 2.5 us of skew between split peers, and the reduce. The fill costs about 2 us.
- Stage count barely matters, and fewer stages were slightly faster. A latency-bound ring would speed up with more stages, so this argues against a latency limit. Something throughput-like is in the way, and we could not identify it without a profiler. By our averages nothing obvious is saturated: HBM about 1.8 TB/s, L2 to SM about 4.1 TB/s, tensor pipe about 31%. The loop SASS has no `DEPBAR 0` and no extra barrier inside it.
- Suspects, none proven: the DRAM rate under the access order of the compressed stream (unpooler at 7 stages: 4-tile copies took 43.3K cycles against 46.9K for 2-tile copies, about 8%); register-A HGMMA throughput; and the per-unit `MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S` that `fence.proxy.async.shared::cta` lowers to.

## Verdict

NO-GO. We did not launch a third check, and we did not submit either file.

To beat gq by the 3% margin on the pooler, the main loop would have to reach about 700 to 750 cycles per unit (measured about 1500), and the epilogue would need a different design (we did not work out which). Those are two unknown-cause fixes. A score-level gain would need pooler and unpooler each about 4 to 5 us under today's picks, which means rs128 kernels of 15 to 16 us against the 31 to 37 us we measured.

The other layers do not change this. The o layer already lost 1.7x (18.7 against 10.85 us raw). A qkv_gate version, which we did not build, would take about 18 units per CTA at roughly 1300 cycles (about 13 us), plus 2 us fill, 5 us epilogue and the corrections: over 20 us, against about 14 us for gqx (18.2 us raw).

Both scores (1.505 and 1.491) are noise on the board's own kernels, because rs128 was never picked. Neither shows per-cell movement.

One regret: the phase timers went into version 2, and the first check was spent on a diagnosis by arithmetic that turned out to be mostly wrong. If we did it again the timers would be in version 1.

We would reopen this only with a profiler stall breakdown of the main loop on an H200 (warp-state, shared-memory and tensor-pipe metrics in Nsight Compute), not with another blind variant.

## Two smaller notes

- `fence.proxy.async.shared::cta` shows up in SASS as a full CTA memory barrier plus a view-async fence. We did not isolate its cost inside a hot loop. If you add a proxy fence to every stage of a weight ring in another kernel, time it on the judge before assuming it is free.
- If an m=128 path like this is ever shipped, it should cover a contiguous bucket (33 to 128), the self-test should include a mid-bucket value such as 64 or 100, and the entry should time 127 and 129 once during prepare, so that nearby shapes are not tuned by accident.

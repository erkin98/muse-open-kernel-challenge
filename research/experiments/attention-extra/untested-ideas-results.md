# Five more attention ideas and what became of them

This file records five attention ideas from a list of untried suggestions (one new judged check, one compiler test, one earlier check re-read and two desk assessments) with the numbers behind each verdict.

The base is the public attention entry we ran unchanged (2.605x, job c20df44ab71d4005; a second run of the same code gave 2.589x, job 705c7dd9f34446dd). The shorter list that ties these into the rest of the attention work is in [ideas-tried-and-results.md](../../attention/ideas-tried-and-results.md), and the judged attention work is in the experiments/attention folder next door. The ideas for the MoE, bf16 and decode-layer problems are in [untested-ideas-moe-and-bf16.md](../general/untested-ideas-moe-and-bf16.md).

Scored rows below use short names: d1s is decode 1 with a sliding-window layer, v32f is verify 32 with a full layer, and so on (d = decode, v = verify, the number is the batch, s = sliding, f = full).

## A1: packed half-precision softmax and FP16 QK^T at q_len 8

Idea: the exponential runs on the special-function unit (MUFU). If a packed half2 or bf16x2 exp2 handles two values per instruction, the number of MUFU operations halves at q_len 8, and an FP16 QK^T would feed it.

Test: we wrote tiny PTX kernels and read the SASS. ptxas 12.9.86 (PTX 8.8, sm_90a) lowers `ex2.approx.f16x2` to two `MUFU.EX2.F16` plus a PRMT, and `ex2.approx.ftz.bf16x2` to two `MUFU.EX2.BF16` plus a PRMT. NVRTC 13.2 does no better: `h2exp2` on a half2 or bf162 compiles to two conversions (HADD2.F32 or PRMT) and two MUFU.EX2, so the library does not emit a packed op. A public repository's SASS built with CUDA 13.0.88 shows the same, and lee101's public notes report the same result.

Per pair of values the packed path costs one HFMA2, two MUFU, a PRMT and one F2FP conversion: five instructions. The current path costs two FFMA, two MUFU and one F2FP: also five. There is nothing to save.

One side result: an FP8 wgmma with FP16 accumulate (`m64n32k32`, e4m3 inputs) assembles into a single QGMMA. We did not measure it.

Verdict: no-go. The mechanism is refuted by the SASS. We tested only the exp half of the idea. Whether an FP16 QK^T pays on its own was not measured, but without the packed exp it has no motivation.

## A2: bit-trick exp2 producing the e4m3 P codes directly

Idea: build the e4m3 code of exp2(x) with integer arithmetic instead of the MUFU plus a float-to-e4m3 convert.

This was already measured in an earlier attention check of ours (job 24a52be614264969). That version computed a linear-mantissa approximation with one FFMA (magic constant) plus a shift and add, then converted four values at once, on interior tiles, with a byte test and a warp vote for edge cases. Result: v16f 101.0 to 101.1 us, v32f 184.1 to 183.6 us. Identical-code runs of the base gave 101.0 and 101.6 for v16f, 184.1 and 185.6 for v32f, so the movement is inside the spread.

The variant that writes the byte directly only drops the convert, which is about half an instruction per element. We did not spend a check on it.

Verdict: no-go. It is the same mechanism as that earlier check.

## A3: four workspace copies for decode 1 sliding

Idea: in the single-request sliding plan, 32 CTAs write partial results into one workspace copy using atomic reductions on the same addresses. Spreading them over four copies would remove same-address serialization, which we estimated at about 17 ns per operation over 32 CTAs, up to 0.5 us. We predicted d1s would drop by 0.2 to 0.3 us from baselines of 4.94 and 5.17 us. The prior was low, because the base author had chosen one copy on purpose, and an earlier note says four copies were worth trying only on top of a winning cold-code warm-up, which had lost.

Change: one line in the tuned-plan table, `nc=4` for the key (batch 1, q_len 1, sliding). Nothing else.

Job 66be38f978b54004, file sha256 prefix 259f132d, cluster nyc3b. Passed 16/16, 2.567x (95% interval 2.529 to 2.606). The identical-code base runs were c20df44ab71d4005 (2.605x) and 705c7dd9f34446dd (2.589x), both on nyc3b. Yours-time per row (us), this run first:

|Row|nc=4 file|705c7dd9f34446dd|c20df44ab71d4005|
|-|-|-|-|
|d1s|5.53|5.17|4.94|
|d1f|10.89|10.66|11.27|
|d16s|8.39|8.54|8.37|
|d16f|87.14|86.98|86.58|
|d32s|10.24|10.25|10.00|
|d32f|166.50|165.79|165.55|
|v1s|6.43|6.59|6.80|
|v1f|14.20|14.47|14.31|
|v16s|11.63|11.63|11.39|
|v16f|101.95|101.64|101.04|
|v32s|14.90|14.65|14.85|
|v32f|186.30|185.62|184.09|

The eleven control rows moved by a factor of 1.0003 and 0.9998 against the two baselines. d1s, relative to the controls, came out at x0.926 and x0.864, so 0.36 to 0.59 us slower, outside the 0.23 us spread between the two identical-code runs.

Verdict: no-go, and one copy stays. Our reading (inference, no per-phase profile of the finalize) is that four copies cost more in the final merge than they save in reduction serialization.

## A4: a 128-byte bulk copy per KV row

Idea: load each KV row with one 128-byte bulk (TMA-style) copy.

We did not run this. A back-of-envelope estimate puts the heavy full rows (decode 16 and 32) at 86 to 90% of the ceiling for 128-byte random-row reads already, so the headroom is small. lee101 reports that the bulk 128-byte row copy was negative because of the copy-operation rate. It would also need an async-proxy fence, and the replay race it could introduce can only be ruled out with a long soak (30,000 to 60,000 graph replays) that we cannot run without a spare GPU.

Verdict: no-go. The expected gain is under the noise floor, there is a corruption risk, and we cannot soak-test it.

## A5: the two exact-key tuned overrides (assessment only)

The base has a tuned-plan table with two exact keys, keyed on (batch, q_len, sliding, KV pairs): verify 32 sliding (v32s) and decode 1 full (d1f). The challenge rules discourage tuning that only works at exactly the scored sizes, and the final review reruns the scored and nearby shapes. So we asked what each override is, and whether a rule could replace it.

What each override does:

- v32s. The generic plan uses 8 splits (it targets 2 CTAs per SM once there are 24 or more requests). The override uses 4 splits, which is `sms // requests`, one CTA per SM, the formula the file already uses for 16 to 23 requests. A CPU run of the plan function shows the generic plan stays at 8 splits at B = 31 and B = 33, so the override is an isolated value at exactly B = 32.
- d1f. The override equals the plan the file already builds for one request at q_len 8: 252 splits (from the headroom function), 8 workspace copies and a cooperative finalize. The generic q_len 1 branch would give 264 splits, 4 copies and no cooperative finalize. One request is its own regime (B of 2 and up goes to the flat grid), so this is a regime rule written as a key, not a peak at one size.

Generic plans for the v32s family (q_len 8, window 2048), splits by batch B:

|B|Splits|
|-|-|
|8|33|
|15|17|
|16|8|
|17|7|
|24|11|
|28|9|
|31|8|
|32|8 (the override sets 4)|
|33|8|
|34|7|
|36|7|
|48|5|
|64|4|
|128|2|

We wrote a rule-based replacement: a single full request uses the q_len 8 grid, and 32 or more requests at q_len 8 sliding use one CTA per SM, while 24 to 31 keep 2 CTAs per SM. Over a grid of B from 1 to 512, q_len 1 and 8, window 0 and 2048, and 1, 2 or 4 KV pairs (6,144 points), the final plans are identical at all 12 scored shapes. At 177 unscored points they differ, all of them q_len 8 sliding: pairs 1 with B from 33 to 132 (100 values), pairs 2 with B from 16 to 66 (51), pairs 4 with B from 8 to 33 (26). Those neighbours are unmeasured. Shipping unmeasured changes at nearby shapes is exactly what we wanted to avoid, and a judged check cannot time unscored shapes. The only way to measure them would be a prepare-time self-timing on synthetic data.

Risk read: v32s is the exposed one, as an isolated key. d1f is a regime and low risk. If a reviewer counts exact keys as tuning, the cost is the v32s gain, which we have not measured ourselves. Nothing here involves caching or reduced precision.

Options: (1) keep as is; (2) replace both keys by the rule above; (3) delete the keys and let v32s revert to 8 splits, which needs a check to see the loss and which we did not run.

Verdict: assessed, no check run. We would keep the base as it is. If zero exact keys is wanted, replace the d1f key only, since that rule changes no unscored point, and leave v32s. We have a proposal file carrying both rules, but it was not checked and has no measurement behind it.

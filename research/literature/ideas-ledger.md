# Ideas ledger

This file lists the optimisation ideas we priced against our four entries (attention, bf16, moe, decode_layer), what we checked for each, and the verdict, including every negative result.

Two bars govern the verdicts, and they are separate. First, a change is only visible if it exceeds about 0.5 us on a 5 to 15 us shape or about 1% on a 100 to 190 us shape, and a headline score carries a 95% interval of at least plus or minus 1.5%, so an idea needs roughly 2.5 to 3% expected whole-score gain to justify a judge check. Second, a check only counts as a win if it beats the board entry's confidence-interval upper bound on the same shapes. Public board rows keep each team's best draw, which sits about one standard deviation above reruns, so we discounted them. Gains below are whole-score numbers unless marked per-shape. One useful conversion: 1% on the MoE kernel is about 0.46% on decode_layer.

Verdict words: NO-GO means measured or refuted by static evidence. CLOSED means we stopped on earlier evidence and did not re-measure. ASSESSED means we examined it and changed nothing. UNTESTED says why we did not test it.

## Part 1: ideas we checked in detail

### Attention

|Id|Idea|What we checked|Result|Verdict|
|-|-|-|-|-|
|A1|Packed half2 softmax with FP16 QK^T at q_len 8|ptxas SASS of the packed ops, and whether the FP16-accumulate wgmma assembles|ex2.approx.f16x2 becomes two MUFU.EX2.F16 plus a PRMT, and ex2.approx.ftz.bf16x2 two MUFU.EX2.BF16 plus a PRMT. A score pair costs 5 instructions (1 HFMA2, 2 MUFU, PRMT, F2FP) against 5 now (2 FFMA, 2 MUFU, F2FP). A wgmma m64n32k32 f16.e4m3.e4m3 does assemble. lee101 found the same|NO-GO. The mechanism is refuted by SASS|
|A2|Bit-trick exp2 into e4m3 P codes|The diff of our first check: a linear-mantissa exp2 (FFMA with a magic constant plus shift and add) then an e4m3x4 convert, on interior tiles, with a byte test and warp vote|Job 24a52be614264969: v16f 101.0 to 101.1 us, v32f 184.1 to 183.6 us. Identical-code runs spread 101.0 and 101.6, and 184.1 and 185.6|NO-GO. A direct-byte variant would only drop the convert, about 0.5 instruction per element. It is the same mechanism as the closed FMA-emulated exp2|
|A3|nc = 4 workspace copies for d1s|The plan code deliberately gives nc = 1 for sliding L = 1 below 24 requests|Job 66be38f978b54004: 2.567x (2.529 to 2.606), 16 of 16 passed. d1s 5.53 us against 5.17 and 4.94 in two identical-code runs. Controls shifted by x1.0003 and x0.9998|NO-GO. d1s was 0.36 to 0.59 us slower, nc = 1 stays|
|A4|One 128-byte bulk copy per KV row|d16f and d32f already run at 86 to 90% of the 128-byte random-row ceiling. lee101 found a bulk 128-byte row copy negative (TMA op rate). It needs fence.proxy.async, and a soak of 30,000 to 60,000 replays is impossible without a GPU|none run|NO-GO: below noise, corruption risk, no way to soak|
|A5|Exact-key tuned overrides (v32s, d1f)|Compared the plan table against a range rule over 6,144 grid points on a CPU. The d1f override is the q_len 8 single-request plan. The v32s override is an isolated key at batch 32 (sms // requests, a formula the file uses for 16 to 23 requests)|A proposed range-rule file (sha 96ef225d) is identical at all 12 scored shapes, and 177 unscored points differ, all q_len 8 sliding|ASSESSED. Keep as is unless we want zero exact keys. Then use a d1f-only rule, which changes no unscored point. The v32s rule needs measurements at its neighbours. No check was run, since a check cannot time unscored shapes|

### decode_layer

|Id|Idea|What we checked|Result|Verdict|
|-|-|-|-|-|
|D1|Raise the PDL token cap from 32 to 128 for glue kernels|A Triton audit: all the PDL kernels (nine by a script count, ten in our notes) call the wait, and pre-wait loads exist only in the first boundary, route, route-split and second boundary kernels (weights, stream input, local_start, and data from two kernels back). At T = 128 all dense picks are cuBLAS, the glue variant is off, and the MoE uses its main build|Job 304f0ebb702346d2: 2.106x (2.075 to 2.138), 13 of 13 passed. Target rows v16s and v16f against controls: x1.0014 (against baseline a006fb) and x0.9824 (against f6918a). Controls shifted together by +2.4% and +2.0%. The prepare-time A/B at T = 128 gave -0.3 us (floors -0.3 and +0.6), at T = 64 -2.3 us|NO-GO. No movement beyond the controls' common shift. Keep the cap at 32|

A follow-up closed the confound. The first check's prepare block ran extra captures and an empty_cache, so a clean test was our v3 plus only that block. Job c918dc92e4b949f7 (sha a32fd969): 13 of 13, 2.0856x, controls +0.52% against the same-cluster baseline. The earlier +2.0% control shift was run noise: five runs of the same runtime code span 2.4% on the 10-row control geomean. The A/B at T = 128 again showed zero (+0.4 us). The first check's headline 2.106x sat above the board entry's interval upper bound of 2.090, but the per-shape condition failed, so we did not count it as a win. The two baselines ran on different judge clusters (f6918a on nyc3b, a006fb on sfo3, the check on nyc3b), but the control shift against the same-cluster baseline was not a cluster effect.

### moe

|Id|Idea|What we checked|Result|Verdict|
|-|-|-|-|-|
|M1|Stream-K flush constant 3 to 4|Closed first by earlier evidence (below noise), then measured in-process in the diagnostic run|0.1% to 0.5% faster at 17 to 256 tokens|CLOSED, under our 1.5% bar|
|M1b|Per-unit overhead constants in the stream-K cost model|Code read only. The base file has no per-unit overhead term. Public constants differ by 2x between standalone and in-layer use (80 and 70 against 160 and 140), so a blind value has no basis|none|UNTESTED. Needs new cost-model code in a 198 KB kernel and a prepare-time constant scan on the GPU. Expected under 0.5% of the score|
|M2|8-op E2M1 dequant against the current 9-op form|A search found an 8-op layout (I, rotl10, rotl13, then X = ((q & 0x1b001b) * C) & M with C = 2^2 + 2^7 + 2^12): 7 ALU plus 1 IMAD, against 7 ALU plus 2 IMAD. SASS of the tiny2 and mid builds shows 15.69 instructions per HMMA, ALU pipe 7.75 (x2 cycles = 15.5), FMA 6.75, so ALU cycles roughly equal issue slots|Model: issue falls 15.69 to 14.69 but ALU stays 7.75, so the loop bound moves 15.69 to 15.5 at best (-1.2%), and the kernel is DRAM-bound at 4.6 TB/s|NO-GO on SASS evidence|

Two caveats on M2. The SASS counts come from the clang-to-ptxas route (166 registers against 149 and 153 in the nvcc build), so 7.75 ALU and 15.69 issue per word are approximate. The close also rests on a throughput model that does not cover a latency-bound loop with 2 warps per SMSP. The verdict stands because the bound is at most 1.2% of the loop either way. An ALU-light variant (IMAD.WIDE for the rotates) was the only remaining lever at that point. We later tried the 8-op form on the wgmma loop and it lost (job 84b9fccd4a804a48, 2.222x against 2.36x).

|Id|Idea|What we checked|Result|Verdict|
|-|-|-|-|-|
|M3|T = 1 deferred normalisation|Notes only. It is a deep redesign of the o, pooler and router kernels (another team's prose puts the gain near 8 us at d1s and d1f)|none|UNTESTED. A multi-kernel redesign of a 570 KB file with no GPU and one shot. The only idea above noise on its own (+1.7%)|
|M4|5-bit scale code|The skinny loop is ALU-bound. SASS shows 7.75 ALU per word, a 6-bit field costs one more ALU op per scale, and 5-bit costs the same or more|none|NO-GO|
|M5|Uniform wgmma descriptors|SASS of a build (clang route) showed 1024 HGMMA with descriptors filled by R2UR from vector registers before each group. lee101's judge comparison of two versions was noise. Expected 0 to 0.3%|Correction: with NVRTC 13.2 plus ptxas reproducing the judge's 96 registers and 0 spills, there is one UIADD3.64 per HGMMA and no non-uniform descriptor math left. The R2UR reading was an artifact of the clang route|NO-GO. Below detectability|
|M6|Column-major or split-expert reuse|A tiles-per-expert table: 73% of v256 routings split an expert into 2 or more jobs (6.2% extra weight streams), 30% at v128, 35% at d16. How many re-streams hit L2 is unknown|none|UNTESTED. Upper bound +0.8% of the score. Needs a GPU diagnostic first, then a stream-K order change, and sits below one check's noise. A first step would be a prepare-time diagnostic comparing one expert in every token's top-8 with four experts at 64 tokens|

### bf16

|Id|Idea|What we checked|Verdict|
|-|-|-|-|
|B1|Router split constant RK_SPLITS[2] = 16|The base table is {1: 16, 2: 12, 3: 10, 4: 8} from a sweep. Headroom ceiling 0.1%|NO-GO. The sweep chose 12|
|B2|A validated route for unscored m (compliance)|Read the dispatch|ASSESSED. It is already bucket-based with a cuBLAS fallback. Two unscored m values appear in the self-test (33 and 200). Unscored m are not timed|
|B3|Sub-12-bit codes|ZipServ gives 11.3 bits on average with popcount plus data-dependent loads per element (see gemm-moe-and-compression-papers.md)|NO-GO|

## Part 2: ideas priced on paper, by board

These come from reading the papers, the public repos and the forum before any check. Status words are as of that reading, and later results are noted where we have them.

### Attention (2.605x)

|Idea|Whole-score gain|Status|
|-|-|-|
|Packed half2 softmax with FP16-accumulated QK^T|Ceiling about 3.2% if the 14.3% and 10.0% verify-full gaps closed. Realistic 0 to 1%|Refuted by SASS (A1)|
|Bit-trick exp2 into e4m3 P codes (used in a public file)|None recorded|No movement (A2)|
|nc = 4 workspace copies for d1s|At most 0.5%|No-go (A3)|
|128-byte bulk copy per row, 128-key tiles|Under 0.3%|No-go or not pursued (A4)|
|Conditional softmax rescale (FA4)|Unknown; nobody isolated the rescale cost|A rare-rescale path is already in the entry (found in the code after an earlier note had tagged the idea untested). What FA4 adds beyond it, a per-warp decision and a dedicated correction warpgroup, was never tried|

Closed on measurement: cold-code warming (our check scored 2.576x), FMA or polynomial exp2 (+24 to +105 us), V transpose in the producer, merge and balance tricks (including the TMA bulk-reduction merge and the batch-balanced verify kernel in public files, and any tree merge), uneven partition, a producer-share probe, a pingpong or one-CTA FA3-style pipeline (116.4 against 100.5 us), deeper stages. Not applicable: the FA1 to FA3 gains, Ring Attention, PagedAttention, TMA tensor loads, block quantisation, incoherent processing (lossy).

### bf16 (1.491x)

|Idea|Whole-score gain|Status|
|-|-|-|
|Swap-AB register-A wgmma decoding the 12-bit pack at m = 128 (a public design by chenwainuo, used inside keell0renz's file)|Ceiling 6.0%, 2.7% at half capture; expected value 0.3 to 0.5%|Closed earlier as the idea of putting 12-bit weights inside the m = 128 GEMM, where alygg77 lost 2.4 to 12 us. New evidence reopened the question: BF16 partials are now allowed, and a public design scored 1.5742. chenwainuo's own repo later showed the realised gain at m = 128 is about 4 us on two of five layers, roughly +1.2% of the bf16 score, not 6%. We built it anyway: two judge checks scored 1.505x and 1.4915x, and the kernels ran about twice as slow as the existing picks. NO-GO|
|Router K-split 16 at m = 128; split-K router for m = 8 to 32|0.1%, up to 1% at half capture of 2.0%|NO-GO. A Triton router from an X post already lost our A/B at every m|
|cuBLASLt pins; a per-(layer, m) tuner under CUDA graphs|None|Closed in practice. Our three checks gave 1.497, 1.484 and 1.492x, every H200 config is algo 66, and our autotune with cold-L2 graphs already exists|
|Validated route for every unscored m from 2 to 384|Compliance, not speed|Same as B2|
|Sub-12-bit codes (ZipServ 11.3 bits, a public codec at 11.07 to 11.32 bits)|At most 5.8% fewer bytes, m <= 32 only|NO-GO|
|Evict-first hint on the small-m weight stream|Negative on the standalone board|Checked and rejected|

Dense bf16 wins ported into decode_layer stay closed, because weights are L2-warm only in the standalone run.

### moe (2.329x)

|Idea|Whole-score gain|Status|
|-|-|-|
|Scale-sharing K-permutation, 6-bit lossless scales and a warp-parallel stream-K prologue|Our first candidate measured +0.6% raw, about +1.4% against controls (2.3431x, interval 2.3080 to 2.3782)|Done. Public numbers: the K-permutation +1.25% of the moe board, 6-bit scales about +2% at 32 to 256 tokens, the prologue +0.9% at 32 tokens|
|6-bit scales on the skinny builds|Unknown|Not run; closed on an op count (one more ALU op per scale, see `../moe/ideas-tried-and-closed.md`). keell0renz shipped it off. Another team's single judge check was +0.77% worse in score with no visible gain (see public-repos-survey.md). Treat as a warning|
|Flush constant 4, per-unit stream-K cost 80/70, ring pacing at M <= 64|Each under 0.5% (our estimate)|Flush constant 4 measured 0.1% to 0.5% in-process (closed); the other two untested|
|bs8 ring-issuer warp|+1.2 to 1.7% at bs8 on the authors' boxes, about 0.2 to 0.3% whole score|Untested; would need a single-size gate and tests at bs 4, 6 and 10|
|Per-expert counters; BF16 per-route buffer|None recorded|Untested. The BF16 allowance covers split-K and split-KV partials, not per-route outputs|
|Stream-K grid size and a collaborator cap|Unknown|Untested; pays only at 3 or more CTAs per tile|
|MULDEC|+2.3% claimed|Not actionable; the mechanism is not public|

Closed: column-major and SplitK ordering, 12-bit weights inside the m = 128 GEMM, more than 2 PDL pre-wait stages, fence cuts (+0.23%), a 4th ring slot, a dynamic tail, the WGS32 tile, side streams, L2 prefetch, and keell0renz's dead list (fused-slice MoE 5 to 13% slower, dual-lane and per-unit barriers 8 to 20% slower).

### decode_layer (2.096x)

|Idea|Whole-score gain|Status|
|-|-|-|
|Port the MoE weight-byte cuts into the decoder's MoE|Realistic 0.5 to 1.0% (the moe check times 0.46), partly spent recovering the FP32-router cost|Allowed, not closed. Weights are cold in both settings, so it transfers. Later our v4 MoE variant measured net zero on the layer|
|A tuner at T = 128 and a pinned pick for 33 <= T <= 128|Unknown; another team called it their largest single change|We read our picks at T = 128 afterwards: none flipped between runs, so there was nothing to pin (see `../decode-layer/moe-port-and-other-ideas.md`)|
|PDL cap 32 to 128|About +0.1 to 0.3%|NO-GO (D1)|
|Early PDL trigger placement|0 to 1% (one team +0.9%, another at most 0.736 us per pair)|Capped by public measurements|
|cuBLASLt picks chosen by in-layer timing at T = 128 and 256|Small|Open. An isolated winner ran 1.5 us slower in the real pipeline|
|Pre-wait hoist of barrier init and table copy|+0.2 to 0.5% (our estimate)|Do not start. Same family as the failed hoisting check|

Closed: cgemv in any form (our check failed 3 of 13 tests, and another team moved 2.1898 to 2.2007), dense bf16 ports, BF16 router input, L2 hints at T = 1 (d1 25.7 to 26.6 us with hints against 24.5 to 25.8 without), a megakernel, side streams, cross-stage PDL (zero or worse), L2 prefetch of MoE weights (x0.99 to 1.00), 4-cluster qkv (x0.87), stacking unchecked wins.

## Part 3: ideas that came from the public repos

|Idea|Where|Result|
|-|-|-|
|Pin all T <= 32 dense picks to the bf16_proj route|ruwwww|Their run scored 2.0751x (interval 2.044 to 2.106, 2 attempts), against our own checks on the same upstream base at 2.096x and 2.105x. No gain. Porting would also pin two entries that our file overrides afterwards, so ordering must stay|
|128-key tiles for verify-full at batch >= 16|infraax|x1.16 and x1.13 on their two weakest shapes (3 seeds, jobs 9206dc7a and 2d42052a), on a Triton-era kernel at 1.77x. Our tile is 64 keys and the producer ring, wgmma N and registers assume it. A deeper KV pipeline measured +2.5% worse on d16f elsewhere. Even a 5% transfer on v16f and v32f would be about 1%. Unmeasured|
|Pre-wait L2 prefetch in glue kernels|infraax|+2.0 us at T = 1 (light prefetch), -3.9 us (full prefetch, bandwidth congestion), below their bar. Evidence against|
|Packed half2 exp2|AutoclickerI|Two MUFU ops, confirmed by our SASS (A1)|
|Sub-12-bit and block codes|ozgureyilmaz, chenwainuo|Agree with the closed list|
|Fused hi/lo router reads for m > 64|ozgureyilmaz|Their arithmetic caps the whole score near 1.306x even with the router at its floor, but that cap is for their 1.17x file. For us it touches the bf16 router cells at m = 128 and 256 only. Under 0.5%|
|Cold-code warming, pushed partials, hoisted late scalar loads|Kushagrabainsla|Warming measured no gain on our entry|

Nothing from the twenty public repos and six blog posts we checked was new, legal and above the bar. The most useful outputs were negatives.

One cheap experiment-design idea is described in tuning-and-automation-papers.md: test a different candidate in each contiguous shape range in one check, with default-range controls, so that one judge run answers several questions.

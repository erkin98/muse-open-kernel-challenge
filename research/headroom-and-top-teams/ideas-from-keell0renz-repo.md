# Ideas from the keell0renz repo

This file summarizes what we learned from reading keell0renz's public repo (https://github.com/keell0renz/muse-open-kernel-challenge) and ranks the ideas from it that we might reuse.

Everything is in prose. No code from the repo is quoted or copied, and anything we reuse needs our own clean-room implementation.

The repo has three commits: 3435290 (Nov 1, the kernels), 8e30e1b (Nov 1, README standing) and 890809c (Nov 2, a README note). Its run was Oct 30 to 31, before the Nov 3 rules update. We compare against vjeux's public baseline (f35a41e), which is also our base. We read the text only and never ran or imported anything. "Verified" means we read it in the cited file, and "inference" means it is our reading and the source does not say it.

## What the repo is, and what it is not

Verified contents: README, LESSONS, NOTICE, LICENSE and five kernel files (decode layer 980 KB, bf16 643 KB, moe 361 KB, prefill 370 KB, attention 281 KB). There is no notes folder, no attempt log, no results table, no orchestrator script and no audit folder. The kernel headers name files that are not in the repo: the author's generator scripts for each snapshot and a design framework document. The bf16 file embeds a module copied from chenwainuo (NOTICE cites chenwainuo's public submission at 3d418e1). The notes that module mentions (an m=128 plan, a codec layout note, version notes pw11 to pw34), plus the rs128 and codec E designs, belong to chenwainuo's repo, not to keell0renz. So the process described in an infographic we saw (lanes, supervisor, keep checks, 153 attempts) is not documented in the repo.

|board|README row (Oct 31, 17:14 CET)|judge reruns of the published file|
|-|-|-|
|decode layer|2.2659 (#38 of 1085)|check 2.2641, submit 2.2532. The row is the same kernels minus the last +0.27% change.|
|attention|2.7168 (#35), a high draw|same-box screens 2.63 to 2.66|
|bf16|1.5876 (#26), the pre-fix twin|check 1.5742, submit 1.5690|
|prefill|1.9134 (#29)|submit 1.9141|
|moe|2.4203 (#58), a high draw of a predecessor|submits 2.3884 and 2.3893|

The README mentions about 460 logged experiments, most of them measured dead, with small wins stacked on top of each other. On Nov 2 the author says the whole process is automated, and describes their current number-one submission on one board as exploiting an answer cache on their own judge replica. We ignore that last item. The live rules forbid caching outputs.

Two attribution doubts, both inference. The infographic's score trail (2.1815 to 2.2664, #144 to #42) is close to this repo's 2.2659 at #38, but we cannot prove it is the author's. And a public post claiming bf16 at 1.569x matches this repo's bf16 submit (1.5690), yet it credits other repos and claims its own kernel at m=128 to 256, where this repo uses chenwainuo's kernel and alygg77's cuBLASLt pins. It may be a coincidence, so we do not attribute it to this author.

## How they worked

Judge facts they built the loop on (verified, LESSONS and file headers): the judge reads about 1 GiB to make L2 cold before every replay (260 µs), subtracts a fixed 4.1 µs graph overhead, and feeds fresh inputs on every repetition. Noise is about 0.63% SD per run and plus or minus 1.5 to 1.9% between two checks. A board row keeps the best draw, so rows are biased upward by about 1 SD. Tiny shapes (about 5 µs) move the geomean as much as 300 µs ones.

Their method rules (verified):

- Measure the ceiling first with a bare-ring skeleton that moves the same bytes.
- Use same-codegen controls: compile a variant against a build with the same edit switched off, because unrelated edits in a hot loop shift nvcc codegen by about 5%.
- Decide changes under 1% with paired A/B runs of 30+ repetitions per shape, and use the judge only as a conformance check. A real 1% takes 6 to 8 judge runs per arm.
- Build N variants in the untimed prepare and time them back to back on identical inputs, which gives 10 to 40 experiments per GPU session.
- Compile on CPU into a shared cache so GPU time is only timing.
- Look at per-SM timelines from globaltimer and smid stamps (start, phase marks, PDL wait, end) instead of tables of averages.
- Tweaking one lineage gives low single-digit percent. The large steps came from structural changes: weight re-encoding, per-size builds, stream-K MoE.

The kernel headers also show a stacked-snapshot habit (verified). Every kept change is a named snapshot built on the previous one, with an experiment number (E173 through E457) and a generator script. The numbers rise in the order data-flag hand-off (E173), 6-bit scales (E260+), barrier removal (E300+), ring pacing and flush constant (E371 to E374), bs8 issuer warp (E418, E447), warp-parallel prologue (E457), so the headers double as a rough chronology (inference).

The infographic (unattributed; inference whenever we tie it to the repo) describes six lanes, each fed a single current best, with keep checks applied in order: clean code, beats noise twice, cold compile under 270 s, stable against itself. It reports 153 attempts: 37 kept (24%), 68 judged not faster, 30 that produced nothing, 12 refused for code rules, 4 refused for compile time and 2 refused for unstable timing. We read the checks as three paired runs of the candidate and of the current best (both must win beyond noise), a self-timing of the candidate to show the setup does not drift, 270 s as a margin under the 300 s prepare limit, and a static scan for banned items (encoded or precompiled GPU code, caching across calls, memset in the timed function). The repo defines none of these.

What we could adopt (inference):

1. Keep one current-best file per board and make every candidate a one-flag delta of it, so a check always has a same-codegen control.
2. Build 3 to 6 candidates inside prepare and A/B them in-process, but trust that only as a filter.
3. Run a static gate list that mirrors the refusal classes before any judge run.
4. Never stack two unchecked wins.
5. Do not copy the three-run rule literally; three judge runs cost us hours. Paired A/B on the box only helps if you have a GPU, which we do not.

## What each board's file contains

### Decode layer

Base is alygg77's decode layer (6cae49a), itself built on vjeux's f35a41e. The stack on top, newest first (verified from the header):

- stack_all1 removes the grid barrier before the MoE final pass (per-expert completion counters, called MFIN), adds 6-bit block scales in the wgmma MoE builds (SC6), a component named PRF32 whose meaning we cannot read from the source, and PDL up to T=128 (ours stops at 32). Its header also records a negative: L2 cache hints at T=1 were dropped because the judge showed decode 1 at 25.7 to 26.6 µs with them against 24.5 to 25.8 without.
- st2l moves the deferred-flush constant in the wgmma MoE from 5 to 4 (E374). st1l is that constant 5 plus weight-ring pacing at M <= 64: a stage is issued only after the stage two back has landed (E371, E372).
- ybfl stores and re-reads the MoE per-route output buffer as BF16 instead of FP32 in the wgmma builds.
- cepil_u160 gives the in-layer MoE stream-K cost model a fixed per-unit overhead of 160 (phase A) and 140 (phase B). The standalone MoE file uses 80 and 70. The header credits zhebrak (SK_CEPI).
- pcx_h1 is a "pre-wait control" of the projection kernels: knobs for how many weight stages go out before the griddepcontrol wait, a start delay, per-stage pacing in clock cycles, and a hoist of barrier initialisation and the escape-table copy ahead of the wait, with the x tiles of the pre-issued stages loaded right after it. The published per-role config table is empty, so what ships is the default two pre-wait weight stages plus the hoist.
- bt_sc6 adds lossless 6-bit block scales in the layer's wgmma MoE builds for T >= 17. stack_df1 hands off by data flag from MoE phase A to B in the 9 to 16 token build.
- chenwainuo's rs128 GEMVs are used at T=128 and 256 for qkv and at T=128 for the unpooler.
- A graft block at the end adds an 8-CTA-cluster residual boundary, a Gluon row boundary, a one-warp-per-(token,2) route finish and cuBLASLt algorithm picks for qkv at m=256 and for the dense layers at m=128 and 256. It is gated on exactly T in 1, 8, 16, 32, 128, 256. The README says the 2.2659 row excludes the last +0.27% change; the source does not say which block that was (inference: this graft).

Measured numbers the source gives: a PDL trigger right after the producer's last weight issue is worth +0.9% on the layer, 6-bit scales about +2% on the MoE at 32 to 256 tokens, the warp-parallel stream-K prologue +0.9% at 32 tokens and +0.5% at 128. A comment says a cuBLASLt pick for the unpooler at m=128 ran 1.5 µs slower in the real pipeline than the chosen one. Nothing else has a gain recorded.

### MoE W4A16

The base is alygg77's MoE (6cae49a), unchanged underneath. On top, newest first (verified): ps1 is a dedicated bs8 build with a ring-issuer warp (E418; +1.2 to 1.7% at bs8 on their replica boxes, E447). mf_a is MFIN plus B4L (a single-read final) in the main wgmma build (E300+); B4L defaults to off in the published code, so only MFIN ships. The decode file times the MFIN build against a barrier twin at 32, 128 and 256 tokens inside prepare (4 seeds, 24 replays, twins compiled on a CPU thread) and keeps the winner. The MoE file only prints that comparison, so MFIN ships unconditionally there. st1f4 is flush constant 4, st1 is the BF16 per-route buffer with flush constant 5 and ring pacing at M <= 64, and cepi_u80 is the 80/70 per-unit cost. Earlier snapshots added 6-bit scales for bs >= 17 (E260+) and the data-flag hand-off in the 9 to 16 token build (E170+).

SC6 on the skinny (small token) builds exists in the source but its switch is off in the published file. The reason is not given (inference: it did not win, or was not worth the risk). The only per-size patch layer is bs8. The base author's comment on the flush constant says that choosing 3 lets the group's own atomics and stores drain, worth +0.5 to 1%. There are no numbers for MFIN, ring pacing, the flush constant or the cost term.

### Attention decode

Verified from the header and routing code: the vjeux base plus a TMA bulk-reduction merge (an async reduce-add of a partial from shared memory into the FP32 workspace, with 72-float workspace rows) for the q_len-1 per-request kernels, credited to Madhavan113's repo, plus ZhimaoL's verify-attention kernels. Routing: full-layer verify (q_len 8) at batch 16 to 24 runs a batch-balanced cooperative kernel (stream-K over requests, in-kernel merge); sliding-layer verify at batch 1 runs the second kernel plus a merge; the sliding batch 8 to 64 route and the full batch 1 route are switched off. So only two scored cells (verify 16 full and verify 1 sliding) leave vjeux's kernels. Their edit to ZhimaoL's code replaces his piecewise-linear exp with an exact exp2 on packed half precision (his version is behind a switch that is off). His bit-trick exp builds the e4m3 byte from the exponent argument: exponent floor(y)+7, mantissa 8 times the fraction, error 0 to 6% mid-octave, which sits inside e4m3's own 6.25% rounding. The choice predates the Nov 3 update, which now allows an approximated exp2 for FP8 P. The repo has no measured numbers beyond the README screens.

### bf16 projections

A per-(layer, m) dispatcher over four public kernels (verified, routing table near line 11323):

|m|qkv_gate|o|pooler|unpooler|router|
|-|-|-|-|-|-|
|1|vjeux|vjeux|vjeux|vjeux|vjeux|
|8|chenwainuo|chenwainuo|chenwainuo|chenwainuo|NathenFernandes|
|16|vjeux|chenwainuo|chenwainuo|vjeux|NathenFernandes|
|32|chenwainuo|vjeux|vjeux|vjeux|NathenFernandes|
|128|chenwainuo|NathenFernandes|chenwainuo|chenwainuo|vjeux|
|256|alygg77 (pinned cuBLASLt)|alygg77 (pinned cuBLASLt)|alygg77 (pinned cuBLASLt)|alygg77 (pinned cuBLASLt)|vjeux|

At m=128, three of the four big layers go to chenwainuo's module. Its newest path (rs128) is a swap-AB, register-A split-K wgmma kernel for m exactly 128 and 256. Each CTA takes 128 weight rows by all tokens. Two math warpgroups decode the lossless 12-bit pack straight into wgmma A registers. The activations are the shared-memory B operand, loaded through 2D TMA. One full barrier per ring slot is shared by a weight bulk lane and an x TMA lane. K is split over CTAs on a one-wave grid, and partials are added with a bulk-tensor reduce-add into an output the kernel zeroes itself, so there is no memset node. The module also has a prepare-time "bake" that times each new path against the older one per (layer, m) and keeps the winner. So whether rs128 is the runtime path in a given cell is decided on the judge box (inference; the repo has no bake output). At m=256 the table sends the four layers to cuBLASLt pins, so rs128 matters there only for the decode layer's own qkv.

Also in the header: the router at m=128 got a K-split of 16 (grid 128, was 12) from a sweep. And in "v6", every m from 2 to 384 that is not a scored size gets its own validated route built in prepare (pinned kernels called with the real m, plus cuBLAS on a K-major weight), while the scored sizes take exactly the v5 path. That is a ready-made answer to the rule against tuning that only works at exact scored sizes.

The copied module's docstring (so chenwainuo's work) also describes a lossless prefix-code variant, "codec E", at 11.07, 11.17 and 11.32 bits per weight for 1, 2 and 4 substreams, against 12. The code length is decided by the first 3 bits, with an 8-byte table and the exponent pair read from a 256-entry shared LUT. Each pack is proven bit-exact by a reference unpacker before use, and its timing against the 12-bit path is baked per shape.

Prefill (submit 1.9141) was not studied, since it is not one of our boards.

## The ten ideas, ranked by expected gain per effort

Their run predates the Nov 3 rules update. "Untested" below means the idea was absent from our own moe and decode_layer files as of Nov 4.

1. PDL for the glue kernels up to T=128 (decode_layer; verify 16 is T=128). Each glue kernel launches while its producer drains and waits (griddepcontrol.wait) only before it reads producer data. Our base caps this at T=32 and theirs raises the cap to 128, with a comment that T=256 loses about 1%. No gain is recorded for it alone. The effort is one constant plus a check that every kernel used at T=128 is PDL-safe, and that check is the real work. We have never launched a glue path at T between 64 and 128 under PDL, and a kernel that reads producer data before its wait would race. That is the failure family of our first decode_layer judge run (job a44e46117b9449f8, which passed only 10 of 13 tests under PDL), so the risk is medium, not low. Check statically first, keep to a contiguous range and test 64 and 96 once. We would expect +0.1 to 0.3% overall if the launch gap at 128 is about 1 µs (inference). The item we dropped earlier was "more than 2 pre-wait stages", a different knob.
2. Stream-K constants in the wgmma MoE. There are two to copy. The first is the flush constant: the deferred flush of the previous segment's atomics and epilogue sits behind commit group N of the first stage, vjeux's default is 3, and theirs ended at 4 after a sweep of 3 to 7. The second is a fixed per-unit overhead in the cost model of the range partition (80 and 70 standalone, 160 and 140 in the layer). The vjeux base has no such term, and LESSONS says a 12.5% cost error costs 7 to 9%. No gain is recorded. Effort is tiny and the correctness risk is low, but standalone and in-layer optima differ, so each needs its own value, and a prepare-time A/B is only a filter.
3. Weight-ring pacing at M <= 64 in the wgmma MoE: the producer issues stage n only after stage n-2 has landed, instead of a three-stage burst. It ships in the files that scored 2.3884 and 2.3893, with no gain recorded. It sits next to our dropped fourth-ring-slot item without being the same. Cheap to try, with a small chance that the burst was hiding latency.
4. A bs8 MoE build with a dedicated ring-issuer warp, aimed at the verify 1 load (T=8). They measured +1.2 to 1.7% at bs8 on their own replica boxes, not on the official board, which would be about +0.2 to 0.3% overall. Medium effort and medium risk. It is a single-size gate, so bs 4, 6 and 10 need a regression check.
5. MFIN: per-expert completion counters instead of a grid barrier before the MoE final pass (moe at 32, 128, 256 and the decoder's MoE at T >= 17, where a 1% gain moves decode_layer by about 0.46%). The final pass waits on the counters, with ready and done flag sets that alternate by launch parity because the flags are never re-zeroed. No gain is recorded, but the decode file keeps a paired timing against a barrier twin, so it must have won on their box at some sizes. We have not tried it. It is not our dropped MoE fence cuts (+0.23%), which moved fences instead of removing a barrier, but it needs the same soak testing (2000 queued replays per shape). Effort and risk are both high, and livelock and stale-flag failures are the classic bugs here.
6. A bit-trick exp2 that produces e4m3 P codes directly (attention; a 1% gain in the layer's attention moves decode_layer by about 0.14%). It drops the MUFU and the conversion together, and the Nov 3 text allows FP8 P with an approximated exp2. No gain is recorded. We had dropped an FMA-emulated polynomial exp2 earlier; our first attention check had also tried a linear-mantissa exp2 feeding e4m3 codes with no change, which a later review put in the same class as this idea. Medium effort. Accuracy has to stay at BF16 level, so the oracle needs testing at several magnitudes.
7. cuBLASLt algorithm pins for the dense layers at T=128 and 256 inside the decoder, chosen by timing the whole chain (a wrapper plus a tuner that keeps the base unless a variant wins by 0.4 µs). A comment shows an isolated winner running 1.5 µs slower in the real pipeline, so the choice has to be made inside the layer. The repo records no gain; our survey of public repos records alygg77's decode 32 at 2.0 to 2.2 µs, and 2.8 to 3.4 µs on the full-context loads. It stays open for us, because it differs from our dropped dense bf16 GEMM item: the choice is made in the real chain. Medium to high effort, medium risk.
8. YBF: a BF16 per-route output buffer in the wgmma MoE, which halves the bytes of the final pass. It is cheap to build, but the rules allow BF16 partials for split-K and split-KV, and a per-route expert output is a different object. We could not confirm that SGLang keeps it in BF16. The rule status needs a check first, and precision on the correctness gate after that.
9. A pre-wait hoist of barrier setup and table copy in the projection kernels (all decode_layer loads; a 1% gain in the projections moves decode_layer by about 0.32%). This has the same danger as our failed first run, where x loads were hoisted above the wait: nothing that reads activations or a previous kernel's outputs may happen before the wait. We would expect +0.2 to 0.5% (inference). Medium effort, high risk until proven.
10. A validated route for every non-scored m, plus the m=128/256 kernel set (bf16). The bf16 board scores 1.5742 (check) and 1.5690 (submit) against our 1.491, which is +5.2 to +5.6%, with no per-cell numbers. Our "12-bit inside the m=128 GEMM" item is closed because it lost for alygg77. But the kernel here is chenwainuo's, a different design whose split-K partials are reduced in BF16, which we had flagged as a precision blocker and the Nov 3 text now allows. The evidence sits in chenwainuo's repo (not read) and only counts if the bake picks the 12-bit path per cell. Effort is very high and we have no local compiler. This has the highest ceiling of the ten and the lowest gain per effort, so it ranks last.

Not ranked, because they touch our dropped "merge and balance" item: the TMA bulk-reduction merge for q_len-1 attention kernels and the batch-balanced cooperative verify kernel for full layers at batch 16 to 24. The bulk reduce into a self-zeroing workspace also meets our red.add soak failure (about 1 in 10 to 20k replays). SC6 on skinny builds shipped off, which is a warning rather than proof. Two other public posts (one with bf16 1.595x using a 12-bit pack and moe 2.382x with a "MULDEC" graft, one describing 12-bit skinny matrices with fused routing, which vjeux already does) look like different or later team states.

What happened to these after Nov 4, for the items we tested: raising the PDL cap to 128 (item 1) scored 2.106x as a check and, read against the unchanged rows, was noise (job 304f0ebb702346d2). Flush constant 4 (item 2) measured 0.1% to 0.5% faster in-process, under our bar. The rs128 kernel (item 10) scored 1.505x and 1.4915x and ran about twice as slow as the existing picks. We tested none of the others.

## What failed for them

The repo has no list of the 68 "not faster" or 30 "found nothing" attempts. What it carries (verified) is LESSONS and inline comments. All of the following were measured dead by them, and each needs a check against our own dropped list before reuse.

- A megakernel or persistent whole layer: a cross-SM flag hop costs 1.4 µs idle and 6.3 µs under 4.3 TB/s background load, which caps it below chain plus PDL.
- A fused-slice MoE (A and B per I-slice in one CTA): six variants, 5 to 13% slower.
- The belief that the MoE streams at 3.3 TB/s instead of 4.4 is a myth, they say. The steady rate equals the bare ring, and the losses are events: a 3 µs ramp, an A-to-B dip, slow phase B at bs1, a 6 to 9 µs tail.
- Prefetching the MoE's first weights into L2 during the previous kernel's tail: x0.99 to 1.00. With phase A fully L2-resident the MoE gains 0.4 to 0.65 µs of 22.
- 4-CTA clusters splitting K for the 96-CTA qkv GEMV: x0.87 at decode 1, because clusters at 1 CTA per SM do not all fit and need two waves.
- Speculative expert prefetch, L2 persistence windows, lookahead rings, twin L2 policies, ring depth, unit size and helper-warp knobs: at optimum or worse.
- A dual-lane wgmma MoE and per-unit barriers: 8 to 20% slower.
- Approximations: attending only the last 2 to 16k tokens on full layers failed the router check on every shape, and dropping low-weight expert routes broke the residual bound (0.17 to 0.7 against 0.15). The router check is on the discrete decision (top-8 within 0.05), which pins everything before the router to near-exact.
- L2 hints at T=1 in the layer were slower on the judge.
- Switched off in the published files, which we read as negative (inference): SC6 on skinny MoE builds, the B4L single-read final, and the sliding batch 8 to 64 and full batch 1 routes of ZhimaoL's attention kernel.
- Dense: only the pooler shape wins with a row-block by K-half split. T=256 sliding routes of the dense kernels were 10 µs slower than cuBLAS. Several unpooler and o cuBLASLt picks were slower in the real pipeline than in isolation, which is the same lesson we learned: an isolated tuner cannot see PDL overlap.

## Traps when porting

- We grepped all five files for base64, zlib, lzma, cubin and fatbin and found none. All CUDA is readable source. The 12 attempts "refused for code rules" are therefore not explained by the published files (inference: they never got kept).
- Caching across calls: the module-level caches hold only compiled extensions and scratch, as the decode docstring states. We saw no activation caches.
- Memset in the timed function: the bf16 file has a self-zeroing path with no memset node, while the older path used a caller-zeroed output via a memset node in the graph. The live rules say no memset in the timed function, so a port must use the self-zeroing form, which then meets our red.add failure. Their router workspace is sized at prepare and its counters are zeroed there.
- Exact-shape gates: the decode graft gates on T in {1, 8, 16, 32, 128, 256}, bf16 rs128 on m exactly 128 and 256, attention on batch ranges 16 to 24 and exactly 1, and the MoE bs8 patch on one size. Their bf16 v6 adds a validated route for every other m; decode and moe do not (no fallback pick visible). Under the live rule against tuning only at exact scored sizes, any port must dispatch on contiguous ranges and verify the shapes just below and above.
- The BF16 per-route buffer (idea 8) needs a rule check, as above.
- Router input precision: the decode file builds the router input in a separate input-norm kernel and runs the router as BF16 hi plus lo weights with FP32 accumulation (verified). We could not confirm from the text whether that input stays FP32 for T <= 8.
- Timing: their exact exp2 and BF16 buffer may have been shaped by the older rules text, and the approximate-exp2 and FP8-P allowances are now explicit.

## Calibration of the decode-layer path

Our decode_layer entry is vjeux's baseline unchanged, at 2.0960x. Alygg77's judged score is 2.2058 per our survey, which is +5.2% over ours. The keell0renz file scores 2.2641 (check) and 2.2532 (submit): +2.6% and +2.1% over alygg77's number, and +8.0% and +7.5% over ours. The infographic's top is 2.2664, and the top-50 cut was about 2.261 on Oct 31 (12:15 PT). So almost all of our 7.9% gap to that cut is alygg77's layer work plus keell0renz's stack, and the stack alone is roughly +2% to +2.6%.

Their file is not built on vjeux directly. Its header names alygg77's layer (6cae49a) as the base, and the file differs from vjeux's by about 5,250 added lines and 83 removed. The infographic says their lane started at 2.1815 on Oct 30 (6 PM PT), below alygg77's 2.2058, so it was probably an earlier state of the same base (inference). The +3.9% from 2.1815 to 2.2664 cannot be split by step, because the repo has no per-step gains. Items that touch only the MoE and the glue are portable once our base moves to alygg77's lineage. Items that touch the dense kernels (rs128, cuBLASLt pins) run into our dropped note that standalone dense wins overstate the in-layer gain, except the cuBLASLt picks, which they chose by in-layer timing.

## Could not verify

We searched all five kernel files, the README, LESSONS and NOTICE for MULDEC (and spelling variants) and found it nowhere; the only hits were substrings of "graft". The nearest thing is that the MoE scale decode in this repo is already multiply-based: scale bytes become half-precision pairs with one integer multiply-add per pair, and the nibble decode (DQ7) is 7 ALU ops plus 2 shifts per 8 weights. A multiply decode of the 4-bit codes would be one more step on that line, but nothing here says what the other team did (guess).

Also unverified: the process as the infographic shows it; the infographic's real author; per-step gains; the bake result for rs128 on the judge; the router-input precision at T <= 8; who zhebrak is; anything about the 68 and 30 negative attempts; and whether any of this is in the newest private variants of our own moe and decode_layer files. As of Nov 4 none of our moe or decode_layer variants contained MFIN, flush constant 4, the cost term, ring pacing at M <= 64, the issuer warp, the BF16 buffer or the in-prepare MFIN A/B switch.

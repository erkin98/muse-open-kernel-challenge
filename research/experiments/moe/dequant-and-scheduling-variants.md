# Dequant, scheduling and finalize variants for the MoE kernel

This file records each change we tried on the W4A16 MoE kernel, what we expected, what the judge showed, and a bug that cost us one check.

Build names and shape codes are the ones defined in fixed-cost-anatomy.md: tiny, tiny2 and mid are the skinny builds (up to 4, 8 and 16 tokens), wk6p is the wgmma build for 17 tokens and up, and d1, d16, d32, v1, v128, v256 are the six scored shapes. The changes are a skinny scale merge, a late generation store, an 8-op dequant, a finalize rewrite and an early weight start. Base file v3 has sha256 prefix c0828b74. Identical-code ratios for v3 are d1 3.061-3.114, d16 1.955-2.001, d32 1.667-1.668, v1 2.189-2.279, v128 2.604-2.623, v256 2.866-2.898.

The lesson that runs through the whole file: static instruction counts did not predict speed. The change that cut instruction count and ALU work the most made the judge run slower by 5% to 13%.

## 1. Skinny scale-pair merge (SGP)

Change: in the skinny builds, one PRMT and one IMAD build both rows' scales for a chunk into one register (row g in the low half, row g+8 in the high half), and the four HMUL2 take the halves as operand modifiers (`.H0_H0`, `.H1_H1`), the way the wgmma loop already does. Same values, same multiplies. Only tiny and tiny2 take it.

Static: registers 149 to 140, no spills, 32 fewer instructions (16 PRMT and 16 IMAD) in each of tiny and tiny2, 256 HMUL2 with each half modifier. mid and wk6p compile to PTX identical to v3. A CPU test showed the new register's halves equal the old values bit for bit for all 28,322 pairs of positive normal scale bytes and both chunk parities.

Expected: about 3% fewer instructions in the skinny loop, so 0 to 0.5% on d1 and v1.

Measured as in-process arms (graph replay, ratio to the base build; below 1 is faster). The first run is the check that failed to load wk6m and the second is the check that fixed it; both are described in section 5.

|arm|M|first run|second run|
|-|-|-|-|
|tiny with SGP|1|0.988|1.008|
|tiny with SGP|4|0.997|1.002|
|tiny2 with SGP|4|-|0.997|
|tiny2 with SGP|8|0.998|0.999|
|mid with SGP|12 / 16|1.001 / 1.003|-|

The sign flipped between runs at M=1, so it is noise. Verdict: no effect.

## 2. Late generation store (GENLATE)

Reading the SASS of v3 we saw thread 0 store the launch generation (`S.gen`, `S.ssp`) right behind its cold volatile load of that value. Warp 0's lanes park at the reconvergence barrier, so warp 0 issued its routing loads about one cold-miss latency late and the first CTA barrier in `route()` waited. GENLATE keeps the load first but moves the store behind `route()`, followed by one extra CTA barrier. In SASS the store moves from offset 0xcb0 to 0x36a0, behind the routing barriers.

The ordering argument: all reads of the generation and the sum-of-squares copy index (zeroing of the other copy, counter reset, epilogue reads, the final barrier counter) are in consumer threads after `route()`, hence after the new barrier. The producer warps never read them. The generation cannot change before the grid barrier of this launch, so loading before storing is harmless.

It was applied to tiny and tiny2, to a new build wk6q (wk6p plus GENLATE, intended for 97 tokens and up), and later to mid (as mid2). Expected: 0.5 to 1 us per call at v128 and v256.

Measured:

|arm|first run|second run|
|-|-|-|
|wk6q / wk6p over M 17, 24, 32, 48, 64, 96, 128, 256|0.998, 1.001, 0.999, 0.998, 1.001, 1.002, 1.000, 1.000|0.994 to 1.002|
|mid2 / mid at M 9, 12, 16|0.996, 0.992, 1.000|0.998, 0.997, 0.997|

Outputs matched (relative L2 against wk6p at 97, 128 skewed and 257 tokens: 0, 6.9e-6, 0). Verdict: zero on the wgmma builds, 0.3% on mid. Our reading of the SASS found something real, and it does not show up in time. We have no explanation for that.

## 3. Eight-op E2M1 dequant (DQ8)

### What it is

The wgmma build decodes each 32-bit word of eight 4-bit codes into packed bf16 pairs. The base uses a 9-op form (7 ALU plus 2 IMAD). An exhaustive search (an exact cover over 2,470 register forms, including two-piece forms, 4.5 million covers) found no pure rotate-and-mask tiling and nothing with fewer than 7 ALU ops at 9 ops or fewer. The best was an 8-op form with 7 ALU plus 1 IMAD, which needs the codes stored in a different bit order. It reads `r0 = q & M`, `r1 = rotl10(q) & M`, `r2 = rotl13(q) & M` and `r3 = ((q & 0x1b001b) * 0x1084) & M`, with the rotates written as `mul.wide.u32` (lo or hi). The new bit order is a pure bit permutation of the judged order, applied to the single K-reordered copy of the weights, so there is no extra weight memory.

The 1-tile stage of the wgmma loop is 303 instructions per thread before the change: LOP3 85, HMUL2 64, IMAD 41, SHF 34, UIADD3 17, WARPGROUP 16, HGMMA 16, LDS 6, SYNCS 4 and about 20 others, with 126 ALU-pipe ops. Static effect on wk6p: 22,920 to 22,408 instructions; the 1-tile phase-A loop 303 to 287 instructions per stage; ALU-pipe instructions 126 to 95 (-25%); FMA-pipe 105 to 120 (32 IMAD.WIDE per stage); `IMAD.WIDE` count 387 to 1411 and `SHF` 1597 to 654; HGMMA count unchanged at 512; 96 registers and no spills.

### Why we believed in it

Two static arguments pointed opposite ways. An earlier analysis of the skinny loop found ALU cycles roughly equal to issue slots (7.75 ALU ops and 15.69 issue slots per code word, counted on a compiler route that gives 166 registers against the judge's 149 to 153, so approximate), so removing one IMAD could not help; it bounded the gain at 1.2% of the loop. Our wgmma analysis said the ALU pipe was 83% busy at 303 issue slots and would drop to 66% at 287. An in-process A/B of the K-reordering alone (8.3% fewer instructions) had moved M=32 by -0.4%, M=128 by -0.3% and M=256 by -1.8%, so DQ8 was predicted at +0.3% (d32), +0.2% (v128) and +1.2% (v256), about +0.2% of score. Neither argument predicted a loss.

### Proof of exactness

The new decode equals the layout table for all 2^32 words, and a negative control was caught. DQ8 applied to the permuted word equals the judged decode register by register for all 2^32 words. The packer round trip and a full `_wgs_copy` round trip on random 4096-by-4096 weights matched the judged copy bit-permuted, with identical scale bytes. A single Python constant drives both the packing and the compile flag, so layout and decode cannot disagree.

### Result

We ran the stack (SGP, GENLATE, wk6q for M of 97 and up, and DQ8 on wk6p and wk6q) as one check, job 84b9fccd4a804a48. It passed 5/5 and scored 2.222x (CI 2.188-2.255), 6,053.0 us. Prepare took 43 s (builds 35 to 42 s each, 140 registers on tiny and tiny2, 96 on wk6p and wk6q). The decision table was written before the run: d32 at or below 1.664 meant DQ8 hurts.

|case|SGLang us|ours us|ratio|vs mean of three identical-code runs|
|-|-|-|-|-|
|decode 1|80.4|26.4|3.044|-1.2%|
|decode 16|195.4|99.4|1.965|-0.7%|
|decode 32|213.9|135.8|1.576|-5.5%|
|verify 1 (8 tokens)|151.0|68.5|2.205|-1.3%|
|verify 16 (128 tokens)|454.0|196.1|2.316|-11.3%|
|verify 32 (256 tokens)|633.2|253.4|2.499|-13.4%|

d1, d16 and v1 are within their identical-code noise (d1 sits 0.5% under the range; d16 runs the unchanged mid build and acts as the control). The wgmma rows lose 11.7 us at 32 tokens, about 16 us at 128 and about 39 us at 256, so the loss grows with M. The stack is a regression, never to be submitted or stacked on.

Attribution is clean only for d32, where wk6p carried DQ8 alone. v128 and v256 also carried GENLATE on wk6q, so those two rows are confounded within this check. A later run measured GENLATE at zero, so DQ8 is the likely cause of nearly all of it.

Our explanation, which is inference and was not isolated: the IMAD.WIDE rotates move work from the ALU pipe to the FMA pipe, and before the change the pipes were balanced within 20% (126 ALU, 105 FMA, 303 issue slots per stage), so shifting work can create a new bottleneck. We have no per-pipe throughput number for IMAD.WIDE. Another oddity is that by our cost table this loop is issue-bound but idle about 40% of the time waiting for DRAM (about 0.73 us of consumer work per 1-tile stage against 1.16 us of DRAM time), so a slower loop should not have cost this much at M=32. We cannot square that.

Rule we took from it: no claim from op counts. An edit to the wgmma loop stays only if its paired in-process arm shows at least 1.5% at M=32 with the sign repeating at 24 and 48.

## 4. Variants closed without a candidate check

- IMAD.WIDE rotates alone (`OKC_DQW` prototype, 32 per stage). It moves 23 ALU ops per stage to the FMA pipe at equal instruction count (ALU 126 to 103, FMA 105 to 128). With the loop issue-bound and DRAM-bound at 1 to 2 tiles, we expected no gain. DQ8 later used the same rotates and lost.
- Four commit groups (GS=4) for jobs of up to 2 n-tiles in the wgmma loop. The build compiles to 96 registers but with 32 B of stack, 144 B of spill stores and 308 B of spill loads (also with only the 1-tile case). The double-buffered A fragments need 32 registers that the 96-register cap does not have. Closed.
- 5-bit scale codes. The scale offsets span 33 values, which does not fit 5 bits, and a 6-bit field already costs one more ALU op per scale in the ALU-bound skinny loop.
- Uniform wgmma descriptors. We first read R2UR before each HGMMA group in the SASS and suspected non-uniform descriptor math. That came from a compiler route that does not reproduce the judge's register counts. With NVRTC 13.2 plus ptxas, which does (96 registers, no spill), there is one UIADD3.64 per HGMMA and no non-uniform descriptor math left. lee101's judge A/B of the idea had also been noise.
- A stream-K flush constant of 4 instead of 3: measured 0.1% to 0.5% faster in the diagnostic run (see fixed-cost-anatomy.md), under our 1.5% bar.
- Per-unit overhead constants in the stream-K cost model: untested. The base has no per-unit term, and keell0renz's constants differ by 2x between their standalone and in-layer runs, so a blind value has no basis. It would need new cost-model code in a 198 KB kernel and a prepare-time scan on the GPU, for under 0.5% of score.
- Column-major or split-expert reuse: untested. 73% of v256 routings split an expert into two or more jobs, 6.2% extra weight streams; v128 30%, d16 35%. We do not know how many re-streams hit L2. Upper bound +0.8% of score, and it needs a GPU diagnostic before any scheduler change.

## 5. Final-pass rewrite and early weight start (wk6m)

### The three edits

The diagnostic run (fixed-cost-anatomy.md) showed the final pass after the grid barrier taking 4.4 to 5.4 us on wk6p, and the weights starting about 4 us in. Three flagged edits went into a new build for M of 17 to 96, wk6m:

- FIN2: accumulate each route's sum of squares with `red.global.add.f32` in the phase-B epilogue (consumer threads, once per unit, warp and n-tile, by whichever CTA finishes the unit) into the launch's parity copy of the sum-of-squares array, then run a single-pass, column-parallel final pass as the skinny builds do. Expected: final pass 4.4 to 5.4 us down to about 1.5 us, so -2.5 to -3.5 us at d32.
- FINPF: prefetch the positions and top-k weights before the barrier. Only usable with FIN2.
- WEARLY: issue the first weight TMA stages before the activation gather finishes, behind a new named barrier (id 8, count 64: the weight producer warp syncs and the activation producer warp arrives). Expected: -1 to -1.5 us.

With GENLATE the total was expected to be -4 to -6 us at d32, +3% to +5%. SASS: wk6m 24,944 instructions, 96 registers, 8 B of stack with 24 B of spill stores and 60 B of loads, all outside stage loops (the nearest local-memory access is at least 105 instructions from any HGMMA).

Safety reasoning, condensed. FIN2 reds are program-ordered before the CTA reaches its grid-barrier arrive (CTA barrier, then thread 0 fence and atomic, the existing shape that already publishes the skinny builds' atomics). The read happens only after the acquire exit and a second CTA barrier, with `ld.global.cg`. The next launch zeroes the other parity copy, never one in use. For WEARLY, barrier ids 1 to 7 were already used and 8 is unique; the wait-for graph has no cycle (the consumers arrive on 7 without blocking and join on 2, the activation warp waits on 7 and arrives on 8, the weight warp waits on 8, and none of those waits depends on weights); ring memory is untouched by generic accesses before the first TMA of a launch.

### Check one: wk6m never ran (job 10650a018faf4076)

The first check carried the whole treatment, with DQ8 turned off, plus in-process arms. It passed 5/5 and scored 2.350x. The judge log said it all in one line: the build of wk6m failed with an ImportError, "dynamic module does not define module export function (PyInit_okc_moe_w4a16_v2_okc_nw16_..._okc_wgsfin21_o...". The module name was built from every -D flag in the define tuple, and with the extra flags it reached 202 characters. The loader derives the init-function name from at most 200 characters of the module name, so a longer name cannot match the symbol.

The prepare step had a fallback for exactly this, so the file served `M<=16: mid2, M<=96: wk6p, rest: wk6p` and d32 ran v3 code (ratio 1.65, a diagnostic-run reading of unchanged code). We got a useful but small result and lost a check on a naming limit. Ratios against the identical-code ranges: d1 3.11, d16 2.00 (served by mid2), d32 1.65, v1 2.21, v128 2.61 and v256 2.85.

The arms that did run: GENLATE on wk6 about zero (above), mid2 -0.4% to -0.8% at M of 9 and 12, SGP at most -1.2% at M=1. The silent fallback also meant the 5/5 pass said nothing about wk6m.

Fix: derive the module name from a hash of the define tuple (20 hex characters, 40 in all), and log "imported as <name>" for every build plus a "builds kept, left out" line before the dispatch line. We added a 40 s deadline to the first launch of each new build and a self-check of every arm against wk6p at 32 and 96 tokens. Rule for any future file with many -D flags: never build a module name from the flags.

### Check two: the measurement (job 7bff417c13bd4edf)

The same CUDA source with the name fixed. It passed 5/5 at 2.345x (CI 2.309-2.380). Paired ratios: d1 3.100 (24.9 vs 77.2 us), d16 1.983 (96.4 vs 191.3), d32 1.654 (128.0 vs 211.8), v1 2.206 (67.9 vs 149.8), v128 2.596 (176.0 vs 456.9), v256 2.852 (213.4 vs 608.8). All inside the identical-code ranges except v128 (-0.5%) and v256 (-0.9%), which run v3 code in a diagnostic run, as in the first check. All seven builds imported with 40-character names. Self-check relative L2 against wk6p was at most 4e-5. No hang, prepare 50 s plus diagnostics up to 105 s.

Rules written before the run: an edit goes forward only if its ratio is 0.990 or lower at M=32 and 0.995 or lower at M=24 and 48, pooled over 17 to 96 also 0.990 or lower, outputs clean, and the mechanism visible in the timeline.

In-process arms (graph replay, ratio to the reference of the pair; above 1 is slower):

|arm|M=17|24|32|48|64|96|128|256|
|-|-|-|-|-|-|-|-|-|
|FIN2+FINPF (wk6m_gf / wk6q)|0.992|0.997|1.005|1.004|1.000|1.015|1.026|1.016|
|WEARLY (wk6m_gw / wk6q)|1.007|1.006|1.001|1.007|1.008|1.011|1.006|1.006|
|all three (wk6m / wk6q)|1.003|1.002|1.005|1.009|1.009|1.025|1.032|1.020|

wk6m against wk6p at M 16, 20, 28, 40, 80, 97, 112: 1.001, 0.996, 1.004, 0.998, 1.023, 1.016, 1.024. GENLATE on wk6 again 0.994 to 1.002. mid2 over mid at M 9, 12, 16: 0.998, 0.997, 0.997. Controlled fits at M=32: wk6m 27.2 + 6.26 E against wk6p 27.8 + 6.18 E, so the fixed cost drops 0.6 us and the per-expert cost rises 0.08 us, and the slope loss eats the gain once more than 8 experts are touched. Verdict on every lever: no-go.

### Mechanism

FIN2 did what it was built for at the end of the kernel and the saving was paid before the barrier. Timeline for wk6p against wk6m at 32 tokens, three sampled routings:

|quantity (us)|wk6p|wk6m|
|-|-|-|
|final pass after the barrier, max|4.9 / 4.6 / 4.6|2.6 / 2.6 / 2.6|
|phase B end, max|143.4 / 100.9 / 110.8|145.9 / 102.1 / 111.1|
|barrier release|144.4 / 101.6 / 111.9|147.5 / 103.2 / 112.4|
|last CTA done|149.0 / 106.2 / 116.5|149.8 / 106.2 / 114.9|

The shuffles and float reductions sit in the epilogue of the CTAs that finish phase B last, so a 2.3 us cheaper final pass moved the barrier release later by 0.5 to 3.1 us. WEARLY moved the first weight issue 0.5 us earlier (4.9 to 4.4 us at 32 tokens, level with the join at 4.4 us) but the stage ranges from the activation warp, ready at about 4.2 us, bound it, and the end-to-end arm shows nothing. Rule of thumb we took: judge a barrier-side change by the barrier-release stamp, not by the stage it removed.

## 6. A static finding worth keeping: weight-ring arrives

The wgmma build has 32 weight-ring empty-barrier arrives (`SYNCS.ARRIVE` on offset 0x428). In all 32, the last shared-memory load of the stage before the arrive was issued 72 to 95 instructions earlier and its scoreboard is waited on 13 to 31 instructions after issue, by its decode consumers, which is before the arrive. So in the compiled code the arrive never executes with a shared load of that stage still in flight, and the formal proxy-fence gap between generic loads and the next TMA write has no practical window with this toolchain. It is a scheduling property of this ptxas, not a guarantee. We re-ran the check on the DQ8 file and on the wk6m builds (32 of 32 each time).

## 7. Ideas parked without a build

- Finisher-latency prefetch at the end of phase B, about 2 us at d32, v128 and v256. It needs a new `cp.async.bulk` and mbarrier protocol in the replay-race area, and we had no GPU to test it.
- Final-pass staging through shared memory and token pipelining for 97 tokens and up: about -2 us at v256 and -0.7 us at v128, +0.2% of score at best.
- Weight-producer start decoupling: bounded by the activation-ready flag chain, at most 1.2 us (WEARLY later moved the first weight issue 0.5 us with no end-to-end effect).
- T=1 last-arriver final pass and staggered pollers: at most 0.3 to 0.4 us each.
- Split-expert reuse, +0.8% upper bound (see section 4).
- Persisting L2 across replays: against the cold-L2 intent of the benchmark, and another team measured it dead.
- Runtime-tunable stream-K cost table scanned by a prepare-time A/B: unknown sign.

## What we would try next

The fixed-cost target is still the right one, but the lesson of section 5 is that the work has to come off the CTAs that finish last. The late finishers are mostly the CTAs holding a hot expert, which spreads A-done and B-done by 8 to 11 us. We have no tested design for that. A zero-cost-dequant ablation inside prepare would bound any dequant idea before building it: if the loop with decode removed does not speed up, no decode change will.

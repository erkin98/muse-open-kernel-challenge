# MoE experiment ledger

This file lists every MoE experiment we ran or closed, one row per experiment, with job ids, numbers and verdicts.

Shapes: d1, d16, d32 are decode with 1, 16 and 32 tokens; v1, v128, v256 are verify with 8, 128 and 256 tokens. Builds: tiny (up to 4 tokens), tiny2 (5 to 8), mid (9 to 16) and wk6p (17 and up). Base file v3 has sha256 prefix c0828b74 and scored 2.362x (job c663b9dfc629461b, CI 2.327-2.397); a rerun, 03072c0ab2854c2e, scored 2.344x (CI 2.309-2.379). Identical-code paired ratios: d1 3.061-3.114, d16 1.955-2.001, d32 1.667-1.668, v1 2.189-2.279, v128 2.604-2.623, v256 2.866-2.898. The narrative write-ups are fixed-cost-anatomy.md and dequant-and-scheduling-variants.md.

## Run on the judge

|id|file or build|what it tested|result|verdict|
|-|-|-|-|-|
|Diagnostic run, job 36fb48423c7d4e62|v3 plus a diagnostic block in prepare (sha256 prefix 3399d587)|Per-CTA timelines of all four builds, controlled-routing fits, pure-stream bandwidth, small tuning arms|2.363x (CI 2.328-2.398), 5/5, d32 1.668 (the fourth identical reading). Wgmma build: intercept +21 us over a pure stream, slope 6.14 us per expert (the stream rate). Final pass after the barrier 4.4 to 5.4 us on wk6p against 0.8 us on the skinny builds|Measurement. Fixed cost is the target, not bytes|
|Stack with DQ8, job 84b9fccd4a804a48|sha256 prefix 1093a6e5: SGP and GENLATE on tiny and tiny2, GENLATE on a new wk6q build for 97 tokens and up, 8-op dequant (DQ8) on wk6p and wk6q|Whether the stack helps|2.222x (CI 2.188-2.255), 5/5. d1 3.044, d16 1.965, d32 1.576 (-5.5%), v1 2.205, v128 2.316 (-11.3%), v256 2.499 (-13.4%)|Regression. DQ8 closed, never stack on it. Only d32 attributes cleanly to DQ8|
|Treatment check, job 10650a018faf4076|sha256 prefix 37e958b6: GENLATE on mid (mid2), wk6m for 17 to 96 tokens (FIN2, FINPF, WEARLY), DQ8 off, in-process arms|The wk6m treatment on d32|2.350x, 5/5. wk6m failed to import (module name 202 characters, limit 200). d32 ran v3 code: ratios d1 3.11, d16 2.00, d32 1.65, v1 2.21, v128 2.61, v256 2.85|Lost to a naming bug. Arms that did run: GENLATE about 0, SGP at most -1.2% at M=1|
|Treatment check, job 7bff417c13bd4edf|sha256 prefix 7a3168fc: the same device code with a hashed module name|FIN2, FINPF, WEARLY, wk6m, mid2, SGP, GENLATE as in-process arms and the scored rows|2.345x (CI 2.309-2.380), 5/5. d1 3.100, d16 1.983, d32 1.654, v1 2.206, v128 2.596, v256 2.852. wk6m / wk6p 0.996 to 1.004 at M 20 to 40, slower above M=48|All no-go. Nothing built from this goes to a candidate|

## In-process arms (paired graph replays, ratio to the reference arm)

|id|arm|result|verdict|
|-|-|-|-|
|FCI 4|Stream-K flush constant 4 instead of 3, wk6p|0.1% to 0.5% faster at M 17 to 256 (diagnostic run)|Closed under our 1.5% bar|
|mid NS=4|One more weight-ring slot on mid|1.2% slower at M=16, 0.3% at M=24|Closed|
|Grid below 132|128, 124, 120, 112 CTAs|mid +1.1%, +4.2%, +6.1%, +11.7%; wk6p at M=32 +1.3%, +1.5%, +3.5%, +8.0%|Closed. Use every SM|
|Dispatch boundary|mid above 16 tokens|Ties at 17 and 20, slower from 24 (+3.2% at 24, +10.5% at 32, +20.5% at 64)|The 16/17 boundary stays. At M=16 wk6p is 5.7% slower than mid|
|Ring depth in the pure stream|3, 4, 5 slots of 36 KB|4.26, 4.28, 4.27 TB/s at 537 MB|Ring depth does not limit the stream|
|SGP|Skinny scale-pair merge on tiny, tiny2 (mid as a check arm)|tiny at M=1 0.988 then 1.008, M=4 0.997 then 1.002; tiny2 at M=8 0.998 then 0.999; mid 1.001 and 1.003|Noise. Static: 149 to 140 registers, 32 fewer instructions|
|GENLATE on wk6|Generation store moved behind `route()`|wk6q / wk6p 0.998 to 1.002 (first run), 0.994 to 1.002 (second)|Zero|
|GENLATE on mid|mid2 / mid|0.996, 0.992, 1.000 at M 9, 12, 16 (first run); 0.998, 0.997, 0.997 (second)|About 0.3%|
|FIN2 with FINPF|Sum of squares via fp32 atomics in the phase-B epilogue plus a single-pass final|wk6m_gf / wk6q: 0.992 at M=17, 1.005 at 32, 1.026 at 128. Final pass after the barrier 4.9 to 2.6 us, but the barrier release moved 0.5 to 3.1 us later|No-go: the saving moved in front of the barrier|
|WEARLY|Weight TMA before the activation gather finishes|wk6m_gw / wk6q 1.001 at M=32, 1.006 to 1.011 elsewhere. First weight issue 0.5 us earlier|No-go: the stage ranges bound it|
|wk6m (all three)|Stack of the above for 17 to 96 tokens|wk6m / wk6q 1.003 at M=17, 1.005 at 32, 1.025 at 96, 1.032 at 128. Fit at M=32: 27.2 + 6.26 E vs 27.8 + 6.18 E|No-go|

## Static analysis or desk work only

|id|idea|evidence|verdict|
|-|-|-|-|
|8-op dequant on the skinny loop|One IMAD fewer in the E2M1 decode, all builds|Search found an 8-op form (7 ALU plus 1 IMAD) against the base's 7 ALU plus 2 IMAD. Per code word: 7.75 ALU ops, 15.69 issue slots, so ALU cycles match issue slots and one op saved bounds the gain at 1.2% (counts from a compiler route that gives 166 registers, so approximate)|Closed on the skinny loop. DQ8 on the wgmma loop was later tried and lost|
|IMAD.WIDE rotates alone (DQW)|Move rotates to the FMA pipe at equal instruction count|ALU 126 to 103, FMA 105 to 128 per stage. Issue-bound and DRAM-bound|Closed, never run|
|GS=4 groups for jobs of 2 n-tiles or fewer|Four commit groups|96 registers but 32 B stack, 144 B spill stores, 308 B spill loads (the 32-register A double buffer does not fit)|Closed on spills|
|5-bit scale code|Smaller scales|Scale offsets span 33 values, which does not fit 5 bits; a 6-bit field costs one more ALU op per scale|No-go|
|Uniform wgmma descriptors|Hoist descriptor math|Reading from a wrong-route build suggested R2UR; with the judge-matching build there is one UIADD3.64 per HGMMA. A judge A/B by lee101 was noise|No-go, below detectability|
|Stream-K per-unit overhead constants|Add a per-unit term|The base has none; keell0renz's constants differ 2x between standalone and in-layer runs. Under 0.5% of score|Untested|
|Split-expert reuse (column-major)|Re-stream less|73% of v256 routings split an expert into 2 or more jobs (6.2% extra streams), v128 30%, d16 35%. Upper bound +0.8%|Untested, needs a GPU diagnostic first|
|Finisher-latency prefetch at the end of phase B|`cp.async.bulk` plus mbarrier|About 2 us at d32, v128, v256. New protocol in the replay-race area|Parked|
|Final-pass staging through shared memory for 97 tokens and up|Token pipelining|About -2 us at v256, -0.7 us at v128, +0.2% of score at best|Parked|
|Weight-producer start decoupling|Start weights earlier|Bounded by the activation-ready flag chain, 1.2 us at most. WEARLY later gave 0.5 us with no end-to-end effect|Parked|
|T=1 last-arriver final pass, staggered pollers|Smaller tail on the skinny builds|At most 0.3 to 0.4 us each|Parked|
|L2 persistence across replays|Keep weights resident|Against the cold-L2 intent of the benchmark; keell0renz measured it dead|Not touched|
|PDL on the MoE file|Raise a PDL cap|The file runs with PDL off, so the cap is moot|Closed|
|Runtime-tunable cost table scanned in prepare|Retune the stream-K table at start|Unknown sign, needs a paired A/B setup|Parked|
|Weight-ring arrive ordering|Is the empty-barrier arrive ever issued with a shared load in flight?|In all 32 arrives of wk6p the last load had been waited on before the arrive (72 to 95 instructions earlier, scoreboard waited 13 to 31 later). Re-checked on the DQ8 file and the wk6m builds|No practical window with this ptxas; a scheduling property, not a guarantee|

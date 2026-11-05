# alygg77 repository digest

This file digests the public alygg77/okc-muse-spark-kernels repo at commit 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0: what its kernels change against the vjeux base, how its judge runs compare with ours, every measured win and closed lever in its documents, and the timing method it publishes.

It is the best-documented public repo for the muse boards because it ships judge numbers and a long list of negatives. Everything here is described in our own words. We did not copy or port any of its code. We read its documents in full and read its kernels only as a diff against vjeux's f35a41e, for understanding. Licence: Apache-2.0, plus nathenfernandes Fernandes's MIT text in a licences folder.

## What is in the repo

|File|How we read it|
|-|-|
|README.md, RESULTS.md, results.json|Read in full. results.json numbers are exact, case fields are condensed into labels|
|docs/NEGATIVE_RESULTS.md (31 KB), docs/H200_MEASURED_FACTS.md (32 KB)|Read in full|
|Its index text file, NOTICE, LICENSE, the nathenfernandes Fernandes licence|Read. Credits and licence texts|
|timing_kit/README.md|Read in full|
|timing_kit/paired.py, pool_runs.py, stalls.py, __init__.py|Method only, not saved|
|kernels/ (attention 155 KB, bf16 383 KB, decode_layer 840 KB, moe 326 KB, prefill 293 KB)|Diffed against vjeux f35a41e|

## What the kernels change, in prose

Attention and prefill. The only diff is two added comment lines (the vjeux attribution and a submission notice). So alygg77's 2.653x attention row is vjeux's kernel, and its gap to our 2.605x run of the same kernel is pure judge noise.

Moe (about +112 and -8 lines):

- KPERM3. Activation loads are re-indexed to a permuted K order. The per-stage scale fetch becomes one 8-byte load per lane per stage, built once per stage (the old loop reloaded per load group). At prepare time the NVFP4 codes are permuted along K inside every 256-K stage, the gate/up row order is re-mapped so the phase-A epilogue still stores at the usual positions, and the scale table is permuted to match.
- A helper-warp change in the 9-to-16-token build: an alternate source in which the helper (producer) warp is physical warp 0, made by rotating the logical thread index. It has no separate number.
- A per-thread CUDA primary-context bind before the kernel function, credited to Matsvei.
- Documented gain: +1.25% of the moe board.

Decode_layer (about +2,740 and -30 lines). The base is untouched except small hooks, and everything else is appended as layered overrides of prepare, the after-attention path and the kernel function:

1. KPERM3 in the decoder's MoE, same as above. The Marlin bank is no longer packed, since native experts cover the supported shapes.
2. A dedicated MoE build for 16 < M <= 32 (called B4FIN). The phase-B epilogue writes fixed-order sum-of-squares partials per route and 16-column group, and the final pass is column-parallel, over (token, 1024-column segment) items, with the 8 routes' y loads issued before the reduction.
3. Glue for T in {1, 8, 16, 32, 128, 256}: an 8-CTA cluster boundary kernel, Gluon row-boundary kernels, a one-warp-per-(token, 2) route finish, pre-computed alpha and beta residual-gate vectors, and the o projection pinned to the bf16_proj kernel for T <= 16.
4. cuBLASLt plans for T = 128 and 256. The qkv at m = 256 uses a fixed algorithm (id 66) built from attributes, and the pooler-router, unpooler and o use picks from a sweep. At T in {128, 256} the kernel function runs qkv through this plan and then vjeux's attention.
5. A prepare-time chain tuner. For each T it times 5 complete after-attention variants (glue on or off, boundary warps 16, 8 or 4) on synthetic inputs routed to the busiest expert range, with a cold read before each replay. It keeps the default unless a variant wins by at least 0.4 us. It also pins static bf16 sub-kernel choices for T <= 32 and uses the fused pooler plus router for T <= 16.
6. Sliding-specific cluster and tile finishing kernels.
7. Small-token MoE builds with the helper warp at physical warp 0 (tiny, tiny2, mid) and a final prefetch flag in the tiny build.
8. QKV at m = 2 to 8: a narrow build whose escape corrections are read directly per lane, keeping the PDL wait.
9. A per-thread context bind in the kernel function.
10. An exact FP32 router at T <= 8 (next section).
11. Deferred normalisation at T = 1 (no first-boundary launch). The o-projection epilogue writes FP32 rows for the router, BF16 rows for the pooler and partial sums of squares. The route kernel forms the norms and combines, and the fused pooler interleaves 4 FP32 router stages into its weight stream.
12. A final prepare pins the T <= 8 chain to the exact-FP32-router path every time, so the tuner can never pick an older chain that fed the router a BF16-rounded copy.

Bf16 (outside our two main problems, +2,618 lines): nathenfernandes's MIT routes and a small router, pinned cuBLASLt large-m configs (some transposed), independent per-warp escape corrections without atomics, "iv" router codec builds, custom O-layer and router-layer kernels, a big-m qkv kernel, context binds. The O-layer, router-layer and big-m qkv sources are zlib plus base64 payloads, which the live rules ban, so nothing from that file is usable.

## Router feed at T <= 8

- In the vjeux base (and so in our entry before we fixed it), boundary 1 computes the normalised activation in FP32 but stores only a BF16 copy. The pooler and router both read that copy at every T, and the router uses BF16 hi plus lo weight rows with FP32 accumulation. At T <= 8 the router input is therefore BF16-rounded.
- alygg77 also writes an FP32 copy of the normalised activation at T <= 8, and a later override recomputes it exactly because an earlier version still had about 2e-3 relative error. The router runs as an FP32 by FP32 GEMV, or as the interleaved FP32 stage in the fused pooler at T = 1. Its documents say SGLang's reference feeds the router an FP32 activation and FP32 weights at T <= 8 and BF16 plus hi/lo above 8 tokens. Matsvei found this and alygg77 checked it against the task baseline.
- The live rules let the router's FP32 weights run as BF16 hi plus lo at any batch size. Their line on activations (stay BF16 or FP32 wherever SGLang's kernel keeps them in BF16) is worded for BF16 sites and does not name the FP32 router input, so reading it as meaning that FP32 stays FP32 where SGLang keeps FP32 is our inference. Our entry passed 13 of 13 judge tests with the rounded input, so the tolerance accepted it, but the organisers read top entries.
- Recommendation we followed: at T <= 8 feed the router the FP32 activation and keep the hi/lo weights. The FP32 activation is already in registers in boundary 1, so a second FP32 output costs about 2x the x-bytes on 1 to 8 rows, and alygg77 measured the interleaved version at board parity.
- Their measured facts: the exact compliant router has a maximum absolute difference of 8.6e-6 and 7.6e-6 against FP32 F.linear, and top-8 ids are identical in 882 of 882 cases. The judge prices the FP32 router at v1 about twice as high as local runs (3.8% against 1.8%).

## Per-case comparison with our judge runs

Our runs are the profiles c20df44ab71d4005 (attention), 41f596feaa60434e (moe) and 9b0d0973f11a4758 (decode_layer). Theirs come from results.json (judge, 2026-10-29). The ratio column is their speedup divided by ours, so above 1 means they are faster.

Noise. The attention rows are the same kernel on both sides, so they measure judge noise between two runs days apart. The ratios run from 0.941 to 1.055 and the board level is 1.8%. We treat per-case gaps under about 5% and board gaps under about 2% as noise. Their own pad tests put single-layout noise at about 1.5 us at v32 and 0.7 us at d1, v1 and d32.

Attention (identical code, so this is noise calibration):

|Case|Ours us|alygg77 us|Delta us|Ours x|alygg77 x|Ratio|
|-|-|-|-|-|-|-|
|d1s|4.94|5.19|-0.25|4.253|4.002|0.941|
|d1f|11.27|10.95|+0.32|2.791|2.890|1.035|
|d16s|8.37|7.91|+0.46|3.254|3.434|1.055|
|d16f|86.58|83.65|+2.93|2.312|2.415|1.045|
|d32s|10.00|9.70|+0.30|3.107|3.279|1.055|
|d32f|165.55|159.93|+5.62|2.330|2.422|1.039|
|v1s|6.80|6.77|+0.03|3.415|3.430|1.004|
|v1f|14.31|14.25|+0.06|2.518|2.567|1.019|
|v16s|11.39|11.77|-0.38|2.564|2.540|0.991|
|v16f|101.04|99.98|+1.06|2.022|2.070|1.024|
|v32s|14.85|14.82|+0.03|2.321|2.359|1.016|
|v32f|184.09|182.13|+1.96|2.111|2.155|1.021|

decode_layer (alygg77 is the vjeux base plus the changes above):

|Case|Ours us|alygg77 us|Delta us|Ours x|alygg77 x|Ratio|
|-|-|-|-|-|-|-|
|d1s|77.53|69.33|+8.20|2.220|2.497|1.125|
|d1f|82.49|74.43|+8.06|2.193|2.445|1.115|
|d16s|135.13|129.35|+5.78|1.975|2.060|1.043|
|d16f|212.62|203.96|+8.66|2.057|2.119|1.030|
|d32s|157.66|149.13|+8.53|1.668|1.750|1.049|
|d32f|309.22|304.34|+4.88|1.966|2.025|1.030|
|v1s|108.99|110.12|-1.14|2.378|2.443|1.027|
|v1f|117.39|117.16|+0.23|2.313|2.400|1.038|
|v16s|255.00|244.17|+10.83|2.062|2.160|1.048|
|v16f|344.45|327.81|+16.64|2.082|2.116|1.016|
|v32s|338.21|303.85|+34.36|2.198|2.319|1.055|
|v32f|495.90|464.10|+31.80|2.172|2.241|1.032|

How we read the decode table:

- Clearly beyond noise: d1s and d1f, at about 8 us and 12%. This is the T = 1 glue path (deferred normalisation, no first-boundary launch, cluster boundary, route finish, QKV escape reads, helper warp zero). KPERM3 does not run at d1.
- At v1 (T = 8), alygg77 is no faster. It pays for the exact FP32 router (priced by the judge at about 3.8% at v1), so the glue savings are spent there.
- Probably real, but only 3 to 5.5% per case: d32s, v16s and v32s, where KPERM3 runs (judge d32s +3.1%, v16s +5.2%), plus B4FIN at d32 and the cuBLASLt picks at v32.
- The SGLang baselines also moved between runs: ours at v32s was 743.5 us against their 704.7 us. That is why the speedup ratio is the fairer column.

moe (alygg77 is the vjeux base plus KPERM3 and the MID helper warp zero):

|Case|Ours us|alygg77 us|Delta us|Ours x|alygg77 x|Ratio|
|-|-|-|-|-|-|-|
|d1 bs1|25.09|24.64|+0.45|3.083|3.162|1.026|
|d16 bs16|93.99|92.27|+1.72|1.976|2.047|1.036|
|d32 bs32|129.18|123.96|+5.22|1.625|1.670|1.028|
|v1 bs8|65.05|63.61|+1.44|2.221|2.238|1.008|
|v16 bs128|180.09|175.24|+4.85|2.557|2.676|1.047|
|v32 bs256|216.19|207.76|+8.43|2.841|2.963|1.043|

The loads that run the WGS build (d32, v16, v32: +2.8%, +4.7%, +4.3%) match alygg77's same-cluster KPERM3 read (+2.6%, +3.9%, +3.1%) plus noise. The d1, d16 and v1 loads sit in the +0.9 to +1.8% cluster-offset band that they documented. Board: 2.4015 against 2.329 (+3.1%), of which about +1.25% is KPERM3 and the rest is noise or cluster offset.

## Measured wins in the facts file (attention, moe, decode_layer)

- KPERM3 on moe: +1.25% of the board; judge d32 +2.6%, v16 +3.9%, v32 +3.1%. Locally d32 is 2.05 us, v16 3.01 us and v32 5.92 us faster. Decode-loop instructions per weight word fall from 20.12 to 18.44.
- KPERM3 on decode_layer: local +0.68% of the board; judge d32s +3.1%, v16s +5.2%, other changed loads +0.5 to 1.2%.
- cuBLASLt picks inside the layer: pooler-router at m = 256 gains 1.32 us, unpooler at m = 256 0.78 us, o at m = 256 0.40 us, pooler-router at m = 128 0.96 us, unpooler at m = 128 0.19 us, all bitwise equal. Combined, v32s is faster by 1.97 to 2.16 us and v32f by 2.8 to 3.4 us. They shipped these despite listing the sweep as having little headroom.
- Interleaved FP32 router stage: board parity (0.06 to 0.09% slower); d1 faster by 1.0 us, v1 slower by 1.9 us. The alternatives cost 0.23% to 5.0% of the board.
- Prefill only: KPERM-style scale sharing plus a 2-CTA cluster split-merge gives t1s 7.1 us and t1f 6.9 us faster (+0.35%); a fine-plan build for T 2,048 to 4,096 gives t4s 4.0 and t4f 2.6 us; a 2-CTA cluster merge of two key splits gives about 2 us per call.
- Bounds, not wins: d1s takes 73.5 us against a 44.6 us byte floor, v1s 109.9 against 77.5. The d32s budget is MoE 84.0, dense GEMVs 51.5 (floor 34.9), glue 8.8, attention 9.8 us. The v32s budget is MoE 188.2, dense 83.4 (floor 72), attention 14.8, glue 18.4 us. The WGS expert stream loses 3 to 6 us in ramp, 3 to 6 in phase-A waves, 4 to 12 in shadows and 7 to 11 at the end. Verify attention slows by 50 to 80% next to any co-resident GEMM.

## Closed levers (negative results)

decode_layer:

|Lever|Result|
|-|-|
|Pooler and router stream started before attention|2.1 to 7.7 us slower; each join costs at least 2 us|
|Side-stream splits|2.1 to 3.2 us slower; Marlin halves up to 35 us slower|
|Skipping glue launches, more PDL pre-wait stages, early trigger|Each skipped launch saves 1.2 to 2.0 us, all three 4.0 to 4.8 us. Two pre-wait stages are optimal (four or six slow attention by 1.7 us and a boundary by 0.9 us). An early trigger gains at most 0.736 us|
|L2 prefetch of the next weights|2.6 to 5.1 us slower|
|Launch-table search|0.15 us, noise|
|cuBLASLt sweep|About 2 us per layer (shipped anyway)|
|Removing split-K and its combine|Exposed combine is 0.00 us; ceiling 0.12%|
|Stream-K or more CTAs for the qkv GEMV|Saturates near 96 CTAs; ceiling at most 0.6 us; fix-up tail 1.5 to 4 us|
|Lane-major repack or producer-warp decode for the d1 codec GEMV|+0.78 to 1.02 us and +6.3 to 6.5 us slower|
|nathenfernandes's T = 128 QKV block in the decoder|Board -0.09% to -0.16%|
|+3 MB of prepare scratch|0.5 to 1.3 us slower on SASS-identical code, so reuse existing buffers|
|FP32 router variants|-2.27% (separate GEMV) to -5.0% (cuBLAS), -0.23% (fused tail); interleaved is at parity|

moe:

|Lever|Result|
|-|-|
|Fourth ring slot|Within plus or minus 1 us|
|Tail fixes|Dynamic claims cannot win; dispersed ends about 1 us; a refit table 0.8 to 4.6 us slower on the judge|
|Fence moves|About 0.13 us per fence. Removing all fences gave +0.234%, which is noise, and on decode d32f it was 1.35 us slower|
|WGS32 tile|2.2 to 3.0 us slower|
|d16 second read|1%|
|Expert byte compression|At most 5% of bytes|
|Marlin, fused-lean or SIMT replacements|10 to 149 us slower|
|Prefetching the MoE's first weights into L2|x0.99 to 1.00|
|Local gains on non-WGS loads|Did not show on the judge|

attention:

|Lever|Result|
|-|-|
|Balanced dynamic pair split, weighted splits, polynomial exp2 (2.1e-7), reordered exp work, a one-CTA FA3-style pipeline|The SMs finish at the same time either way; one-CTA was 116.4 against 100.5 us. The bound is per-SM consumer throughput: exp 40.5%, QK plus V transpose 17.5%, PV barrier 12.7%, data waits 12.5%|
|V transpose in the producer warps|v16f slower by 0.55 to 0.87 us (the transpose is already hidden behind QK wgmma latency)|
|Finalise capped at 4 rows (3 variants)|Board -0.045%|
|Wave balancing for one-request full attention (11 arms)|All lost; merge median 0.96 to 1.5 us. Fixed latency: 2 DRAM round trips before the stream and 3 L2 round trips after (2.5 and 2.8 us). Same-address red.v4.f32 costs about 17 ns per op|
|Merging 128 to 318 splits in one CTA or one cluster|+60 and +96 us; the cluster design was 1.1 to 1.2 us slower. Parallel tree merges were not measured|
|Page sorting, length-sorted persistent queue|+162 to 276 us; +3.5 to 70 us|
|Folding verify Q prep into the split core|About 1.6 us; ceiling 0.09% of the board|

Judge-side lessons from the same documents:

- A stand-in kernel predicted 6.6x the real gain.
- Cross-cluster comparisons fake gains: a +4.19% read was really +2.4%, and a moe +2.46% was +1.25%.
- A local +0.04% became -1.7% on the judge.
- Unchanged files spread 1.6 to 2.2%.
- Layout effects under 1.5 us at v32 (0.7 us at d1, v1 and d32) move with the padding.
- A/A noise reaches about 3 us per shape on moe (0.7 to 3.1 us), and one judge run cannot show a gain under about 2%.

## Timing kit method, in prose

- Arms. The harness takes three arms named base, candidate and aa. The aa arm must be a byte copy of the base, and their source MD5s are compared before and after the run, aborting on a mismatch or a change. Each arm supplies prepare, run (graph-safe), check (raises on a wrong result) and refresh (writes the same fresh inputs into every arm).
- Captures (version 2 defaults). Eight captures per arm, prepared in rotated arm order. Before each capture's prepare, a seeded random throwaway allocation of 0 to 32 MiB (in 512-byte steps) is made and kept alive, so every capture gets a different memory layout, and a check asserts that no two captures share tensor storage. Each capture is an eager warm-up followed by a CUDA graph with begin and end timing events inside the graph, so host jitter between calls is not timed. After capture the output is poisoned with NaN or int-min and the graph is replayed and checked, so correctness must come from a real replay. Every graph is re-checked after 100 cold replays on new data.
- Rounds. 48 rounds, each pairing base with candidate and base with aa in random order, each pair run as ABBA or BAAB on fresh inputs. A 256 MiB read-only torch.sum flushes L2 before every replay. There is no fixed-cost subtraction.
- Statistics. Each round gives a saving (base minus candidate, in us) and a ratio. Rounds are averaged per capture. The 95% interval is a Student-t interval over the 8 per-capture means (df = 7), because rounds that share a capture share its memory layout. A gain is counted only if there are at least 8 captures, the interval's low end is above 0, and the mean gain exceeds the largest absolute per-capture A/A mean. A round-level bootstrap interval is reported for reference.
- pool_runs. States too large for 8 captures run as 2 or more processes with 4 captures each. Pooling requires version-2 results, identical source MD5s, distinct seeds (so the offsets differ), and at least 8 pooled captures, and the same counted-gain rule is applied to the pooled per-capture means.
- stalls. Summarises Nsight Compute CSV per kernel (duration, DRAM TB/s, top stall-reason ratios: barrier, long and short scoreboard, wait, membar, math throttle) and CUPTI traces into per-kernel time sums. It warns that stall ratios are not fractions of wall time.

What we could adopt: the in-graph events, the random allocation pad per capture, and the A/A arm with the rule that a gain must beat the largest A/A mean.

## results.json beyond RESULTS.md

Per board it adds kernel_sha256 (decode 8f17613b...f4a, moe b3307089...281, bf16 290a86ab...b90, prefill ced91e7e...d20, attention e6d929d4...124), score_us and baseline_us totals (decode 10,254.9 and 22,620.3; moe 5,600.0 and 13,448.5; attention 1,419.6 and 3,766.3), the board's rank-50 speedup at read time (decode 2.1885, moe 2.3867, attention 2.6469, bf16 1.5604, prefill 1.8022), and per-shape microseconds to 3 decimals. The judge job ids are decode a61afb9687f14eff, moe e1bcad5a55cd4a29, bf16 f07235a6ca8a46e2, prefill 7f87198d13a34650 and attention 1c8d81a0dbd94ee0.

## External links inside the repo

- vjeux/okc-2026: our base.
- NathenFernandes/muse-bf16-proj: default branch has README, LICENSE, a results folder (ranking history, source hashes, v18.json) and a tool named expand_source.py. We did not read the two JSON files (bf16 data only) or the tool. The name suggests the submission is stored condensed and expanded, so it needs a rule look before anyone borrows bf16 code from there.
- openkernelchallenge.dev: the rules, already saved. We did not run the install script.
- A coordination site named in the README, which we did not use.
- apache.org: licence text only.

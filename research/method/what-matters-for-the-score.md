# What matters for the score

Which parts of each problem move its score, which problem is cheapest to climb on, and the optimization leads we ranked, including the ones we had to correct.

The numbers come from the judge's own per-kernel profiles of our four board runs, combined with the case weights from the task files. We wrote a small script that rebuilds each problem's score from the profile's per-case times, and it reproduced the judge exactly (decode_layer 2.096, attention 2.604, bf16 1.492). That is why we trust the shares below. Scores are the state around midday on Nov 4.

## How much each part moves the score ("pull")

The pull of a part is the average, over the six loads, of its share of the weighted load time. Making a part x% faster raises the problem score by about pull × x%.

Where kernels overlap (the profile prints "GPU busy" lines for PDL or streams), shrinking one kernel may not shrink wall time one for one, so those pulls are upper bounds. In the tables the six columns are decode 1, decode 16, decode 32, verify 1, verify 16 and verify 32.

### muse_decode_layer (board 2.096×)

|Part|Pull|d1|d16|d32|v1|v16|v32|
|-|-|-|-|-|-|-|-|
|MoE (`moe_fused`)|0.457|27%|45%|43%|52%|55%|52%|
|projections (`sk2_kernel`; cuBLAS at large m)|0.318|49%|30%|27%|31%|27%|26%|
|attention (`attn_kernel`)|0.137|8%|17%|23%|7%|13%|15%|
|glue (`_boundary1`, `_boundary2`, `_route_split`, `_route`)|0.088|16%|8%|6%|10%|6%|7%|

The MoE share is the largest by far, which makes any gain in the moe kernel count twice: in full on the moe board, and at 0.457 pull inside decode_layer.

### muse_attention_decode (board 2.605×)

|Part|Pull|d1|d16|d32|v1|v16|v32|
|-|-|-|-|-|-|-|-|
|full-context cases|0.678|44%|78%|85%|42%|76%|81%|
|sliding-window cases|0.322|56%|22%|15%|58%|24%|19%|

### muse_bf16_proj (board 1.491×)

|Layer|Pull|d1|d16|d32|v1|v16|v32|
|-|-|-|-|-|-|-|-|
|pooler|0.326|39%|29%|29%|35%|32%|31%|
|qkv_gate|0.226|25%|21%|20%|25%|22%|23%|
|unpooler|0.222|10%|29%|29%|11%|27%|27%|
|o|0.122|15%|11%|12%|15%|10%|11%|
|router|0.104|11%|11%|10%|13%|9%|8%|

### muse_moe_w4a16 (board 2.329×)

One kernel, and each of the six loads counts 1/6. Every case runs at 45 to 57% of its floor speed. The moe profile showed time growing by about 6 to 7 µs for each distinct local expert touched, which looked like a parallelism limit and turned out to be bytes (see below and `lessons-learned.md`).

## Which problem is cheapest to climb

The per-problem rank we would reach if that problem's score rose by the given amount (boards read between 11:40 and 11:55 on Nov 4):

|Problem|Now|+1%|+3%|+5%|+10%|+20%|
|-|-|-|-|-|-|-|
|bf16_proj|#355 of 1,699|315|256|182|14|1|
|attention_decode|#304 of 1,331|217|116|66|24|4|
|moe_w4a16|#397 of 1,387|352|176|84|28|1|
|decode_layer|#449 of 1,255|435|401|362|75|4|

A real +3% gains about 190 places on attention and about 220 on moe, which makes those the cheapest rank. Decode_layer barely moves below +5% and then jumps near +10%. The overall rank combines the four problems, and its formula is not published (see `challenge-rules-as-we-read-them.md`).

## Leads, ranked by evidence

Each lead carries its status. Two of the early estimates were wrong, and we say how.

### decode_layer

1. The moe kernel inside the layer. At 0.457 pull this is the main lever, and any improvement to the moe kernel carries over.
2. A faster 1-token GEMV for qkv, o and the unpooler, at T = 1 only. Our first estimate was +9.5%. It assumed the projections read uncompressed weights through a separate router. Both assumptions were wrong. The layer's `sk2_kernel` already reads the same 12-bit lossless format (a `COMP=true` template argument in the kernel name), and the 18.7 µs kernel is the fused pooler plus router reading about 59 MB compressed. The faster GEMV only handles m = 1. The real gain is qkv at T = 1 (11.7 down to about 6.7 µs) plus smaller gains on o and the unpooler, 4 to 8 µs on decode 1 in total, or about +1 to 1.5% overall. It would be chosen by timing in `prepare()` and default to the old path. When we built a version of this, its first check failed 10 of 13 cases (a load hoisted above the PDL wait; see `replay-race-hazards.md`).
3. Starting more weight stages before the PDL wait. The pooler could stream about 19 MB during `_boundary1` instead of about 6 MB, and the routing kernel's warp count could be tuned. Others' measurements say two stages is already the optimum (see "Closed by others' numbers" below).
4. Glue fusion, worth at most 0.088 pull.
5. Any attention win, worth 0.137 pull.

Dropped: fusing the router into the pooler GEMV. It is already fused in the file for T up to 16, and the author kept it separate at T = 32 after measuring.

Decode_layer needs about +10% to move rank much. Without a large moe win we expected small rank gains there.

### attention_decode

1. Load balance on full-context cases. The scored cases use a context-length spread meant to mimic coding-assistant sessions, so lengths vary per sequence. If the work is split per sequence, SMs finish unevenly. Splitting by KV tokens across all 132 SMs (stream-K, in the style of LeanAttention) should fix that. Full-context cases run at 63 to 76% of floor speed, and they carry 0.678 pull. We have no result for this lead in these notes.
2. Register and MUFU relief under the Nov 3 rules: FP16 accumulation of QKᵀ and P·V, FP8 P, and polynomial exp2. We tried a linear-mantissa exp2 and it gave nothing (2.550× on that check). Others report a ceiling of about 7% for polynomial exp, and report FMA-emulated exp2 as slower by 24 to 105 µs.
3. Batch-1 sliding, a 4.2 µs kernel against a 0.1 µs floor. This is a latency chain, not a bytes problem, so the idea is more CTAs over the 2,048-key window and a shorter serial chain. Cold-code warming was the attempt here and gave no gain (`how-the-speedups-happened.md`).

### moe_w4a16

Our first guess, limited parallelism, was wrong. For M of 17 and up the kernel already cuts the work into 132 equal-cost stream-K ranges (about 8 CTAs per expert at decode 32 and 5 at verify 32). The 6 to 7 µs per expert is byte cost: 28.31 MB at 4.6 TB/s is about 6.15 µs.

The measured loss is at unit boundaries: GPU-scope release fences (about 0.9 µs per unit per warpgroup), the epilogue and split-unit atomics, and a 3.5 to 6.9 µs spread at the end of a phase. That per-CTA timing comes from alygg77's stamps on the same base.

1. KPERM3, a lossless K-permutation of the NVFP4 codes, so each scale pair is decoded once per four k16 steps. alygg77 reports +2.6% at decode 32, +3.9% at verify 16, +3.1% at verify 32, roughly +1.25% overall.
2. Fewer fences: one release per CTA range instead of four per unit.
3. Approximate SiLU and FP16/BF16 split partials, allowed by the new rules, expected small.

Picks happen in `prepare()` at a measured win of at least 1% per M bucket (the in-prepare A/B noise is about 0.5%). The bar to submit is still a win over the board entry outside its 95% interval.

### bf16_proj

The one lever left untried in these notes was 12-bit weights inside the m = 128 GEMM. The pooler at verify 128 takes 22.3 µs against a 14.6 µs floor at 16 bits and would be about 11 µs at 12. It needs a new tensor-core kernel and we had no local compiler for it, so we judged it high risk. Others have since measured it as a loss (below).

Already ruled out in our own runs: cuBLASLt pins, a split-K wgmma GEMM, K-major cuBLAS, and extra gq/gqx instances. Our clean-room CUDA is slower than the borrowed file on every shape, and Huffman or DFloat11-style coding is slower at small batch.

### Closed by others' numbers

From alygg77's measured negatives, as quoted by trxphxle:

- 12-bit weights inside the m = 128 GEMM: `o` and the unpooler lose 2.4 to 4.1 µs, and the pooler takes 32.6 µs against 20 to 21.
- More PDL pre-wait stages: closed, two is the optimum.
- Cutting moe fences: about noise (+0.23%).
- Polynomial exp has a ceiling of about 7%.
- Moe: a fourth ring slot, a dynamic tail, a WGS32 tile, byte compression and moving fences are all closed.
- The best open attention lead at the time was cold-code warming for the batch-1 sliding cases. We tried it and it did not help.

## Picking what to work on

1. Run the sensitivity script on the latest judge profile. Choose the part with the largest pull that also has the most room to its floor.
2. Estimate the rank payoff from the table above before committing checks.
3. Bundle several candidates into one checked file, as described in `measuring-on-the-judge.md`, so one trip through the queue answers more than one question.

# Literature and public notes checked against the kernel

We read the main decode-attention papers and some public notes before spending judge runs, and this file records what each says about our kernel and what a public page for a neighbouring problem found.

In the table, "verified" means we read the relevant sections. "Inference" means we are going from notes or memory.

## Papers and posts

|source|mechanism|where it sits in our kernel|status and verdict|
|-|-|-|-|
|Flash-Decoding blog (crfm.stanford.edu, 2023-10-12)|split the KV sequence, attention per split, log-sum-exp per split, then reduce|flat equal-share split-KV grid plus a finalize|verified; already in, nothing to add|
|2311.01282 FlashDecoding++|asynchronized softmax with a unified maximum, no partial-softmax sync|fixed reference plus a rare conditional rescale (QK-norm bounds the logits)|verified (sections); already in|
|2407.08608 FlashAttention-3|warp specialization, FP8, 132-SM divisibility to avoid wave quantization; no decode or KV-cache numbers in the text|producer/consumer ring, FP8 P, flat grid sized to the SM count|verified (sections 4.1, C.1, C.2); already in|
|2501.01005 FlashInfer|load-balanced scheduler, tile-size selection; page size 1 (vector-sparse) within 1% of contiguous KV at decode|equal-share grid, gather per row|verified (sections); already in. It also says that the row gather alone is not the limiter|
|2205.14135 FlashAttention|IO-aware tiling, online softmax|basis of the kernel|verified; nothing new|
|FlashAttention-4 ideas (conditional rescale, polynomial exp2 split between MUFU and FMA)|cheaper exp|our ideas on packed half2 softmax and bit-trick exp2|inference (via notes). The bit-trick exp2 measured no change; packed half2 has the same instruction count per pair in SASS|
|2605.30571, batch-1 decode overhead|a fixed per-step overhead keeps batch-1 decode far from peak|the judge already replays CUDA graphs; our 1.4 us fixed cost and 3.0 us floor say the same thing|verified (skim); nothing new|
|LeanAttention (2405.10480), Hopper microbenchmarks (2402.13499)|stream-K style equal split of attention work; DSMEM and L2 latency numbers|the flat grid already splits by tile count; cluster paths are closed|inference from memory; not opened. Neither changes the arithmetic in [gap-to-the-leaders.md](gap-to-the-leaders.md)|

Result of the reading pass: every mechanism in these sources is already in the kernel or on the closed list. None gives a number for our actual configuration, which as far as we know is 2 KV heads, 64-dimensional heads (so a 128-byte row in FP8 per token, though we did not re-derive that), FP8 KV and page size 1 on Hopper. There is therefore no external bound for the chain shapes either, and the 3.0 us floor we use is our own estimate.

## An outside experiment page on a neighbouring problem

A public page at https://posttrainllm.com/experiments/openkernel-attention-20261023 records a closed experiment dated 2026-10-23, with a longer write-up at https://posttrainllm.com/docs/learn/openkernel/. We read both for ideas and did not open their submission code.

The experiment is about fusing the attention preparation (QK-norm, RoPE and the cache-slot writes) in front of the existing SGLang attention, not about a new attention kernel. Their scored H200 run, f141c8d265234a99, reached 1.1357x aggregate (95% interval 1.1187 to 1.1528x) with correctness passing. The lesson they state is that the fusion has to keep BF16 rounding, the exact RoPE table, the cache-slot writes and the fallbacks for non-unit scales. The experiment closed at rank 283 of 749, short of its top-100 target.

Their write-up adds:

- A final decoder-layer composition of 1.2503x.
- A best attention candidate called joint-head FP8 paged attention at 1.3723x (interval 1.3517 to 1.3995x). It is paged GQA split-K with FP32 online softmax and a split reduction, FP8 Q/K/V converted to BF16, and BF16 probabilities. Dispatched broadly it measured only 1.0564x, and long-context verify at batch 16 and 32 regressed to 0.553x and 0.537x. Their single-request sliding decode reached 2.05x.
- Negative results: merge, norm and gate fusion at 1.3068x; a wider key tile at 1.2293x (255 registers, 96 spills in the large full decode); shared-load verification attention at 0.8432x (254 to 255 registers, no spills).
- A profile note that attention is about 50 to 55% of summed kernel time in large full-context decode.

|item|theirs|ours|
|-|-|-|
|attention score|1.37x best kernel (1.14x with the prep fusion)|2.6211x|
|single-request sliding decode|2.05x|4.14x (decode 1 sliding)|
|probabilities|BF16|FP8 P, allowed by the rules|
|batch 16 and 32 verify|regressed to about 0.55x|2.05x and 2.11x on the full layer|

We read their problem as matching ours in shape, and on that basis our kernel runs about twice as fast. We did not verify that the two benchmarks use identical shapes, so treat the comparison as indicative.

Verdict: nothing new for us. Everything they list (split-K over paged KV, FP32 online softmax, a split reduction, fused QK-norm, RoPE and cache write) is already in our kernel. Their negatives match lessons we already held: a wider key tile spills registers, and shared-load verify attention loses.

## Public notes by other teams (ideas only)

We took ideas from these and did not copy code. Credit where it applies:

- Kushagrabainsla: warming the finalize code. The notes report a merge going from 6,928 to 2,524 cycles and a kernel from 10.8 to 8.9 us. We tried a version, see [linear-exp2-and-cold-code-warming.md](linear-exp2-and-cold-code-warming.md); no effect on our kernel.
- arashattar: 32 KB of straight-line code costs about 1 us when cold, and code size moves timings by 0.2 to 3 us. Polynomial exp2 variants measured as slower in their numbers (+24 to +105 us as reported).
- alygg77: same-address `red.v4.f32` operations serialize at about 17 ns each; decode 1 full and verify 1 full spend three L2 round trips after the last merge; the judge's L2 is cold at the start of a replay; 3 MB of extra scratch slowed unchanged shapes by 0.5 to 1.3 us; a degree-5 polynomial exp2 did not pay.
- VkaZas: partitioning at the sliding shapes. We checked it on paper and found the critical path unchanged, see [launch-plan-sweep.md](launch-plan-sweep.md).
- lee101's openkernels repository: an L2 prefetch-size hint on 64-byte half-row copies helped on an H100 (350 to 311 us) and did nothing on an H200 (157.7 against 157.6 us). That is the evidence behind our decision not to try the hint on the 128-byte rows.
- keell0renz's repository lists its own ten ideas across all boards. Only one of them touches attention, and it is an exp or q_len 8 idea we had already covered.
- vjeux's public entry (Apache-2.0) is our base; its git history documents several of the dead ends we later hit ourselves.

A community discussion is also the source of the roughly 4.1 TB/s random 128-byte read ceiling that our bounds use. We never re-measured it.

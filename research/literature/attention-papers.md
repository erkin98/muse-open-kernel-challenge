# Attention papers and FP8-KV decode on an H200

This file covers what the attention papers say that matters for the muse_attention_decode board, which of their mechanisms our entry already has, and how we priced the one softmax idea the papers left open.

None of the headline numbers below was measured on an H200. FlashAttention 1 and 2 report A100, FlashAttention-3 reports H100 SXM5 at prefill and training shapes (sequence 512 to 16k), online softmax reports V100, FlashInfer reports A100 and H100 in f16, PagedAttention reports A100, Ring Attention reports A100 and TPUs. No paper measures head_dim 64, 16 query heads over 2 KV heads, page size 1, FP8 E4M3 KV and 80K context. We used the papers for mechanisms, not magnitudes.

## The problem we were pricing against

- Shape: batch 1, 16 or 32; q_len 1 (decode) or 8 (verify); 16 query heads over 2 KV heads; head_dim 64; page size 1; sliding window 2048 with RoPE, or full context around 80K tokens; FP8 E4M3 cache with scales equal to 1 in the scored cases. One KV row is 2 heads x 64 B = 128 B.
- Our per-case times in microseconds: d1 sliding 4.94, d1 full 11.27, d16 sliding 8.37, d16 full 86.58, d32 sliding 10.00, d32 full 165.55, v1 sliding 6.80, v1 full 14.31, v16 sliding 11.39, v16 full 101.04, v32 sliding 14.85, v32 full 184.09.
- Full-context cases carry 0.678 of the attention score. The full-layer share of each load is 44, 78 and 85% for decode at batch 1, 16 and 32, and 42, 76 and 81% for verify.
- Full-context decode at batch 16 and 32 already runs at 86 to 90% of the random 128-byte-row read ceiling, which another team measured at about 4.05 to 4.1 TB/s (sequential reads reach about 4.46 TB/s).

## Mechanisms and headline numbers

FlashAttention (2205.14135). An IO-aware exact attention. Q, K and V are tiled into on-chip SRAM, the N-by-N score matrix never goes to HBM, and an online softmax lets the reduction work without seeing the whole row. HBM traffic is Theta(N^2 d^2 / M) against Theta(Nd + N^2) for standard attention, and the paper proves no exact algorithm does better across SRAM sizes. On A100: 15% faster BERT-large training than the MLPerf 1.1 record (17.4 against 20.0 minutes on 8 GPUs), up to 3x end to end on a medium decoder-only transformer at sequence 1K, 2.4x on long-range arena. In the Fig. 2 case (the same medium decoder, sequence 1024, head dim 64, 16 heads, batch 64) standard attention moves 40.3 GB in 41.7 ms against 4.4 GB in 7.3 ms.

FlashAttention-2 (2307.08691). Three changes. The output stays un-normalised inside the loop, with one divide at the end and only the log-sum-exp stored. The query-sequence dimension is also parallelised across thread blocks, so small batch times heads still fills the GPU. Inside a block, Q is split across warps while all warps read K and V, instead of FA1's split of K and V, which forced shared-memory round trips. On A100 it reaches up to 230 TFLOPs/s forward (73% of peak), 1.7 to 3.0x over FA1, and 225 TFLOPs/s per GPU training a 2.7B-parameter decoder at 8k context. The same code on H100 without TMA reached up to 335 TFLOPs/s. The paper says non-matmul throughput can be up to 16x below matmul throughput.

FlashAttention-3 (2407.08608). Three Hopper techniques. A producer warpgroup issues TMA loads into a circular shared-memory buffer and returns registers with setmaxnreg while consumer warpgroups run wgmma. Softmax is overlapped with the GEMMs, both across two warpgroups (pingpong, via named barriers) and inside one warpgroup (a two-stage pipeline). FP8 gets an in-kernel V transpose with ldmatrix and stmatrix, a byte permute to turn the FP32 accumulator layout into the FP8 operand-A layout, per-block scaling, and a random-sign Hadamard rotation of Q and K to spread outliers. Numbers on H100 SXM5, FP16 or BF16: 1.5 to 2.0x FA2 forward, up to 740 TFLOPs/s (75% utilisation), FP8 close to 1.2 PFLOPs/s. FP8 RMSE is 9.1e-3 against 2.4e-2 for per-tensor FP8. Ablation at batch 4, sequence 8448, 16 heads, head dim 128: 661 TFLOPs/s with both techniques, 582 without GEMM-softmax pipelining, 570 without warp specialisation. Pingpong alone moves 570 to roughly 620 to 640. The paper lists inference-specific optimisation and a persistent FP8 kernel as future work.

Online normalizer (1805.02867). Carry a running max and a running sum, rescale the sum by exp(old max minus new max) when the max moves. That cuts safe softmax from 4 to 3 memory accesses per element, and the (max, sum) combine is associative and commutative, so partials can be merged in any order. This is the root of every split-KV merge. On a V100 it was about 1.3x over safe softmax at vector size 4000.

FlashInfer (2501.01005, version 0.2). KV cache layouts (page tables, radix trees, tree masks) are stored as block-sparse rows. Shared-prefix rows can use a second, larger block format. Kernels are FA2 or FA3 templates with query tiles {1, 16, 32, 64, 128} and KV tiles {32, 64, 128} picked by heuristics. A CPU planner, inspired by Stream-K, assigns KV chunks to CTAs by a longest-first priority queue with cost alpha*lq + beta*lkv, and attention plus the cross-chunk contraction run as one persistent kernel with a fixed grid and fixed workspace offsets so CUDA graphs work. Results on A100 and H100 in f16: 29 to 69% lower inter-token latency than the Triton backend in SGLang, 28 to 30% lower for streaming (sink-token) attention with a fused RoPE-plus-attention kernel. The scheduler ablation on H100 with Llama 3.1 8B in SGLang gave 8.63 ms against 13.89 ms without balancing and 11.08 ms for Triton, on inputs of 4096 to 16384 tokens. With e4m3 KV on another serving stack the median inter-token latency was 10.92 against 12.56 ms for the default backend. On Hopper it uses TMA only for contiguous KV and uses 128-byte cp.async for paged KV, because TMA cannot handle non-affine access. Its planner avoids atomic aggregation on purpose, to keep the output deterministic.

PagedAttention (2309.06180). KV is cut into fixed blocks found through a per-sequence block table. The paper is mostly about memory management (near-zero fragmentation, copy-on-write sharing, preemption), not arithmetic. On A100 it gives 2 to 4x serving throughput over FasterTransformer and Orca, and its attention kernel is 20 to 26% slower than FasterTransformer's contiguous one. Default block size is 16.

Ring Attention (2310.01889). Shards a long sequence over a ring of devices and passes KV blocks around while computing. Up to 256K tokens on 8 A100s for a 7B model, 4096K in the 32-GPU table. The running text says "over 1 million tokens", which is a small inconsistency inside the paper.

## Where the gains do not apply to decode

- FlashAttention 1: the saving comes from not materialising an N-by-N score matrix. At q_len 1 to 8 a fused kernel never writes it anyway. The block-sparse variant skips blocks, which the live rules forbid.
- FlashAttention 2: parallelism over query blocks gives a decode kernel nothing, because there is one query tile. Split-KV does that job. Only the divide-once-at-the-end tweak transfers.
- FlashAttention 3: pingpong and the two-stage softmax-GEMM overlap target compute-bound prefill, and the TFLOPs/s figures are prefill numbers. Block quantisation and incoherent processing are accuracy tools for quantising fresh Q, K and V.
- Ring Attention: our problem is one GPU and one layer.
- PagedAttention: the memory-manager gains sit outside one attention call, and the page size is fixed by the judge.
- FlashInfer composable formats and JIT variants: the workload has no shared prefix and one fixed variant.

The cases that do have measurable gaps are the verify-full ones, v16f and v32f (101.0 against 86.6 us and 184.1 against 165.6 us for the same KV bytes, so 14.3% and 10.0% of the verify time is extra), and the latency-chain cases d1s, v1s, d1f and v1f. No paper addresses the second group. FlashInfer's scheduler targets throughput on mixed lengths, not a fixed chain of two DRAM round trips before the stream and three L2 round trips after (another team measured those chains at 2.5 to 2.8 us).

Nothing read offers a way to read fewer bytes, and the rules forbid lossy KV and skipping keys that must be attended. So the memory-bound decode cases have at most about 10 to 14% left, by the row-gather ceiling above.

## What was already in our entry

We checked each transferable mechanism against the entry's code. Verified in the code unless marked.

|Mechanism|Where it comes from|Status|
|-|-|-|
|Warp-specialised producer with setmaxnreg and mbarrier stages|FA3|In the entry: one extra producer group of 128 threads issues every K/V copy, consumers are 128 threads (q_len 1) or 256 (q_len 8), 64-key tiles, 3 to 4 stages, 2 CTAs per SM|
|FP8 QK and FP8 PV wgmma at q_len 8|FA3|In the entry. P is built directly from the FP32 score registers, four exp2 results packed to e4m3 at a time|
|FP8 layout permute and producer-built V transpose|FA3|In the entry. Moving the transpose into the producer warps made v16f 0.55 to 0.87 us slower, so that is closed|
|Equal-share flat grid across requests|FlashInfer balancing|In the entry: all requests' keys are cut end to end into equal shares (264 CTAs, with a per-boundary charge)|
|Fixed-reference split merge with last-CTA finalise and a self-zeroing workspace|FlashInfer, Flash-Decoding|In the entry, with FP32 red atomics. QK-norm bounds the logits, which is why a fixed reference is safe|
|Page-table rows staged into shared memory by bulk copy|public notes|In the entry. K/V tile rows themselves use cp.async|
|GQA group times q_len folded into M|FlashInfer|In the entry: at q_len 8 that is 64 rows (8 tokens x 8 heads) by 64 dims|
|Rare conditional rescale, row sum on the tensor core via an extra accumulator column|FA4, FA2|In the entry. The sums live in an m64n72 PV helper (64 value columns plus 8 extra), which is our reading of the code|
|Two CTAs per SM overlapping softmax with GEMM|FA3 pingpong|In the entry at CTA granularity instead of warpgroup granularity (inference)|

Not usable under our rules: TMA tensor loads for paged KV (non-affine gather at page size 1), block quantisation and incoherent processing (re-quantising the cache is lossy, and a Hadamard on Q alone breaks QK^T since past K cannot be rotated), composable shared-prefix formats, JIT.

FlashInfer's deterministic no-atomics merge would answer the replay-failure risk described in cuda-docs-and-replay-hazards.md. But the closest measured analogue, one CTA merging 128 to 318 partials, lost 60 to 96 us, so it is a robustness option and not a speed one.

The live rules (as of Nov 3) allow queries rounded to E4M3 but nothing coarser, P stored in E4M3 at any power-of-two scale with an approximated exp2, BF16 or FP16 split-KV partials, and FP16 accumulation in QK^T and PV. The FP8 KV cache stays FP8 and every key must be attended.

## Softmax and exp2: the one idea the papers left open

The cheap, numerically safe tricks are all in the entry already: fixed reference instead of a running max, divide once at the end, row sum on the tensor core, a rare conditional rescale, power-of-two scaling of P. The tricks that move exp to another pipe have been measured as a loss or as nothing:

- Our own first attention check replaced the MUFU exp with an FFMA-plus-LEA linear-mantissa exp2 at q_len 8, plus a byte test and warp vote on interior tiles. Job 24a52be614264969: v16f 101.0 to 101.1 us, v32f 184.1 to 183.6 us. Two runs of identical code spread 101.0 and 101.6, and 184.1 and 185.6, so this is no movement.
- Another team (arashattar) measured FMA-emulated exp2 in the flat kernel at +24 to +105 us. A different team's degree-5 polynomial did not pass its own gain gate.
- One public profile of the flat q_len-8 full verify kernel splits the consumer loop as exp 40.5%, QK plus V transpose 17.5%, PV barrier 12.7%, data waits 12.5%. That leaves 16.8% unattributed.

FA3 puts the H100 SXM5 at 989 TFLOPs/s of FP16 matmul against about 3.9 TFLOPs/s of special functions (16 MUFU ops per SM per clock). For head dim 128 that is 512x more matmul FLOPs than exponentials but 256x lower exp throughput, so exp can take about half a matmul's cycle time, and more in FP8 because only the matmul speeds up. FA4's motivation is the same imbalance, 2.25x larger on B200. That is why FMA-emulated exp2 pays on Blackwell and not on our hardware.

The untested idea was to cut the number of issued softmax instructions with packed half-precision math, fed by an FP16-accumulated QK^T, instead of moving work between pipes. Our arithmetic for instructions per pair of scores:

|Path|Instructions per 2 elements|Total|
|-|-|-|
|Current (FP32 scores)|2 FFMA, 2 MUFU, 1 e4m3 convert|5|
|FP32 accumulate, packed math only after the subtract|2 FFMA, 1 convert to f16x2, 1 ex2.f16x2, 1 e4m3 convert|5, no saving|
|FP16 accumulator from the wgmma|1 HFMA2, 1 ex2.f16x2, 1 e4m3 convert|3, or 4 if ex2.f16x2 lowers to two MUFU ops|

So the idea only pays with an FP16 accumulator. Before spending a check we set two static gates that need no GPU: assemble the PTX for sm_90a and count SASS instructions per pair, and rerun the CPU numerics model of the q_len 8 path with FP16-accumulated scores.

Numerics, from format arithmetic and still to be confirmed by simulation. The f16 exp2 has about 2^-10 relative error, far below the E4M3 rounding of P (3 mantissa bits, a step of roughly 6%). The risk is the FP16 accumulator, whose rounding hits the raw dot product before the offset is subtracted, and that error is largest for the keys with the largest logits, which dominate the output. The problem statement gives q = rmsnorm(q) x 5.473 and k = rmsnorm(k) over 64 dims, so |q| is about 43.8, |k| about 8, and a raw q.k can reach about 350. The f16 spacing in [256, 512) is 0.25. After the 1/8 score scale and the log2(e) factor that is about 0.045 in the exp2 exponent, a worst-case step of roughly 3% in P on the top keys (half of that as rounding error). That is the same size as E4M3 rounding of P, against a judge margin of 0.09 to 0.15 relative L2 (our CPU model sat at 0.07 to 0.09). Typical dot products are far smaller, which helps. PV has to keep FP32 accumulation across tiles, since f16 range and the rare-rescale offsets make cross-tile f16 accumulation unsafe, so FP16 accumulation can only be used on QK^T.

Gain model (upper bound). The score is a geometric mean over six loads, and the full-layer share of v16 and v32 is 76 and 81%, so the score moves about 0.26 times the fractional speedup of v16f and v32f. Closing the whole verify-versus-decode gap would give about 3.2%. A realistic outcome is a third of that, and zero if the gap is not issue-bound. The PV barrier and data-wait shares above argue that part of it is not issue-bound.

What happened. The SASS gate settled it. On our toolchain, ptxas turns ex2.approx.f16x2 into two MUFU.EX2.F16 plus a PRMT, and ex2.approx.ftz.bf16x2 into two MUFU.EX2.BF16 plus a PRMT. A pair costs 5 instructions (1 HFMA2, 2 MUFU, PRMT, F2FP) against 5 now (2 FFMA, 2 MUFU, F2FP). NVRTC 13.2's h2exp2 does not emit the packed op at all. A wgmma m64n32k32 with f16.e4m3.e4m3 does assemble. lee101 and another team's measurements agree that packed half exp lowers to two MUFU ops on this hardware. Verdict: no go, and the check was never spent. That is a cheap way to lose an idea, and we would price the next one the same way: assemble first, then decide whether it deserves the judge.

## Ideas as ranked before the SASS result, and what became of them

|Idea|Expected gain then|Outcome|
|-|-|-|
|Packed half2 softmax with FP16-accumulated QK^T on v16f and v32f|Upper bound 3.2%, realistic 0 to 1%|No go on SASS evidence, see above|
|nc = 4 workspace copies for d1s (relieves same-address red.v4.f32 serialisation, about 17 ns per op over 32 CTAs, up to 0.5 us of queueing)|At most 0.5% of the score|Tested, job 66be38f978b54004: 2.567x (2.529 to 2.606), 16 of 16 passed. d1s came out 5.53 us against 5.17 and 4.94 in two identical-code runs; controls shifted x1.0003 and x0.9998. No go, one copy stays|
|One 128-byte cp.async.bulk per K/V row instead of 8 x 16-byte cp.async|Under 0.3%|No go. d16f and d32f are already at 86 to 90% of the row-gather ceiling, a public note found it negative, and it needs fence.proxy.async with no way to soak without a GPU|
|128-key KV tiles|A few tenths of a percent, sign uncertain|Not pursued. TILE = 64 is baked into the wgmma n64 shapes, the V transpose and the index slots. A deeper pipeline measured worse elsewhere (see public-repos-survey.md)|
|Hierarchical tree merge for d1f and v1f using the (max, sum) associativity|Unknown|Closed with our other merge tricks. Single-CTA and cluster merges lost, and the finalise is 2.5 to 2.8 us of 11 to 14|

One more negative from our own checks: warming the merge code in an idle producer warp (cold-code warming) scored 2.576x against the 2.605x entry, with d1s at 5.1 against 4.94 us. A public repo measured a large gain from the same trick in a different kernel design, so the mechanism may be real and may simply not matter on ours.

## Open questions

- Nothing we read says whether the v16f and v32f consumers are issue-bound or latency-bound. That was the deciding unknown for the softmax idea.
- No source quantifies q_len 8 at head_dim 64 on any GPU, so the 10 to 14% verify gap has no paper-side reference.
- We never confirmed from a source whether TMA gather (several rows per instruction) exists on Hopper.

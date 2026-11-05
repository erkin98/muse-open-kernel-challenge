# lee101/openkernels: what the repo holds

The public repo at https://github.com/lee101/openkernels stopped at judge scores below ours, and this file records its numbers, how it was built, the measurements worth keeping, two compliance traps and what we could borrow.

Everything below was read directly in the repo's files unless marked inference. We cloned it, read it with grep and sed, and never built or ran any of it. No code is copied here.

## Short answer

Nothing in it clears judge noise on any board we entered. It is a from-scratch effort that stopped on October 28 at judge scores below ours. Our entries are built on vjeux's solutions, which already have ones-row sums, FP8 P and lazy rescale, so almost every attention technique in the repo is either in our file already or measured by the author as a loss.

Where it is useful:

1. Negative evidence. Packed half2 exp does not buy MUFU throughput on Hopper; wgmma-RS for small M does not pay; fusing residual glue into GEMV tails lost; a 12-bit RS-wgmma GEMM at m = 128 loses to cuBLAS by 1.6 to 1.7x.
2. One small untested MoE micro-idea, an 8-op E2M1 placement against our 9-op decode, with an expected gain of 0.3 to 1.0% on the MoE and 0.15 to 0.45% on the decode layer. Later outcome: the repo's method notes (`notes/method.md` at the repo root) record that an 8-op dequant variant regressed, 2.222x against 2.36x. We believe it is this idea, but the notes do not say which placement was used (inference).
3. Two traps for anyone borrowing from it: a speculative sequence-length guess in attention, and a BF16 router input in the decode layer.

## The repo

- Author lee101, the only committer (357 commits). Apache-2.0. It carries the task files for all five problems and its README says the kernel track closes at the same moment ours does.
- First commits October 23, last commit October 28 (commits per day: 8, 61, 44, 71, 97, 76). By November 4 it was 6.5 days stale and predates the November 3 rules update.
- 98 MB with history. About 107 attention snapshots (v10 to v85), 60 for the MoE, 35 for the decode layer, 30 for bf16 and 25 for prefill. Final working files: attention 348 KB, decode layer 736 KB, prefill 776 KB, MoE 224 KB, bf16 96 KB.
- It has its own judge-like local script, wrappers for rented GPU pods, a calibrated GPU model, and a 220 KB development log (`docs/ideas.md`). Unrelated notes on other projects in `docs/other_projects.md` are not relevant.
- It is not derived from vjeux, alygg77, keell0renz or chenwainuo: a grep over all text files for their names and for the vjeux repo finds nothing.

## Scores

Scores come from `okc check` judge runs. The README table is stale; later figures are in `docs/ideas.md` and the gap analyses.

|board|their best|where|our entry then|board leader on Nov 4|
|-|-|-|-|-|
|attention|about 1.99 (v81: 1.991, v83diag: 1.9917)|`docs/ideas.md`|2.605|3.339|
|MoE|1.9749 (v56d), parked|`docs/moe_gap_analysis.md`|2.329|2.782|
|decode layer|1.974 (v75)|`docs/ideas.md`|2.096|2.551|
|bf16|1.459 (v22), parked|`docs/bf16_gap_analysis.md`|1.491|1.711|
|prefill|1.4868 (pf25diag)|`docs/ideas.md`|not entered|n/a|

By our arithmetic their figures are 60 to 85% of the board leaders' and 2 to 24% below ours.

## How they worked

- A local RTX 3090 Ti for correctness only, with a software emulation of wgmma and cluster paths built on `mma.sync`. Short rented H100 pods (about $1.6 to $4 for 25 to 60 minutes) for timing. The judge for the headline number. No performance counters were available on the pods.
- One snapshot per idea, with a "diag" twin that prints timelines on the judge's H200: per-CTA globaltimer stamps and ablation switches (no exp, no convert, no loads, no MMA). One judge check yields many in-process A/B lines. That is the same trick as keell0renz's in-prepare A/B, and both need a GPU in the loop for each iteration.
- Prepare-time autotuning on the judge GPU with synthetic routings that mimic the judge's generator, with a 2% rule: keep the incumbent unless a candidate wins by more.
- One lesson worth copying: a prepare-time timer that flushed L2 with a write-heavy operation picked the wrong candidates, and they switched to a read-only flush.

## What they measured

### Attention decode

- Softmax denominators from an extra wgmma against a tile of ones: loop from 786 to 735 SASS instructions, verify 16 full 160.8 to 156.9 us, verify 32 full 302.2 to 293.7 us. Already in our file.
- FP8 P and FP8 V transposed for the PV matmul: verify 16 full 130.7 to 122.0 us, verify 32 full 240.0 to 227.4 us. Already in our file. Their producer-side V byte-transpose deadlocked on the H200 and was dropped.
- L2 128-byte prefetch hint on half-row copies helped on an H100 (350 to 311 us) and did nothing on the H200 (157.7 against 157.6 us).
- Balanced splits for full layers gained 1 to 2% on verify 16/32 full and hurt the sliding layer (prologue 4 to 6.7 us). In-request work stealing lost (190 against 163 us on an H100).
- Cluster merges over distributed shared memory lost five times. One batch-1 case won (decode 1 sliding 12.35 to 7.2 us, verify 1 sliding 9.4 us), but our entry is faster there (4.9 and 6.8 us).
- A partial merge inside the CTA gained a little (decode 16 sliding 15.17 to 14.53 us, verify 1 full 24.93 to 23.81 us) and uncovered three data races during development, one of them NaN from float bytes landing in a ring slot that was still read as e4m3 V.
- A ptxas scheduling trap: ptxas hoisted the load of the sequence length (a cold DRAM read) above the first CTA barrier, so every warp waited a DRAM round trip at the barrier (1.3 to 2.0 us in every verify shape). Their fix reads the length through a non-uniform zero from shared memory after the barrier. The judge did not confirm a gain (1.8282 against 1.8429, then 1.8399, which is noise).
- The packed half2 exponent is emulated in hardware with two MUFU.EX2.F16 ops plus a permute, so it gives no gain. Ablations on verify 16/32 full: no exp -7.5% and -6.1%, no V convert -11%, no KV loads -17.4%, no MMA -22%, everything removed -58%. They sum to the total, which says the consumer chain is serial rather than pipe-bound (their reading; inference). Exp on the FMA pipe for a quarter of the elements ran 4% slower, lazy rescale gained 1 to 2%, an integer-pipe V convert ran 7% slower.
- Other negatives: ptxas serializes wgmma when accumulators are zero-initialized or registers are short; unrolling the step loop 4x bloats code and serializes wgmma; starting QK(i+1) before softmax(i) loses (821 against 758 ns per step) because it needs more than the 112 to 120 consumer registers.
- Floors: random 128-byte row gather tops out at 3.85 to 3.92 TB/s on the H200, and one `cp.async.bulk` per 128-byte row reaches only 0.6 to 1.15 TB/s because of the TMA operation rate.

### BF16 projections and router

- A lossless 12-bit pack (a 3-bit offset of the exponent from a 16-value window plus escape lists, 99.993% coverage, 0.75 of the BF16 bytes), decoded in registers with 9 operations per 8 values, in a swap-AB `mma.sync` GEMV. Good to m = 16. Ours has the same family.
- Exponent entropy of the generator's weights is 2.54 bits, so the ideal lossless ratio is 0.659. Their 11-bit fixed code leaves 1.35% escapes.
- At m = 32 cuBLAS wins or ties (qkv 13.67, o 6.62, pooler 17.0, unpooler 16.6 us). At m = 128 three attempts at 12-bit wgmma-RS GEMMs lost: qkv 25.4 against 15.35 us, pooler 31.9 against 19.6, unpooler 32.0 against 18.7, o 10.5 against 6.65. A cuBLASLt search over 440 configurations kept the default for unpooler at m = 256. Both confirm items we had already closed.
- Router: for m >= 128, one cuBLAS GEMM on the stacked hi and lo weights plus an add took 11.4 against 17.7 us at m = 256, later 7.1 us (m = 128) and 9.2 us (m = 256). A dedicated kernel for m <= 32 takes 4.4 to 6.0 us.

### MoE (NVFP4)

Their final design picks a path per token bucket in prepare: a register-ring GEMV up to 16 tokens, a ring kernel at 32 and a wgmma-RS kernel with weights dequantized straight into A registers from 128 up, with a stream-bound floor analysis of 4.29 to 4.43 TB/s for 1D bulk copies of 16 KB or more (2D TMA boxes are about 8% lower).

- The 8-op E2M1 placement: one masked copy of the code word, two rotated-and-masked copies and one copy built with a masked multiply, for 8 integer operations per word of 8 weights instead of 11, checked exhaustively over the 256 leftover inputs and on 20k random words. Judge: v54d 1.8802 to v56d 1.9749 (+4.9% by their own attribution), with consumer instruction count falling from 22.9 to 19.9 per word. Nothing in 17 surveyed open kernels beats 11 operations on sm_90 (Marlin, SGLang's baseline, uses 15). Ours is a 9-op layout (7 ALU plus 2 shift-by-multiply), so the comparison is 9 against 8. Caveat: their old 11-op sequence also had 7 ALU-pipe ops, so their +4.9% probably came from removing FMA-pipe and issue-slot work (inference), and it rests on one judge run. Our decode comment says the ALU pipe is the saturated one, and the 8-op layout keeps 7 ALU ops. The gain would exist only if our consumer is issue- or FMA-bound.
- Lossless scale compression: the E4M3 block scales carry 3.71 bits of entropy on the generator, only 32 distinct codes (95 to 126), top-16 coverage 98.7%. A 5-bit code minus a per-expert base with an 8-bit fallback would save 4.2% of bytes, a 4-bit variant with escapes 5.5%. They never built it. alygg77 and keell0renz ship 6-bit scales, so 5-bit would add at most 1.4% fewer bytes (inference).
- Passing the ring slot base through a warp-wide max so ptxas builds all wgmma descriptors on the uniform datapath: 23.5 to 22.9 instructions per word, judge equal within noise.
- One wgmma group per 64-K chunk with two alternating A register sets and wait_group 1: 27.9 to 23.5 instructions per word, judge 1.8137 to 1.8740 (mixed with other changes). We could not tell from the source whether ours has the same per-word count (inference).
- Measured losses: a warp-specialized producer/consumer ring (1.188 against 1.339); 12 warps x 3 stages; L2 prefetch of the next phase's weights; a fused split-K kernel (wins only at M = 1: 52 against 55 us); last-arriver combine (M = 8: 96 against 77 us); more consumer warps or two CTAs per SM, which lower the SM clock under the power cap (1497 against about 1650 MHz); a 17th producer warp (M = 1: 38 to 39 against 35 us); wgmma-RS for small M (+2.5 to 3 instructions per word of descriptor traffic); dequantizing into shared memory on the producer side (about 7x the HBM bytes in shared traffic); folding activation scales (the scale is per output row and K block, so impossible).
- Their unproven hypothesis: the remaining gap is issue efficiency at 55 to 60% of capacity, not instruction count or bytes. Closing it needs per-instruction stall data, which they and we both lack.

### Decode-layer glue

- Their component budget (in-layer): glue is a fixed 19 to 35 us per load, 20% of decode 1. MoE dominates every load except the full-context shapes.
- Fusing residual and RMS rows into the last-arriver tails of GEMVs at T = 1 lost: decode 1 sliding/full went from 83.4/93.3 to 90.2/98.7 us (o projection 5.4 to 14.8 us because the tail runs in a 64-thread CTA). Folding the glue into the next GEMV's prologue was estimated under 3 us and not built.
- PDL: kernels with spinning merges wait at the top and do not trigger; cluster launches need `cudaLaunchKernelEx` with the programmatic attribute (the first version without it timed flat). keell0renz extends the cap to T = 128; theirs is T <= 8.
- They built the layer from standalone kernels, and many standalone wins vanished in the layer (router: their dedicated kernel 38.5 us against cuBLASLt 35.5 us at T = 128 on an H100).

## Compliance if you borrow from it

We checked it against the live rules page. No obfuscated source or embedded binaries, no lossy weights or KV, no skipped keys, no cached or replayed results (self-cleaning counters instead of memsets, as ours). Two concerns:

1. Tuning to exact scored sizes. The attention submission guesses the sequence length as the page-table capacity minus L, and a comment calls the guess "exact for the scored fixed-length" cases. A checked slow path handles a wrong guess. It is correct for any input, but the speedup exists only for the generator's structure, and the gates are batch 1, a window, page size 1 and L in {1, 8}. Their dispatch tables in the decode layer and MoE are per-bucket picks made with synthetic routings that copy the judge's generator. If anything is ported, dispatch on contiguous ranges.
2. BF16 router input in the decode layer: their glue builds `h` as a BF16 tensor and feeds the router from it at every T, including T <= 8 where SGLang keeps FP32. That is the same pattern as the vjeux file's (see `../decode-layer/fp32-router-input-rule.md`). Their task files predate the November 3 rules update, so whether it counts as non-compliant under the current rule is our reading (inference).

Also: prepare was 154 s in one attention/MoE version and extension builds took 91 to 112 s per file on a loaded machine. That is under the 300 s cap but not by the margin keell0renz keeps.

## Ideas ranked by expected gain

All of them are under the 2.5 to 3% bar other teams' reports suggest is needed to see a gain through judge noise.

|idea|status|estimated whole-score gain|
|-|-|-|
|8-op E2M1 placement against our 9-op|untested when read; later regressed on the MoE board (see the short answer)|MoE 0.3 to 1.0%, decode layer up to 0.45%|
|5-bit lossless scale code with a per-expert base|untested, same family as the 6-bit scales|MoE 0.3 to 0.7% at most|
|check SASS for a sequence-length load hoisted above the first barrier|untested, static check only|attention 0 to 0.5% (their judge did not confirm)|
|uniform-datapath wgmma descriptors|untested, noise on their judge|MoE 0 to 0.3%|
|one wgmma group per 64-K chunk with alternating A sets|probably already in ours (inference)|MoE 0 to 0.5%|
|one-pass FP32 router at m = 128/256 on stacked hi/lo|untested; bf16 board only|bf16 0.1 to 0.3%|
|packed half2 softmax exp|refuted: still two MUFU ops|ceiling about 1%, realistic 0 to 0.3%|
|wgmma-RS for small M, last-arriver combines, fused split-K|their measured losses|0|
|glue fused into GEMV epilogues or prologues|their measured loss (decode 1: 83.4 to 90.2 us)|0|
|cluster merges, balanced splits, work stealing, producer V transpose, FMA exp, L2 prefetch|closed or lost on the H200|0|
|12-bit RS-wgmma GEMM at m = 128; cuBLASLt pins|re-confirmed losses, 1.6 to 1.7x slower than cuBLAS|0|
|ones-row sums, FP8 P / V^T, lazy rescale, 128-byte row rings|already in ours|0|
|speculative sequence guess, exact-gate dispatch, synthetic-routing tuning|compliance risk|do not import|

The MoE is the only board with a transferable idea, and it is worth about one instruction per word.

## Could not verify

- Whether the MoE judge uses the same scale histogram as the repo's generator (only 32 distinct scale codes).
- Our per-word instruction count in the wgmma MoE loop (it needs SASS of the built module).
- Whether an 8-op layout can also keep two shifts on the FMA pipe (6 ALU plus 2 shift-by-multiply). We did not construct one.
- Their final decode-layer figure beyond v75 (v34 and v35 are unjudged in the log) and their final attention figure after v83.
- The git history was not read commit by commit; results come from `docs/ideas.md`, not from commit messages.
- Router-input precision at T <= 8 in their composed layer, beyond the dtype of `h` in their glue file.

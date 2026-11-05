# How the speedups happened

Where each of our board scores came from, what we wrote ourselves, what we borrowed, and what we tried afterward that did not move a score.

## The short version

At midday on Nov 4, every score on our four boards came from vjeux's public Apache-2.0 entry (his okc-2026 repository at commit f35a41e), submitted with a 15-line attribution header and nothing else that changes speed. The speed was his. Later that day and on Nov 5 we submitted modified versions for three problems: attention (a flat 8-token path and BF16 vector reductions, now 2.621×), moe (K-reordered weights with 6-bit block scales, now 2.362×) and decode_layer (an FP32 router input up to 8 tokens plus the same moe changes, now 2.083×). The gains over his entries are small: about +0.6% on attention and +1.4% on moe, and decode_layer ended slightly lower because the FP32 router rule costs a little. Our own code, written without reading his, reached 1.150× as its best bf16 board entry and 1.258× on its best judge check. That is well short of the borrowed 1.491× on the same problem.

The scores below are the state around midday on Nov 4. Later submissions changed some of these numbers, and this document stops at that point. The decode_layer entry 9b0d0973f11a4758 was withdrawn afterwards because its router input was not FP32 up to 8 tokens. For how to read a score line and its interval, see `measuring-on-the-judge.md`.

|Problem|Board entry|Score (95% interval)|Rank|Leader|
|-|-|-|-|-|
|muse_bf16_proj|6c7a0061944e4339|1.491× (1.469–1.514), 9/9|#355 of 1,699 (#341 earlier that day)|1.694×|
|muse_attention_decode|c20df44ab71d4005|2.605× (2.565–2.644), 16/16|#304 of 1,331|3.339×|
|muse_moe_w4a16|41f596feaa60434e|2.329× (2.294–2.364), 5/5|#397 of 1,387|2.764×|
|muse_decode_layer|9b0d0973f11a4758|2.096× (2.065–2.127), 13/13|#449 of 1,255|2.540×|

Overall: #449 of 1,957 with all 4 problems solved. It was #1,201 with 3 solved and #1,451 with 1. The overall formula is not published; the API returns ranks only.

Our own code on muse_bf16_proj: the best board entry is v3 at 1.150× (job a5ac82b328cd4734), and the best check is clean-room CUDA at 1.252× (e80520a1edb0468b), which is off the board. Neither beats the borrowed entry.

## Our route on muse_bf16_proj: Triton, then CUDA

The problem is `out = x @ W.T` for five projections on one H200, at m = 1, 16, 32 decode tokens and m = 8, 128, 256 verify tokens. Four of the layers are BF16 and the router is FP32. For m up to 32 this is a memory-bandwidth problem. The 17 to 67 MB of weights have to stream from HBM and the arithmetic is tiny. At m = 256 it starts to be compute-bound.

### Triton checks

|Step|Job|Score|What changed|What we learned|
|-|-|-|-|-|
|Check 1|76c338ff8cc64f7b|0.908×|unmodified starter file|the starter is slower than SGLang|
|Check 2|a74a6c2953e84703|1.105× (1.089–1.122)|Triton router (BF16 hi+lo split, split-K); other layers on cuBLAS|SGLang's router is the weakest baseline kernel: 10 to 26 µs for a 256-row matrix|
|Check 3 (v3)|a58c6e3b903744ea|1.139× (1.122–1.156)|tuned router configs; Triton BF16 kernels at m ≤ 8|small m beats cuBLAS by 1.2 to 1.3× per shape|
|v3 on the board|a5ac82b328cd4734|1.150× (1.133–1.168), #1,409|same file|first board entry|
|Check 4 (v4)|e96f52235171459f|1.112×|Triton at m = 16 and 32 as well|plain Triton loses to cuBLAS at mid m (qkv_gate 0.76×, pooler 0.93×); reverted|

Three things capped Triton.

First, bandwidth. v3's pooler kernel keeps about 40 KB of weights in flight per CTA and reaches about 3.3 TB/s. cuBLAS's mid-m kernels keep about 128 KB in flight (inferred from the cuBLAS kernel names, not verified) and reach about 3.6 TB/s. That second figure is measured: the unpooler at m = 16 moved 67.1 MB in 18.5 µs. The H200's floor is 4.8 TB/s.

Second, launch count. Split-K needs a second reduce launch, which costs 1 to 2 µs per shape.

Third, the design itself. An offline compile for sm_90 showed v3 using 16-byte `LDGSTS` loads and `HMMA.16816` tensor-core instructions, with no TMA, no wgmma and no spills. The code generation was fine. The design was the ceiling.

We also built a small model that rebuilds the judge's score from per-shape times. It said:

|Change|Model score|
|-|-|
|nothing (v3 as it stood)|1.151×|
|one launch per shape|1.201×|
|4.0 TB/s and 2 µs overhead|1.357×|
|4.3 TB/s and 1 µs overhead|1.567×|

So the lever was moving weight bytes faster, or moving fewer of them, and cutting fixed costs. More arithmetic would not have helped.

### What public write-ups showed

A research pass over the board and public write-ups found a crowded band: 436 entries between 1.50× and 1.60×, most of their best runs dated after vjeux published a complete Apache-2.0 entry on Oct 27 and 28. The other public entries at or above 1.5× are alygg77's (built on vjeux's) and NathenFernandes's (MIT licensed). The public READMEs describe the same four-part stack:

1. A CUDA GEMV compiled with `load_inline`, with the weights repacked in `prepare()` so each lane streams 16 bytes straight from HBM.
2. Lossless 12-bit BF16 weights: the low byte kept raw, the sign and exponent mapped through a small window, and out-of-window weights patched exactly from an exception list. The rules allow it, and it reads 25% fewer bytes.
3. The router as one fused kernel.
4. Selective cuBLASLt or wgmma work at m = 128 and 256.

To see which part mattered, we applied nathenfernandes's published per-shape speedups to our own SGLang timings, one layer at a time:

|Layers adopted|Score|
|-|-|
|pooler alone|1.273×|
|plus qkv_gate|1.357×|
|plus router|1.420×|
|plus unpooler|1.486×|
|plus o|1.520×|

The pooler is 67 MB and appears in every load, so it is the biggest single lever. We also measured the weights' entropy at 10.46 bits per value. A 12-bit code is about 1.5 bits above that, so the headroom is real.

### Clean-room CUDA

Our own CUDA route times a set of candidate kernels per shape in `prepare()` and keeps a CUDA kernel only where it beats v3 by more than 3%. We worked in order: plan, tests first, self-review, then a judge check.

|Check|Job|Score|What it added|
|-|-|-|-|
|1|0b9bb452effa4cd4|1.206× (1.188–1.224)|28 CUDA GEMV kernels; wins on every BF16 shape at m ≤ 8 (1.16 to 1.31×)|
|2|e80520a1edb0468b|1.252× (1.234–1.271)|our own 12-bit pooler (decode m=1: 21.5 to 15.0 µs, 1.43×); 12-bit qkv and pooler at m=16; CUDA router at m ≤ 8 (1.94×, and 5.09× at verify m=8)|
|3|827732e7c9474ab9|1.250×|x staged per group in shared memory at m = 16 and 32, plus 12-bit twins; staging was about 30% slower, so dropped|
|4|6f09dc063d2f4745|1.258× (1.239–1.276)|best clean-room check|
|5|51ea12fc37e14f27|1.243× (1.224–1.262)|same kernels as check 4 with dead code removed; the gap is noise|

The router at decode m=1 went from 10.4 to 5.4 µs, which is the 1.94×. That is one cell, not a problem score.

Where the CUDA kernels beat v3 in the end: all BF16 layers at m=1 and m=8, the pooler and qkv at m=16, and the router at every m ≤ 32. The 12-bit format helped only the pooler at m ≤ 16 and qkv at m=16. Still on cuBLAS: `o` and the unpooler at 16 and 32, qkv at 32, and everything at 128 and 256. The final file is check 5's. We did not submit it, because it is below the 1.491× we already had and the board keeps only the best run.

Per load at clean-room check 2:

|Load|Score|
|-|-|
|decode m=1|1.419|
|decode m=16|1.215|
|decode m=32|1.104|
|verify m=8|1.740|
|verify m=128|1.098|
|verify m=256|1.062|

The failure we could not fix: at m = 16 and 32, staging x before the weight stream stalled it, 10 to 70% slower, so mid-m stayed on cuBLAS.

## The borrowed route

For all four boards we took vjeux's public entry. What we did:

- Found the public base and got a byte-exact copy: a git clone at f35a41e, with sha256 and git-blob hashes matching the GitHub pages.
- Added a 15-line Apache-2.0 attribution header.
- Checked each file against the rules: graph-safe `kernel_fn`, no compressed source, the precision rules, and that attention's FP8 probabilities match SGLang's FA3 FP8 path.
- Ran one `okc check` per problem and then submitted.

Each file body is byte-identical to upstream. In the bf16 entry our one added option, a route to the v3 router, is a speed no-op because the judge never picks it.

The judge's own profile for the bf16 entry shows a few things about the file:

- The pooler at decode m=1 takes 13.68 µs. The floor for reading the 16-bit weights at 4.8 TB/s is 13.99 µs, so this beats the floor, which is only possible by reading fewer bytes. That fits the 12-bit format.
- At m=1 each shape is one custom GEMV kernel (qkv_gate: 6.7 µs of GPU time).
- The router is 2.4 to 5.0× faster than SGLang at every m.
- The weak spots are m = 128 and 256, at roughly cuBLAS speed (0.99 to 1.13×).

Per load, the borrowed bf16 entry scored:

|Load|Score|
|-|-|
|decode m=1|1.73|
|decode m=16|1.56|
|decode m=32|1.49|
|verify m=8|1.92|
|verify m=128|1.23|
|verify m=256|1.17|

The two large-m loads hold the problem score down. If both reached 1.45 the score would be about 1.59×.

The other three problems came from the same base with the same header. Attention scored 2.605×. It is fastest on sliding-window decode (decode 1 sliding at 4.25×) and weakest on full-context verify (2.02 to 2.11×). Moe scored 2.329× on the board (on its check: decode 1 at 3.13×, decode 32 at 1.62×, verify 32 at 2.82×). Decode_layer scored 2.096× on the board, from 1.67× to 2.38× per case, with decode 32 sliding the weakest at 1.668×. Adding the fourth problem moved our overall rank from #1,201 to #449.

## Keeping the two routes separate

The clean-room kernels were written without reading the public kernel source. What we know about the public stack comes from READMEs, nathenfernandes's published per-shape CSV, thethings' write-ups and the judge's profile tables for our own runs. We kept the two routes apart on purpose, so that the 1.258× is a statement about what we could build alone. The line-by-line review of the borrowed kernels in `replay-race-hazards.md` came afterwards, once the clean-room work had ended.

## Trying to beat the borrowed entries

After the four entries went in, we tried to beat each by more than the judge's noise. As of these notes, none of it cleared the judge's noise.

### bf16 at large m

The weak loads are m = 128 and 256, so that is where we worked. Check 1 (cuBLASLt with pinned configs) scored 1.497×. Check 2 (a split-K wgmma GEMM, plus cuBLAS on a K-major copy) scored 1.484×. Every candidate lost to the picks already in the file. A third check with more instances of the existing tiles (job 1dd4d6bbe0214964) was still queued when these notes were written, and we have no result for it. Also ruled out in our notes: Huffman and DFloat11-style coding, which is slower at small batch.

### Attention

Check 1 combined a linear-mantissa exp2 with deeper pipelines for q_len 1 and scored 2.550× (2.512–2.588), 16/16, which is worse. The deeper pipelines slowed full-context decode by 3 to 4 µs and the exp2 change gave nothing. Check 2 (job a58eeeeaf93c4bed) tried cold-code warming: idle warps run the final merge once with every load and store switched off, so the real pass runs from a warm instruction cache. Four batch-16 and batch-32 full-context cases ran byte-identical machine code as controls. Result: 2.576× (2.537–2.614), 16/16, no gain. The batch-1 sliding cases did not drop (decode 1 sliding at 5.1 µs, verify 1 sliding at 6.8 µs, no lower than before) and the controls stayed within ±2 µs. We dropped it.

### Moe

We built candidates from our own formulas, with ideas credited and nothing copied: a K-permutation of the NVFP4 codes (alygg77's idea), 6-bit lossless block scales (keell0renz's idea), and a warp-parallel stream-K prologue. A prepare-time A/B keeps a candidate only on a win of 1% or more against an A/A control, capped at 200 s so prepare stays inside its budget. All static gates passed. The first check (job 2eaae7ab93824649) was queued and we have no result for it in these notes. Two other moe candidates are covered in `lessons-learned.md`: one with a cheaper decode regressed, and one never loaded.

### Decode_layer

Check 1 failed 10 of 13 cases. The new 1-token GEMV's loads had been hoisted above the point where it waits for the previous kernel, so it read an input before it was written. We dropped it. Separately, we found that the board entry feeds the router a BF16-rounded activation at 8 tokens and below, where SGLang keeps FP32. A compliant v3 (an unrounded FP32 copy for the router, falling back to a separate FP32 GEMV, never BF16) was written, and at the time of these notes it had only been checked for syntax.

### What other teams' write-ups added

Passed along as ideas only:

- One write-up says the judge grades the decode_layer increment, computed as stream minus carried. We did not pin down what those two terms mean in the harness.
- One team lost judge runs because prepare plus build took 437 s.
- Replay-only bugs (float atomics into a buffer that zeroes itself, missing proxy fences) fail about once in 10,000 to 20,000 replays. See `replay-race-hazards.md`.
- A local A/B is a filter and nothing more.
- Emulating exp2 on the FMA pipe in attention is slower, by 24 to 105 µs.

## What the numbers said was left

- bf16: the gap to the board's roughly 1.55× cluster and the 1.69× leader sits at m = 128 and 256 (1.17 to 1.23 per load). Our clean-room CUDA kernels are far behind the borrowed file except at m ≤ 8 and on the router.
- Gaps to first place: attention 2.605× against 3.339× (a factor of 1.28), moe 2.329× against 2.764× (1.19), decode_layer 2.096× against 2.540× (1.21).
- The overall rank combines the four problems, so a better attention or moe entry moves it more than another bf16 percent.

`what-matters-for-the-score.md` has the per-part sensitivities behind these targets.

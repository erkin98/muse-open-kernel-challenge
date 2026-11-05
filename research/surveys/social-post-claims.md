# Claims other teams posted on X, traced to what is public

This file traces the performance claims and technique outlines that other teams posted on X, with what we could verify, what we could not, and what each means for our four boards.

Everything below is third-party text, read from public pages. We did not click, post, sign up for anything or run anything from these posts, and we reproduce no code. The pattern is plain: almost every claim names an account but few have any public mechanism. Only two posts have technique detail (the bf16 1.569x list and the router GEMV post), and only one has an official-check screenshot (the router post).

## Summary

|Account and date|Claim|Public detail|Our assessment|
|-|-|-|-|
|@anshulcreates, Oct 24|Router GEMV in Triton: m = 1 from 10.3 to 5.1 us, m = 8 from 26.3 to 11.9 us. Total 1.055x|A technique description and an official-check screenshot. No repo|Medium to high reliability. Our router is already 3.84 and 5.19 us|
|@AdiverseK, Oct 31|bf16 1.569x|A five-point outline. No repo, no screenshot|Plausible outline. Number equals keell0renz's submitted 1.5690 (coincidence or reuse, our inference). The gap decomposes into the rs128 design|
|@NivasT49781, Oct 30|bf16 1.595x with a 12-bit pack, MoE 2.382x with a MULDEC graft|Three lines, no explanation|Low. MULDEC has no public mechanism|
|@karthikeyansuku, Oct 29 to Nov 2|MoE 2.017x, later 2.211x. Better NVFP4 dequantisation|No code or mechanism|Below our 2.329x. No reason to chase|
|@utterwqlut, Nov 2|A first GPU kernel reaching 1.3x|Three ideas; one reply with a hint|Sits below our entries|
|RohanArun (@RohanArun), Oct 24 to Nov 3|Top 2 in 7 hours; about +5% from hyper-parameter search; a statistical model plus a Lean-based pruner|None. No code, no mechanism|Low. Not traceable to a shape, board or mechanism|
|An unattributed note on isolated wins|cgemv gains that died in the full decode layer|No public text found|Used as a rule|

## @anshulcreates's router GEMV post (Oct 24, quoted again Oct 27)

The text, plus an attached official-check screenshot: total 1.055x over SGLang, 95% interval 1.039 to 1.071, 4,580 us, 9 of 9 passed (https://x.com/anshulcreates/status/2114040355196686371). Router rows: decode 1 (m = 1) 10.3 us stock against 5.1 us his (2.03x); verify 1 (m = 8) 26.3 against 11.9 us (2.22x). Decode 16 and decode 32 show 1.09x each and verify 16 shows 1.07x, while verify 32 shows 0.93x (17.4 against 18.7 us). All other projection rows are within about 2% of 1.0x. His own forecast was 1.10x to 1.15x.

His description: a Triton kernel with 256 programs, one per output, weights streamed once in contiguous chunks, reuse across the 8 tokens, software prefetch, a preallocated output, and the old path for m greater than 8. A bound estimate puts the router at 1.8 us. A later thread of his (https://x.com/anshulcreates/status/2115084231432966638) states that only m = 1 and m = 8 beat the stock call and m = 16 did not, and that anything above 8 tokens uses the old path. So only the m = 1 and m = 8 router rows in the screenshot reflect his kernel. The rows above 8 tokens are the stock call and show roughly plus or minus 9% run-to-run variance, cluster offset, or effects of his other changes (fusion, a DeepGEMM swap). The two posts do not conflict. The same thread links two non-kernel papers (arXiv 2605.10481 and 2608.14380). An Oct 23 post of his says it was his first GPU kernel.

For us: the router is about 0.10 of the bf16 pull, and the total he reached is only 1.055x, which agrees. Our entry already has a small-m router path and a fused pooler-router. His Triton router lost our A/B at every m. The headroom from 3.84 and 5.19 us to his 1.8 us bound is untested.

## @AdiverseK's bf16 1.569x (Oct 31)

The post, as a five-point list (https://x.com/AdiverseK/status/2116519029494149293): bf16 is memory-bound at batch 1 to 32; weights are stored losslessly in about 12 bits (exponents cluster) and decoded in the kernel for about 25% less traffic; the base is open-source kernels from alygg77, vjeux and nathenfernandes; for batches 128 to 256 there is his own TMA and wgmma kernel plus pinned cuBLASLt configs plus a special FP32 router path; and an auto-tuner times every option per layer and batch size under CUDA graphs. No replies, no repo, no link.

Public sources for the parts he credits: nathenfernandes's MIT bf16 repo (a team's run of its v18 passed at 1.532x in a check and scored 1.521x, per forum thread 5); pinned cuBLASLt algorithms with rebuild, verify and fall-back self-tests, described in prose in the bigboateng README and quoted in forum thread 11; forum thread 125 mentions cuBLASLt algo-66 pins for prefill and the o-projection row-major layout as known levers on a 1.55-class file.

Arithmetic. The weighted SGLang baseline for bf16 follows from RohanArun's numbers as about 4,830 us, which matches another team's official record. So our 1.491x is about 3,240 us, 1.569x is about 3,079 us, a gap of about 160 us, and 1.613x is about 2,995 us. His gains over a vjeux-based file must come from his points 4 and 5, which nobody has published.

Rule reading. An auto-tuner that times every option per exact (layer, batch) reads close to the live ban on tuning that only works at the scored sizes, if the pick is a lookup on exact keys. His claim is not evidence that it is allowed. In our hands it was also closed in practice: a re-tuned bf16 dispatcher predicted +6.6% locally and returned -5% on the judge (forum thread 147).

## @NivasT49781: 12-bit pack and MULDEC (Oct 30)

Three lines and a tag (https://x.com/NivasT49781/status/2116231340161188087), with no explanation of either term. Searches on X find MULDEC only in this post, a web search finds nothing, and the forum files we saved contain no match for MULDEC, 1.569, 1.595, 2.382 or 2.017. A 12-bit pack is a lossless exponent-clustering scheme, which the live rules explicitly allow, and the idea appears in three of these posts (@AdiverseK, @NivasT49781, @utterwqlut), which suggests it came from a common public source. The vjeux file we build on already has it. MULDEC could mean a multiply-based decoder, but nothing confirms that, so we treat it as unknown. If real and transferable, 2.382x over our 2.329x is about 2.3%. Our moe entry already decodes FP4 by field placement and the E4M3 scales by a byte permute plus an integer multiply.

## @karthikeyansuku: MoE 2.017x to 2.211x

The Nov 2 post (https://x.com/karthikeyansuku/status/2117243370808508918) says better NVFP4 dequantisation, fused kernels and scheduling tweaks, validated on a rented Nebius H200, still a long way from the top. A later post (https://x.com/karthikeyansuku/status/2117772776161558714) gave 2.211x on the MoE board with no detail. Earlier posts: 1.82x to 1.92x on Oct 30, and an Oct 29 post (rank 613 to 438, 1.4x to 1.8x, a dequantisation path, with a write-up promised after the challenge). These are all below our entry.

## @utterwqlut and the hand-written question

@utterwqlut's Nov 2 post (https://x.com/utterwqlut/status/2117234769746657540) listed three ideas for a first GPU kernel at 1.3x. The only reply with content, from @bitcoinduke, says he hit the same wall around 1.33x on the projections, that about 1.6x comes from hand-written per-shape CUDA kernels because split-K tuning still pays for an extra reduce launch, and that judged times per shape should be compared. A separate thread by @jino_rohit on Oct 26 (https://x.com/jino_rohit/status/2114693263299170774) asked whether anyone was hand-writing kernels. Replies reported about 1.204x and about 1.3x hand-written, with no technique named. By those numbers hand-written work stalled around 1.2x to 1.3x on a board they did not name, far below our entries.

## RohanArun

- Oct 24: a top-2 claim in 7 hours with no board or score named, unpublished CUDA tricks collected since 2009, and a trick that spreads a convolution over many GPU cycles to get a fuzzy version of the data with a sliding compression level (https://x.com/RohanArun/status/2113794964953948336). No mechanism, code or number followed. If it means lossy weight or KV compression it is banned.
- Oct 27 (https://x.com/RohanArun/status/2114830856393027974): his own numbers were bf16 1.443x to 1.453x (3,346.95 to 3,324.00 us), MoE 2.013x to 2.033x, decode layer 1.736x to 1.785x and attention 1.788x to 1.801x. The only public description of his queue and lock system is a paragraph here: a hook-based setup so workers are never idle and constant full GPU use. The forum title lists overall #62 on Oct 27 while the X post the same day says #172, which we could not resolve.
- Nov 2 and 3 (https://x.com/RohanArun/status/2117256734687326243, https://x.com/RohanArun/status/2117446433716478226): 2,000 experiments, a hyper-parameter search worth about +5%, a genetic-algorithm dashboard, a small statistical model trained on thousands of benchmark runs to short-list candidates, and a Lean-based pruner for combinations ruled out by memory and physics bounds. He reached #16 on bf16 on Nov 3. Nothing is released, and he said the surrogate tool will be open-sourced later. A 5% gain on one of six loads would move the score about 0.8%, which is one judge run of noise, and the method needs thousands of GPU runs that we do not have. Its one transferable part is bound-based pruning (compute bytes over the 4.8 TB/s floor per shape to prune the search), which we do in cheaper form.
- Forum record for his team on Nov 4: bf16 1.613 (#26), attention 2.679 (#123), moe 2.422 (#116), decode layer 2.322 (#64), prefill 1.952 (#33), overall #64. These are above ours on all four of our problems. The step from 1.453x on Oct 27 to 1.613x on Nov 4 on bf16 is larger than the extra 5% his search posts claim, and the rest most likely came from adopting public kernels (inference). In forum thread 40 his side offered only design ideas for attention (fold the GQA query heads into the wgmma M dimension, FP8 operand layouts, split-KV, no gather into a contiguous copy) and said there was no public write-up and no measurements.

### Starter-guide changes in his repository (pull request 6)

The repo is a native macOS app. Its README says the five MUSE problems are built-in projects, that the guides contain no private tuned kernels, and that there is no score guarantee. Pull request 6 (merged, one commit, +35 lines) only adds four numbered guidelines to each of five starter-guide files. We read the diff and judged each item against the live rules. The verdicts below are our own judgment, not sourced facts.

|Guideline|Verdict|
|-|-|
|Attention: pre-allocate output and scratch in prepare()|Inapplicable as worded. The live rules say the attention problem has no weights and never calls prepare(), so scratch would have to be created lazily from the call's own shapes, with no memset inside kernel_fn and nothing cached from input data|
|Attention: fuse RoPE, QK-norm and FP8 quantisation into one pre-pass before the paged KV write|Legal if the cache stays E4M3 and rounds like SGLang's. A public MIT repo (bigboateng) describes a fused QK-norm, RoPE, cache store, split-K and exact LSE merge in one kernel, so it is likely already covered|
|Attention: align loads to 64 or 128 byte segments|Harmless generic advice. Forum thread 40 adds the only number: about 4.0 TB/s with 8 lanes by 16 B per 128 B row against about 0.8 TB/s with one row per thread, quoted from another team|
|Attention: replace branches in the causal loop by predicated selects|Legal, risky for speed. On sliding layers it reads out-of-window blocks that could be skipped. Skipping keys outside the window is allowed, skipping keys that must be attended is not|
|bf16: pre-allocate in prepare()|Legal, but derive buffers from the state argument. Never bake raw addresses or captured graphs, because the judge clones the prepared state 2 to 4 times|
|bf16: pre-transpose weights so loads are contiguous 128-byte lines|Legal (lossless repack). The repo's own starter text before the PR says a layout-only repack showed no gain and the router dominated|
|bf16: specialise m = 1 into a streaming GEMV|Legal and sound; vjeux's file already has a small-m path|
|bf16: unroll statically and drop boundary masks|Risky against the live rule on tuning that works only at exact scored sizes. Keep a masked fallback and test sizes just below and above each boundary|
|MoE: pre-allocate routing tables|Legal. Do not grow prepare memory: +3 MB cost 0.5 to 1.3 us on unchanged shapes. A self-zeroing float red.add workspace failed about 1 in 10 to 20k replays|
|MoE: replace if/else on token ids and expert counts by predication|Wrong for MoE. Predicating away the skip of empty experts adds work|
|MoE: decode NVFP4 words in interleaved stages to cut register pressure|Plausible and rule-clean, not measured anywhere public. bigboateng's README says NVFP4 is already decoded in registers via inline PTX|
|MoE: fuse SwiGLU in accumulator registers|Standard, legal, almost certainly in every top entry|
|decode_layer: reuse one contiguous residual workspace|Needs care. The layer boards grade the increment (stream minus carried), so an in-place overwrite that aliases the carried input would corrupt that check|
|decode_layer: overlap the MoE router projection and the attention output projection on side streams|Wrong. The router consumes the post-attention hidden state, so it depends on the output projection. Side streams are also on our closed list, and VkaZas measured no kernel overlap at T = 1|
|decode_layer: fuse residual add and RMSNorm into the preceding projection's epilogue|Legal. Leading public lineages already fuse the o projection with the boundary, and another team found that breaking that fusion cost their gain|
|decode_layer: keep routing on device with no .item() or sync|Correct and required. The guidelines never mention the FP32 router input at T <= 8, so a reader following them could round it to BF16|

Overall the guidelines are generic and our entries are past them, with two exceptions to avoid: the stream overlap and the exact-size unroll.

### The hosted GPU queue

The getsupers experiments page (https://app.getsupers.com/experiments) is a hosted GPU cluster queue and marketplace. The guest view showed zero credits, a gate saying one free experiment per registration IP, a promo to clone the starter repo, and a one-line install that pipes a remote Python script into the interpreter to submit a kernel file. At our read it showed one active runner labelled for the dense bf16 problem, one running job (428 s) with 0% GPU load and 3.1 GB of VRAM, five pending jobs, job ids that begin with a small priority number, and four of five recent completed runs ending with return code 124 (a timeout code). Platform counters read 444 users and 3,162 jobs. That is one snapshot, so it neither proves nor disproves a constant-utilisation claim. The free quota is stated two ways, one run per registration IP on the page and five free runs in the README. We did not use it, because it uploads kernel source to a third party and runs remote code locally. His other repository (rohanarun/virtual-human) is an unrelated multi-scale human-body simulation.

## A note on isolated wins that died in the full layer

A note circulated saying cgemv improved the QKV and output projections but only applied to 2 of 12 workloads, with decode-layer scores of 2.1898 to 2.2007, 2.1704, 2.1859, one GEMM at 2.219 and 2.1981 after stacking. We could not find its text anywhere public. Two candidate origins by inference: VkaZas's team (alygg77 lineage), whose forum post lists a T = 1 unpooler and a T = 1 QKV as cgemv and a pinned tuner pick, with a decode_layer board speedup of 2.218; or the keell0renz team, whose decode climb from 2.1815x to 2.2664x brackets every number in the note and who had a lane porting standalone GEMMs. Against the second, their README and LESSONS never mention cgemv, and their bf16 file only has a cgemv name for an m = 1 12-bit GEMV. We cannot choose. The note's lessons match what we measured: our own cgemv check failed its three one-token tests (10 of 13 passed), standalone wins do not carry into the layer, an autotuner's gains may be noise, and stacking two proven wins can fail.

# Survey of public solution repos

This file records what we found in other teams' public repositories for the four muse boards: how we found them, where they sat on the leaderboards, and one section per repo with its claimed score, technique, licence and what we took from it as an idea.

We used these repos for ideas only. We did not copy or port any code, and nothing below reproduces any. Some repos were read through their READMEs, notes and file trees and never opened at the kernel level, and each section says which. Where a statement is verified in the cited source we say so, and where it is our inference we say that.

## How we looked

- Every submission file starts with a header line of the form `# okc: problem=<name>`. GitHub code search for that exact string, per problem, returned 120 files for muse_decode_layer, 110 for muse_attention_decode, 124 for muse_moe_w4a16 and 114 for muse_bf16_proj at the first pass. A loose query for `okc: problem=muse` reported 842 files on a later pass (GitHub's count varies; we saw 1.8k earlier). GitHub serves at most 5 pages per query, and page 6 returns nothing.
- GitHub does not index files over 384 KB, so the biggest entries never appear. The vjeux decode_layer file is 568 KB.
- The loose query's pages show 19 owners: lee101/openkernels, rohanarun, and two forks of it (xrey167, aviyanp), Madhavan113, lilaclilac09, PostTrain, ozgureyilmaz, Kushagrabainsla, 4gatepylon, tommyyliu, Ranoobaba, keell0renz, xharshvardhan, msaroufim, Mithun-033, vjeux, ruwwww and alygg77. Excluding all 19 with `-user:` filters leaves one file, a blog post (hieptran1812/my-website) that matches the header string in prose. Some owners (trxphxle, NathenFernandes, bemlerlabs, AutoclickerI) appear only under the per-problem exact queries, so a per-problem pass is the way to see them.
- Repository searches: `Open Kernel Challenge` gave 36 results over 4 pages (page 4 is unrelated noise). `Muse Spark Open Kernel` gave 19 results, later 18. Variants (`openkernelchallenge`, `muse spark kernel`, `okc muse`) added a few small repos.
- No top-20 team on any board owns a repo that carries an `okc:` header file. pjmullin3, tjiang-fair and jt914 have no public repo on the topic, LeshengJin has only forks of sglang and flashinfer, and olety has nothing on it. So what the leading teams do cannot be read from GitHub. Our inference is that they keep their source private. One bemlerlabs note records that on Oct 28 pjmullin3 led attention, decode and moe with a single kernel set, and that the public job-profile API refuses other users' jobs, so those kernels were never visible even to a team with a large pipeline.

## Where the public repos sat on the boards

Boards read on the morning of Nov 4. Cells are rank and speedup over SGLang.

|Owner|bf16|attention|moe|decode_layer|
|-|-|-|-|-|
|rohanarun|#25 1.613|#123 2.679|#116 2.422|#61 2.322|
|keell0renz|#76 1.590|#74 2.717|#119 2.420|#140 2.271|
|aviyanp (fork of rohanarun)|#163 1.572|#206 2.634|#210 2.391|#228 2.250|
|alygg77|#307 1.511|#162 2.653|#166 2.401|#350 2.206|
|ruwwww|#336 1.500|#314 2.603|#382 2.337|#466 2.075|
|Madhavan113|#351 1.493|#343 2.588|#338 2.357|#362 2.200|
|vjeux (our base)|#327 1.501|#282 2.610|#403 2.327|#444 2.102|
|trxphxle|#417 1.442|#372 2.567|#409 2.325|#383 2.176|
|erkin98 (us)|#355 1.491|#304 2.604|#397 2.329|#449 2.096|

lee101, tommyyliu, Ranoobaba, Kushagrabainsla, xharshvardhan, lilaclilac09, ozgureyilmaz and Mithun-033 sat lower.

The leading boards at the last read, from the challenge home page: bf16 1.73x (gogowhy), attention 3.34x (pjmullin3), moe 2.79x (pjmullin3), decode_layer 2.58x (gogowhy), prefill 2.15x (LeshengJin). bf16's top ten were all 1.67x to 1.73x.

## The base: vjeux/okc-2026

Apache-2.0. Head commit f35a41e, last commit Oct 28, one issue (a thank-you note from keell0renz). Our four entries are this base plus later changes, so most public repos near us are the same code with edits. Verified: several repos ship the base byte for byte. Rows: attention #282 2.610x, moe #403 2.327x, bf16 #327 1.501x, decode #444 2.102x.

## alygg77/okc-muse-spark-kernels

Apache-2.0, with nathenfernandes's MIT licence text in a licences folder. We read its documents in full (README, RESULTS, results.json, a measured-facts file, a negative-results file, the timing kit); the kernel files were read only as a diff against the vjeux base. The details are in alygg77-repo-digest.md.

- Claimed: attention 2.653x, moe 2.401x, bf16 1.511x, decode_layer 2.206x.
- Verified: its attention and prefill files are the vjeux files with only a two-line header added, so its attention row is our kernel and the gap to our 2.605x is judge noise. Moe adds a scale-sharing K-permutation (KPERM3) and an alternate helper-warp build. Decode_layer appends about 2,700 lines of layered overrides.
- What we took as ideas: KPERM3 (+1.25% of the moe board), the T = 1 glue path, cuBLASLt picks at m = 128 and 256, and the exact FP32 router at T <= 8.
- Rule problem: its bf16 file stores CUDA source as zlib-compressed base64 payloads decompressed at run time. The live rules ban compressed, encoded or obfuscated source and code run from such data. We borrowed nothing from that file. Teams whose files descend from it may be affected at the final review (inference).

## keell0renz/muse-open-kernel-challenge

README and LESSONS (Apache-2.0) and NOTICE read; kernel files deliberately not opened. This is the strongest public lead on decode_layer and the only one with judge reruns.

- Claimed (reruns): decode 2.2641 on a check and 2.2532 submitted; moe 2.3884; bf16 1.5742 on a check and 1.5690 submitted; attention screens of 2.63 to 2.66 on the same box. The board row of 2.7168 is a high draw by their own account. The attention row is the best single draw, and the same-box screens sit in the same band as the vjeux file's 2.605 to 2.653. A graphic posted for their team showed decode climbing from 2.1815x to 2.2664x between Oct 30 and 31. Whether that graphic is theirs is open, since we could not read the original post.
- Technique stack, in prose. The decode layer is built on alygg77's base. On top of it: rs128 swap-AB GEMVs credited to chenwainuo, 12-bit lossless BF16 weights in TMA-ring GEMVs, a PDL trigger placed right after the producer's last weight issue (+0.9% on the layer), per-batch-size MoE builds, a 4-warpgroup stream-K wgmma MoE for 17 or more tokens, a warp-parallel stream-K prologue (+0.9% at 32 tokens and +0.5% at 128), and lossless 6-bit NVFP4 block scales (+2% on the MoE at 32 to 256 tokens). The attention file contains ZhimaoL's q_len-8 verify kernels. The notice credits a TMA bulk-reduction merge of the split-KV partials to Madhavan113.
- ZhimaoL's public branches were deleted on Nov 1, so that CUDA survives only inside keell0renz's file. Searches for its marker names found nothing else.
- Dead list from their LESSONS: a persistent whole-layer megakernel, fused-slice MoE (5 to 13% slower), speculative expert prefetch, L2 persistence windows, a 4-CTA cluster split-K qkv (x0.87), dual-lane warpgroups and per-unit barriers (8 to 20% slower), attending only the last 2k to 16k tokens, dropping low-weight routes. The last two fail correctness. They also ship the 6-bit scales on the skinny MoE builds switched off, reason unstated.
- Not borrowable: a remark about caching answers is a comment about judge design, never a technique. Carrying state between calls or doing work outside the timed call is banned.
- Missing: no per-step gains, no logs, no confirmed router-input precision, and no bake output showing which bf16 cell uses rs128.

## Kushagrabainsla/muse-spark-open

No licence found, so ideas only. We read an excerpt of EXPERIMENTS.md (experiments 15 and 16 were omitted) and never opened the code.

|Technique|Where|Their number|Our status|
|-|-|-|-|
|Cold-code warming: an idle producer warp runs the merge routine once with stores off before the real merge, so its instructions are already in the instruction cache (the judge flushes L2, so once-per-CTA code otherwise comes from DRAM)|Sliding verify, one-launch cluster kernel|Merge 6,928 to 2,524 cycles, kernel 10.8 to 8.9 us locally. Judge verify sliding batch 1 and 16: 12.3 and 21.6 to 9.9 and 17.2 us, bundled with pushed partials, a RoPE table and an approximate sigmoid|Legal if nothing is carried between calls. Our check on the vjeux base scored 2.576x with d1s at 5.1 against 4.94 us. Closed|
|Pushed partials: each split writes its partial straight into the merging CTA's shared memory over the cluster|Sliding decode and verify in one launch|8.9, 12.1, 16.6 to 8.5, 11.4, 15.8 us on their base|Different kernel design (1.5x on the board)|
|Hoisting late scalar loads (a scale the compiler sank to the end became a cold DRAM read of about 5,500 cycles) and one DSMEM read per thread|One-launch sliding kernel|Part of a bundle: sliding decode 13.3, 18.7, 21.1 to 9.1, 12.3, 16.6 us on the judge|Open as a check: look for cold loads in the epilogue|
|Resolve each chunk's page-table rows into shared memory up front in one coalesced pass|Split-KV full attention|Full batch-16 kernel 187 to 107.5 us. Random 128 B row gather 3.7 TB/s against 2.3 TB/s for 64 B rows|Probably already in the vjeux base (verified later: page-table rows are staged by bulk copy)|
|FMA-pipe exp2 for half the elements, 128-key stages, trimming a roughly 16K-cycle prologue|Full verify|Not built. They estimate full verify at full-decode speed would give about x1.15 on that problem|Open idea; our exp2 variants lost|
|Two KV heads per CTA for full verify, a one-launch full decode, a CUDA merge after the split kernel, an FA3 num_splits retune|Full|Ties at best, about 1%|Closed on their base|

## trxphxle/okc-2026-muse-kernels

README and a 55,556-character FINDINGS.md read (the reader cut the first pass at 50,000 characters; the tail turned out to be a table of non-kernel repositories). Its rows: bf16 #417 1.442, attention #372 2.567, moe #409 2.325, decode #383 2.176.

FINDINGS surveys 24 public repos against the alygg77 base and concludes that no public piece was established as an eligible, reproducible improvement over pinned alygg77. Results that touched our plans:

- decode_layer: more PDL pre-wait stages do not help (two is optimal, four or six slow attention by 1.7 us and a boundary by 0.9 us). Removing glue launches saves 1.2 to 2.0 us each, 4.0 to 4.8 us for all three at D1 and V1 sliding, which is the realistic cap for glue fusion. A pooler and router on a side stream is 2.1 to 7.7 us slower. L2 prefetch loses 2.6 to 5.1 us. An exhaustive cuBLASLt search is worth about 2 us per layer.
- bf16: a 12-bit codec inside the m = 128 GEMM loses (O and unpooler 2.4 to 4.1 us slower, pooler 32.6 against 20.0 to 21.4 us). A tighter codec than 12 bits is worth at most 0.5 to 1% of the board. m = 256 split-K loses 3.3 to 4.6 us.
- moe: a fourth ring slot, a WGS32 tile, a dynamic tail and a cost-table refit are closed. Fence relocation gave +0.234% on the MoE and nothing on the decoder. Expert compression is about 5% of bytes in practice.
- attention: equalising full-verify pair end times leaves SM completion unchanged, so the bound is consumer throughput. A producer V transpose and single-CTA or cluster merges lose. Polynomial exp (infraax) is 12 to 21% slower with an exp ceiling near 7%, and removing all exp keeps 87 to 91% of attention time in prefill. Deeper full-attention lookahead (lee101) lost.
- judge artifacts: +3 MB of scratch slows unchanged code by 0.5 to 1.3 us, padding can move V32 by 1.5 to 2.5 us, the same dense file scored 1.5259 to 1.5502x, and a cross-cluster +4.19% was really about +2.4%.
- open leads it listed: cold-code warming; dense per-thread escape cursors with a predicated-XOR decode (chenwainuo, Apache-2.0, about 1 us at M128, held back by a BF16 split-K reduce that the new rules may now allow); a two-level multi-CTA attention merge (tommyyliu); a full-K output-column BMM for the pooler at M32 (Xuefeng-Zhu, Apache-2.0, about 10% on the pooler on its own base); and custom WGMMA O-projection paths at m = 128 and 256 in alygg77's standalone dense file (O 6.96 and 9.26 us) that the decoder lacks.
- It restored the BF16 boundaries that the vjeux upstream skips, at about 0.6% cost. Its precision notes later softened the wording about disqualification.

## bemlerlabs/okc-2026

About 24,000 files, 20,000 of them an Oct 30 H100 backup, with a 314 KB log, a falsified-ideas registry, per-board notes, daily briefs, a judge-profile model and a champion manifest. Parts are in German. No licence file in the tree we listed, so ideas only. The repo returned 404 on Nov 5, removed or made private. We read the notes through a shallow clone of the markdown files and never opened kernels.

- Claims: decode 2.2703x (rank #107, a best draw; their other 15 draws sit at 10.01 to 10.10 ms, against a best of 9,963.8 us), bf16 1.584x on Nov 3, working base is keell0renz's files plus alygg77's.
- Attention score model (verified by us against alygg77's per-case times): total = sum of (case time x weight) / 8, with weight 46 for each of the six sliding cases and 16 for each of the six full cases. That gives 1449.6 us against 1446 us on the board for us, so sliding cases are about 22% of our time (324 us) and full cases 78% (1126 us).
- Their attention base is the keell0renz file. Their 24-draw family has a median of 1427.8 us (2.638x), minimum 1415.7, standard deviation 0.64%. The vjeux-style file has a median of 1451.6 us (2.595x) over 5 draws on the new judge image. So that family is about 1.6% better than the vjeux family. The families were drawn on different judge-image days, so part of the gap may be an image offset (inference). Board rows use the minimum draw and final ranking uses medians.
- Six sliding-case experiments on that base all lost: a DSMEM cluster merge (+22% to +55% per case), a one-tile split for multi-request sliding cases (+31% and +40%), a cooperative grid barrier on d16s and d32s (+3% and +8%) and on d1s (+11%), and 8 instead of 4 splits on v32s (+4%).
- Full-case experiments were flat within 1 to 2%: a 4-stage KV pipeline (+2.5% worse on d16f), split plans for batch-1 full (+15.6% worse), a cooperative finalise for q_len 1 (nothing), a switched TMA-producer path for batch up to 32 (only v32f moved, about -1.1% in three draws, but the aggregate median was +2% worse). They conclude full cases are bounded by the random 64 to 128 byte row gather of a page-size-1 table. Sorting the page table inside the call broke graph capture (a replay did not finish in 60 s, plus a correctness failure on one check). They state about 2.0 TB/s on full cases, while our d32f time (165.5 us) over the 606 MB implied by the judge floor gives about 3.7 TB/s. We read their figure as a different byte count (inference), not as headroom.
- Decode MoE bytes: the 12-bit lossless weight code is at 12.008 bits per weight all-in, the entropy floor on the harness weights is 10.46 bits (measured), and an 11-bit fixed code overflows the escape budget (1.25 to 2.5% escapes against a 0.5% cap) and loses bytes on the biggest matrix.
- 6-bit scales were live in their decode bank for the large-token MoE builds (T >= 17), with in-judge twin ratios of 1.025, 1.029 and 1.021 at T = 32, 128 and 256. They then built the same for the skinny builds. Judge check 402045a4cf644295: score 10,040.9 us against the bank's 9,963.8 (+0.77% worse, 2.253x). Targeted cases d1s +0.7%, d16s +3.0% and v1s +2.1% slower, but untouched cases moved as much (v16s +3.85%, d32f +2.69%, d1f -1.74%), and their notes put single-draw spread at 0.8 to 1% and cross-day offset at 1 to 3%. Their in-prepare MoE twin runs at T = 32, 128 and 256, which are the large builds that had 6-bit scales in both arms, so it says nothing about the skinny builds (our correction of their reading). Honest reading: one check, inside noise, no gain visible. They concluded skinny MoE is not byte-bound, which one check does not prove.
- A SGLang-style cluster boundary kernel for T >= 17 as decode glue: check 3fe4b704a87c47e8, 2.266x (interval 2.232 to 2.300), no better than their bank, dropped.
- Draw spread on identical code: their moe case verify-1 varied 63.9 to 67.1 us (5%) across four judge runs of byte-identical code. They call rerolling with header mutations a banned class and say a winning mutation is noise.
- Their decode plan notes list levers summing to about 1% beyond their bank and say top-10 on decode (-8.6%) is not reachable from known levers.

## chenwainuo/okc

Apache-2.0. We read the README and handover notes through a shallow clone. This is the missing source of the bf16 swap-AB 12-bit design behind keell0renz's m = 128 cells. Best judged result in its README: bf16 1.542x.

- Design in prose. Every BF16 weight is stored in 12 bits: sign, 7 mantissa bits and a 4-bit exponent offset inside a 16-exponent window per matrix. Weights outside the window are escapes, about 0.01% of weights, kept in a per-thread sorted list and applied exactly after decode. For m = 16, 32 and 128 the kernel is a swap-AB split-K GEMM. Each CTA owns 128 weight rows and all tokens. Two math warpgroups decode the packed bytes straight into the A registers of wgmma, x arrives as the shared B operand through a 2D TMA, weights arrive by bulk copy into a 5-slot ring, K is split over about one wave of CTAs, and partials are summed by a BF16 bulk tensor reduce-add into an output the kernel zeroes itself. For m <= 32 a register-direct GEMV with a predicated-XOR decode is used instead. The prepare step builds every candidate path per layer and m, checks each against the reference, and bakes the winner by timing under graphs.
- Judge run 846e25d9: at m = 128, the pooler went from 25.31 to 22.62 us in the in-prepare bake and 18.16 us judged against about 20.2 for the cuBLAS-era path. The qkv went from 19.49 to 16.70 us in the bake and 12.79 judged against about 15.0. Unpooler tied and stayed on cuBLAS. The o projection lost 1.9 to 2.1 us and stays on cuBLAS. All m = 256 variants lost 3 us. The step from their pw33 (1.4498x) to pw34 with this kernel (1.4666x) is +1.2%, so the whole design is worth about +1.2 to 1.3% on bf16 before the other cells.
- Cost breakdown at m = 128: bare weight stream 17.98 and 11.17 us (pooler and qkv), the full kernel 21.9 and 15.1 us, so decode, wgmma, barrier handshake and the reduce-add epilogue cost about 4 us over the stream. Unmeasured levers they list: a stream-K grid on 132 SMs and a 2-CTA multicast of x.
- Closed on their side: variable-length codecs (11-bit fixed needs 1.3% escapes, the pair-Huffman code is issue-bound), a separate decode warp feeding an SS-wgmma, clusters (a launch costs about 1 to 2 us fixed), bulk rings for small m, hand bf16 pipelines at cuBLAS geometry, CUTLASS mixed-input.
- For us the realised gain is about 4 us on two of the five layers, roughly +1.2% of the bf16 score, not the 6% ceiling we had priced. Effort is very high (a new wgmma kernel plus codec, no GPU), so we did not pursue it.

## Madhavan113/ParadigmKernelChallenge

No licence file seen. Rows: attention #343 2.588, decode #362 2.200, moe #338 2.357, bf16 #351 1.493. We read the notes folder only. It holds design memos of vjeux's published kernels (attention, MoE, projection) written for re-implementation, a decode-layer map and a stream-K note. The attention memo describes our own base accurately: a single launch, red.add into a self-zeroing workspace with every split rescaled to a fixed reference so no running max is exchanged, a flat kernel with a producer warpgroup, and a speculative sequence length for batch 1. It confirms that the base's merge is the float red.add structure, and that the base author tested clusters, cooperative finalise and copy counts. Nothing newer than our base.

## ruwwww/open-kernel-challenge

No licence in the tree. Claims: attention 2.6032x (job 2827c6e6), moe 2.3370x (3ed21907), prefill 1.7950x, decode 2.0751x (f60b136c, interval 2.044 to 2.106), bf16 1.5056x (c512f73a; the board row shows 1.4999x, job 66361ede). Ranks on our snapshot: attention #314, moe #382, decode #466, bf16 #336, overall #466.

- Verified by blob hash: the attention file and the moe file are identical to vjeux's published files, so they add nothing.
- The decode file matches vjeux's except the first 200 lines. The change is a rewritten dense-pick table that pins qkv, o and unpooler to the bf16_proj route and the pooler to a fused-router variant at every token count 1, 8, 16 and 32. It pins picks already present in vjeux's code and is not a new kernel. Their 2.0751x on 2 attempts is inside vjeux's range, so the pin shows no visible gain (inference).
- Because the rest of the file is vjeux's, it keeps the base's router-input handling: a BF16-rounded activation at T <= 8, where the live rules want FP32. Do not read their 2.0751 as a reachable compliant target.
- Their decode profile for the judge run (per case, SGLang against theirs in us): d1 sliding 172.98 vs 77.59, d1 full 182.85 vs 84.17, d16s 267.15 vs 138.83, d16f 435.02 vs 212.35, d32s 262.66 vs 158.10, d32f 612.11 vs 313.72, v1s 268.27 vs 113.83, v1f 267.47 vs 115.76, v16s 539.76 vs 263.55, v16f 716.24 vs 348.39, v32s 698.51 vs 328.39, v32f 1062.32 vs 489.29. Kernel mix at T <= 32: the MoE kernel takes 23 to 88 us, four swap-AB bf16 GEMV kernels take 4.6 to 22 us, two boundary kernels take about 4.3 us each (4.4 to 5.3 us at T = 32) and a router-split kernel about 4 us, so glue is about 13 us per layer at T = 1 (16% of 80 us). At T = 16 and 32 verify the dense GEMMs are cuBLAS kernels of 15 to 38 us each plus a split-K reduce of 3 us, and the MoE takes 122 to 226 us.
- The bf16 file (633,375 bytes, 7,194 lines) says in its header that large CUDA source strings are zlib-compressed to fit a 1 MB limit. It has 42 calls to a helper doing base64 decode plus zlib decompress, and 33 lines longer than 3,000 characters. That is the compressed, encoded source the live rules ban, so its 1.5056x is not reachable by us. Its README describes the technique in prose: 12-bit lossless weights with a sparse correction table for escapes (under 0.2%), a swap-AB tensor-core GEMV (mma m16n8k16, activations as the A operand, weights streamed through registers) with no cross-CTA atomic reduction, and the router as BF16 hi plus lo with FP32 accumulation. All of it is known to us from chenwainuo and keell0renz.
- The serving patch file is empty.

## AutoclickerI/muse-spark-open-kernels

Apache-2.0 with a third-party notices file. README claims: decode 1.889x (best run, Oct 27), moe 2.032x, attention 1.670x, bf16 1.202x, prefill 1.454x. Our snapshot shows decode 1.7428x #652, moe 1.6939x #851, attention 1.6696x #801, bf16 1.1873x #1163, overall #658. The repo stopped on Oct 27, so the README is higher than the later board rows, and the board row is the latest attempt, not the best (inference). Well below our entries on every problem.

Its FINDINGS file is 371 KB (2,736 lines) with receipts for about 490 judge runs and a board model saying the overall rank follows the decode board. Techniques in prose (README): PDL-chained Triton swap-AB split-K GEMVs with L2 prefetch before the wait; lossless 12-bit dense weights (16 exponents cover 99.993%, saving 3.9% of the layer at decode 1 against 1.1% A/A noise); a persistent NVFP4 expert kernel with 36 KiB tiles and 8-bit recoded scales (the recoded bank takes 0.88 to 0.91x the raw time); a per-cell attention pick between their own FP8 split decode, a verify kernel with a page ring and clusters, FA3, and ports from operatorstack and lee101; cluster-per-token glue at T <= 32.

Measured negatives worth keeping, from the attention section of FINDINGS:

1. The packed-half exp2 lowers to two MUFU ops on this hardware, so it saves no issue slots. Our SASS later agreed.
2. A native FP8 design with a Q-major SS layout lost because V-transpose and barrier work took about 990 cycles and softmax about 760, against 200 and 208 for the two GEMMs. It is issue-bound, not memory-bound.
3. The licensed public attention ports (operatorstack under MIT, lee101), tested on their S19 baseline across 12 cells with 8 seeds each, were 1.23 to 1.26x slower on their file, with A/A intervals covering 1. The direction of that ratio is our reading of the table's decision to retain their S19 baseline (inference).
4. A page ring beat FA3 at verify 16 and 32 (141.6 against 191.9 us and 250.6 against 378.9 us) but not at q_len 1.
5. A long run of MoE recipes lost: a vector-writer variant +1.3 to +1.9% slower, four-arm and pair variants +2.2 to +3.4%, a 16-warp CTA with a shared B cache +20 to +23% slower. All were measured on their weaker 2.0x MoE, so transfer is unclear.

Compliance note: a glue plan in the file moves the BF16 rounding of the pooled activation ahead of the pool and router GEMM and rescales in FP32 after. If that feeds the router GEMM at T <= 8 it is a BF16-rounded router input, which the live rules disallow. It is a plan, with no result seen. We did not read the closed-recipes table (63 rows), SOTA.md or the 513 judge JSON files, because the team scores 0.17 to 0.35x below us and their negatives do not transfer.

## infraax/okc-kernel-challenge

Apache-2.0. RESULTS: 22 scored submissions in 3 days with no local GPU. Final: attention 1.770x (job 1f0a0c8f, #739), moe 1.486x (99f12b5d, #963), decode layer 1.433x (efc7af96, #858), bf16 1.170x (7506407, #1270), overall #892. The tree holds hypotheses, lessons, method, results and 60 or more judge-run profile files. Kernel source was not in the part of the tree we listed.

Techniques (HYPOTHESES.md): a two-launch fused split-KV attention; full-attention layers treating two KV heads as one 16-row wgmma problem; a unified MoE expert kernel at 3 CTAs per SM with activations staged once per CTA (+8 to 13% on their MoE); pooler shared memory halved (147 to 74 KB per CTA) so the router and pooler co-reside at T <= 8 (decode layer 1.271 to 1.387); a router fixup kernel at T = 1.

Measured negatives, with job ids:

- Exp2 and FP8 convert are not the attention verify bottleneck. Twins with no exp or no convert show exp at about 7% and convert at 1 to 4% of the kernel (job 9206dc7a).
- Splitting exp2 between MUFU and an FMA polynomial (FA4 style) was 12% to 21% slower (same job).
- A per-warp cp.async ring prefetch in the MoE GEMM gave 2.7x bytes in flight but dropped residency from 3 to 2 CTAs per SM, net -22% (job 698aa839).
- A warp-specialised MoE GEMM (1 bulk-copy producer warp, 12 consumer warps, mbarrier ring) was copy-issue bound at 80 copies per stage and ended at 0.82x of the old kernel, with consumers capped near 2 TB/s (3 runs). A design review had predicted 1.3x.
- A light L2 weight prefetch before the PDL wait was +2.0 us at T = 1 and full prefetch -3.9 us from bandwidth congestion (job fcbc85a4).
- T = 1 quick wins (grid width, late trigger, smaller staging, route on a side stream) gave -1.4, -0.5, -6.0 and +1.1 us (job f31c9839). No winner.
- A tensor-core GEMV for BF16 projections with many 16 B loads in flight reached 0.2 to 2.5 TB/s against 2.6 to 3.7 TB/s existing (job e2a5099d) and was killed.
- Fusing the post-FFN norm into the unpooler kernel with a grid barrier took 10.0 us, equal to the two kernels it replaced. Without the barrier (last-CTA scalars plus an apply kernel) it scored 1.425 against 1.433 (job ce45f6ae).
- About 160 Triton GEMV configs for qkv_gate gave at most 3% against cuBLAS.

One positive reading we could not use: BN 128 tiles for verify-full at batch >= 16 gave x1.16 and x1.13 on their two weakest shapes (3 seeds, plus or minus 1 us; jobs 9206dc7a and 2d42052a). Their kernel is Triton-derived at 1.77x and ours is at 2.6x with 64-key tiles baked in, so it is unmeasured on our base.

A calibration lesson from their LESSONS: first-principles predictions were off by 1.5 to 2x (MoE v5 predicted 1.72x, measured 1.486x), while predictions anchored on judge-measured medians landed within about 1%.

## lee101/openkernels

Also copied under 4gatepylon/LearningML. Contains moe submissions (v39, v41d, v47d), build tools and attention submissions. We know it mostly through other teams' notes and through a separate deep dive. Findings that touched us: deeper full-attention lookahead lost; a bulk 128-byte row copy was negative (TMA op rate); a judge comparison of two moe versions (v53d against v52) was noise; its survey counts Marlin's FP4 dequant at 15 operations per 8 values against our 9; and its SASS evidence for packed half-precision exp2 matches ours.

## tommyyliu/okc

Rows: attention #484 2.2336x, moe #494 2.1692x, decode #541 1.899x, bf16 #559 1.3375x. Their scored attention file is 1.559x, and a check run of 1.717x was never submitted. README: Triton, one launch per call, two KV heads packed per program with a block-diagonal query so every gathered cache row is one contiguous 128-byte read (their measurement: random 64-byte reads top out near 2 TB/s on H200, 128-byte reads near 3.0 to 3.3 TB/s), new tokens on an extra split, a two-level tree merge with self-resetting counters, P.V in FP16 with FP32 accumulate. Insights: full verify dropped from 410 to about 325 us through unmasked interior tiles plus a register cap that fits 4 programs per SM, with a floor near 200 us. Full decode is at the measured random-gather ceiling. Sliding d1s at about 5.4 us is a chain of three dependent DRAM round trips plus a roughly 2 us merge tail. Everything is below our file; the 128-byte row read is our idea A4.

## Smaller repos, read at README or tree level

|Repo|Claim or content|Note|
|-|-|-|
|ozgureyilmaz/okc-muse-kernels (Apache-2.0)|bf16 only. Best check v027 1.1804x, scored 1.1718x (job 20251c5f, #1258). The 1.5696x in an earlier survey is not their score but the rank-10 threshold they quote from the live board on Oct 27. 49 files, 42 KB of notes|Negatives, on their cuBLAS-era baseline so only direction transfers: explicit K-partial split-K for O at m = 16 and 32 was much slower (9.9 against 7.2 us and 7.0 against 6.6 us), unpooler 21 to 22 against 18.5 us; all-small-M TinyGEMM regressed O at m = 1 and 8 (10.7 against 7.1 us); a transposed torch.mm changed layout only (1.115x); DeepGEMM native SM90 GEMM at m = 128 qkv_gate 15.97 against 16.98 us (-6%), at m = 256 about baseline. Replacing every router call by its floor caps their score near 1.306x. Unmeasured ideas (a 12.25-bit block code, compile-time M for DeepGEMM, fused hi/lo router reads for m > 64) are already in our bf16 line|
|Ranoobaba/okc-campaign|296 files. attention 2.3995x #439 (29 attempts), moe 2.1523x #511, decode 1.961x #516, bf16 1.3472x #541|From LEARNINGS (read in full): judge-timed shape time is about kernel GPU time plus 4 to 5 us fixed cost per launch inside the graph (an m = 1 qkv GEMV 10.7 us on GPU timed about 16.7 us). Sliding decode-1 at 15.5 us was insensitive to tile size, stages and split count (a launch-and-latency chain). A one-launch merge with an unrolled loop is needed because a runtime loop over splits serialised about 1 us per partial load. FP8 x FP8 QK is accepted by the checker with 4 to 7% error against an 11 to 15% bound. 64-row Triton programs over full context were 3x slower than FA3 (unexplained). Their MoE CUDA kernel was instruction-bound at about 3.5 instructions per weight (prmt dequant 1.5, scales 0.9, hmul2 0.5, mma 0.25, against about 2 for Marlin). A hand mma.sync bf16 skinny GEMM was 1.5x slower than cuBLAS because too little data was in flight per SM (7% occupancy)|
|Mithun-033/Muse-Spark-Open-Kernel-Challenge|bf16 only. README 1.255x (3,847.9 against 4,830.8 us); board row 1.3559x #523 after 38 attempts|Naive Triton 0.694, custom Triton for M <= 32, tile tuning with two-phase non-atomic split-K (1.158 to 1.209), then adoption of a community 12-bit lossless code (lop3 plus prmt decode). Nothing beyond existing bf16 entries|
|theluckystrike/muse-spark-open-kernels|One 1,649-byte placeholder README saying code lands after Nov 4. Rows: attention 2.0902x #539, moe 2.0606x #571, decode 1.9536x #521, bf16 1.3932x #473|Method line: several variants timed inside prepare() per check. Nothing else|
|GiulianoIntelligence/Muse-FlashSpeed-Kernel|moe only, Triton, best 0.2376x|One useful fact: some Triton tile configs silently miscompiled on the judge GPU (output exactly zero at small batch) while passing the interpreter, so every config change must be validated on the judge. Does not apply to our CUDA kernels|
|KingCobra52/muse_bf16_optimization|8 files, README 25 bytes, submission.py the size of the starter|Empty|
|xharshvardhan/open-kernel-challenge|Overall #647 of 1271 on Oct 27. Dense 1.1991x, experts 1.3749x, attention 1.2990x, decoder 1.2252x|Far below ours|
|lilaclilac09/open-kernel-challenge|Generic README. bf16 1.0707x #1612|Nothing|
|Reedo3, nibague, jacobpmiller, MadanChaollaPark, Aravind-11, iyerramji, sid19991, arudomanenko, Snehan2k2, ScrashOff, zimo0110|Starter-level or empty repos, or scores of 1.1 to 1.9x (decode 1.437x, moe 1.632x, an early moe 1.896x, attention 1.33 to 1.51x)|Nothing|
|Xuefeng-Zhu/open-kernel-challenge (Apache-2.0)|1.18 to 1.24x|Earlier notes credit it with FP8-QK scale folding, page lookahead and a full-K output-column BMM for the pooler|
|operatorstack/open-kernel-challenge-solutions (team bigboateng, MIT)|Final solutions, no speedup figures extracted|Its README describes fused QK-norm, RoPE, cache store, split-K and an exact LSE merge in one kernel, NVFP4 decoded in registers via inline PTX, and pinned cuBLASLt algorithms with rebuild, verify and fall-back self-tests. A reply on the forum links its MIT attention source|
|NathenFernandes/muse-bf16-proj (MIT)|README, results history, source hashes, and a tool named expand_source.py|A forum thread records the v18 file scoring 1.521x on submission and 1.532x on a check. Per trxphxle, nathenfernandes's small-M routes are worth +0.6% and others +1.25%. The tool name suggests the submission is stored condensed and expanded, which deserves a rule look before anyone borrows from it. We did not open expand_source.py|
|msaroufim/openkernelchallenge-audit|Compares the judge to GPU MODE's KernelBot (same seeds, exit codes, file layout)|About the organisers, not entrants. No effect on us|
|danyloooah-web/okc-claims|Encrypted claim archives|Not usable|
|Lurrobert/guild|A repo for the coordination forum|No kernel content. Lurrobert sits at #31 on attention and #33 on decode|
|hieptran1812/my-website, content/blog/machine-learning/gpu-kernel-engineering|Six explanatory posts (inside the judge, the challenge decoded, one layer on one H200, roofline for decode, setting up the lab, Muse Spark for kernel engineers). Rows: attention 1.894x #655, moe 1.8508x #717, decode 1.6288x #724, bf16 1.2466x #820|No kernel source and no measured negatives in the parts we read (two post headings, the judge post's section 5, the layer post's section 6). Facts worth having: the three component problems sum to 22,045.6 us against 22,620.3 us for the whole layer (gap 2.5%, from four ops timed only in the layer: the two norm-residual boundaries, the router sigmoid, bias and top-8, and the pre-expert norm). The judge's static scan rejects timing patches, harness reads, processes and network before running, and the after-the-fact review removes cached or lower-precision work. Matches the live rules page|
|rohanarun/Auto-research repo and forks (xrey167, aviyanp)|A macOS app with starter guide files for the five problems. Board rows are above ours (table above)|We did not study any kernel code from it. Its starter-file changes are reviewed in social-post-claims.md. It advertises a hosted GPU queue (app.getsupers.com) that needs sign-up and uploading code to an outside service, which we did not use|
|PostTrain, thethings|Saved and not re-read for the later notes|Earlier reads found nothing new|

## Rule problems we noticed in public files

We recorded these so nobody borrows them by accident.

- Encoded source. alygg77's bf16 file and ruwwww's bf16 file both store CUDA source as zlib plus base64. The live rule bans compressed, encoded or obfuscated source and code run from such data.
- Router input. The vjeux base feeds a BF16-rounded copy of the normalised activation to the router at T <= 8. SGLang's reference feeds an FP32 activation and FP32 weights at T <= 8 and BF16 plus hi/lo weights above 8 tokens. The BF16-rounded copy differs by about 2e-3 relative. alygg77 writes an FP32 copy and recomputes it exactly. The live rules let the router's FP32 weights run as BF16 hi plus lo at any batch size, and say activations stay BF16 or FP32 wherever SGLang keeps them in BF16. Keeping FP32 where SGLang keeps FP32 is our reading, and our entry had to be changed to match. ruwwww's decode file keeps the rest of the base and so has the rounded input too, and AutoclickerI's glue plan risks it.
- Exact-size tables. Pick tables keyed to the scored token counts (ruwwww's, and a pin inherited from vjeux in our own decode file) meet the organisers' nearby-shape reruns. Dispatch should be a contiguous range.
- Answer caching and any state carried between calls are banned. Cold-code warming is fine only inside the call, with stores off.
- Rerolling an unchanged file or a header-mutated file to catch a lucky draw reads as gaming, and final ranks come from reruns. We did not do it.

## Ideas worth re-implementing, ranked at the time

We ranked these by how much measured evidence stood behind them, and each would have had to be written fresh in our own code.

1. decode_layer: the techniques behind alygg77's decoder changes, especially the T = 1 glue path and KPERM3. Evidence: same base plus documented changes, 5.2% above us on the board; per-case lead 11 to 12.5% at d1 and 4 to 5.5% at d32s, v16s and v32s, well beyond the 2 to 5% noise band. Caveat: their v1 is no faster than ours, because the FP32 router costs time there.
2. decode_layer and moe: keell0renz's stack (6-bit scales, warp-parallel prologue, PDL trigger after the last weight issue, per-size builds). Their decoder is about +2.5% over alygg77 on the judge.
3. decode_layer: cuBLASLt picks at m = 128 and 256, about 2 to 3.4 us on v32s and v32f measured on the judge.
4. attention: cold-code warming for once-per-CTA phases in the sliding kernels. The mechanism was measured at 1.9 us on a 10.8 us kernel. Our check found no gain.
5. attention: full verify (v16f and v32f), the weakest case at 2.02 to 2.11x. v16f takes 101 us against 86.6 us for d16f over the same keys. Everything around balance and transpose is closed, and the levers left sit on the consumer side.

What we actually got from this: the KPERM3 layout, 6-bit scales and the stream-K prologue went into our moe work and were measured; cold-code warming and the packed-half exp2 were tested and failed; the rest is covered by the closed lists in ideas-ledger.md. If we read these repos again we would start with the measured-loss lists in the alygg77 and bemlerlabs notes, because the negatives were worth more to us than the ideas.

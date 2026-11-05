# Research gaps and answers

This file lists the open questions we chased in public sources, what each one turned up, a summary of a third-party note about isolated kernel wins that died in the full decode layer, and a short table of what our own headroom studies concluded.

Most of the answers are negative or thin. We kept them anyway, because a closed question saves the next person an afternoon.

## Gap ledger

|gap|status|source|finding|what it means for us|
|-|-|-|-|-|
|1|closed|A bachelor thesis on brute-force autotuning caches (https://theses.liacs.nl/pdf/2024-2025-JansenTThomas.pdf)|It analyses brute-force cache files (convolution, dedispersion, a CLBlast-style GEMM, Hotspot) on A100, A4000, A6000 and AMD GPUs. Block size, unroll and tiling factors matter, and rankings change from GPU to GPU. It offers no new cheap search method.|No new lever. We should not import its ranking to H200. Our search space is already narrowed to bytes-in-flight parameters.|
|2|closed|https://arxiv.org/html/2512.22147v1 (preprint of a PAKDD chapter)|An automated rewriting loop: extract the hot kernel into a minimal program, time it with a trimmed mean, then run error-repair and pattern-hint passes. On PolyBench it reports 5.29x standalone and 5.05x integrated, against 2.10x for a direct one-shot rewrite. On HPC kernels the figures are 1.78x standalone and 1.25x integrated. Its own table shows the drop from standalone to integrated.|Supports our rule that standalone wins do not carry over to the full layer. We already run this kind of loop (a local test plus a judge check).|
|3|closed (arXiv version read; the peer-reviewed page was not opened)|https://arxiv.org/abs/2509.07506|A four-role pipeline (test, profile, plan, code) over 5 rounds on H100. Three SGLang kernels reach 1.26x, 1.25x and 1.46x, average 1.32x. A single-role version gets 1.08x. Speedups range from 1.00x to 1.57x per shape.|Nothing shown for attention, MoE or GEMM. A merge-hoist idea is untested (we have not read our own merge code closely, and it may already be in the entry). A fast-math SiLU is untested and rated small.|
|4|closed, verdict unchanged|The authors' replies in a Reddit r/MachineLearning thread about DFloat11 (an 11-page printout)|At batch 128 on A100 it only reaches parity with BF16 (1.02x), and at batch 1 it is about 40% slower. It decodes with Huffman lookup tables. The authors say 8-bit quantization is faster and smaller but lossy, and point to Deep Compression and ZipNN for storage-only use. No kernel detail in the thread.|Not a fit for bf16 at m=1 to 256. A fixed 12-bit packing is already in our entry. Nothing new.|
|5|partly; no public detail|One participant's Nov 2 and 3 posts and repo (including its pull request 6), the kerneltuner docs (https://kerneltuner.github.io) and the abstract of a PhD thesis by Willemsen|The author's method is unreleased (a GA dashboard, a surrogate model, a Lean-based pruner, 2000 runs). We have tuners of our own: an override table in attention, prepare-time A/B ranges in moe, a per-T tune in decode_layer and an autotune in bf16.|A process change we have not tried. The plan is one setting per contiguous range per judge run, with controls, carrying winners forward and filtering by the bound. Check the decode_layer tune pick at T=128. The organizers' final review reruns nearby shapes, so behaviour at range boundaries matters.|
|6|closed|arXiv 2605.10481 and 2608.14380|A post by @anshulcreates on Oct 27 cited them. They are not kernel papers (they cover constraint drift and checkpoint rollback in autonomous systems).|Nothing for the boards.|
|7|partly; origin unclear|A community-forum post, a collection of other teams' posts, and a text search of keell0renz's repo|The cgemv note described below has two candidate origins (inference). One is VkaZas, from alygg77's lineage: T=1 qkv and unpooler as cgemv, a pinned tuner pick, decode at 2.218. The other is keell0renz: the decode climb from 2.1815 to 2.2664, a decoder-ports lane, and cgemv in the bf16 file. The note's text is not public.|No action. We already treat the note as a rule: cgemv stays out.|

## The cgemv note

A short list of six attempts to carry isolated wins into the full decode layer reached us as a copy with no source named. The numbers are that team's, not ours, and we do not know whose they are. We summarize it here instead of reproducing it, and we use it for ideas and warnings only.

|attempt|result|
|-|-|
|cgemv in the full decode layer, for the QKV and output projections (projection times nearly halved in isolation)|2.1898x to 2.2007x, barely moved|
|cgemv for QKV only, expected to save about 9.6 µs|2.1704x, a 0.9% regression (the expected gain was only about 0.59%)|
|cgemv fused with the router (projection, residual, norm, routing and top-8 in one launch)|2.1859x, below baseline|
|7 winning standalone GEMMs ported in|6 tied or lost to cuBLASLt, one reached 2.219x|
|dense GEMMs that looked 10 to 19% slower in the layer than standalone|no real improvement|
|stacking the unpooler and sliding-attention wins|2.1981x, the score fell|

The note blames two causes, and we find both believable. Some changes broke a fusion that was already paying off (the output projection feeding the router), so they cost about what they saved, and several touched only a few of the twelve workloads (the first attempt covered 2 of 12), which leaves a gain too small to see against run-to-run noise. The other cause is that standalone timing flatters a kernel: its setup cost is paid somewhere else and its weights sit warm in L2, and neither is true inside the layer. For the stacked pair the note says the tuner's apparent gains were noise to begin with.

The note does not say what baseline each score is measured against. The first row suggests 2.1898x was the starting point.

How it lines up with our own results: our first decode_layer judge run (job a44e46117b9449f8) tried cgemv in the full layer and failed 10/13 under PDL, because the x loads were hoisted above griddepcontrol.wait and used ld.global.nc. Our notes do not say what the 13 counts. The note says the same idea earns no score even when it runs, so cgemv stays out of decode_layer in all three variants.

Three lessons we took from it:

- Do not port a standalone win from the bf16, attention or moe board into the decode layer expecting the same gain. Weights are warm in L2 when a kernel runs standalone and cold in the layer.
- Do not trust an autotuner's pick unless it repeats on a second run.
- Keep a compliance change (we had a variant with an FP32 router at zero speed change) separate from speed ideas, so that a judge run can attribute its result.

## Outcomes of our own studies

|study|result|outcome|
|-|-|-|
|Headroom model of all four boards|Reproduces the four reported scores. Ceilings: attention 52%, bf16 15%, moe 56%, decode_layer 56%.|Done with no judge runs spent. See `headroom-model-all-boards.md`.|
|Attention parameter search|No candidate built.|No-go. The heavy full shapes sit at 1.11 to 1.34 of bound with the cheap levers already used, and the chain shapes are below the resolution of a judge run. A research note on attention agreed.|
|bf16 static probe|Router ceiling 3.2%. The rs128 design at m=128 has a ceiling of 6.0% (2.7% at half capture).|Not pursued. It was a static study only, and rs128 is a hard kernel to write and a design that has already been tried and dropped.|

Board entries behind these studies:

|entry|board file sha256 prefix|result|judge run|
|-|-|-|-|
|attention|cf04259b|2.6045x (interval 2.565 to 2.644)|c20df44ab71d4005|
|bf16|1a2bf6ba|1.4914x (interval 1.469 to 1.514)|6c7a0061944e4339|

# What a public post claimed, and what we could check

This file summarizes a public post about climbing the leaderboards, the pull request and starter files it points to, and where each claim stands against what we verified and against our own entries.

The review was neutral on purpose. The post promotes a tool and a workflow, and most of it makes claims we cannot test. Where we could test something, we did. The short result: nothing in the post, the pull request or the starter files was new kernel knowledge for us, and the one concrete lever in the thread was a change we had already measured.

## What we looked at

- The post: https://x.com/RohanArun/status/2114830856393027974, posted 1:25 AM Oct 27, 2026, 5,922 views when we read it. Thirteen replies were visible.
- Pull request 6 on the author's public repository for his app. It was merged by its author, head commit 7a14f0d, merge commit 17ec0bd, +35 lines in five files, no comments, no reviews. Its own validation note says the change is Markdown additions plus catalog and instruction integrity tests.
- The repository at its head when we read it (b94e66e, 527 tree entries, not truncated): the starter prompts, a research-procedure file, a setup template, a continuation prompt, the README and a benchmark folder.
- Three bullets that circulate alongside the post but come from a different post (see the last section).

Everything below was read, not run. We did not sign up for the hosted service the post mentions and did not post anything.

## The claims in the post

|Claim|What we could verify|What we could not|
|-|-|-|
|Board moved Oct 26 to 27: decode layer 1.736x to 1.785x (rank 178 to 143), MoE 2.013x to 2.033x, bf16 1.443x to 1.453x, attention 1.788x to 1.801x, prefill unchanged at 1.303x|The post says so. At the time we read it our own entries scored 2.605x (attention), 2.362x (MoE), 2.059x (decode layer) and 1.491x (bf16), all above those numbers. The rank gains are small microsecond moves (attention improved by 15.6 us).|Nothing about how the gains were produced.|
|CUDA best practices shipped as pre-written starter prompts, and a workflow that follows an auto-research.md guide|Pull request 6 exists and holds what is described below (verified). The repository has no file named auto-research.md. The README says the app creates an editable starter guide of that name for each project.|Which source files feed that guide. We guess (inference) it is built from the starter prompts, the research-procedure file and the setup template. We did not read the Swift source.|
|With the tool, beating the SGLang baseline should be easy|The pull request contains no code and no measured result.|The claim itself.|
|A fleet of automated workers directed by a human expert, said to be twice as fast as having the direction automated too|It is a stated claim.|There is no data for it in the post.|
|A hook-based system so that workers are never idle|Neither the post, the pull request nor the README describes one. The only matching feature we found is the app's keep-researching option: after a successful round it starts another only if that round concludes another experiment is justified. It is off by default and does not survive an app restart (verified from the README).|That this option is the author's hook. That is inference.|
|A queue and lock system keeping GPUs at constant 100% use|It is not in the repository. In later posts the author links a hosted experiment queue. The setup template documents its HTTP interface: submit a name and a shell command, and ask how many free experiments are left.|How it works inside. We did not use it: it means sending source to a third party and signing up.|
|Teams trading H200 compute for GPGPU tricks; first hours of human insight into memory and hardware, then microseconds|They are claims.|No technique is named anywhere.|

## The thread

Only a few replies carry content.

- flowers.engg asked about run-to-run noise of the scorer. The author said he had not measured it, but the same kernel had shown differences of about 100 us, and a lucky draw of the random input could hide a better method. He added that teams can exploit this simply by "re-submitting for a better draw" (RohanArun). For the record, the platform returns the earlier result for byte-identical files, and we never tried to fish for a better draw (see [check-launch-lock-wrapper.md](check-launch-lock-wrapper.md) for how a repeat looks).
- joseph asked what the automated lead got wrong. The author's answer: it keeps forgetting that most of the other kernel optimizations can be combined into the full decode kernels, and reminding it was a big breakthrough. This is the only concrete lever in the whole post. It is how our decode entry is built, and we ran the one open piece of it: porting the MoE byte cuts into the decode-layer MoE. Job 042deb1ca7a94df6 scored 2.090x against 2.072x and 2.059x for two runs of the base, and the target shapes were flat. Details in [pdl-cap-and-other-checks.md](../decode-layer/pdl-cap-and-other-checks.md).
- Pickler.fun asked whether the fleet shared kernels. The author said there were teams of three or more per leaderboard, who sometimes combined results, split work and messaged each other.
- vjeux replied with a link to his own hill-climbing setup (https://x.com/Vjeux/status/2115117930233171981): runs stalled and the final code was published. That code is the base of our attention work, so it adds nothing new.
- The remaining replies gave scores (prefill 1.69x, projections 1.53x, MoE 2.21x, a top-20 overall rank) or remarks, and named no technique.

We did not read the quoted earlier post about reaching second place in seven hours.

## Pull request 6: the items

The pull request adds a numbered list of four jump-start items to each of five starter prompts (attention, bf16, MoE, decode layer, prefill). Today's repository head has a fifth item in every kernel prompt, a pre-flight step that points to the free runs on the hosted queue. We machine-searched the four prompts for our problems, the research-procedure file and the setup template: none contains the word split or the string 12-bit.

Paraphrased items and what we found when we compared them with our entries (the prefill prompt is not one of our problems):

|Item|What it says|Our entry|
|-|-|-|
|Attention 1|Pre-allocate output and scratch buffers in prepare, no allocation in the decode step|The attention entry has no prepare step. Its workspace is cached, grows to a floor and is zeroed by the kernel. Timing is by CUDA graph replay, so allocation is outside the timed path (inference). Expected gain: none.|
|Attention 2|One fused pre-pass for RoPE, QK-norm and FP8 quantization before the KV write|Already how it works. Each CTA builds the block's new K and V (normalized, RoPE, FP8) and writes them. Another public entry uses the same design.|
|Attention 3|Align per-block loads along the head dimension to 64 or 128 byte segments|Overlaps the 128-byte bulk-copy idea we tested separately ([untested-ideas-results.md](../attention-extra/untested-ideas-results.md)). Not re-derived.|
|Attention 4|Replace if/else in the causal loop with predicated selects|Our branches are per block and uniform across a warp (inference). The sliding-window rows are our fastest because blocks outside the window are skipped, and predication would still read them. Negative.|
|bf16 1|Pre-allocate outputs and scratch in prepare|Nothing to gain under graph replay, same reasoning as attention 1.|
|bf16 2|Pre-transpose weights in prepare so reduction-axis loads are contiguous 128-byte lines|The prompt's own earlier text says a layout-only repack of one ordinary projection showed no gain and the unchanged router dominated. The large-m path is cuBLAS on the given weights, and a transposed view costs nothing (inference). The skinny path already reads our own 12-bit layout. Zero to negative.|
|bf16 3|Specialize M=1 into a streaming GEMV with register accumulation|We have it: a compressed m=1 GEMV, a skinny kernel for m up to 32, and a router read from the 12-bit stream.|
|bf16 4|Unroll statically and drop boundary masks when dimensions divide evenly|K per layer is already a constant. Removing masks in a memory-bound loop saves issue slots, not bytes. At most 0.2% (inference).|
|MoE 1|Pre-allocate routing tables, intermediates and outputs in prepare|Same reasoning as above.|
|MoE 2|Replace if/else on token ids, expert counts or slot masks with predication|Skipping empty experts saves work and predication gives that up. The prompt itself says to handle empty local contributions. Negative.|
|MoE 3|Decode NVFP4 words in interleaved stages (word, MMA, word, MMA) to lower register pressure|The premise fails: the main builds use 96 registers and no local-memory spill. The skinny loop is limited by the ALU pipe, and interleaving does not reduce the op count. At most 0.3% (inference).|
|MoE 4|Form silu(gate)*up in accumulator registers|We have it. The first phase fuses GEMM1 and SwiGLU.|
|Decode 1|Reuse one contiguous pre-allocated residual workspace across stages|The carried residual is T x 4096 x 4 bytes, at most 0.5 MB at T = 32, and sits in L2 (inference). There is no DRAM round trip to save.|
|Decode 2|Overlap independent stages, such as router projection and attention output projection, on side streams|Impossible here, since the router reads the output of the o projection and the first residual boundary. Another team measured no gain from side streams at T=1.|
|Decode 3|Fuse residual add and RMSNorm into the preceding projection's epilogue|Boundary work is already one fused kernel, with per-T variants timed in prepare. A full-row norm needs a cross-CTA finish, which we do with a release-acquire counter. A cluster-based boundary kernel for T of 17 and up gave no gain in another team's judge log. Our own recovery of the FP32-router cost at T of 8 and below was rated about +0.2% at best.|
|Decode 4|Keep routing on device, no `.item()`|Required for graph capture, and done. Our entry also keeps the router in FP32 at T of 8 and below, which the prompt never mentions.|

Counting the sixteen items on our four problems: eight are already done or give no gain at replay time (attention 1 and 2, bf16 1 and 3, MoE 1 and 4, decode 3 and 4); five are wrong or negative for us (attention 4, bf16 2, MoE 2, decode 1 and 2); two are at noise level (bf16 4, MoE 3); one overlaps another test of ours (attention 3).

## The other files at the repository head

- The research-procedure file reads like a decent checklist. In paraphrase: read the current problem list and leaderboard before spending anything; freeze the candidate and its hash before each remote attempt, and recover a launched job with a status call instead of relaunching; identical source can return an earlier result and is not a new measurement; judge a candidate only by the whole-problem number with the same timing definition in both arms, and treat a single-shape win, a sweep lead or a static gain (fewer registers, fewer instructions) as a hypothesis; before timing a variant, check that it changes the emitted code; compose verified gains as one new whole candidate instead of adding percentages; keep a promotion ledger of incumbent, challenger, hash, job and decision. We adopted two of these: hashing the SASS of a candidate against the incumbent before spending a check, and the ledger (see the launcher write-up).
- The setup template (blob f13926c9, 6079 bytes) tells the user to confirm the account before each remote attempt and to hash each file before a check. It also documents the hosted queue's interface.
- A login-flow file, a continuation prompt (blob a7395397, 1957 bytes) and an app-housekeeping file hold nothing technical. The README (blob d1bf4e7c, 32103 bytes) describes parallel working modes, cloud sandboxes, a benchmark suite and the free-runs promotion.
- A benchmark results file (blob 47775310) reports 30-minute single-trial runs of automated optimizers against the benchmark's own public baselines on a local H200. The best results were 2.654x on attention, 1.560x on bf16, 2.172x on decode layer and 2.377x on MoE. A calibrated-targets file lists reference speedups of 2.288x (attention), 1.389x (bf16), 2.165x (decode layer), 2.453x (MoE) and 1.569x (prefill), and is flagged as not distributed. These are not official challenge scores, and we could not verify that the benchmark baseline is SGLang, so we treat them as not comparable (inference). They contain no technique we could use.
- Not read: three generic policy files, the benchmark protocol and validation files, the candidate source files under the benchmark folder (other parties' code, produced by 30-minute attempts on the public starters), the Swift sources and the hosted queue's own page.

## The three bullets from a different post

A separate post of Nov 2 by @utterwqlut (https://x.com/utterwqlut/status/2117234769746657540), a first GPU kernel project at about 1.3x over SGLang, had three bullets that circulate with this material. They are not in the thread, the pull request or the prompts. Comparing each with our entries:

- Fusing the routing layers. On the MoE problem this does not apply: routing is an input there (the task statement hands in the expert lists and gates), and there is no router or top-k to fuse. The decode layer already has a fused pooler plus router kernel with a release-acquire finish, and the bf16 entry has a small-m router path.
- 12-bit compression for skinny matrices. We have this. The bf16 entry packs weights losslessly into 12 bits for the skinny kernels. Using 12-bit weights inside the m=128 GEMM was the experiment in [split-128-experiment.md](../bf16/split-128-experiment.md), and it lost.
- Sweeping Triton split-K kernel parameters. Our kernels are native CUDA, not Triton. The analogue is already swept: the router's K-split table came from a sweep, and the entry times its alternatives as CUDA graphs on the judge. The one reply with content on that post (from bitcoinduke) said that about 1.6x needs hand-written per-shape CUDA because split-K pays for an extra reduce launch. A new sweep at the scored sizes would tune to those sizes, and one check per problem cannot settle a sweep.

## What we took from it

- Nothing new on kernels. The only concrete lever (combine the component optimizations in the full decode kernels) is how our decode entry is already built, and its MoE byte-cut port measured net zero.
- Two process ideas were worth taking: gate each candidate on whether its emitted code differs from the incumbent's, and keep a promotion ledger.
- We still do not know where the 7.5 to 8% gap between our decode-layer entry and the leading entries comes from. Nothing in this material explains it.

## Limits

Engagement counts change over time, and the figures above are as read. We saw the replies as the site showed them to a signed-in reader. We read the pull request and files from their public pages and did not clone or run anything.

# Research notes

The working notes behind the kernels in this repo, cleaned up after the challenge. Each file starts with a one-line summary. Numbers come from the judge unless a note says "inference" or "assumption"; where our sources disagreed or a fact came second-hand, the note says so. Scores are for the state around the end of the kernel track (2026-11-05), and some files describe earlier states and are marked as such.

Start with `method/measuring-on-the-judge.md` for how to read a score and its noise, then the folder for your problem.

## method

How we measured on the judge, lessons, replay-race hazards, the rules as we read them.

|file|what it covers|
|-|-|
|[Challenge rules as we read them](method/challenge-rules-as-we-read-them.md)|A plain-words summary of the Muse Spark Open Kernel Challenge rules we worked under, with organizer guidance kept apart from our own interpretation.|
|[How the speedups happened](method/how-the-speedups-happened.md)|Where each of our board scores came from, what we wrote ourselves, what we borrowed, and what we tried afterward that did not move a score.|
|[Lessons learned](method/lessons-learned.md)|The technical mistakes and surprises from our kernel work that are most likely to save someone else time.|
|[Measuring on the judge](method/measuring-on-the-judge.md)|How the judge scores a run, how noisy that score is, and the habits that kept us from chasing noise.|
|[Replay race hazards in graph-captured kernels](method/replay-race-hazards.md)|A static review of the ways our four board kernels could produce wrong output only under CUDA-graph replay, written up as a guide to what to look for.|
|[What matters for the score](method/what-matters-for-the-score.md)|Which parts of each problem move its score, which problem is cheapest to climb on, and the optimization leads we ranked, including the ones we had to correct.|

## attention

Attention decode: cost model, gap to the leaders, ideas tried, launch-plan sweep, judge noise.

|file|what it covers|
|-|-|
|[Cost model and per-shape breakdown](attention/cost-model-and-breakdown.md)|Where the attention score comes from, shape by shape: the weights, the bounds we compared ourselves to, and a two-parameter model of the heavy full-attention shapes.|
|[What the gap to the leaders requires](attention/gap-to-the-leaders.md)|Arithmetic only: how much time each rank target would take off our 2.6211x, and which combinations of shape improvements could get there.|
|[How the top teams reach 3x on attention](attention/how-the-top-teams-reach-3x.md)|This file sets out what the public record and our own numbers can say about the teams above 3.0x on the attention board, with a clear line between what we know and what we are guessing.|
|[Ideas tried and what they returned](attention/ideas-tried-and-results.md)|This file lists every attention idea we measured or ruled out on paper, with its result, followed by our ranking of the headroom that is left.|
|[Judge runs and noise](attention/judge-runs-and-noise.md)|Identical code does not produce identical judge numbers, and this file records how far they move, what that hides from us, and how we read results because of it.|
|[Launch-plan sweep over all twelve shapes](attention/launch-plan-sweep.md)|One round of eight judge runs tested different launch-plan settings (split counts, workspace copies, finalize mode) on all twelve shapes at once, and this file has the per-shape results.|
|[Linear exp2, deeper pipelines and cold-code warming](attention/linear-exp2-and-cold-code-warming.md)|Our first two judge checks on top of vjeux's public entry (a cheaper exp2 with deeper pipelines, then cold-code warming) are written up here, along with how we validated kernels without a GPU and...|
|[Literature and public notes checked against the kernel](attention/literature-and-public-notes.md)|We read the main decode-attention papers and some public notes before spending judge runs, and this file records what each says about our kernel and what a public page for a neighbouring problem...|

## moe

MoE W4A16: the K-reordered weights with 6-bit scales, results, ideas tried and closed.

|file|what it covers|
|-|-|
|[muse_moe_w4a16: ideas we weighed and closed](moe/ideas-tried-and-closed.md)|Most of the MoE ideas we did not ship are listed here with the numbers or reasoning behind each, along with two open questions.|
|[muse_moe_w4a16: K-reordered weights and 6-bit scales](moe/kperm-and-6-bit-scales.md)|The wgmma build of the MoE kernel got three changes, K-reordered weights, 6-bit scales and warp-parallel stream-K ranges, and this file covers how they work, how we checked them without a GPU and...|
|[muse_moe_w4a16: judged runs](moe/results.md)|The judged runs that led to our submitted MoE file, from the vjeux baseline at 2.332x to 2.362x, with per-shape speedups and our reading of them.|

## decode-layer

Whole decoder layer: results, the FP32 router rule up to 8 tokens, the PDL failure, the MoE port.

|file|what it covers|
|-|-|
|[muse_decode_layer: a GEMV that read its input too early](decode-layer/cgemv-failure-under-pdl.md)|Our first decode-layer attempt, a compressed-weight GEMV for one-token projections, failed 3 of 13 tests on a programmatic dependent launch (PDL) bug, and this file records what happened and what...|
|[muse_decode_layer: FP32 router input up to 8 tokens](decode-layer/fp32-router-input-rule.md)|The vjeux decode-layer file rounded the router's input to BF16 at 8 tokens or fewer where SGLang keeps FP32; this file covers the rule, our fix, how we checked it without a GPU and what it cost.|
|[muse_decode_layer: the MoE port and the other ideas we weighed](decode-layer/moe-port-and-other-ideas.md)|Porting the MoE byte cuts into the decode layer turned out to be a wash, and this file covers the port, why we read it that way, and the other ideas we weighed with their numbers.|
|[muse_decode_layer: judged runs](decode-layer/results.md)|The decode-layer kernel scored between 2.059x and 2.105x in the passing runs below, which also show how much of the per-shape movement is noise.|

## bf16

Dense bf16 projections: weight packing, router and small-m kernels, large-m results, techniques and verdicts.

|file|what it covers|
|-|-|
|[Judge environment and validation risk](bf16/judge-environment-and-validation.md)|This file covers what we know about how muse_bf16_proj is timed, what the judge image should contain, which ready-made GPU libraries and DSLs are worth trying, and how to keep a kernel from...|
|[Judge results for muse_bf16_proj](bf16/judge-results.md)|This file lists every judged run we have a record of for muse_bf16_proj, with per-shape times for the runs of the file we submitted, the byte floors per shape, and a note on how noisy the numbers are.|
|[Large-m plan and results](bf16/large-m-plan-and-results.md)|This file covers our attempts to speed up the m = 128 and m = 256 verify loads on top of the vjeux file: why those shapes are the weak spot, the plan, five judge checks, and the results, which...|
|[Lossless weight packing for the BF16 projections](bf16/lossless-weight-packing-and-formats.md)|This file covers why a lossless 12-bit weight format pays on the decode shapes, how the public format works, where it loses time, and how to keep it exact on weights we have not seen.|
|[Router and small-m kernels](bf16/router-and-small-m-kernels.md)|This file covers the FP32 router, the small-batch dense layers (m up to 32), split-K reduction options, launch gaps, and what we learned about Triton's limits on these shapes.|
|[Techniques considered and verdicts](bf16/techniques-considered-and-verdicts.md)|This file lists every technique we looked at for muse_bf16_proj, with what the evidence says and where it ended up.|
|[Where the bf16_proj score comes from](bf16/where-the-score-comes-from.md)|This file covers the leaderboard shape for muse_bf16_proj, what each score level implies about bandwidth and fixed overhead per call, what separated our early 1.14x from the 1.5x entries, and what...|

## experiments

Experiment records: what we changed, how we measured it, the numbers and the verdict.

|file|what it covers|
|-|-|
|[Five more attention ideas and what became of them](experiments/attention-extra/untested-ideas-results.md)|This file records five attention ideas from a list of untried suggestions (one new judged check, one compiler test, one earlier check re-read and two desk assessments) with the numbers behind each...|
|[Attention experiment ledger](experiments/attention/experiment-ledger.md)|This file lists every attention idea we tested or ruled out, one row per experiment, with the job id, the numbers and the verdict.|
|[In-process timing lab for the flat attention kernels](experiments/attention/lab-ab-method-and-results.md)|This file describes the timing lab we built into an attention submission, what it measured on the judge's H200, and how the paired A/B arms it carried compared with the judge runs that followed.|
|[Latency chains in the small attention shapes](experiments/attention/latency-chain-study.md)|This file covers the batch-1 and sliding-window attention shapes, where the kernel is a chain of dependent memory round trips, and the experiments we ran on them.|
|[Twelve-bit weights inside the m=128 GEMM (the rs128 experiment)](experiments/bf16/split-128-experiment.md)|This file records two judged checks of a register-sourced wgmma kernel that reads our 12-bit weight packing for the 128-token bf16 projections, what the phase timers showed, and why we stopped...|
|[Raising the PDL cap in the decode layer, and how much identical runs differ](experiments/decode-layer/pdl-cap-and-other-checks.md)|This file records the judged checks of a higher programmatic-dependent-launch (PDL) token cap on the decode-layer problem, the follow-up that separated the change from run-to-run noise, and the...|
|[A locked, logged launcher for `okc check`](experiments/general/check-launch-lock-wrapper.md)|This file describes the small shell wrapper we put around `okc check` so that several people could launch judged checks at the same time without stepping on each other, what it enforces, and what...|
|[What a public post claimed, and what we could check](experiments/general/claims-in-a-public-post.md)|This file summarizes a public post about climbing the leaderboards, the pull request and starter files it points to, and where each claim stands against what we verified and against our own entries.|
|[Untested ideas for the MoE, bf16 and decode-layer problems](experiments/general/untested-ideas-moe-and-bf16.md)|This file records what we found when we went through a list of ideas nobody on the team had tried yet for the MoE, bf16 and decode-layer problems, with the evidence for each verdict.|
|[Dequant, scheduling and finalize variants for the MoE kernel](experiments/moe/dequant-and-scheduling-variants.md)|This file records each change we tried on the W4A16 MoE kernel, what we expected, what the judge showed, and a bug that cost us one check.|
|[MoE experiment ledger](experiments/moe/experiment-ledger.md)|This file lists every MoE experiment we ran or closed, one row per experiment, with job ids, numbers and verdicts.|
|[Where the MoE kernel's time goes](experiments/moe/fixed-cost-anatomy.md)|This file collects the diagnostic measurements we took on the W4A16 MoE problem: the per-launch timeline of each build, the fits that separate bytes from fixed cost, the small tuning arms, and how...|

## headroom-and-top-teams

How much room each board has, what the public record says about the top teams, ideas from one public repo.

|file|what it covers|
|-|-|
|[Headroom model for all four boards](headroom-and-top-teams/headroom-model-all-boards.md)|This file compares the measured per-shape times of our four board entries against estimated practical bounds, to show where a kernel could still get faster and what one shape is worth to the score.|
|[Ideas from the keell0renz repo](headroom-and-top-teams/ideas-from-keell0renz-repo.md)|This file summarizes what we learned from reading keell0renz's public repo (https://github.com/keell0renz/muse-open-kernel-challenge) and ranks the ideas from it that we might reuse.|
|[Research gaps and answers](headroom-and-top-teams/research-gaps-and-answers.md)|This file lists the open questions we chased in public sources, what each one turned up, a summary of a third-party note about isolated kernel wins that died in the full decode layer, and a short...|
|[What the public record shows about the top attention entries](headroom-and-top-teams/top-teams-public-record.md)|This file records what we could find in public about how the leading muse_attention_decode entries reach about 3x (not much that is direct), the board facts we could verify, and some arithmetic on...|

## literature

Papers and vendor documents read, what each changed, and the ideas ledger.

|file|what it covers|
|-|-|
|[Attention papers and FP8-KV decode on an H200](literature/attention-papers.md)|This file covers what the attention papers say that matters for the muse_attention_decode board, which of their mechanisms our entry already has, and how we priced the one softmax idea the papers...|
|[NVIDIA documentation and the replay hazards other teams reported](literature/cuda-docs-and-replay-hazards.md)|This file records what the PTX ISA, the CUDA Programming Guide, the CUTLASS pipeline page and the FlashAttention-4 blog say about programmatic dependent launch, async-proxy fences, setmaxnreg and...|
|[GEMM, MoE and weight-compression sources](literature/gemm-moe-and-compression-papers.md)|This file covers Stream-K, MARLIN, Machete, the PyTorch column-major MoE blog, DFloat11, the fused-Huffman GEMV repo and ZipServ, and what each means for the 4-bit MoE and BF16 projection boards...|
|[Ideas ledger](literature/ideas-ledger.md)|This file lists the optimisation ideas we priced against our four entries (attention, bf16, moe, decode_layer), what we checked for each, and the verdict, including every negative result.|
|[Papers read and what they changed](literature/papers-read-and-what-they-changed.md)|This file lists the papers, vendor articles and library documents we read for the four kernel boards, how deeply we read each one, and whether it changed anything in our entries.|
|[Automated-tuning papers, a tuning thesis, and a plan for searching per-range settings](literature/tuning-and-automation-papers.md)|This file covers the Astra paper, the Minimal Executable Programs preprint, a Leiden thesis on tuning parameters, the Kernel Tuner documentation, and a plan for testing several shape ranges in one...|

## surveys

Surveys of public solution repos, forum and social notes, and how deeply each source was read.

|file|what it covers|
|-|-|
|[alygg77 repository digest](surveys/alygg77-repo-digest.md)|This file digests the public alygg77/okc-muse-spark-kernels repo at commit 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0: what its kernels change against the vjeux base, how its judge runs compare with...|
|[Community and forum notes](surveys/community-and-forum-notes.md)|This file summarises what other teams posted on the challenge's community forum and on public pages (the challenge site, X, Hacker News, GitHub) during the last two days of the kernel track, and...|
|[Survey of public solution repos](surveys/public-repos-survey.md)|This file records what we found in other teams' public repositories for the four muse boards: how we found them, where they sat on the leaderboards, and one section per repo with its claimed...|
|[Claims other teams posted on X, traced to what is public](surveys/social-post-claims.md)|This file traces the performance claims and technique outlines that other teams posted on X, with what we could verify, what we could not, and what each means for our four boards.|
|[Source coverage](surveys/source-coverage.md)|This file lists every source we collected for the four muse boards, how deeply we read it, how far we trusted it, and what we never read.|
|[What the top attention scores imply](surveys/top-board-arithmetic.md)|This file works through the attention score arithmetic to see what the leading 3.1x to 3.3x entries must be doing differently from our 2.605x, and whether anything public lets us close that gap.|

## other-teams

Notes on other teams' public repos and two blog posts (ideas only, no code copied).

|file|what it covers|
|-|-|
|[alygg77/okc-muse-spark-kernels: what the repo shows](other-teams/alygg77-okc-muse-spark-kernels.md)|Our K permutation and our FP32 router fix both started from alygg77's public repo (19 files, Apache-2.0, commit 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0), and this file covers what they changed on...|
|[Two blog posts people pointed us at](other-teams/blog-posts.md)|Two public posts were sent to us during the challenge, and this file says what each claims and whether either helps a kernel score.|
|[lee101/openkernels: what the repo holds](other-teams/lee101-openkernels.md)|The public repo at https://github.com/lee101/openkernels stopped at judge scores below ours, and this file records its numbers, how it was built, the measurements worth keeping, two compliance...|
|[Findings from other teams, by team](other-teams/scattered-findings.md)|Other teams' public repos and write-ups told us the things in this table, each with a pointer to where it shows up in our notes.|

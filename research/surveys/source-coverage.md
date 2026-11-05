# Source coverage

This file lists every source we collected for the four muse boards, how deeply we read it, how far we trusted it, and what we never read.

We did not read everything to full depth, and the tables below say so. Where two of our passes disagree about depth, we say that too.

## How much we trusted each kind of source

The most useful single lesson was that only repos with judge reruns earn trust, and social posts do not. Public board rows keep each team's best draw, which sits about one standard deviation above reruns, so a board row alone is weak evidence.

|Source|What it claims|Reliability|Relevance to us|
|-|-|-|-|
|Live rules page and the problem listing|The allowed and banned list; run limits|High, primary|Binds all work|
|keell0renz repo and its graphic|Reruns: decode 2.2641 (check), 2.2532 (submit); moe 2.3884; attention screens 2.63 to 2.66 (row 2.7168 a high draw); bf16 1.5742 (check), 1.5690 (submit)|High for scores; medium to low for process (no per-step gains, no logs)|Source of decode and moe stack ideas. Its decode layer is built on alygg77's base, not vjeux's|
|alygg77 documents|Judge-measured negatives; KPERM3 +1.25%; d32s +3.1% and v16s +5.2% per shape; v16f consumer loop 40.5% exp|High on measurements. Its bf16 file embeds zlib and base64 source, which is banned|Ideas only, never code|
|NVIDIA PTX ISA, CUDA guide, FA4 blog|PDL, proxy fence, setmaxnreg and mbarrier rules|High. Cannot explain a 1-in-10,000 failure rate. FA4 targets Blackwell|Backs our correctness audits|
|Seven attention papers|Speedups on A100 and H100 at prefill-size shapes|High as papers, but none measures head_dim 64, 16 query over 2 KV heads, page size 1, FP8 KV or H200|Low. Every usable mechanism is in the entry|
|Stream-K, MARLIN, Machete, the serving library's FP4 Marlin path|Even K split; a 4-bit Ampere pipeline; wgmma needed at batch 128+|High. No skinny, grouped or Hopper-FP4 data|Low|
|PyTorch MoE blog and a rank-175-to-80 post|SplitK +18 to 20%; column-major 2.7 to 4.4x|Medium. FP16 Triton on Mixtral; the poster is unidentified|None. About M/32 tokens per expert means one M-tile per expert|
|DFloat11, lszxb, ZipServ, the Reddit thread|About 11 bits per weight; fused GEMV at 80 to 90%; an 11.3-bit fixed code|High as papers. DFloat11 does not fuse decode (about 2x slower at batch 1)|None. Our 12-bit form is the cheap variant|
|Astra, Minimal Executable Programs, Leiden thesis|Automated tuning of small SGLang kernels (1.32x, H100); a loop on PolyBench (5.29x); parameter sensitivity on A100 and AMD|Real, but off-target hardware and kernel class|None for scores|
|Anshul's router post|Router 10.3 to 5.1 us (m = 1), 26.3 to 11.9 us (m = 8); total 1.055x|Medium to high: the only post with an official-check screenshot|None. Ours is 3.84 and 5.19 us|
|RohanArun's posts, PR #6, getsupers|About +5% from 2,000 experiments; a Lean pruner; a GPU queue; board scores above ours|Low: no code or mechanism; PR #6 is +35 generic lines, some wrong|None; getsupers uploads code to a third party|
|Aditya, Nivas, Karthikeyan, Dhruva|Outlines of bf16 and moe wins|Low: no repo, no screenshot|Aditya's gap decomposes into rs128. MULDEC unknown. The other two sit below ours|
|Community forum threads (42, 120, 154)|Fence, refill and replay hazards; cross-stage PDL gains zero|Medium: peer claims without a primary source; the PTX text confirms the rules in principle|Origin of our hazard rules|
|Note on isolated wins that died in the full layer|cgemv failed in three forms|Medium; matches our own check (10 of 13 shapes failed)|Closed rule|
|Weight-space blog|Editing network weights without training|Not applicable|None|
|Earlier generic pastes and links (CUDA checklist, Triton guide, fused softmax part 2, Modular handbook, ENCCS reduction, KernelFoundry, Cursor's blog, 2603.29010)|Generic tuning advice; shape-specialised BF16 GEMM up to 9% over cuBLAS at small M (B200)|Triton guide unreliable (its tl.cache_read, tl.shared_memory and tl.barrier do not exist in Triton 3.9). The rest low or none. Cursor's kernel is Blackwell-only|None for scores. Cursor's per-shape tuning meets the nearby-shape rerun, so keep general fallbacks|
|Public competitor repos (vjeux, NathenFernandes, trxphxle, Kushagrabainsla, thethings, PostTrain)|Base kernels; bf16 v18 (1.521x scored); a non-kernel FINDINGS table; cold-code warming; 128-byte row gather 3.7 against 2.3 TB/s at 64 bytes|vjeux: Apache-2.0 base. Kushagrabainsla: no licence, ideas only. thethings and PostTrain were saved and not re-read for these notes|Already absorbed|

Two facts to keep straight. The name Astra in the weight-space blog and in RohanArun's posts is a model name, not the Astra kernel-tuning paper 2509.07506 (the link between the blog and Rohan is our inference). And the keell0renz decode layer sits on alygg77's code, so most of its roughly 8% gap to our decode entry is alygg77's work (+5.2%) plus their own stack (+2.6%).

## Papers and documents

The paper-by-paper depth table is in papers-read-and-what-they-changed.md. In short: the kernel papers were read at method or section depth; about 25 non-kernel papers were skimmed; figure-only numbers were never extracted; two items (arXiv 2610.07334 and 2603.05451) were not read.

One conflict we could not resolve: an earlier pass reports reading the Leiden thesis in full and sections 3 to 5 of the Minimal Executable Programs preprint, while a later ledger still lists both as not read and says the preprint body was behind a bot check. We used the earlier pass's numbers.

## Repositories and notes

|Source|How we read it|What we did not read|
|-|-|-|
|alygg77|README, RESULTS, results.json, both docs, the index text file, timing kit READMEs, NOTICE and licences in full. Kernels as a diff against vjeux's f35a41e. Tree at commit 6cae49a6 has 19 files|Kernel files were not studied beyond the diff|
|keell0renz|README, LESSONS, NOTICE, commit list (3 commits, no notes)|Kernel files, deliberately|
|Kushagrabainsla|An excerpt of EXPERIMENTS.md (experiments 15 and 16 omitted), repository list|Code|
|trxphxle|README and FINDINGS.md (55,556 characters; the first pass stopped at 50,000, a later fetch read the tail, whose last few hundred characters were clipped but listed only non-kernel repos)|Its code|
|bemlerlabs, chenwainuo|Markdown files through shallow clones: README, handover notes, briefs, a 314 KB log in part|Kernels. the bemlerlabs repo later returned 404, so we could not read more|
|Madhavan113|Notes folder only|Kernel source|
|ruwwww|README, file tree, git blob hashes against vjeux's files, the first 200 lines of the decode file diffed, header and helper lines of the bf16 file, the judge profile|The rest of the bf16 file|
|AutoclickerI|README, parts of the 371 KB FINDINGS (attention section)|The 63-row closed-recipes table, SOTA.md, 513 judge JSON files|
|infraax|RESULTS, HYPOTHESES, LESSONS|Kernel source (not in the part of the tree we listed)|
|ozgureyilmaz, Ranoobaba, tommyyliu|READMEs and notes (Ranoobaba's LEARNINGS in full)|Code|
|hieptran1812 blog|Directory listing, headings of two posts, one section of the judge post and one of the layer post|setting-up-the-lab, roofline, Muse Spark for kernel engineers, the-open-kernel-challenge-decoded in full|
|NathenFernandes|Tree, README, ranking history|The two JSON files, expand_source.py|
|Small repos (theluckystrike, GiulianoIntelligence, KingCobra52, Mithun-033, xharshvardhan, lilaclilac09, Reedo3, nibague, 4gatepylon)|README or tree level|Everything else|
|Not opened at all|ZhimaoL's sglang fork (branches deleted Nov 1), gauravjain14's two kernel repos, operatorstack source, lee101 kernel source, ozgureyilmaz's combined README figure||

GitHub search coverage is described in public-repos-survey.md. The main limits: files over 384 KB are not indexed, private repos are invisible, and GitHub serves about 5 pages per code query.

## Community sources

The forum copies we saved covered 14 of 36 threads, read in full as JSON. Threads 18, 36 and 105 were never saved, so claims quoted from them are second-hand. On the last morning we re-read the feed and the one active thread (120), but read only the first 420 to 1,000 characters of each new reply. Older threads (105, 18, 49, 108, 4) were not reopened. The X posts were read on the pages, with replies only partly loadable and often only the first screenfuls. We did not read Reddit. A hosted-queue page and a few repo pages were read as one-time snapshots.

## What no note could verify

- No mechanism is public for MULDEC, for better NVFP4 dequantisation (Karthikeyan), for Nivas's 12-bit variant, for Aditya's m = 128 and 256 kernel and tuner, for Dhruva's routing fusion, or for Rohan's Lean pruner and surrogate model. The 5% hyper-parameter-search claim cannot be traced to any shape or board.
- The keell0renz repo has no per-step gains, no logs, no bake output showing which bf16 cell uses rs128 and no confirmed router-input precision. Whether the graphic posted for their team is theirs is open.
- No attention-paper number was measured on an H200 by us. Our headroom bounds rest on assumptions (4.1 TB/s practical, 1.4 us fixed, 3.0 us floor, the last quoted second-hand from the forum) and are upper bounds.
- Tile counts per CTA in the moe kernel were unmeasured, so the Stream-K collaborator cap has no evidence. The PyTorch blog's tile-index bug is a reading of the source, not a run.
- Whether the judge runs two jobs on one problem concurrently is unknown.
- Whether ex2.approx.f16x2 issues as one MUFU op was unknown until our SASS check showed two.
- Two note conflicts were settled toward the note that actually read the files: cuBLASLt pins (the bf16 probe over the other-teams note) and conditional rescale (the attention-paper read of the code over the saved-references tag that said untested).

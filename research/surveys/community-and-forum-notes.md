# Community and forum notes

This file summarises what other teams posted on the challenge's community forum and on public pages (the challenge site, X, Hacker News, GitHub) during the last two days of the kernel track, and what, if anything, it changed for us.

Everything here is other teams' text or our own reading of public pages, and none of the performance claims was reproduced by us. We read the forum and did not post, reply or vote. Where a claim is verified against the cited source we say so. Everything else is a team's own report.

## The forum at a glance

On the first read we counted 36 unique threads across the top, hot and new feeds, and 15 active teams in 24 hours (Lurrobert, prateekm1007, aaln, nikebp, gaidar0yegor, arkintof, SatyamKluisz, QuantIntellect, CapCap, arashattar, haian1910, 2018kguo, elmatonex and us). Several posters sat above us on the boards: Lurrobert (overall #33, attention 2.84) and prateekm1007 (bf16 #44). Our own introduction, thread 156, had no replies on either read. Offers of free or dedicated H200 time (rohanarun's getsupers, CapCap and others) came up repeatedly. We did not use them, because that means uploading our code to a third party.

## Judge mechanics other teams verified

- Thread 93 (prateekm1007, checked against the reference file with line numbers). The layer boards (decode_layer and prefill) grade the increment, stream minus carried, so a local test that compares stream directly is too lenient. Boards containing attention use an adaptive bound, min(cap, max(floor, factor x e4m3_err + slack)), with REL_L2_TOL 0.15, factor 2.5, floor 0.03, and slack 0.015 on attention only. On bf16 the router is checked at 1e-4 and the probe rows at 1e-3. aaln corrected one claim: the probes are a canary for weight precision and indexing, and a correct split-K passes them. The MoE path that skips checking is unreachable in scored runs, so a MoE failure is always real.
- Thread 11 (Lurrobert). The judge calls kernel_fn on 2 to 4 clones of the prepared state. Never bake raw addresses, descriptors or captured graphs in prepare().
- Thread 12 (Lurrobert). An H200 sits at about 700 W and 1.665 to 1.725 GHz under GEMM load (emeraldant measured 1.70 to 1.79 GHz inside MoE GEMMs). Plan floors at about 900 TF/s and 1.75 GHz, not at the spec sheet.
- Prepare and build budget. VkaZas (thread 125) had two judge runs killed with a message that prepare and build had consumed the benchmark budget: 215 s of base prepare plus 123 s of attention builds plus 98.5 s of dense builds is 437 s. preonxtrun reports about 48 to 54 s of one-time initialisation on the first call of the WGS path, on top of compile time. alygg77 had one MoE file take 273 to 312 s of prepare against the 300 s limit.
- Noise mining. nikebp says the judge dedupes on a comment-normalised content hash and that the best of the 10 scored runs a day counts, and describes a repeatable method for mining the noise band (+0.86% on their prefill row). We did not do this. Resubmitting an unchanged file to catch a lucky draw reads as gaming, and final ranks come from reruns.

## Replay-only correctness bugs

Thread 42 (jakozaur, later adopted by prateekm1007) described three failures that appear only in graph replays: mbarrier memory reuse in Gluon warp specialisation, a missing fence.proxy.async before the empty-barrier arrive in hand-written cp.async.bulk rings (1 to 19% of launches wrong at some stage counts), and a float red.add into a self-zeroing workspace that gave wrong tokens about once in 10,000 to 20,000 replays. The full account, with our reading of the NVIDIA documentation and the soak arithmetic, is in cuda-docs-and-replay-hazards.md. Our attention merge uses float red.v4.f32 atomics and our MoE uses counters in the same pattern, and we cannot soak without a GPU here, so we noted the exposure.

## A local A/B is a filter, not a verdict

gaidar0yegor (thread 147): a local cold-L2 A/B predicted +6.6% for a re-tuned bf16 dispatcher, and the judge returned -5%. They also found that a CUTLASS route for the decode_layer o projection gives no gain (and flashinfer headers may be missing on the judge image), and that raising the MoE MID_MAX_M from 16 to 32 gives none (5,624.5 to 5,700.0 us, inside noise).

## decode_layer, per stage

From VkaZas and kiro (thread 120, the alygg77 lineage), taken from an H200 judge profile of a sliding layer.

|T|Whole step|Stage times in us|
|-|-|-|
|1|70.8 us|MoE 22.4, fused pooler plus router 20.0, qkv 9.0, o plus first boundary 6.5, attention 5.0, unpooler slice 3.5, route 2.6, final boundary 1.9. No kernel overlap is visible|
|8|not given|MoE 57.8, pooler plus router 19.0, qkv 9.3, attention 7.2, o 5.9, unpooler slice 5.4, cluster boundary 3.2, route 1.9, final boundary 1.8|

- Their three small changes on that lineage: pinning prepare's choice for 33 <= T <= 128 (the tuner picked a losing variant), a cgemv for the T = 1 unpooler, and a cgemv for the T = 1 qkv. They say the pinning was worth more than any kernel edit that day. The cgemv gains were about 2.7 us on qkv and 1.3 us on the unpooler slice.
- SatyamKluisz, H100 Nsight Compute at T = 1: qkv at 61% of HBM peak, pooler plus router 66%, o 43%, unpooler 30%, MoE 42%. The step averages 58%, so the headroom sits at the stage boundaries.
- VkaZas: cross-stage PDL prestreaming gains zero or worse. An extra PDL-chained boundary costs 0.4 to 0.7 us (trigger early) or 0.83 to 1.5 us (late). grid.sync costs about 1 us, so megakernel fusing nets zero. Folding the final boundary into the last-CTA epilogue was 30 to 120 us slower.
- Sudhendra (thread 114, decode end to end): L2 prefetch from a graph side branch lost 0 to 18%, matching our closed list.

## MoE

- VkaZas (thread 120). Per distinct local expert: 27.0 MiB, a DRAM floor of 6.362 us at 4.45 TB/s, their time 8.212 us per expert (R-squared 0.99), SGLang 23.23. An NVFP4 dequant lower bound is about 3.94 us. About 1.85 us per expert is exposed. The traced mechanism: the consumer waits for W-full, then X-full. The phase-B X producer waits on doneAs, so a held W slot returns no credit and stalls the weight producer. The bubble is about 0.764 + 0.153 x (maxtok / meantok) us per expert. Their lowest-risk arm is to publish phase A's partial earlier; it was not yet measured.
- aaln (thread 23): compile-time ring-stage index specialisation gave +1.57% (overlapping intervals). nikebp adds that on a 96-register base ptxas folds the constants into wgmma descriptor immediates and rematerialises ops per stage, so it fails on register-tight bases.
- preonxtrun (thread 125): a flush-warp redesign of release, count and read-back lost 9%. A SLIM 5-bit scale compression gave a ratio of about 1.00x.

## Attention

- aaln and luna-research (thread 48) report per-case numbers worse than ours: verify full at batch 16 and 32 at 163 and 314 us (aaln's diagnostic 113 and 235), decode-1 sliding 9.7 or 6.0 us. Ours are 101 and 184 us and 4.9 us.
- arashattar (thread 154): FMA-emulated exp2 in the flat kernel costs +24 to +105 us. A cold instruction cache: 32 KB of straight-line code costs about +1 us. Modules with identical SASS differ by plus or minus 0.06 to 0.09 us. Sliding kernels read 0.3 to 0.8 us slower on the judge than on Modal.
- VkaZas (thread 125): an uneven tile-to-CTA partition left 112 of 132 CTAs effective on one sliding path (33 tiles partitioned as 5, 5, 5, 5, 5, 5, 3, 0). Random page-table reads top out at about 4.05 to 4.1 TB/s against about 4.46 TB/s sequential. The producer-share probe is blocked because setmaxnreg needs a full 128-thread warpgroup.

## The last morning, Nov 5

We re-read the public pages after the final submissions.

Challenge site.

- The updates page was identical to the Nov 4 snapshot. Its newest entry was still Nov 3, 3:45 pm PT. There was no new notice, no deadline change and no removal or disqualification notice.
- The rules page text hash (whitespace-normalised sha256, prefix 5d5260ce3c994144) was equal to the Nov 4 snapshot, so no rule had changed since the Nov 3 12:15 pm PT guidance.
- The problem pages were unchanged apart from a leaderboard block that now renders.
- The home page still showed the kernel track as Oct 22, 11 am to Nov 4, 11:59 pm PT, and the serving track as Nov 10, 9 am to Nov 19, 11:59 pm PT.
- Top of the boards: bf16 1.73x (gogowhy), attention 3.34x (pjmullin3), moe 2.79x (pjmullin3), decode_layer 2.58x (gogowhy), prefill 2.15x (LeshengJin). On Nov 3 these read 1.69, 3.34, 2.76, 2.51 and 2.25. The bf16 top ten were all 1.67x to 1.73x.

Forum. The feed listed 30 threads and the highest id was still 156 (ours, with 0 comments), so nothing new had been created. Only thread 120 (SatyamKluisz and kiro, decode layer at T <= 8) had new activity: 18 comments, 6 more than in our earlier save, all from adityakm24 and prateekm1007. We read the first 420 to 1,000 characters of each.

- adityakm24 (decode lineage, 2.317x best draw). (1) A lab that calls generate_input with local_start = 0 and the default router runs the MoE 8 to 23 us too light, because the judge sets local_start per input and a router seed per run. Fixing that reversed their verdicts (cuBLAS plan pins +0.12%, which is neutral). (2) A rare prepare() crash in the alygg77-lineage compress path: a tiny weight fails the wdec + delta == wtrue check, the function returns None, and a KeyError on 'pooler' follows. (3) Negatives on true routing: the WGS table is flatter at T > 128 (ratio 1.023 to 1.034) and steeper at 1.005 to 1.012; MID at M <= 32 gives 1.06 to 1.09; MAIN at M <= 16 gives 1.04 to 1.05; pre-touching pages gives 0.999 to 1.018, so there is no TLB cost. (4) A small MoE-board win over 3 paired checks: M = 8 WSBLK 3, M = 16 WSBLK 4 with NSF 2, and T <= 32 tables x1.5 gave +0.44% mean and decode-32 +1.35%. Tables x2 gave nothing more.
- adityakm24 also posted per-case decode-layer medians over 9 judge submits of one version: decode 1 full 71.0 us, d1 sliding 64.3, d16 full 199.8, d16 sliding 124.4, d32 full 299.3, d32 sliding 146.4, verify 1 full 111.4, v1 sliding 101.5, v16 full 325.6 (the rest was cut off). It is useful only as a comparison row.
- prateekm1007 had no before-and-after yet for their release-wait fixes (publish A's partial earlier, decouple the W slot return). Their wait-skew sweep shows route spread of 6 to 10%, flat in M, with release-wait skew from 1.6x to 16.5x median to max, and the max-wait CTA burning 26 to 31% of its phase-1 compute spinning at M >= 128. Lab-only (not judge) toggles for the NVFP4 K-permutation and scale sharing gave +3.9 to +4.8% at stage level on the moe stage.
- Thread 123 (nikebp) asked for 2 to 3 hours of an H200 for a 2-CTA cluster prefill attention kernel. Thread 10 (compute offers, pinned) had nothing new.

X. We found no organiser post about the challenge after the Oct 22 launch tweet, and no deadline extension. A third-party recap about 13 hours old restated the Nov 3 notices (approximate BF16-accurate math, BF16 or FP16 partial sums and FP16 attention accumulation allowed, serving Nov 10 to 19, the serving board hidden, maintainers judging patch quality) and said kernel submissions close that night at 11:59 pm PT, matching ours.

- RohanArun, 2 hours old: his team (getsupers) was overall #38, and he said the top 100 all clear 2.30x over SGLang. His process notes: a genetic algorithm with a 0.05% minimum per advance, 3 confirmation runs on the H200 above noise, Nsight plus a Lean verifier to guide candidates, and a small statistical model to rank candidates before benchmarking. No kernel technique.
- Arthur (@arthur_hyper88), 1 hour old: decode layer 2.27x, #190 overall, 18 ideas through the gates on the last night, all 18 rejected, using a same-run control, an A/A no-op test and a stale-data replay. Process only.
- Other posts (a first-iteration bf16 at 1.13x, a 1.33x bf16) were noise. An article on load-balanced compression for DeepSeek throughput on B300 (@dhruvmt2) was re-shared. It is a different Dhruv from the Nov 2 post noted elsewhere, concerns a different workload, and we read only the snippet.

Other pages. Hacker News had a single item, titled Muse Spark Open Kernel Challenge with $100k in prizes (posted by ckraeuter, 4 points, 0 comments, 13 days old). vjeux/okc-2026 had no commit after Oct 28 (f35a41e is still head) and one issue. A GitHub search sorted by update showed KingCobra52 (3 hours, a starter-level repo), zimo0110 (22 hours, a moe starter), AutoclickerI, Mithun-033, then keell0renz (2 days), ruwwww (3), Kushagrabainsla (4), trxphxle (4) and alygg77 (6). No top-team repo changed in the last 12 hours, and bemlerlabs/okc-2026 now returned 404. We did not get to read Reddit, and a web search for the challenge returned only GitHub pages, so concluding that nothing notable was posted there is an inference, not a verified result. We did not open GPU MODE.

## What it changed for us

No rule or deadline changed, and no technique found was above the noise bar.

1. adityakm24's MoE retune (+0.44%, 3 paired checks) is inside our single-run noise of about 0.8%, and it is tuned to specific M values. Our dispatch rule is contiguous ranges only. Not worth a final check.
2. prateekm1007's K-permutation and scale-share toggles: we already carry the scale-sharing K-permutation and 6-bit scales, and our v4 MoE variant measured net zero on decode_layer.
3. adityakm24's lab-versus-judge routing note matters only if our own A/B harness used the default router seed and local_start = 0. That claim comes from the judge's reference per the post and is unverified.
4. The rare prepare() crash only matters for code descending from the alygg77 compress path. We never borrowed from it.
5. The negatives match our closed list: L2 prefetch from a side branch (0 to -18%), pre-touching pages, cross-stage PDL.

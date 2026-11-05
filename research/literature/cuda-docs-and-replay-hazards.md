# NVIDIA documentation and the replay hazards other teams reported

This file records what the PTX ISA, the CUDA Programming Guide, the CUTLASS pipeline page and the FlashAttention-4 blog say about programmatic dependent launch, async-proxy fences, setmaxnreg and atomics, set against the correctness hazards that other teams reported on the community forum.

The competitors' technique claims rest on their own measurements, not on cited literature. The only literature that can confirm or contradict them is NVIDIA's documentation plus a handful of classic papers, so we read the documentation. The result: NVIDIA's text confirms the three hazard rules in principle, adds some qualifications, and cannot explain a 1-in-10,000 failure rate. No document we read says a competitor technique is wrong, and nothing breaks a live rule.

We read PTX ISA 9.4 (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), CUDA Programming Guide chapters 4.5, 4.12, 4.2 and 5.7 (https://docs.nvidia.com/cuda/cuda-programming-guide/), the CUTLASS pipeline page (https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html), the FA3 paper body and the FA4 blog. We did not read the PTX memory-consistency chapters (proxy definitions, causality order), only the instruction pages and the Async Proxy subsection.

## Programmatic dependent launch (griddepcontrol)

- The wait modifier blocks the thread until all prerequisite grids in flight have completed and their memory operations are performed and visible to the current grid.
- The launch_dependents modifier lets the runtime schedule dependents once every CTA of the grid has issued it or finished. The dependent may start before the primary completes, with no guarantee that it does.
- A release fence that precedes launch_dependents in program order synchronises with the start of the dependent grid when both share a memory synchronisation domain and the fence scope is gpu, or when the scope is sys. So data written before a fence-plus-trigger is documented as visible at dependent start, before any wait.
- The page also says that when the prerequisite grid uses launch_dependents, the dependent must use wait for correct functional execution.
- The programming guide says secondary blocks may launch before the primary's data is visible, so the secondary must call the grid-dependency synchronise (the runtime name of the wait) before using primary results. If the primary never triggers, the trigger happens implicitly when its blocks exit. Overlap is opportunistic, and relying on concurrent execution is unsafe and can deadlock. The suggested pre-wait work is independent preamble such as constant loads or buffer zeroing.
- In CUDA graphs PDL is a programmatic kernel-to-kernel edge, produced by stream capture of a launch with the programmatic-serialisation attribute.
- ld.global.nc is described only as a load through the non-coherent read-only cache, possibly higher bandwidth and longer latency. The page says nothing about coherence with concurrent writers.

The documents say nothing about performance. Two competitor numbers on trigger placement therefore stay uncontradicted and unconfirmed: one team measured +0.9% on the layer from triggering right after the producer's last weight issue, and another measured early triggers at most 0.736 us per real kernel pair, with two pre-wait stages optimal and four or six slower.

Our reading. Early triggering and early launch are legal as long as every read of data the producer writes after the trigger comes after the wait. Reading producer output before the wait is the unsafe case, which is exactly why our earlier hoisting attempt on the decode layer failed (it hoisted loads above the wait). The generic rule explains the failure. The non-coherent load path adds an undocumented coherence question but is not the root cause. A documented legal route to hoist some producer data above the wait does exist (write, release fence at gpu scope, then trigger), but it costs a fence per CTA, and a gpu-scope fence was measured at about 0.13 us each in a MoE build. The documents do not say whether a wait in the third kernel of a chain also covers the first kernel's writes when the second kernel skipped its own wait.

## Async proxy, fences and mbarriers

- cp.async.bulk and its reduce variant run in the async proxy. Touching the same location through two proxies needs a cross-proxy fence, and fence.proxy.async is the tool. Completion of a bulk copy is followed by an implicit generic-to-async proxy fence, so a copy's result is visible to ordinary loads once completion is observed.
- Proxy fences are bidirectional per proxy kind, take effect inside a single thread (no scope qualifier), and need sm_90 or higher.
- mbarrier.arrive defaults to release semantics at CTA scope. The page describes no proxy-fence effect of an arrive.
- The programming guide's TMA example orders shared-memory writes by ordinary threads before a bulk copy by having every thread run the proxy fence and then a block-wide barrier before thread 0 issues the copy. It shows only the generic-write-then-async-read direction, not read-then-refill.
- cp.async.bulk completes via complete-tx on the mbarrier with release semantics at cluster scope. The guide advises few, large bulk copies.
- mbarrier.inval: a barrier must be invalidated before its memory is used for anything else, and any mbarrier operation other than init on memory that does not hold a valid barrier is undefined.
- wgmma operations run in the async proxy and their completion is followed by an implicit generic-to-async fence. The guide models cp.async (LDGSTS) as async thread operations, not async-proxy operations.
- The CUTLASS pipeline page describes producer_acquire as blocking and consumer_release as non-blocking. The paragraphs we extracted do not mention a proxy fence.

Where the read-then-refill hazard can occur (inference from the text above). Only when the refill is a cp.async.bulk and the consumer read that stage with generic-proxy loads (LDS, ldmatrix, ld.shared). Stages consumed only by wgmma are read in the async proxy, whose completion carries an implicit fence, so they should not need an extra read-then-refill fence. cp.async.cg refills are async-thread operations, not async proxy. Because the proxy fence takes effect inside a single thread, each thread that did generic reads should execute it before signalling the empty barrier. That last point is our reading of the per-thread wording plus the guide's all-threads-fence pattern, not a quoted rule.

By grep on the vjeux base files (idea-level counts, not traces): cp.async.bulk appears 1 time in attention, 8 in the MoE, 4 in bf16 and 11 in the decode layer. fence.proxy.async lines: MoE 7, attention 12 (plus 11 wrapper call lines), bf16 1, decode layer 24 (plus 14 wrapper calls). Every site we inspected in the MoE and attention was the write-then-wgmma-read direction. We did not trace the release-path sites. Exposure therefore concentrates in rings read with generic loads: the MoE weight ring (weights are decoded in registers, which implies generic shared-memory loads, not verified) and the dense GEMV TMA rings. Attention has a single bulk-copy site and is least exposed.

## setmaxnreg

The instruction must be executed by all warps of a warpgroup, and the warpgroup must synchronise before the next one. Execution by only some warps is undefined. The count must be 24 to 256 and a multiple of 8. An increase blocks until the CTA's register pool can supply it and returns undefined contents. Without a launch-time maximum register count it may have no effect. This confirms a forum report that a producer-share probe (giving the producer a different register share) is blocked, since each role needs a whole 128-thread warpgroup. FA3 uses it to give the MMA warpgroups a bigger register share than the single-thread TMA producer.

## red and the last-block merge

- red is relaxed by default (no ordering) and gpu scope by default. Two atomics are atomic with respect to each other only if each one's scope includes the other. A vector red guarantees atomicity per scalar element, not for the whole access. red.add.f32 rounds to nearest even and, without the noftz qualifier, flushes subnormals on global memory.
- NVIDIA's threadFenceReduction sample puts a thread fence before the ticket atomic and has the last block sum the partials and reset the counter. A NVIDIA forum answer says a fence orders visibility of separate locations, not whether a change is visible at all. The memory-model chapter's message-passing example needs release at a scope that includes the reader.
- Our attention merge, which comes from the public vjeux file, issues red.v4 and red.f32 into the workspace, runs a thread fence in every thread, a block barrier and a single-thread counter add, then the last arriver fences, reads with cache-global loads, zeroes the workspace with plain stores and resets the counter. That is the fence-before-ticket form of the NVIDIA sample, with the self-zeroing structure the forum report describes. Other counters in the same file use acq_rel gpu atomics with two alternating counters and an epoch.

## What other teams reported

One forum poster (jakozaur, thread 42, which another poster then adopted) described three failures that show up only in graph replays:

1. Gluon warp_specialize reuses mbarrier memory, giving an intermittent out-of-range read in roughly 1 of 20 to 1,700 replays.
2. Hand-written cp.async.bulk rings need fence.proxy.async.shared::cta before the empty-barrier arrive. Without it a stage was wrong in 1 to 19% of launches at 2, 3 or 5 stages, none in 10,000 at 4 stages, and zero in 200,000 after adding the fence.
3. Split-K with a float red.add into a self-zeroing workspace read by the last CTA gave wrong tokens about once per 10,000 to 20,000 replays (about 1 in 300 with a 2-stage config). The fix was per-slice partial slots with plain stores, summed by the last CTA in fixed order: 0 failures in 300,000 replays at the same speed.

Their cheap habit: soak at least 2,000 queued graph replays per scored shape, with a host sync every 20, and compare bitwise against a synced replay.

Our assessment of those reports:

- The PTX rule behind item 2 is general (same location, two proxies, fence needed), and the arrive is a CTA-scope release that does not order generic reads against later async-proxy writes.
- The documents cannot explain a 1-in-10,000 failure rate. A missing release before the counter, or an unordered late red, is sufficient to produce one. Whether the reporter's code had the fence is not stated.
- Our four base files contain no mbarrier.inval and no Gluon warp_specialize (the single warp_specialize hit is a comment in the bf16 file), so item 1 targets a path we do not use unless a barrier region is reused for other data.
- The soak size is too small for the failure class it is cited for. With per-replay failure rate p, the chance of seeing at least one failure in N replays is 1-(1-p)^N. For p = 1e-4 and N = 2000 that is about 18%, for p = 5e-5 about 10%, and 95% needs roughly 30,000 to 60,000 replays. A 2,000-replay soak catches the 1-in-300 and 1-in-100 class of bug, not the 1-in-10-to-20k class. Without a local GPU only a prepare-time self-test can soak.
- The MoE and decode-layer counters may share the self-zeroing red.add pattern. We only counted them by grep (MoE 16 atomic lines and 8 fence lines, decode layer 43 and 19) and did not trace them.

## FlashAttention-4, briefly

The blog is by the paper's lead author and targets Blackwell. Tensor throughput goes from 1 to 2.25 PFLOPS between H100 and B200 while the special-function unit count (16 ops per cycle) and shared-memory bandwidth stay put. FA4 therefore splits 2^x between the MUFU and an FMA-pipe emulation (Cody-Waite range reduction plus a Horner polynomial) for a tunable share of every 128 elements. It also rescales conditionally: only when the running max jumps by a large amount, normalising with the true final statistics at the end, decided per warp, with rescaling moved to a dedicated correction warpgroup. FMA-emulated exp2 losing on the H200 is consistent with the blog's own motivation, since the exp-to-tensor imbalance FA4 attacks is 2.25 times larger on B200 than on H100. Conditional rescaling needs neither Blackwell features nor MUFU relief, so on Hopper it is a separate and still-untested question. If it were tried, the dynamic range needs a check: P can exceed 1 until the next rescale, and with P in E4M3 (max about 448) or FP16 accumulation that has to fit the allowed power-of-two P scale. The four consumer-loop shares measured by another team total 83.2%, leaving 16.8% unattributed, which bounds the rescale work at under 17% of the consumer loop if it sits in that remainder (arithmetic on the cited shares).

## Consequences for our entries

In order of risk avoided, then expected gain:

1. Proxy-fence audit of the cp.async.bulk rings that are read with generic loads (MoE weight ring and the dense GEMV TMA rings first; attention least). It gains no speed. The risk avoided is a wrong-output rerun at the organisers' final review. It can be done statically, without a GPU.
2. Soak sizing and the attention merge's float red into a self-zeroing workspace. The reported speed-neutral fix (per-slice slots, fixed-order sum) touches the merge, which we had closed for speed reasons, so it is a correctness decision for the team, not a tuning candidate. It changes accumulation order, which is allowed arithmetic.
3. PDL trigger placement in the decode layer. Legal by NVIDIA's text, capped by the competitor numbers above, expected 0 to about 1% of the layer, and not audited on our base.
4. Conditional softmax rescale in Hopper verify attention. Our entry already has a rare-rescale path, so the open part is FA4's per-warp decision and dedicated correction warpgroup. Untested, and no gain estimate exists because no competitor isolates the rescale cost.
5. Pre-wait hoisting and the producer-share probe stay closed. The documented exception for hoisting costs a fence per CTA, and the probe is blocked by the whole-warpgroup requirement.

## Gaps

- The Stream-K, MARLIN and FlashInfer papers, the NVFP4 format blog and the Hopper microbenchmarking papers were not part of this documentation pass.
- No primary source exists for a forum claim that 32 KB of straight-line code costs about 1 us of cold instruction cache. Two measurements touch it. Our own cold-code warming check on attention found no gain, and a public repo measured a merge routine dropping from 6,928 to 2,524 cycles when pre-warmed in a different kernel design.
- The CUDA graphs chapter, as extracted, said nothing about back-to-back replays of one executable graph or memset-node cost. The figures of 3.2 and 5.0 us for memset-node and last-consumer reset rest only on one team's protocol benchmark.
- We did not verify whether the base files' release paths run the proxy fence per consumer thread, nor where the decode-layer base places its launch_dependents relative to the last weight issue (it appears at several sites, with a pre-wait stage knob set to 2).

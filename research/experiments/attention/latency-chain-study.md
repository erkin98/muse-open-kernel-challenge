# Latency chains in the small attention shapes

This file covers the batch-1 and sliding-window attention shapes, where the kernel is a chain of dependent memory round trips, and the experiments we ran on them.

It holds a static timeline model of d1s, what the in-process lab measured on those shapes, the levers we weighed, and the one that survived: bf16 reductions for the q_len 8 merge, through a lab arm and a judge check.

Shape codes are the ones used in lab-ab-method-and-results.md: d or v for decode (q_len 1) or verify (q_len 8), the batch size, then s or f for sliding-window or full-attention layer. The context is 80,000 tokens everywhere.

## The shapes and their kernels

|shape|kernel instance|SASS instructions|judge kernel time (us)|
|-|-|-|-|
|d1s|`attn_kernel<1,false,13>`|2040|4.2|
|d1f|`attn_kernel<1,false,11>`|2592|9.9|
|d16s, d32s|`attn_kernel<1,false,1>`|2672|6.8, 8.9|
|v1s|`attn_kernel<8,false,15>`|2176|6.0|
|v1f|`attn_kernel<8,false,11>`|2752|13.5|
|v16s, v32s|`attn_kernel<8,false,3>`|2680|10.5, 14.4|

Kernel times are from the profile of board run c20df44ab71d4005. Facts we verified by reading the code:

- One call is one kernel launch. The output allocation is graph-private and there is no memset, so the captured graph has a single kernel node.
- `kernel_fn` launches with `pdl=None`, so the programmatic-dependent-launch attribute is never set and the `griddepcontrol.wait` in the kernel does nothing. There is no PDL chain to shorten, and a second kernel would add a graph node of about 1 us to a shape that takes 4.
- The reference input generator builds page-table rows from a random permutation. K/V row addresses cannot be guessed. Only the sequence length is speculated (page-table width minus q_len, verified on device), and the dependency page index to K/V row costs two DRAM round trips no matter what.
- The planner's cluster branch is dead for every L in (1, 8): four earlier returns cover all cases. The DSMEM-cluster kernel (MODE 2) runs only through an explicit override, and the cluster variants with a speculated length or one tile (MODE 10 and 14) are not instantiated.
- d1s is MODE 13 (speculated length, one tile, plain split-KV). v1s is MODE 15 (the same plus a cooperative finalize). With one tile per CTA there is nothing for deeper pipelining, linear exp2 or V-transpose to work on.

## Static timeline model of d1s

Before any measurement we decoded the stall counts from the SASS control words and built an issue-time model of `<1,false,13>`. Memory waits are not counted, so the straight-line numbers over-estimate.

|phase|SASS|model|
|-|-|-|
|CTA start to the first page-index copy|0 to 113|about 560 cycles (0.3 us): parameter loads, speculated range math, an `fdiv_i` for the copy index (computed at run time even though nc is 1), coordinates|
|trip 1: page indices, qkv rows, seq_lens, scalars|113 to 210|one DRAM round trip|
|page indices landed to last K/V copy issued|about 215, then 454 to 520|about 70 instructions, the branch-free one-tile path|
|trip 2: 64 K rows and 64 V rows|memory|one DRAM round trip, dependent on trip 1|
|K/V landed to first red|865 to about 1260|about 900 cycles (0.5 us): QK (16 HMMA), exp (54 MUFU), PV (16 HMMA), 4 CTA barriers|
|merge: 4 vector reds per thread, CTA barrier, then thread 0 does `MEMBAR.ALL.GPU`, the counter atomic, `CCTL.IVALL`, barrier|1260 to 1280|memory: the membar waits for every red to be acknowledged, then the atomic costs a round trip|
|last CTA: read 16 rows, normalise, gate, store, reset|1400 to 1998|one more L2 round trip plus about 600 instructions, then the stores must drain|

Components of the 4.2 us kernel (estimates): launch, parameters and prologue 0.3, trip 1 0.6 to 0.75, K/V issue 0.1, trip 2 0.6 to 0.75, compute 0.5, merge and atomic 0.9 to 1.0, finalize and drain 0.6 to 0.7. We treat the merge plus finalize as one block of about 1.5 us, 35% to 40% of the kernel. It is the only part that is not inherent. How it splits internally was not known at that point.

The floor is two dependent DRAM trips (1.1 to 1.4 us), launch and teardown (0.3 to 0.5 us; the smallest bf16 kernel on the board takes 2.4 us for an 8.4 MB stream), compute 0.3 us and the best merge we can imagine (DSMEM, about 0.8 us). That sums to 2.6 to 3.0 us. A perfect design is therefore -30% on d1s. A realistic blind gain is -0.3 to -0.5 us (7% to 12%). Our own notes carried an estimate that the leaders sit 30% to 35% below us on chain shapes. When we traced it back, it was an arithmetic scenario, not a measurement, and nothing cheap gets there. If the leaders really are there, they have a different architecture tuned on real hardware.

What a gain is worth, as percent of score per 0.1 us saved in one shape (from our score model): d1s 0.19, v1s 0.14, d1f 0.065, v1f 0.049, d16s 0.043, v16s 0.036, d32s 0.025, v32s 0.021. A uniform 0.1 us off all six sliding shapes is +0.46%. Batch 1 is almost everything, and since a check's interval is about 1.5%, a win on chains that a judge run could confirm would need about 0.8 us off both d1s and v1s.

## What the lab measured

The in-process lab (job 56793dfd379e4741, described in lab-ab-method-and-results.md) stamped every phase of every shape. Times are microseconds after the earliest CTA entry, medians over launches, globaltimer with a 32 ns step.

|phase|d1s|v1s|
|-|-|-|
|entry / part|0.1 / 0.4|0.1 / 0.3|
|idxwait (page indices landed)|0.8|0.9|
|qready (Q group landed)|1.4|1.9|
|tile0 (first K/V tile landed)|1.7|2.0|
|loopend|2.2|2.8|
|reds issued|2.4|3.7|
|arrreq / arrret (fence drain and atomic)|2.5 / 3.4|3.8 / 4.5|
|release (last-CTA finalize start / cooperative barrier)|3.6|5.1|
|finload / end|4.2 / 4.3|5.7 / 5.8|
|span (judge GPU time)|4.3 (4.2)|6.0 (6.0)|

d1s: every phase lands inside the static model. No phase exceeds its model by 0.5 us, and the last two CTAs arrive 0.03 us apart, so there is no straggler. The fence-plus-atomic block is 0.86 us and amounts to two L2 round trips. The tail from loopend to end is 2.1 of 4.3 us on d1s and 3.0 of 6.0 us on v1s, about half of each kernel. The rule we had set before reading the table was to reopen the d1s verdict only if one phase exceeded its model by 0.5 us and a static change could cut at least 0.3 us of it. Nothing met it, so the d1s verdict stayed NO-GO.

The other per-request shapes by span: d1f 9.8, v1f 13.4, d16s 7.2, v16s 10.8, d32s 9.2, v32s 14.2 us. For d1s, each phase is about one L2 or DRAM round trip.

One thing was not in our model. The `reds` step scales with bytes. Median clock64 step per CTA:

|shape|reds step (us)|bytes per CTA|
|-|-|-|
|d1s, d16s, d32s, d1f|0.25, 0.40, 0.40, 0.32|4 KB|
|v1s, v16s, v32s, v1f|0.87, 0.93, 0.92, 0.92|32 KB|
|flat d16f, d32f|0.32, 0.33|not recorded|
|flat v16f, v32f|0.66, 0.83|not recorded|

That is about 21 ns per KB plus roughly 0.2 us fixed. On v1s the red step alone comes to 0.6 us more than the d1s-like model, which was past our reopen bar. We had previously rejected narrower reductions, because alygg77's number for the merge at batch 1 (median about 0.96 us) capped the gain at 0.7 us. The lab changed our mind by showing the step was bound by bytes, so halving them looked worth about 0.35 us.

## Levers weighed and what happened

Details for every lever are in the experiment ledger. The ones that carry a lesson:

### More workspace copies for d1s (nc=4): measured, worse

We had a story: same-address reds queue up in L2 (alygg77 put the cost at about 17 ns per op), so a 32-deep queue on d1s should cost around 0.5 us, and four copies would cut it to 8-deep. We predicted -0.2 to -0.4 us. The judge run (job 66be38f978b54004, 2.567x, CI 2.529-2.606, 16/16) gave d1s 5.53 us against 4.94, 5.17 and 5.13 in three runs of the same machine code, which is 0.36 to 0.59 us slower and outside the 0.23 us spread of the identical-code runs. The other eleven rows shifted by a factor of 1.0003 and 0.9998 against two baselines, so the run itself was clean. The only difference from the base was one tuned-plan entry. The finalize with four copies uses the generic copy loop instead of the single-copy path and does four times the reads and the zeroing. That extra work is the likeliest cause of the 0.45 us (inference; we did not isolate it). Whatever the queue depth is, it is not what dominates the tail. The stamp table later agreed: arrival-to-return is 2 L2 round trips and not queue-bound. We do not know which of the three runs the 5.13 comes from.

### Cold-code warming: measured earlier, null

Warming the per-request finalizes gave d1s 5.1 us (identical-code runs 4.94 and 5.17), v1s 6.8 (6.80 and 6.59), headline 2.576x against 2.605x. Static reasoning agrees: the d1s last-CTA finalize is about 600 SASS instructions. If a cold fetch cost 10 ns per 128-byte line (about 1.2 ns per instruction), warming would have shown 0.7 us. It showed 0.0 plus or minus 0.2, so cold fetch is below about 0.3 ns per instruction here.

### Cluster plus DSMEM merge: not retestable blind

vjeux had measured plain at least as fast as cluster at batch 1 on the judge (6.5 vs 6.6 us, and batch 16 9.2 vs 9.4) on an older base. A fair retest needs MODE 10/14 builds with two tiles per CTA and a cluster launch. We estimated -0.3 plus or minus 0.5 us, which is high effort and not debuggable without a GPU.

### Small ones, below resolution

Skipping the `fdiv_i` copy index when nc is 1 and folding the chunk size (about 0.04 us), issuing the finalize's reset stores right after the workspace loads (0.05 to 0.1 us, touches the shared finalize), and prefetching the workspace, counter and output lines into L2 (0 to 0.1 us, no evidence a miss exists). A bundle of the three is at most +0.3% of score and unverifiable. We did not build them.

### d1f and v1f

d1f (252 splits, nc 8, cooperative finalize, 11.27 us on the judge, 9.9 us kernel, a bound near 6.4 us) is the biggest gap in the chain class, but alygg77's 11 wave-balancing arms all lost, and the configuration is an exact-key override tuned on the judge. v1f is bound by the q_len 8 tile loop, about 10 tiles per SM at 0.55 to 1.0 us per 64-key tile, the same loop the flat v16f and v32f use. No separate chain lever.

## bf16 reductions for the q_len 8 merge

### What we specified and what we built

The spec called for 16-byte vector reds: `red.global.v4.bf16x2.add.noftz` (or the `.v8.bf16` form), checked against NVIDIA's PTX ISA documentation. Those need sm_90, and vector atomicity is per scalar element, which is what an accumulation needs. Eight dimensions per op would have halved both the bytes and the op count.

We built a smaller version. Each lane packs its own 4 dimensions into bf16 and issues one `red.global.add.noftz.v2.bf16x2` (8 bytes), with the same lane exchange, the same op count and half the bytes. A row keeps its 64 bf16 values in the first 128 bytes of the unchanged 256-byte slot. The readers unpack the thread's 16-byte word and zero exactly those 16 bytes, the other 128 bytes are never written, and the denominators stay fp32. So the workspace is all zero when idle in both formats, and kernels of either format can run back to back on the same workspace. The 16-byte form was never built or measured.

Race argument for the shared self-zeroing workspace (valid for the lab twins and for the check-4 build):

1. The writer lane (tig is its index within its group of four) adds dims 8t + 4(tig>>1) + 0..3, which maps to byte offsets 16t + 8(tig>>1), 8-byte aligned. The reader thread zeroes bytes 16*part to 16*part + 15, and the 8 parts tile bytes 0 to 127 exactly. Every byte a writer can dirty is read and zeroed by exactly one thread.
2. A red on a bf16x2 word is one atomic 32-bit read-modify-write at L2, the same granularity as one float in the fp32 vector red. The two bf16 halves of a word are independent sums.
3. Inside a launch the chain is unchanged: every CTA's reds, CTA barrier, thread 0 `atom.acq_rel.gpu` arrival, the others spin on `ld.acquire.gpu`, CTA barrier, then `ld.global.cg` of the slice and zero stores of exactly what was read. The SASS census agrees (barrier, atomic, membar and nanosleep counts all equal).
4. Across launches there is one kernel node per call and PDL is off, so launch N+1 starts after launch N completes.
5. A speculation miss re-runs the segment with the real length before any global write. A non-resident grid falls back to the fp32 MODE 9 kernel, which is self-consistent.

What this does not remove: the base design already carries a risk. Another team reported that float `red.add` into a self-zeroing workspace failed about once in 10,000 to 20,000 replays in their design. The element type neither adds to it nor reduces it. And the 30,000-launch soaks cannot show a race at 31 to 33 partials, because one lost partial moves the launch's relative L2 by less than its natural spread.

### Numerics

CPU model: the writer rounds each partial to bf16, each add rounds the running sum to bf16, arrival order is random. Mean relative L2 of the normalised output against the fp32 sum:

|shape|partials|model mean|measured on the judge, vs the fp32 arm|
|-|-|-|-|
|v32s|4|3.1e-3|3.8e-3|
|v16s|8|4.1e-3|4.3e-3|
|v1s|33|7.3e-3|6.9e-3|
|v1f|31 per copy|7.1e-3 (p99 9.8e-3)|6.6e-3|

The model's worst case was 6e-3 at 8 partials and 1.6e-2 at 31 to 33 (1.0e-2 to 1.1e-2 in the v1f-specific run).

The judge accepts 0.15 relative L2 and the kernels' own error against PyTorch is 0.07 to 0.09. A full-attention batch-1 case at 300,000 tokens gave 7.0e-3 against the fp32 arm (distance to PyTorch 0.0795 vs 0.0792).

### Lab 2 arms (job 7c7e80b4e68f4850)

Paired median span difference to the old arm, in us, with old2 (a second run of the old arm) as the noise of d, and the step medians:

|shape|bf16 d [p25,p75]|old2 d|reds step old / bf16|rel L2 vs old|soak (30,000 launches) med / p999 / max|
|-|-|-|-|-|-|
|v16s|-0.10 [-0.19,+0.06]|+0.13|0.94 / 0.62|4.3e-3|4.3 / 4.3 / 4.3e-3|
|v32s|+0.10 [-0.13,+0.26]|+0.06|0.94 / 0.62|3.8e-3|3.8 / 3.8 / 3.8e-3|
|v1s|-0.03 [-0.13,+0.10]|-0.03|0.89 / 0.62|6.9e-3|6.8 / 7.2 / 7.3e-3|
|v1f|-0.35 [-0.48,-0.22]|+0.10|0.94 / 0.63|6.6e-3|6.6 / 6.8 / 6.9e-3|

The rule written before the run asked for at least 0.25 us (and d(bf16) minus d(old2) of -0.2 or better) on at least three of v1s, v16s, v32s and v1f. Only v1f qualified, so as a four-shape package it was NO-GO.

The reds step fell by 0.27 to 0.32 us on all four shapes, as predicted. The wall-clock gain survived only on v1f. On the others the saving came back downstream: finalize load steps rose by 0.17 us (v16s), 0.06 us (v32s) and 0.05 us (v1s), the loop-end maximum rose by 0.16 to 0.19 us on v16s and v32s, and arrival-to-return rose 0.10 us on v1s. On v1f the arrival-to-return step also fell, from 1.02 to 0.78 us, so there both steps gave something. We have not identified why the other shapes lose time elsewhere.

### Check 4: v1f only (job 1c1107b8a97d4e45)

With one shape qualifying, we ran a judge check of bf16 reds for v1f alone, on top of the check-3 file. The v1f build (`attn_kernel<8,false,11+1024>`, selected by the host table for one request, page size 1, q_len 8, full attention, cooperative residency, more than one tile per split) has a static assert tying the bf16 writer to MODE & 3 == 3, so a bf16 writer can never meet an fp32 reader. The planner picks the cooperative path only when n_splits is 64 or more, so nc is always 8 in this build; the non-cooperative fallback stays fp32. Static results: 19 of 20 kernels SASS-identical to the check-3 build, v1f 2848 to 2752 instructions and 111 to 110 registers, no spills, eight red sites that went from `REDG.E.ADD.F32` to `REDG.E.ADD.BF16x4.RN.STRONG.GPU`, reader loads and zero stores halved (26 to 13 and 27 to 14). All 20 kernels were SASS-identical to a build of the lab source, so the shipped v1f is the code whose stamp twin lab 2 measured. The candidate file has sha256 prefix a3bc7413.

Judge: 16/16, 2.627x (CI 2.588-2.667), 1,433.6 us. Per case in us and speedup: d1s 4.65 / 4.347, d1f 10.61 / 2.939, d16s 8.29 / 3.242, d16f 86.86 / 2.308, d32s 10.39 / 3.047, d32f 166.47 / 2.311, v1s 6.45 / 3.454, v1f 13.92 / 2.592, v16s 11.33 / 2.581, v16f 100.43 / 2.046, v32s 14.64 / 2.352, v32f 183.91 / 2.120. The profile names `attn_kernel<8, false, 1035>` for v1f, which confirms the edit ran. (It would have shown `<8, false, 11>` or the fallback `<8, false, 9>` otherwise.) Require that gate before reading any score.

The reading we wrote down before launch was inconclusive by design. The expected effect is -0.35 us of 13.7, about 2.6% in time. The identical-code envelope for v1f over five runs is 13.67 to 14.47 us (2.506x to 2.629x), 4.9% wide, and the best control-normalised envelope was 3.3% wide. The result, 13.92 us and 2.592x, landed in the upper half of the envelope and the reader script printed inconclusive.

A second reading came after the result and was not written down in advance. The judge's per-case "yours" figure is the raw event median minus a replay-overhead estimate for that run, and the estimate moves between 3.7 and 4.4 us for identical code (for the v1f rows: 4,064 / 3,744 / 4,032 / 4,224 / 4,352 and 4,064 ns). That is plus or minus 0.3 us on a 13.7 us case, as large as the effect. The raw medians come from 60 timed runs with a standard error of 25 to 43 ns and are much tighter once divided by control rows of the same run. Raw v1f medians in ns, with the overhead used for "yours":

|run|raw median|overhead|yours|
|-|-|-|-|
|c20df44ab71d4005 (base)|18,368|4,064|14,304|
|705c7dd9f34446dd (base rerun)|18,272|3,744|14,528|
|lab 1|18,160|4,032|14,128|
|lab 2|18,080|4,224|13,856|
|check 3|17,968|4,352|13,616|
|check 4|17,984|4,064|13,920|

The residual is ln(raw v1f) minus the mean of ln(raw controls). For the five identical-code runs its standard deviation is 0.27% to 0.65% depending on the control set, against 3% to 7% for the ratio-of-"yours" reading. Check 4's shift relative to the identical-code mean, by control set:

|control set|shift|z|
|-|-|-|
|d1s d1f d16s d32s v1s v16s v32s|-1.73%|-2.7|
|d1s d1f v1s v16s v32s|-1.89%|-5.8|
|v1s v16s v32s v16f|-1.70%|-5.2|
|all other 11 rows|-1.72%|-5.8|
|v16s and v32s|-1.47%|-5.3|
|d1f alone|-1.63%|-3.2|

All six agree within -1.5% to -1.9%, about -0.3 us, against -0.35 us from lab 2. In raw terms v1f is flat against check 3 (+0.1%); the chain rows in the check-4 run were 1.2% to 4.2% slower than in check 3, and the claim comes from that normalisation alone. Calibration and caveats:

- Leave-one-out z of each identical-code run against the other four, using the five-row set (d1s, d1f, v1s, v16s, v32s): +0.8, -1.0, +1.8, -1.7, +0.1 (v16s and v32s: +0.8, -0.7, +2.6, -0.9, -0.9).
- Two of the larger control sets are distorted by single rows of one baseline run (check 3's d32s raw time is 2% under its siblings, and its flat rows carry the check-3 edit), so the five-row set and the v16s+v32s pair are the cleanest.
- Placebo: the same residual for every other row against its own other 11 controls gives d1s +0.84%, d1f -0.10%, d16s -1.91% (sd 2.7%), d32s +1.10%, v1s +0.75%, v16s -0.58%, v32s +0.02%, v16f -0.29%, v32f +0.13%, d16f +0.07%, d32f -0.03%, all |z| of 1.3 or less. The sliding rows are noisy enough that d16s alone moved as far as v1f did, so the placebo is not spotless.
- The five baseline runs come from four different files (two base, two lab, check 3) and show a drift: v1f raw time falls monotonically from 18,368 to 17,968 across them, and the controls drift the same way. The control sets were chosen after the data, though we defined all six before computing and report all of them. The t-statistics rest on four degrees of freedom.

Verdict: safe (16/16 pass, the edit ran, no unchanged row regressed), and the direction matches lab 2 at about -0.3 us. The real score effect is about +0.15% (v1f is worth 0.049% per 0.1 us). The headline 2.627x against 2.620x is inside the 2.589-2.676 range and proves nothing.

## What we did not try

- The 16-byte vector form, which would halve the op count too.
- A TMA bulk reduction (`cp.reduce.async.bulk`), an idea we saw in keell0renz's public notes. It could take the reds step to about 0.15 us but needs a swizzled shared-memory staging layout, `fence.proxy.async` and a new ordering proof.
- A third lab for the three small levers, because each was worth 0.04 to 0.1 us (+0.05% to +0.15% of score), unresolvable by a single draw, and one of them touches the shared finalize.
- The cluster and DSMEM path.

## Open questions

- Why does a 0.3 us cheaper reds step turn into -0.35 us on v1f but about zero on v16s, v32s and v1s?
- The 21 ns per KB figure says the red path is bound by egress bytes. We do not know whether L2 atomic throughput for bf16 vectors, or op count, becomes the next limit, since the 16-byte form was not tested.
- Whether the judge's replay-overhead estimate can be pinned down per run, so that sub-3% effects could be read without control rows.

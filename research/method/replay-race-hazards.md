# Replay race hazards in graph-captured kernels

A static review of the ways our four board kernels could produce wrong output only under CUDA-graph replay, written up as a guide to what to look for.

## Scope, and how much to trust it

All four board entries (attention, moe, decode_layer, bf16) run vjeux's public Apache-2.0 kernels with our attribution header. The bf16 file also carries one extra router route that the judge never picks. We read the sources on the afternoon of Nov 4. We did not run anything: no GPU, no judge check, no SASS inspection. The board runs of these files passed correctness on every case. The only measured evidence in this document is jakozaur's public report of replay-only failures in his own kernels. Everything about our files is formal memory-model reasoning plus latency arguments, and we say which is which.

Line numbers below refer to the board-entry files of jobs c20df44ab71d4005 (attention), 41f596feaa60434e (moe), 9b0d0973f11a4758 (decode_layer, since withdrawn) and 6c7a0061944e4339 (bf16), header included. Later submissions changed some entries, and our files differ from upstream commit f35a41e, so use the function and kernel names to find things.

Severity labels:

- HIGH: plausible wrong output.
- MED: correct only if some assumption holds.
- LOW: benign, or a gap in the formal argument that no realistic timing can reach.

For memory-model claims we read the PTX ISA 9.4 documentation. The sections we leaned on:

- 8.6 Proxies: bulk-copy accesses are async-proxy accesses, and a proxy fence is needed to synchronize operations across proxies.
- 8.9.4 Memory synchronization: `bar.sync` synchronizes with `bar.sync` on the same barrier, a release pattern synchronizes with an acquire pattern, and stream order, graph node completion and kernel start/end are synchronizing events.
- 8.9.5 Causality order: base causality extends across proxies only for generic-to-generic accesses, accesses in the same proxy within one thread block, or aliases with an alias proxy fence.
- 9.7.10.28.2 Async proxy: a bulk copy's completion is followed by an implicit generic-async fence, so generic loads see the result. Nothing is said about a generic read followed by an async write.
- 9.7.15.4 membar/fence: proxy fences work inside one thread (no scope), are bidirectional, and compose with other synchronization.
- 9.7.15.16.16 `mbarrier.arrive`: release semantics, CTA scope by default.
- 9.7.10.9 `ld.global.nc`: described only as a load through the non-coherent read-only cache, with no statement about coherence with concurrent writers.
- 9.7.15.1 `bar`: a completed barrier guarantees the thread's earlier accesses are performed relative to the other participants.
- 9.7.15.14 `griddepcontrol`: `wait` blocks until the prerequisite grids have completed and their memory operations are visible.

## Why replay bugs hide

A graph replay launches the same kernels back to back with almost no gap. A race that needs a particular overlap can sit for thousands of launches without showing. Other teams that hit these classes report failures at about 1 in 10,000 to 20,000 replays. The rate depends heavily on the kernel (the proxy-fence failure described under Hazard 2 was far more frequent in the kernel where it was measured). The judge replays each shape far fewer times than that, so a clean judge run says little.

The judge's own tolerances also hide small errors. Attention is compared with a 3 to 15 percent relative L2 error against an FP32 reference, plus a per-row factor of 5. Moe allows 1.5 percent relative L2. The bf16 router tolerance, quoted in a public discussion, is 1e-4. A single-bit float-order wobble is far inside all of these.

## Hazard 1: float reductions into a workspace that zeroes itself

The pattern: every CTA adds partial results into a shared workspace with float reductions (`red`), a counter decides which CTA is last, and that last CTA reads the sums, finalizes the output and writes the workspace back to zero so the next launch starts clean. State travels from one launch to the next, so a partial that is lost or late shows up as damage in the following replay.

Attention uses three variants of this finalize step, and moe and decode_layer have their own. What they share is a four-step chain that has to be correct:

1. Each consumer thread issues relaxed GPU-scope float reductions into the workspace, including 16-byte vector forms.
2. A CTA barrier follows.
3. Thread 0 does one `atom.acq_rel.gpu` add of the CTA's key count (or CTA count) to the counter.
4. The CTA that sees the count complete reads with `ld.cg`, zeroes the workspace with plain stores, and resets the counter.

What we checked, all by reading:

- Zeroing becomes visible to the next replay. Verified against PTX 8.9.4: completion of a graph or stream task synchronizes with the start of the next task in the same stream, and which thread did the zeroing does not matter. Attention requests no PDL, so consecutive replays cannot overlap. LOW.
- Exactly one CTA identifies itself as last. Splits with no keys return before touching a counter, and the shares that do add sum to the total key count, so only one arriver can see `old + mine == n_keys`. The epoch variants alternate two counters; the last arriver resets the other and bumps an epoch word. The bump cannot overtake another CTA's epoch read, because the bump needs that CTA's arrive, and the CTA drains its epoch `cp.async` before arriving.
- The release/acquire chain is legal under the model. Relaxed reds in thread T, then `bar.sync`, then thread 0's acq_rel add (a release pattern), observed by the last CTA's acq_rel read-modify-write (an acquire), then its own `bar.sync`, then the readers' `ld.cg`. Base causality is transitive (8.9.5), so thread 0's release covers T's reds. This is the same shape as a CUTLASS semaphore.

The residual risk is MED, and it is an assumption, not a failure anyone has seen. If the hardware does not honor that cumulativity for vector reductions still in flight from other warps, a late red could land after the last CTA has read and zeroed, and would be lost or would poison the next replay. That is the failure mechanism jakozaur describes. The model says it cannot happen. We found nothing public that confirms the hardware path for vector reds either way.

The attention design deliberately has no per-thread fences, and the parent source inside the decode_layer file shows why: two build switches (merge2 and merge3, both on) replace per-thread fences with the single acq_rel atomic. A per-thread fence call survives in the attention file behind a condition that can never be true. There are two ways to buy insurance:

- Enable that dead call so every thread runs a GPU-scope acq_rel fence before the CTA barrier. This is the fence-before-ticket form from NVIDIA's thread-fence-reduction sample and does not rely on cumulativity. Not measured. The MoE fence moves cost about 0.13 µs per fence, and the cost on the small decode shapes is an open question.
- Replace the atomics with per-split plain-store slots that the last CTA (or every CTA, on its own row slice) sums in a fixed order. That removes the zeroing hazard and makes results bitwise reproducible, but adds per-split partial traffic through L2 for the one-request full-context cases (d1f, v1f).

Neither is a tuning candidate. Both are a correctness decision.

### The same shape in moe

The moe kernel's stream-K fix-ups have shares add partial sums with `red.global.add.v4.f32` into a slot buffer, then count their stages with `red.release.gpu` (the designated-finisher phase) or `atom.acq_rel.gpu`. The finisher waits with `ld.acquire`, runs a GPU-scope acq_rel fence in every thread, reads with `ld.cg`, zeroes the slot and resets its count. The acquire side is fenced per thread. On the release side, all 128 threads post their reds at the end of a segment, but the count is deferred to the next segment after a warpgroup barrier. The code comment says the deferral lets the reds drain and is worth 0.5 to 1 percent. In the common case the reds were posted long before the release, which makes the practical exposure small. The last segment of a phase has less slack. Same cumulativity assumption, MED.

The launch-level machinery is sound as read. Each launch reads a generation word, the last CTA at the grid barrier bumps it, the barrier counter alternates by parity, and each launch zeroes the other parity's counter and the other copy of the sum-of-squares buffer. Per-launch flags are zeroed after the barrier, when every CTA is past the phases that poll them. The barrier is `bar.sync`, thread-0 `__threadfence`, atomic add, acquire spin, `bar.sync`. The spin has no timeout: a CTA that never became resident would hang until the run limit (15 minutes in the task files, 10 for bf16), instead of failing cleanly.

### Float order is not a bug, but it breaks bitwise soaks

Contributions from up to about 32 CTAs arrive in a nondeterministic order, so FP32 sums differ in the last bits between replays. After normalization and BF16 rounding, an occasional output element flips by one step. Our estimate for attention is a few elements per launch at batch 32 with 8 query tokens. For moe, `red.v4.f32` into a slot is exactly commutative for two contributors (zero plus a plus b) and order-dependent for three or more, and the sum-of-squares atomics are order-dependent too. All inside tolerance, so LOW. Any soak test of these kernels has to use a tolerance compare.

### The form that avoids all of this

The bf16 split-K kernels are safe by construction, and the form deserves copying. Each CTA writes its own partial to its own slot with plain stores. A monotonic counter (never reset) identifies the last arriver by `old modulo splits == splits - 1`, or, when the whole grid is co-resident, every CTA waits for the count and sums a slice. The sum over slots runs in a fixed order. No float atomics, deterministic output, nothing to zero. Two conditions must hold: each counter array is used with a single splits value (verified: counters are keyed by the full config, and a layer uses one splits value for all its tile widths), and the counter is never partially restored from a snapshot. Overflow would take about 134 million launches. The decode_layer dense-GEMM glue uses the same idea with per-slice partial buffers summed by the next kernel.

The one exception in the bf16 file is the router GEMM, the only place with a float `red.add` into a buffer cleared in-kernel. Helper warps clear their slice with plain stores, count with `red.release.gpu`, and wait with `ld.acquire.gpu`; a CTA mbarrier then releases the consumers' reds. That chain is legal (clears, release, acquire, mbarrier release, mbarrier acquire, reds). It needs a fixed grid size per counter, which holds. The file's own comment says the sums are order-dependent at about 1e-7 relative, against the 1e-4 router tolerance. LOW, but a bitwise soak must be a tolerance compare here too.

## Hazard 2: refilling a shared-memory slot with a bulk copy

This is the one we think matters most, because it matches a failure someone measured.

A ring buffer in shared memory works like this. A producer thread issues a bulk copy (TMA-style, async proxy) into a slot. Consumers read the slot with ordinary shared loads (generic proxy), then arrive on an "empty" mbarrier. The producer waits on that barrier and refills the slot. The chain is: consumer generic read, `mbarrier.arrive` (release), producer wait (acquire), bulk copy writes the slot.

PTX 8.9.5 counts a causality edge between two accesses to one address only when both are generic, or both use one proxy within one thread block, or they are aliases with a proxy fence. A generic read followed by an async-proxy write is none of those. So by the document, the mbarrier chain alone does not order the read before the write, and PTX 8.6 says a proxy fence is needed. The implicit fence that PTX does document (9.7.10.28.2) covers the opposite direction: copy completes, then generic loads. The pattern therefore leans on implementation behavior, not on a written guarantee. This is the pattern in jakozaur's report, where the loads were `ldmatrix` feeding HMMA in a shared-memory-bound main loop, and a shared load can back up a long way in a loop like that. He measured 1 to 19% wrong launches at 2, 3 and 5 pipeline stages and none in 10,000 launches at 4 stages.

The fix, in its final form: every lane of the consuming warp runs `fence.proxy.async.shared::cta` (one warp-uniform instruction), then `__syncwarp`, then lane 0 arrives. A fence from lane 0 alone would not formally cover the other 31 lanes' loads, because a proxy fence acts inside one thread (9.7.15.4). One caveat on the fix itself: the causality rules in 8.9.5 list no async-proxy-fence case, and the only support is the prose saying proxy fences compose with other synchronization. The fenced form is the documented intent, and it is what jakozaur measured clean (0 wrong in 200,000 launches). It is not a formally derived guarantee. Inference, not measured: the cost is one fence per warp per stage, 0 to 0.3% on the moe cases at 32, 128 and 256 tokens.

### Where each ring in our four files stands

|Ring|Used by|Finding|
|-|-|-|
|Moe weight ring (WGS), 36 KB bulk copy per stage, 3 stages|moe at 32, 128, 256 tokens; identical copy inside decode_layer|MED formal, tiny practical window|
|bf16 `sk2` kernels, slot handed back early|bf16 `o` at 8 tokens, router at 8, 16, 32 tokens; same text inside decode_layer|MED formal, small but not tiny window|
|Skinny WS ring in moe|1, 4, 8, 16 tokens|LOW formal|
|Moe x ring (bulk fill, wgmma read, empty arrive after wgmma wait)|moe large-token path|no finding: async proxy on both sides, same block|
|Attention K/V tiles (`cp.async`, generic)|all attention shapes|LOW|
|Moe phase A/B x slices bulk-copied from global memory other CTAs wrote|moe, decode_layer|LOW formal|

#### Moe weight ring (verified facts)

One lane of the weights warp issues a 36 KB bulk copy after waiting on the empty barrier. The consumers read the slot with ordinary 16-byte shared loads (around lines 2207 to 2215 in the moe file), decode in registers and feed wgmma as register operand A. Lane 0 of each of the 16 consumer warps arrives on the empty barrier at the top of the fourth load group (around line 2217). No proxy fence sits between those loads and the arrive; the proxy fences that do exist in the file are phase-entry and write-then-wgmma fences. decode_layer has a textually identical ring (producer around 5616 to 5640, consumer arrive around 5868). Why the window is tiny (inference, not measured): the last load group's loads are issued one full loop body before the arrive, and the refill is a cold 36 KB fetch from HBM that needs the slowest of 16 warps to arrive, then a producer wake-up, then at least about a microsecond to land. At 36 KB per stage, a shared load staying unperformed that long is not plausible. Three stages is one of the counts where jakozaur saw failures, but his data path differs.

#### bf16 `sk2` early hand-back (verified facts)

In the skinny GEMV, all the shared and `ldmatrix` loads of a unit are issued, then `__syncwarp` and the lane-0 empty-barrier arrive happen before decode and the matrix multiply consume those registers (lines 1213 to 1217 of the bf16 file; same text at 7914 to 7917 inside decode_layer). So the arrive is not ordered after the loads complete. That is by design: the kernel hands the slot back before decoding and multiplying. The producer then refills the slot with two copies. The weights come from HBM and take a microsecond or more. The activation tile comes through a tensor-map load that hits L2 after the first CTA touches it and can land within a few hundred cycles. Inference: the pending loads are about 100 to 200 cycles old at the arrive, so the margin against the activation copy is only about 2 to 3 times, thinner than the moe ring's margin. This is why it ranks first. It is live only for `sk2` configs with one unit per warp. Configs with more units per warp hand the slot back after the multiply is issued, with loaded values already consumed, so they are fine. In decode_layer the exposed shapes are the verify-1 loads (8 tokens) for sure, and the decode loads at 16 and 32 tokens if the tuner picks this path or the fused pooler-plus-router kernel uses the one-unit-per-warp router config. We did not read that fused kernel's body.

The same fix applies at both sites: a proxy fence by all lanes, then `__syncwarp`, then the arrive. The fence has to wait for the pending loads. Inference, not measured: likely under 100 cycles per stage, on two configs.

#### Skinny WS ring

Also formally unfenced. It cannot fail in practice because every warp consumes its loads into the accumulators, stores them, and passes a CTA barrier before thread 0 refills the slot.

#### x ring and wgmma

Where the same proxy sits on both sides (bulk fill, wgmma read, release after the wgmma wait) the chain is ordered. In attention, the K tile for 8-token verify is written by `cp.async` (generic) and read by wgmma (async). The proxy fence follows only the generic stores that patch new rows, not the bulk of the `cp.async` rows. That is a formal write-then-async-read gap. We rate it LOW since the data is complete when the `cp.async` mbarrier arrive fires, and CUTLASS mainloops have the same shape. Going the other way, wgmma read then generic refill is covered by the implicit fence at wgmma completion.

#### Global-memory bulk copies

The moe phase-A/B x slices are bulk-copied from buffers that other CTAs wrote with generic stores and published with release/acquire counters. Strictly there should be a global-form `fence.proxy.async` between the acquire and the copy. The hardware path is L2 only and the stores write through, so LOW formal.

## Hazard 3: mbarrier reuse

The files initialize their barriers once per launch, with `fence.mbarrier_init`. No barrier memory is later reused for data, and none of the four files calls `mbarrier.inval`. Initializing over a barrier that is still valid is common practice but not a documented guarantee. Phase parity flips on ring wrap and consumers derive it from the sequence number, with no retry paths. We rate all of it LOW.

Phase-entry proxy fences are executed by thread 0 only and cover only its own earlier accesses. At kernel start the region is fresh. Two of them are in builds the scored shapes do not use.

## Hazard 4: reading before the PDL wait

With programmatic dependent launch, a kernel may start early and run code before `griddepcontrol.wait`. Whatever runs before the wait may touch only weights, constants and graph inputs. Anything written by the previous kernel must be read after it. Placement of `launch_dependents` is safe so long as no kernel skips its own wait.

- Attention never requests PDL, so its `griddepcontrol` instructions are no-ops. Its `ld.global.nc` loads read only inputs (sequence lengths, scales, RoPE frequencies, qkv) that no kernel writes during a launch.
- Moe has PDL off. The wait is the first statement after the thread id and the trigger sits after both phases.
- decode_layer has PDL on for up to 32 tokens. The Triton kernels trigger and then wait immediately; the embedded attention issues page-index copies (an input) before its wait and reads everything other kernels produce after it; the skinny kernels issue weight stages before the wait and read x after it. In every kernel we read, pre-wait code touches only weights, constants or graph inputs, plus an epoch word that only the previous replay writes in the flat attention kernels. A pre-wait knob that would let the attention read K/V early is off.
- Replay to replay is safe. A graph's last kernel completes before the next graph launches (8.9.4), so PDL edges inside one replay cannot overlap the previous replay's zeroing.

Two things we could not close. First, whether the compiler keeps every producer load below the wait in the generated SASS. Our own decode_layer candidate, a GEMV for one-token projections, failed 3 of 13 judge tests (10 passed) because the compiler hoisted its activation loads above the wait, emitted as `ld.global.nc`. That shows the class is real for CUDA code. The Triton kernels in the layer load through plain `tl.load` after the wait, but we have not looked at their compiled output. Because PTX describes `ld.global.nc` only as a load through the non-coherent read-only cache (9.7.10.9), with no coherence statement against writers in the same or an overlapped kernel, the safe habit is to use that load for inputs only. Second, multi-hop visibility. `griddepcontrol.wait` documents visibility only for the immediately preceding grid, yet decode_layer's second boundary kernel reads a buffer that the first boundary kernel wrote four to five kernels earlier. Transitivity of causality order should cover it, and the layer passed its judge runs, but the wait's text does not state the multi-hop case. That is an assumption.

## Hazard 5: workspace lifetime, growth and cloning

### Attention workspace

A module-global triple of accumulator floats, row sums and integer counters, keyed by device string, allocated once with zeros and sized to at least 32768 rows. Rows needed are batch × pairs × copies × query length × 16. The largest of any scored shape or test is 4096 rows (batch 32, one pair, one copy, 8 query tokens), so growth, which would need more than 32768 rows, never happens (shown by this arithmetic). It matters because a graph captured before a regrow would keep the freed pointer, and freed or reassigned memory corrupts later replays. Counter index ranges (tickets 0 to 4095, epoch counters at 8192 and 12288 plus the index, epoch words from 24576, flat words around 30020) all sit below 32768 and do not overlap. The shared workspace is safe because every launch leaves it zeroed and launches are stream-ordered. One launch that traps leaves it dirty for the rest of the process; the spin loops trap after 2^24 polls.

### Moe scratch

Prepare returns the weights plus three zero-initialized scratch tensors, and the kernel re-zeroes what it uses. The generation word lives in the first tensor together with the second sum-of-squares copy, and the first sum copy is in the second. A clone must therefore copy all three together; a partial restore could pair a dirty sum copy with a generation that selects it. The board run's `prepare_s` was 137.3 s (job 41f596feaa60434e), about 160 s under the 300 s limit.

### bf16 state

Per-config counters, shared split-K workspaces created with `torch.empty` and never zeroed, router counters and a 1 MB CUTLASS scratch. Uninitialized workspace is safe because every slot read in a launch was written in that launch by a CTA that counted before the last arriver read. No raw addresses or tensor maps are cached; maps are rebuilt per call from the call's pointers. A snapshot of the counters is safe at any idle moment, and a partial restore would break the modulo test.

### decode_layer

The same designs as above, with sentinel guard words checked at the end of prepare.

## Findings per entry

|Entry|HIGH|MED|LOW|
|-|-|-|-|
|attention|0|1|6|
|moe|0|2|8|
|decode_layer|0|3|6, plus 1 unverified|
|bf16|0|2|4|

The MED items, by exposure:

1. Early slot hand-back followed by a bulk-copy refill with no proxy fence: the bf16 `sk2` kernels (around lines 1214 to 1217), same text in decode_layer.
2. Moe weight ring release without a proxy fence: moe around line 2217, decode_layer around line 5868.
3. Finalize protocols that rely on one thread's release being cumulative over other warps' in-flight float reductions: attention (around lines 2273, 2337, 2406), the moe fix-ups, and the bf16 wide-kernel barrier. The model and CUTLASS's own semaphore code say it holds.

No replay-only race is proven in any of them.

## What we would do

1. For insurance against the proxy-fence class at near-zero cost, add the three-line fix (all lanes fence, `__syncwarp`, lane-0 arrive) at the sites above. Expected cost 0 to 0.3%, not measured. If it is bundled into a speed candidate, any slowdown cannot be attributed, so name in advance the shapes that would carry the cost: moe at 32, 128, 256 tokens; bf16 `o` at 8 and the router at 8, 16, 32; and decode_layer's verify-1 loads (and perhaps the 16- and 32-token decode loads). Otherwise spend one dedicated check per file.
2. Run an offline soak the moment a GPU is free: 60,000 queued replays per shape with a tolerance compare, plus a stage-count sweep on the bf16 `sk2` ring and the moe ring, since jakozaur's failure rate depended on stage count.
3. Close the two static gaps: the SASS of the moe main build around the empty-barrier arrive (if ptxas makes the arrive wait for the last load group, finding 2 drops to LOW), and the SASS of the decode_layer Triton kernels around the PDL wait.
4. Leave the attention per-thread fence off unless there is evidence of a problem. It would likely cost several percent on the small decode shapes.

## Why we did not build a prepare-time soak

The idea: inside `prepare()`, for each shape bucket, capture the same graph `kernel_fn` will run on synthetic inputs, replay it many times queued (with a host sync every 20 to 50 replays, as the judge times), compare every output with a synced reference replay, and on a mismatch mark the route disabled so `kernel_fn` takes a safe path. Vary timing by flushing L2 before every eighth replay, and free the synthetic buffers afterwards (we once saw about 3 MB kept allocated coincide with 0.5 to 1.3 µs slowdowns on unchanged shapes; the cause was never established, see `measuring-on-the-judge.md`).

### Comparison rule

Bitwise only for the deterministic kernels (bf16 split-K, bf16 `sk2`, dense Triton glue). For attention, moe and the bf16 router GEMM, compare with a tolerance: relative L2 against the reference replay at most 1e-6, no element off by more than one BF16 step, and fewer than 0.1 percent of elements off by one step. A lost or duplicated partial is 1e-3 relative or a wrong tile and separates cleanly.

### Detection power

The chance of seeing at least one failure in N replays at failure rate p is 1 - (1 - p)^N:

|Failure rate p|N = 30,000|N = 60,000|
|-|-|-|
|1e-4|95.0%|99.8%|
|5e-5|77.7%|95.0%|
|2e-5|45.1%|69.9%|
|1e-5|25.9%|45.1%|

So 30 to 60 thousand replays catch the 1e-4 class and only part of the rarer ones. The judge's own replay count per shape is far smaller, so a 1e-5 bug would rarely show in a judged run either.

### Time

Replay cost is the profile's kernel time plus about 4 µs fixed, with a floor near 8 µs per queued launch.

|Entry|Shapes|Kernel µs per round|30,000 replays|60,000 replays|
|-|-|-|-|-|
|attention|12|619 (+48 overhead)|20 s|40 s|
|moe|6|710 (+24)|22 s|44 s|
|decode_layer|12|2,635 (+48)|80 s|161 s|
|bf16|30|333 (+120 overhead, 8 µs floor)|14 s|27 s|

An L2 flush per replay (12 to 16 µs) adds about 16 to 21 percent to these totals and roughly doubles the small shapes. Restricting the soak to shapes that use atomics, counters or the early-release ring is much cheaper: five bf16 cases (`sk2` `o` and router, plus the router GEMM) at about 10 µs each take roughly 3 s at 60,000 replays, and three representative decode_layer shapes (sliding decode 1, sliding verify 1, full verify 16) take about 16 s at 30,000.

### Budget against the 300 s prepare limit

- Moe: 137.3 s already, so a 44 s soak lands near 181 s. Feasible, but the first call into the WGS path added about 50 s on other teams' runs, so time-box it.
- decode_layer: prepare already builds five extensions in parallel (about 85 s by the code's own comment), tunes and self-tests; we found no recorded prepare time. A full 30,000-replay soak (80 s) is risky, so only the restricted one would fit.
- bf16: the CUTLASS build takes 70 to 80 s plus autotune; only the 3 s restricted soak is affordable. A self-test and disabled-route mechanism already exists there.
- Attention: no `prepare()` at all. A first-call self-test would run either inside graph capture or as extra work in the timed call, so a soak is impossible.
- In every case, use a wall-clock deadline (prepare start plus 200 s) and truncate the soak if it is hit. A fallback that needs a second compiled variant costs another 15 to 30 s per build.

### Rule questions we would want settled first

These are our reading of the rules, not organizer statements.

- The rules prohibit caching or replaying outputs. Synthetic inputs and a route flag are not outputs and are not derived from scored data, so that looks fine.
- A per-shape disable could look like shape-specific routing. Disable by route family and contiguous token bucket, never by exact token count.
- The organizers list work outside the timed calls among the things that get entries removed. A soak replays kernels, not outputs, and bf16 and decode_layer already time candidates in prepare, but heavy prepare-time GPU work is a judgment call.
- Attention's only fallback is the slow PyTorch path, so a disabled route would cost placement but keep the entry valid. Moe has an `mma.sync` build with the same counters, decode_layer has FA3, Triton and Marlin alternatives, and bf16 has cuBLAS.

### Our recommendation was not to ship one

It detects only the 1e-4 class, we found no evidence of a bug at that rate, and it adds rule and time risk. If GPU time appears, run it offline instead: 60,000 replays per shape would cost about 40 + 44 + 161 + 27 seconds for the four entries, roughly four and a half minutes. At the time of these notes it had not been run.

## What we could not verify

- Any hardware behavior: whether a single-thread release is cumulative over other warps' in-flight reductions, and whether ptxas waits for load completion before an mbarrier arrive.
- The judge harness: whether it deep-copies prepared state or shares references, whether replays of different shapes share a stream, and whether the first call happens outside graph capture. The findings about moe scratch and bf16 counters assume whole-state copies.
- Prepare times for decode_layer and bf16 (moe's is the one number above).
- That every producer load in the decode_layer Triton glue stays below the PDL wait in SASS.
- Multi-hop visibility under PDL, described above.
- The moe fat path, the cluster path and the old TMA skinny kernels, which we treated as dead for scored shapes by reading the plan and variant selection. We did not run them.

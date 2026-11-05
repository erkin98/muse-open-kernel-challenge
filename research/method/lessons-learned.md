# Lessons learned

The technical mistakes and surprises from our kernel work that are most likely to save someone else time.

Lessons about measurement noise, baselines, duplicate checks and the judge queue are in `measuring-on-the-judge.md`. Memory-ordering hazards are in `replay-race-hazards.md`. This file covers what is left: reading profiles, designing memory-bound kernels, and tooling traps.

## A profile gives times and kernel names, not designs

We got this wrong twice in one day, both times by explaining a number before anyone had read the code.

### The 18.7 µs pooler

In the decode_layer profile a kernel took 18.7 µs. The pooler's BF16 weights are 67.1 MB, so that looked like 3.6 TB/s over uncompressed weights. We took it for a plain pooler and projected a +9.5% gain from porting a faster kernel. Reading the file showed (verified) that the kernel was already using the 12-bit lossless weight format, fused with the router, and that the faster kernel we wanted to port handles only m=1. The template argument `COMP=true` in the profile's kernel name had already said "compressed"; we did not decode it. The kernel reads about 59 MB compressed, which by our arithmetic is nearer 3.2 TB/s than 3.6. The realistic gain was about +1 to 1.5%.

### The 6 to 7 µs per expert

In the moe profile, time grew by 6 to 7 µs for each additional distinct local expert touched. We read that as limited parallelism. The kernel turned out to be stream-K over all 132 SMs already, and the slope was exactly the expert's bytes: 28.31 MB at 4.6 TB/s is about 6.15 µs. Divide the bytes by the bandwidth before calling something a parallelism problem. A slope can be bandwidth as easily as serialization.

The rule we took from both: a gain estimate built on how a kernel works stays marked "unverified" until someone has read that kernel.

## Static instruction counts did not predict speed

A decode variant for the moe wgmma consumer loop (we called it DQ8) used an 8-operation decode. It had 5.3% fewer instructions and 25% fewer ALU operations, with the work moved onto the FMA pipe. On the judge it got slower: 5.5% worse on decode 32 and 11 to 13% worse on the 128-token and 256-token verify cases (job 84b9fccd4a804a48, 2.222× against 2.36×).

Our reading, which is inference: that loop is bound by latency and dependencies, not by issue slots or ALU throughput, so fewer instructions bought nothing. Why it got slower instead of staying flat we did not establish. A SASS-only gain on a loop like that is a hypothesis until a paired measurement says otherwise.

## Moving work across a grid barrier only moves the critical path

The moe kernels end phases with a grid barrier. We tried two ways of shifting work across it (variants named FIN2 and WEARLY). Neither helped. The sum-of-squares reductions sit in the epilogue of the last CTAs to finish, so when the final pass gets cheaper the barrier release simply moves later by about the same amount. Only changes that take work off the last-finishing CTAs can help.

## Memory-bound GEMV design

All of this is from the H200 skinny-GEMM work on the bf16 problem.

- FP32 `tl.dot` with IEEE precision is compute-bound in Triton. The FP32 router ran 2 to 4 times slower. Splitting the weights into a BF16 high part and a BF16 low part with FP32 accumulation avoids it, and the rules allow that form for the router.
- Bytes in flight per SM decide memory-bound speed. The same 8.4 MB read (which matches the size of the FP32 router weights) took 3.0 µs with a 32-wide N tile, a 128-deep K tile and 4 stages, and 5.0 µs when each stage held only 4 KB.
- Staging the activation before the weight stream stalls it at m=16 and 32. It ran 10 to 70% slower, and in one clean-room check about 30% slower, so those shapes stayed on cuBLAS.
- A small local GPU cannot rank H200 configurations. An RTX 3050 Ti saturates at about 6 to 7 KB in flight per SM. Sweeps there only tell you whether a kernel is correct.

## Tooling traps

### An offline compile must copy the JIT's specialization

We compiled a Triton kernel for sm_90 ahead of time, with no attributes on the arguments, and saw scalar 2-byte loads (`LDG.E.U16`). We reported that the kernel loads 2 bytes at a time. With the 16-byte divisibility attribute on the pointers, which the real launch adds, the same kernel used 16-byte `LDGSTS` loads. A sanity check would have caught it: the measured 3.0 µs for 8.4 MB rules out scalar loads. Before you report what an offline compile shows, match the launch's alignment and divisibility, and check one known fact first.

### Extension names can get too long

One moe candidate (c2, job 10650a018faf4076) built its `load_inline` extension name from every `-D` flag. The name grew past what the loader accepts, the module failed to import on the judge, and the check silently ran the previous v3 code. The check therefore measured the old code, not the edit. Name extensions with a short hash of the define tuple, and have the prepare log print which builds actually loaded.

### Loads can be hoisted above a PDL wait

The first check of a new 1-token GEMV for decode_layer failed 10 of 13 cases. The compiler had moved the kernel's loads above the point where it waits for the previous kernel, so it read an input that had not been written yet. This was a candidate of ours and we dropped it. `replay-race-hazards.md` explains why `ld.global.nc` is a likely route for this.

### Read the spec before dropping a design on memory-model grounds

We dropped the Triton pattern in which the last block to finish folds the split-K partials. Our reason was that a single acq_rel counter synchronizes only with the most recent writer. The CUDA memory model text (appendix 5.7) says read-modify-write atomics continue a release sequence, so the pattern is correct when block barriers surround it. We had argued from memory while the primary source sat unread. If a decision rests on a memory-ordering claim and you cannot read the spec, say the claim is unverified and keep the decision provisional.

### Check every borrowed file against the precision rules

The decode_layer board entry (job 9b0d0973f11a4758, 13 of 13 cases passing) feeds the router a BF16-rounded activation at T ≤ 8 tokens. SGLang keeps that activation in FP32 there, so by our reading of the precision rule the entry is out of bounds, even though the judge's tolerance did not flag it. We wrote a compliant v3 with an unrounded FP32 copy for the router, falling back to a separate FP32 GEMV and never to BF16. At the time of these notes it had been syntax-checked and nothing more. `challenge-rules-as-we-read-them.md` has the rule.

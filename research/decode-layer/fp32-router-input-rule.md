# muse_decode_layer: FP32 router input up to 8 tokens

The vjeux decode-layer file rounded the router's input to BF16 at 8 tokens or fewer where SGLang keeps FP32; this file covers the rule, our fix, how we checked it without a GPU and what it cost.

## The rule

SGLang's reference path (the task's `baseline.py`) treats the router differently below and above 8 tokens. Up to 8 tokens, the layer's boundary kernel also writes an FP32 copy of the normalized activation, and the router is an FP32 `F.linear` on that copy with FP32 weights of shape [256, 8192]. Above 8, the router runs as two BF16 GEMMs on a hi/lo split of the weights. The constants are `ROUTER_FP32_MAX_T = 8`, a `split_router = tokens > 8` switch, and a router dtype of FP32 when that switch is off.

The organizers' rules page lets the router's FP32 weights run as a BF16 hi + lo pair with FP32 accumulation at any batch size, so the weight side is fine. For activations it says "Activations stay in BF16 or FP32 wherever SGLang's kernel keeps them in BF16". It does not name the FP32 router input. Reading it as "stay FP32 where SGLang keeps FP32" is our interpretation, and it also leans on the page's general line against lower precision than allowed. alygg77 and a poster on a second public code-sharing site read it the same way, and alygg77's notes quote that poster as saying that a kernel feeding a BF16 activation to the router at 8 tokens or fewer does not follow the reference. We agree.

The tests do not catch it: expert picks stay within the 0.05 tolerance with a BF16-rounded input, which is why it is easy to miss. A BF16-rounded copy differs from the FP32 activation by about 2e-3 relative.

Only decode 1 (T = 1) and verify 1 (T = 8) are affected. The bf16_proj problem is not: its reference generates the router's input in BF16 to begin with.

## What the vjeux file does

In `_boundary1`, the normalized activation `hh = s1 * rsqrt(mean(s1^2) + 1e-5) * (1 + w)` is stored only as BF16. There is no FP32 copy, and the parallel `_bnd1_apply` does the same. At T <= 32 with the "bfpf" pick, `_after_attention` calls the fused pooler+router kernel on that BF16 `h`; the router stages multiply BF16 fragments by the hi/lo weight tiles. The other router paths (`_bfp_gemm("router")` and `dense`) also read the BF16 `h`. The board profiles show the fused kernel (`sk2_kernel`, 18.7 us) running on all four shapes with T <= 8.

## What else we checked at T <= 8

Nothing else in that path rounds below SGLang. `s1` is FP32, and so are the post-attention norm (eps 1e-8) and the pre-FFN norm (eps 1e-5). The routing math is FP32: sigmoid of the logit times 0.1767766952966369 plus bias, top-8, renormalization with eps 1e-15. The pooler's input and output stay BF16, as in SGLang's BF16 pooler GEMM, the MoE input is BF16, and routing weights are FP32.

## Designs and their costs, as measured by alygg77

|design|cost|
|-|-|
|separate FP32 GEMV for the router|2.27% of the score (decode 1 +5.5 us, verify 1 +6.5 us)|
|FP32 router fused into the pooler tail|0.23 to 0.24% (verify 1 +2.4 us, decode 1 -0.8 us)|
|FP32 router stages interleaved in the pooler stream|parity (0.06 to 0.09%)|

## Our fix (v3)

`_boundary1` and `_bnd1_apply` now also write an unrounded FP32 copy `hf` of their BF16 output, behind a constexpr flag. At T <= 8, `_after_attention` always routes on `hf`; none of the BF16 router paths is reachable there.

Fused mode, used when it passes the prepare check: the fused pooler+router kernel's router stages read FP32 rows. The producer warp bulk-copies each token's 512-wide FP32 row for the stage (2 KB) into the shared-memory slot, with an evict-last hint and no BF16 x copy. Tokens 0 to 3 go at 0, 2, 4 and 6 KB (the x-tile area), tokens 4 to 7 at 24 to 32 KB, after the 16 KB of router weights. Each lane keeps its own hi/lo B-fragment registers (hi + lo is exact in FP32) and runs `fmaf` in a fixed order: step j, then k, k+1, k+8, k+9. It keeps one partial per token, carried across the four router stages, and then sums over the four lanes sharing a row with two `shfl_xor` steps. The existing fixed-order warp, quarter and last-CTA reduction runs unchanged. The new code is gated on the router's token width being 8, so the 16 and 32 token paths only gain one kernel parameter.

Gemv mode, the fallback: the pooler runs as the existing pool kernel, then a Triton router kernel computes explicit FP32 products with `tl.sum` (no `tl.dot`, which can run as TF32) of the FP32 input against the hi and lo BF16 rows. It uses 64 programs and waits on the previous kernel before the first load.

Prepare tries fused, then gemv. Each mode runs the real chain at T = 1, 2, 3, 5 and 8 on synthetic inputs, once eagerly and once as a CUDA-graph replay: o projection, boundary under programmatic dependent launch, router. The FP32 copy starts as NaN so any read ahead of the boundary's store would show. A mode passes when:

- the logits' relative L2 error against FP64 of `hf` times the hi+lo rows is below 2e-5 (a BF16 input gives about 2e-3, TF32 about 1e-3);
- `hf` is finite and within 1e-5 of an FP64 recompute from `s1`, so it is genuinely unrounded;
- eager and replay agree bit for bit;
- in fused mode, the pooled output equals the BF16-router fused kernel's bit for bit.

If neither mode passes, prepare raises. It never falls back to BF16.

Any later change to this path should keep the router input FP32 end to end, with no BF16 conversion of any kind, not even a lossless split. That keeps compliance checkable at a glance.

## Checks without a GPU

- Diff review of 31 hunks against the vjeux file. Every hunk is the router fix; nothing else is stacked.
- Registers and spills from NVRTC (nvcc's backend), compiled for sm_90a:

|kernel|vjeux|v3|
|-|-|-|
|fused, token width 8 (416 threads, ceiling 152)|87 regs, no stack, no spill|96 regs, no stack, no spill|
|fused, token width 16|86, 0, 0|86, 0, 0|
|pooler, token width 8, not fused|82, 0, 0|82, 0, 0|

  clang compiles both files and ptxas accepts the PTX, but clang's stack frames on the fused kernel (240 B for vjeux's, 368 B for v3) come from clang's own handling of a grid-constant tensor map and a select chain, so we used NVRTC for numbers.
- SASS order: the only branch to the producer warp comes after `griddepcontrol.wait`, and every bulk copy of the router stages, including the FP32 row loop, sits in producer code behind it. The bulk copies before the wait are the weight prologue stages, which never depend on the previous kernel.
- A CPU emulation of the whole router stream, using the file's own tile-unit and stream builders: the producer's row layout, the per-lane decode, the `fmaf` order, both `shfl_xor` steps and the reduction. Relative errors:

|tokens|vs FP64 x @ (hi+lo)|vs FP64 true FP32 weights|BF16-rounded input|k-permuted control|
|-|-|-|-|-|
|1|1.47e-07|2.25e-06|1.50e-03|9.60e-01|
|2|1.59e-07|2.29e-06|1.58e-03|1.04e+00|
|3|1.69e-07|2.53e-06|1.61e-03|1.03e+00|
|5|1.67e-07|2.30e-06|1.67e-03|9.48e-01|
|8|1.63e-07|2.51e-06|1.64e-03|1.02e+00|

  Lanes agree with each other and hi + lo is exact in all cases. The tolerance of 2e-5 sits 100x above the real error and 75x below the BF16-input error. The k-permuted control is a deliberate wrong mapping, and the emulation does catch it.
- The Triton kernels compile offline for sm_90 in all eight variants. Turning the FP32 copy on adds only stores (32 to 48 global stores in `_boundary1`, 16 to 24 in `_bnd1_apply`). The router kernel's loads are plain `ld.global`.
- Graph safety: `hf` and the gemv output are allocated per call and fully written, there is no new zeroing in the timed call, and no addresses are baked into prepare.
- Another team suggested that a generic-proxy shared read followed by an async-proxy bulk write into the same slot needs `fence.proxy.async`. We did not add one. The FP32 path reads the slot with ordinary shared loads and hands it back through the same empty-barrier arrive as the existing ring. See the open question in `../moe/ideas-tried-and-closed.md`.

## Judge result

Job f6918aa1415a469a passed 13/13 at 2.072x. Fused mode passed on hardware: eager and replay logits errors agreed at every T and the worst was 1.73e-07, against the bound of 2e-05. That is the same size as the CPU emulation predicted (1.47 to 1.69e-07 against FP64 of the hi+lo rows, 2.25 to 2.53e-06 against the true FP32 weights). v4moe's log showed a worst error of 1.70e-07.

## What it cost

v3 against the board run, in the four shapes that use the fix. "Normalized" divides by the median speedup ratio of the eight unchanged shapes (0.9959); "cost" is microseconds at the board's SGLang time.

|case|board|v3|raw|normalized|cost us, raw / normalized|
|-|-|-|-|-|-|
|decode 1 sliding|2.220|2.196|-1.1%|-0.7%|+0.84 / +0.52|
|decode 1 full|2.193|2.159|-1.6%|-1.1%|+1.28 / +0.93|
|verify 1 sliding|2.378|2.255|-5.2%|-4.8%|+5.93 / +5.45|
|verify 1 full|2.313|2.239|-3.2%|-2.8%|+3.87 / +3.36|

Kernel deltas from the profiles (one replay each, vjeux to v3, us):

|kernel|decode 1 sliding|decode 1 full|verify 1 sliding|verify 1 full|
|-|-|-|-|-|
|`_boundary1`|4.6 to 5.2|4.5 to 5.4|4.3 to 5.0|4.3 to 5.0|
|fused sk2|18.7 to 19.2|18.2 to 19.3|18.7 to 20.5|19.5 to 21.2|

At T = 1 the end-to-end cost matches the kernel deltas: the 32 KB FP32 store per token in `_boundary1` (about +0.6 to 0.9 us) plus the fused kernel (+0.5 to 1.1 us). At T = 8 the end-to-end cost is about twice the kernel deltas (+2.4 to 2.5 us). The extra part is inside the swing we saw on unchanged code (decode 16 sliding -2.0% and decode 32 full +2.7% in the same run), so we put the real T = 8 cost at +2.5 to +5.5 us. The FP32 router stages are the main T = 8 cost: 8 tokens x 64 `fmaf` per lane per stage on CUDA cores, over twice the activation bytes. The store is the T = 1 cost.

A second run of the same compliant path (v4moe, identical code at T <= 16) lets us pool the two. Each run's offset is its geometric-mean speedup ratio over the two decode-16 shapes against the board run (0.9866 for v3, 1.0039 for v4moe):

|case|board|v3|v4moe|pooled|raw|normalized|cost us|
|-|-|-|-|-|-|-|-|
|decode 1 sliding|2.220|2.196|2.219|2.207|-0.6%|-0.1%|+0.06|
|decode 1 full|2.193|2.159|2.162|2.160|-1.5%|-1.0%|+0.82|
|verify 1 sliding|2.378|2.255|2.361|2.307|-3.0%|-2.5%|+2.78|
|verify 1 full|2.313|2.239|2.202|2.220|-4.0%|-3.5%|+4.29|

Compliance costs about 0.1 to 0.8 us on the T = 1 shapes and about 2.8 to 4.3 us on the T = 8 shapes. Before the run we had estimated -0.3 to -0.5% overall from the added store and stages. The pooled figures work out to about -0.6% on the geometric mean over all twelve shapes (our arithmetic: the four normalized costs sum to -7.1%, divided by 12). The score drop from 2.096x (vjeux's board entry) to 2.072x (v3) is -1.2% raw and about -0.8% after the control offset, with overlapping intervals.

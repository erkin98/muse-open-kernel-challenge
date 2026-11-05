# muse_moe_w4a16: K-reordered weights and 6-bit scales

The wgmma build of the MoE kernel got three changes, K-reordered weights, 6-bit scales and warp-parallel stream-K ranges, and this file covers how they work, how we checked them without a GPU and what the judge measured.

## Starting point

The base is vjeux's solution (vjeux/okc-2026 at commit f35a41e, Apache-2.0): a native CUDA kernel that fuses routing, gate/up, SwiGLU and down, runs stream-K over the 132 SMs, and keeps NVFP4 weights with one E4M3 scale per 16 weights. From 17 tokens up, each consumer warp decodes FP4 codes straight into the A-operand registers of a register-sourced wgmma. Weights arrive through a ring of 36,864-byte stages filled by `cp.async.bulk`.

We went after two things in that build: the instruction count of the decode loop and the number of bytes streamed per stage. That gave three changes, all for the wgmma build only.

## K permutation

A lane's wgmma A fragment needs particular K columns. In the plain layout those columns fall into 16 different scale blocks per 256-K stage, so each lane decodes 16 scale pairs per stage. If the stage stores the K columns in a different order, a lane's columns fall into 4 blocks and it decodes 4 pairs.

The order we use: stage position P = 64 lg + 16 j + 8 q + 2 t + e holds column 64 t + 16 j + 4 lg + 2 q + e, where lg is the load group (0 to 3), j the step inside a group, and q, t, e are small lane and element indices. Lane (g, t) then sees scale block 4t + j at step j of all four load groups. Each lane reads one 8-byte scale word per stage, stored in a [warp][lane][j][h] layout, and builds its scale pairs with byte permutes.

Because the weights are stored in a new K order, the activations must follow. The gather that fills the x tile and the phase-A epilogue that stores the intermediate for phase B write their values to permuted positions. alygg77's version permutes the loads instead. Ours moves the stores; the 16-byte global loads stay unchanged.

The weight side is done once in prepare. The permutation is a bit-exact reorder of the 4-bit codes, and the code raises unless the inverse map round-trips every code.

## 6-bit block scales

The task's weight generator draws block scales log-normally, clamped to [8, 448] (about log2 ~ N(7, 0.4)). That squeezes the E4M3 bytes into a narrow band, at most 0x50 to 0x7E, so a span of 46 or less. The judge logs agree: bytes 93..126 for gate/up and 94..126 for down in check 1, 94..126 and 95..126 in the v3 run.

We store each scale as a 6-bit code, byte minus the tensor's smallest byte, with separate bases for the gate/up tensor and the down tensor. That is lossless as long as the span is at most 63; prepare measures the span and falls back to the plain copy if it is larger. A stage then carries 768 scale bytes per warpgroup instead of 1,024, and the whole stage shrinks from 36,864 to 35,840 bytes, a 2.78% cut in weight traffic. 35,840 is 16 x 2,240 and the per-warpgroup offset 8,960 is 16 x 560, which keeps the 16-byte alignment `cp.async.bulk` requires.

Four 6-bit codes make 24 bits. For each of the two scale sets we keep 16 of those bits in half of a 32-bit word and the other 8 in one byte of a 16-bit word. Decoding is a field extract plus the same multiply-add the byte decode already used, with the base folded into the addend. A CPU emulation of that decode must reproduce every original byte or the pack step raises.

## Warp-parallel stream-K ranges, and one kernel

The x-producer warp used to compute the stream-K stage ranges on lane 0. We changed it to use all 32 lanes: a shuffle scan over job costs, then lanes 0 to 3 each search one boundary. The values are identical (exact 32-bit sums), only the latency differs. keell0renz reported +0.9% at 32 tokens and +0.5% at 128 for this. We expected less, because vjeux already overlaps that prologue with the x gather.

The wgmma build never reads the BIG switch, so we compile and launch only `moe_fused<false>`, for every M. That removes the switch at 64 tokens and about half the device compile time.

## Every change is behind a flag

All of these flags default to 0 (K permutation, the 6-bit scale format with its two bases, the warp-parallel ranges, the single kernel). With them off, the four original builds and the fallback compile to PTX that is byte-identical to vjeux's (checked with `cmp`). The three candidate builds were:

|build|flags|weight copy|
|-|-|-|
|wk|K permutation, one kernel|reordered codes, per-lane scales|
|wk6|wk plus 6-bit scales|reordered codes, 6-bit scales|
|wk6p|wk6 plus warp-parallel ranges|same as wk6|

Comparing wk with the original isolates the permutation, wk6 with wk the scale format, and wk6p with wk6 the prologue.

## What changed in the device code

If you port this, these are the places it touches:

1. A flag block placed before the WGS section, with static asserts. The stage size becomes 8,192 + 768 bytes per piece, four pieces per stage.
2. The weight producer's source stride, expected-transaction bytes and copy size use the new stage size. The ring slot stride stays as it was.
3. The x gather stores each 16-byte chunk as four 4-byte words at permuted shared-memory indexes.
4. The phase-A epilogue stores the intermediate column at the permuted stage position.
5. A new scale-pair routine does the 8-byte load and decode (or the 6-bit decode). The consumer calls it once per stage after the code loads, which also removes the old per-step scale loads.
6. The x-producer calls a warp-parallel version of the range computation.
7. The launch code runs only the one kernel for all M.
8. A small info function reports registers, local bytes and shared memory from the function attributes, so the judge log shows nvcc's real numbers.

Launch configuration is unchanged: grid 132, block 576 for the wgmma builds, dynamic shared memory from the existing helper (three stages of 36,864 bytes plus the x slots).

On the Python side, the weight packer takes a permute flag and a 6-bit base. New helpers build and verify the permutation, apply it to the codes, and pack and re-decode the scales. Extension names carry a new prefix so builds do not collide.

## The prepare-time A/B in check 1, and why v3 dropped it

Check 1 shipped three candidate builds and chose among them inside prepare, on the judge's GPU. The compile plan: base builds started in parallel threads at t = 0, candidates started only if the base builds finished by 120 s, a candidate still compiling at 170 s was dropped, and nothing new began after 200 s (prepare's limit is 300 s).

Each candidate was first gated against its range's default build for correctness at 5, 8, 9, 16, 17, 32, 33, 63, 64, 128 (including a skewed routing), 191, 192, 256, 257 and 400 tokens (relative L2 at most 5e-3, worst row at most 2e-2). A failing candidate was dropped everywhere. Then each range was timed: the default, the default again (an A/A arm to measure in-process noise), and the candidates, with 24 routings, each captured as its own CUDA graph, 20 replays with a 192 MB write between them to cold-start L2, and a t-interval on the geometric-mean ratio. A candidate won a range only if it gained more than 1%, more than the A/A gap, and its interval's upper end was below 1.

The result (ratios against the original build, lower is faster):

|tokens timed|winner|ratio|
|-|-|-|
|8|tiny2 kept|no candidate won|
|16|mid kept|wk6p 1.0694, i.e. slower|
|32|wk6p|0.9753 (wk6 0.9783, wk 0.9958)|
|128|wk6p|0.9800|
|256|wk6p|0.9753|

At 16 tokens wk6p ran 1.0694 times mid's time (interval 1.0582 to 1.0807), so we never extended it below 17.

The A/B is only a filter. gaidar0yegor reported a local +6.6% that became -5% on the judge, so we compared the judge's per-shape table instead (see `results.md`). Once check 1 had picked wk6p for everything above 16 tokens, v3 froze that dispatch, compiled only the four builds it needs, and dropped the gate and A/B. Prepare fell from 85 s to 42 s, and v3's device source is byte-identical to check 1's, so the 96-register, 0-byte-local numbers from check 1's judge log apply to it unchanged.

The dispatch is by token count only and contiguous: 1 to 4 tiny, 5 to 8 tiny2, 9 to 16 mid, 17 and up wk6p. The prepare gate had run wk6p against the original on the judge's GPU at all the sizes listed above, so nothing relies on the scored sizes alone. Speed at unscored sizes is inference from the structure: the stream-K schedule is the original's, every stage streams 2.78% fewer bytes, and the one added cost (four 4-byte shared stores per 16-byte x chunk in the gather) shrinks as M falls, so wk6p should not lose to the original at 17 to 31 tokens either.

If the scale span is above 63, a scale byte is below 0x08 (no byte-permute decode), the wk6p build fails, or the 6-bit round trip fails, v3 falls back to vjeux's copy and the original wgmma build and logs why. The skinny builds never read the extra weight copies.

## Checks without a GPU

We had no local H200, so everything below ran on CPU or in compilers:

- clang to PTX to ptxas for sm_90a: the original builds produce PTX identical to vjeux's, and the candidates compile to 96 registers with no wgmma-serialization warning. clang's own stack and spill figures do not match nvcc's, so we did not use them; the judge log gave nvcc's numbers (96 registers, 0 B local, 231,168 B shared).
- A CPU emulation of the layout: gather and epilogue maps land column C at the right permuted position; a lane's 16 columns per step are scale block 4t + j in all four load groups; the code words read at the kernel's offsets decode to the right codes (24 random pieces, 32 lanes, 4 groups, 4 steps, 4 registers, 2 halves); the per-lane scale bytes are right; the 6-bit decode equals byte x 16 + 0x5C00 in both halves; the warp prefix sum equals the serial one.
- A whole-flow CPU test of prepare and the call path with a fake extension: eligible scales, dispatch at every M from 1 to 800, 384-token chunking, deepcopy of the state, and each of the four fallback conditions.
- The PTX ISA text on `cp.async.bulk`: size a multiple of 16 bytes and 16-byte alignment on both addresses. The mbarrier transaction-count limit is far above 35,840.

## Compliance

Graph-capturable (no host sync, no value-dependent branching in the call), no memset in the timed call, nothing cached from the test data, NVFP4 stays NVFP4 and both repacks are lossless, BF16 activations stay BF16, readable source with no subprocess beyond `load_inline`, no network, no reference import. Dispatch is by contiguous token ranges, not exact scored sizes. File headers keep the Apache notice and the derived-from line, and credit the ideas below.

## Who measured what

- K permutation, alygg77/okc-muse-spark-kernels (their third variant). On the MoE board alygg77 measured decode 32 +2.6%, verify 16 +3.9%, verify 32 +3.1%, about +1.25% on the score. The decode loop drops from 20.12 to 18.44 instructions per weight word, and their notes say the cheaper scale decode accounts for the whole gain. Their second variant spilled 32 bytes (about 2 us). The single 8-byte scale load and our store-side remap are our own design.
- 6-bit scales, keell0renz/muse-open-kernel-challenge: about +2% at 32 to 256 tokens in their notes, with 33 distinct E4M3 values (alygg77 counts 34 and about 3.71 bits of entropy). Another team reported a 5-bit scale-offset scheme with a compression ratio of about 1.00x, so we left the call to the A/B, which picked the 6-bit build.
- Our own judge numbers for the stack, from check 1 against the board run: decode 32 +2.6%, verify 16 +2.0%, verify 32 +1.4%. Do not mix these with alygg77's figures above; they are different runs of different builds.

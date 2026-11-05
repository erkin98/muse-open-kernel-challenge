# muse_decode_layer: the MoE port and the other ideas we weighed

Porting the MoE byte cuts into the decode layer turned out to be a wash, and this file covers the port, why we read it that way, and the other ideas we weighed with their numbers.

## Sizing before the check

The MoE (`moe_fused`) takes a large share of every scored load. From the board run's profile (46 sliding layers plus 16 full layers per load):

|load|T|MoE share of the load|
|-|-|-|
|decode 1|1|27%|
|decode 16|16|45%|
|decode 32|32|43%|
|verify 1|8|51%|
|verify 16|128|55%|
|verify 32|256|51%|

The 6-bit, K-reordered build from the MoE problem (see `../moe/kperm-and-6-bit-scales.md`) applies at 17 tokens and up, so it can reach decode 32, verify 16 and verify 32. Its standalone judge gains on the MoE board were +2.6%, +2.0% and +1.4% on those shapes. Times the shares above, that is about +1.1%, +1.1% and +0.7% on those three loads and about +0.5% on the geometric mean. That is below one run's noise (about 0.8%) and below the interval half-width (1.5%). alygg77 measured the K permutation alone in the layer on the judge, at decode 32 sliding +3.1% and verify 16 sliding +5.2%, roughly twice our estimate.

The layer's MoE source is the MoE board's source minus comments and two dead defines (50 diff lines, 2,670 against 2,682 lines), so the port is a mechanical merge of device code plus the Python packers and dispatch.

## What v4moe changed

The MoE problem's change set (15 hunks) applied to the layer's MoE source with `patch`. Every hunk landed at an offset of -14 or -12 lines with no fuzz, and the difference between the layer's source and the MoE board's source is the same 51 lines before and after the merge, so the layer's MoE device code now equals the MoE problem's exactly. The new flags default to off, so the small builds compile as before.

On the Python side: the same packers and helpers, a base-selection routine that prints its choices, a switch that prepare sets before the builds start, an info print of registers, local bytes and shared memory for the 17+ token build, and a new extension-name prefix. If the 6-bit build fails to compile or its round trip fails, it falls back to the original copies and the original wgmma build, which is v3's behavior. The file header adds credit lines for the K permutation (alygg77) and the 6-bit scales and warp-parallel ranges (keell0renz).

Everything else is v3: T <= 16, the FP32 router, prepare order and the tuner.

Checks without a GPU: a diff review (every hunk is the MoE build, its weight copies or the header); both main builds compile with clang for sm_90a and assemble with ptxas; the MoE problem's CPU emulation passes with this file; and a whole-flow CPU test confirmed the following.

- The 6-bit copy equals the MoE problem's byte for byte, and the original copy equals v3's.
- The 6-bit copy is 8,960/9,216 of the original size (-2.8%).
- A scale span of 63 gives bases and 64 gives none.
- The dispatch is tiny for 1 to 4, tiny2 for 5 to 8, mid for 9 to 16 and the 6-bit wgmma build for 17 to 400, which is the original wgmma build's contiguous range.
- Both problems use the same weight generator, so every scale byte sits within 46 of the smallest and the 6-bit build is on. The prepare log prints the byte ranges to confirm.

We judged it low-risk. It touches only the 17+ token wgmma build and its weight copies, the byte cuts had been judged as one unit on the MoE board, memory shrinks 2.8%, and the device diff is exactly the MoE problem's, whose gather and stage-range code run where the original's did, so there are no new reads before the PDL wait.

## v4moe result

Job 042deb1ca7a94df6: 13/13, 2.090x (2.059-2.121). The log reported scale bytes 93..126 and 94..126 and a 96-register, 0-byte-local, 231,168-byte build for 17+ tokens, and `moe_fused<false>` now runs at T = 128 and 256. The FP32 router check passed again.

Our first reading was negative. The score rose 0.9% raw (2.072x to 2.090x), but the rise sat on the unchanged shapes: median v4moe/v3 ratio 1.0106 on the six shapes at T <= 16 (range 0.9835 to 1.0470), against 0.9904 on the targets at T >= 32 (range 0.9854 to 1.0257). We predicted +0.5 to +1.3% per target shape, which is below the swing on identical code.

A second look added the board run as another baseline. At T >= 17 the board run and v3 run identical code, so there are two baseline samples for each target:

|case|board|v3|v4moe|where v4moe sits|
|-|-|-|-|-|
|decode 32 sliding|1.668|1.698|1.689|between|
|decode 32 full|1.966|2.010|1.982|between|
|verify 16 sliding|2.062|2.051|2.021|below both|
|verify 16 full|2.082|2.061|2.031|below both|
|verify 32 sliding|2.198|2.179|2.235|above both|
|verify 32 full|2.172|2.166|2.191|above both|

Geometric mean over the six targets: board 2.0164, v3 2.0208, v4moe 2.0168. So the port is net zero within noise: no visible gain, no visible loss. We did not rerun the file. One more run would still leave a 0.5% effect inside the per-shape swing.

## Router-input cost recovery: design check

Compliance costs +0.6 to 0.9 us in `_boundary1` and +0.5 to 1.1 us (T = 1) or +1.7 to 1.8 us (T = 8) in the fused router stages. The best recovery design we found drops the FP32 `hf` store: `_boundary1` writes one FP32 scale per token, the fused kernel's producer copies `s1` rows instead of `hf` rows (same bytes), and the consumer forms the normalized value in FP32. That saves about 0.7 us on the boundary and costs 2 extra multiplies per element per token in the router stages (+256 per lane per stage at T = 8, about 0.3 to 0.5 us), for a net of roughly 0.6 us at T = 1 and 0.2 to 0.4 us at T = 8, about +0.2% on the score. That is a design estimate (inference), not a measurement.

A lossless three-piece BF16 split of the FP32 input onto the tensor cores would be faster still. We ruled it out: the router input stays FP32 end to end, so compliance can be checked at a glance. We did not take this design to a check.

## What the tuner picks at 33 to 128 tokens

VkaZas reported that pinning the configuration for 33 <= T <= 128 was worth more than any kernel edit, because their tuner flipped between runs. We looked at what ours does. Its log is off, so the judge log shows no picks, but T = 33 to 64 use the T = 64 bucket and 65 to 128 use the T = 128 bucket, and only T = 128 (verify 16) is scored in that range.

From the profiles, T = 128 puts qkv, o, pooler+router and unpooler all on cuBLAS (two `tss_128x64` calls, a `tss_96x64`, a split-K `tst_96x64` with its reduce), routes through `dense` with an FP32-output GEMM then `_route`, has the v2 glue kernels off, and runs the MoE kernel's main wgmma build. The pin table already fixes qkv at 128 to cuBLAS, and PDL is off above T = 32. The kernel lists at T = 128 are identical in the board run and v3's run: no pick flipped in either. There is nothing to capture by pinning, and no measurement says another fixed configuration is faster, so a pin was not a justified arm. A real candidate would first need an in-chain A/B at T = 64 and 128 that finds a better fixed configuration.

## Closed or parked

- The compressed GEMV in any form (`cgemv-failure-under-pdl.md`).
- Porting dense or bf16-board GEMM wins into the layer. Standalone numbers on warm L2 overstate the gain. lee101 saw the same: many standalone wins vanished in the layer's overlapped timing.
- Trusting an autotuner pick that did not repeat, and stacking two wins without checking the stack.
- A BF16-rounded router input at T <= 8 (non-compliant).
- Closed by other people's measurements: more weight stages before the PDL wait, side streams for pooler and router, L2 prefetch, launch-table search, a cuBLASLt sweep (about 2 us per layer), removing split-K, stream-K qkv (saturates near 96 CTAs), codec GEMV rewrites at decode 1 (-0.78 to -6.5 us), megakernel stage fusing, folding the final boundary into the last-CTA epilogue, a CUTLASS o route.
- Attention early trigger for the o projection: drafted but never checked; alygg77 found early triggers gained at most 0.736 us.
- An 8-op E2M1 placement against our 9-op decode (idea from lee101/openkernels). The ALU-pipe op count is 7 in both, and the decode comment says the ALU pipe is the saturated one, so we expected 0 to 1% on the MoE and 0 to 0.45% on the layer; it needs a new repack table composed with the K permutation and 6-bit scales, plus an exhaustive decode self-check in prepare. Parked at the time. The repo's method notes record that an 8-op dequant variant later regressed on the MoE board, 2.222x against 2.36x (`notes/method.md` at the repo root).
- Another team (bemlerlabs/okc-2026) reports, in their log: 6-bit scales on the skinny MoE builds at T <= 16 were 0.77% worse on the judge (job 402045a4cf644295) and keell0renz ships them off; a cluster and distributed-shared-memory boundary kernel for T >= 17 gave no gain; their 6-bit twin on the T >= 17 builds measured 1.021 to 1.029 at T = 32, 128 and 256 (judge log 0cf64d4a15434555); their decode work ended with no lever above about 1%; their 11-bit weight code is dead. We read this as support for the skinny-build decision and for v4moe.

## Remaining ideas

Every idea here except the last is worth under 1%.

- A higher PDL cap. keell0renz ships 128; ours stops at 32. First prove that every Triton glue kernel on the 64 to 128 token path waits before its first read of producer data (PTX and SASS order, as we did for v3), then measure in the chain. The repo's method notes (`notes/method.md` at the repo root) record a cap of 128 landing inside the noise.
- cuBLASLt algorithm pins at m = 128 and 256 chosen by timing the whole chain (2 to 3.4 us on verify 32 sliding and full). Needs an in-chain measurement first.
- PDL trigger right after the last weight issue. alygg77 caps the gain at about 0.74 us per kernel pair, and keell0renz reports +0.9% on the layer. Our reading of the PTX documentation (inference) is that an early trigger is legal: `griddepcontrol.wait` makes the earlier kernel's writes visible, and data written before a GPU-scope release fence plus the trigger is visible at the next kernel's start even before its wait. Dependents must still wait before their first read of producer data.
- A prepare-time tuner over the complete chain that keeps the base unless a variant wins by 0.4 us or more.
- T = 1 deferred normalization: no boundary-1 launch; the o-projection epilogue writes FP32 router rows, BF16 pooler rows and partial sums of squares. alygg77's decode 1 shapes run about 8 us faster than ours, so this is worth about +1.7% on the geometric mean, the only idea above noise. It is a deep change to the o, pooler and router kernels, and we did not attempt it.

## Where it ended

The submitted file is v4moe, which scored 2.083x as a fresh judge draw (job 8aa6ec880e394012), below the old entry's 2.096x and compliant at 8 tokens or fewer.

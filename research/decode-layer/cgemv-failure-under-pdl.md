# muse_decode_layer: a GEMV that read its input too early

Our first decode-layer attempt, a compressed-weight GEMV for one-token projections, failed 3 of 13 tests on a programmatic dependent launch (PDL) bug, and this file records what happened and what we took from it.

## What we tried

The attempt (job a44e46117b9449f8, "v2") had three parts, all defaulting to the vjeux file's behavior and moving only on a measured win in prepare:

1. At T = 1, replace the qkv, o and unpooler-slice projections with the m = 1 compressed GEMV from the vjeux bf16_proj solution (same commit, Apache-2.0), in its own small extension, with PDL added: weights load before `griddepcontrol.wait`, the activation after it, and the kernel signals dependents early. Prepare compared each matrix with an FP64 GEMV (relative error under 5e-3), compared the PDL launch with a plain launch bit for bit, and kept the GEMV only where its graph replay was more than 3% faster.
2. For each token bucket (1, 8, 16, 32), issue more weight stages before `griddepcontrol.wait` in the sk2 projection kernels: the vjeux file uses 2, we tried 4 and 6. The pooler kernel launches as soon as the boundary kernel starts, so the extra stages could stream during roughly 4.6 us of boundary time.
3. For each bucket from 1 to 256, run the routing kernel with 2 warps instead of 8, to cut cross-warp barriers in its 8 serial argmaxes.

The weights use a 12-bit lossless encoding: planes plus an escape table for the exceptions. The task allows lossless repacking and compression of weights but not a change of precision or value (NVFP4 stays NVFP4, BF16 stays BF16 or smaller, losslessly). The vjeux file's sk2 kernels already use the same family of encoding. A CPU emulation reproduced the BF16 weights bit for bit on the o and unpooler shapes and matched FP64 to 1e-7.

Before launch we compiled the GEMV with clang for sm_90a and through ptxas (31 to 54 registers, no spills), and checked that the module imports.

## What the judge said

```
tests       10/13 passed
first fail  batch=1 ctx=10 layer=sliding  residual update: relative L2 error
            1.039e+00 > 0.0560456 [bound 0.056]
```

The three failing tests were exactly the three T = 1 tests. Prepare had picked the GEMV for qkv and the unpooler slice at T = 1 and left every other knob at its default. The prepare log:

|T = 1 stage|existing kernel us|GEMV us|picked|
|-|-|-|-|
|qkv|13.7|12.4|GEMV|
|o|9.3|10.4|existing|
|unpooler|8.4|7.5|GEMV|
|whole chain|70.1|69.0||

(We read the pairs as existing kernel then GEMV from the picks.) The sweeps of weight stages before the wait (2, 4, 6) gave 69.3, 69.3 and 70.0 us at T = 1, and routing warps 8 and 2 gave 68.7 and 68.5 us. At T = 8, 16, 32, 128 and 256 the knobs that were swept (weight stages up to T = 32, routing warps at all of them) did not move anything beyond noise. So the failure came from the T = 1 GEMV route alone.

## Root cause

The activation `x` was declared `const __nv_bfloat16* __restrict__`, so the compiler emitted its loads as `ld.global.nc`. In the PTX those loads sit after `griddepcontrol.wait`, but ptxas treats non-coherent loads as invariant and hoists them. In the SASS of the unpooler-slice instance, four 128-bit constant loads (x + 0x0, +0x10, +0x1000, +0x1010) issue before the wait instruction, even ahead of the weight loads.

In the full chain, the MoE kernel writes the tensor the unpooler reads after its own early trigger. The GEMV therefore read the MoE output before it was written. The qkv instance could likewise read its input under PDL overlap.

The prepare self-test ran the GEMV standalone, where `x` had been written long before launch, so it could not see the race.

## Why we dropped it

Removing `__restrict__` keeps the loads behind the wait, and is a one-line change. But the compressed planes add about 47 MB of prepare-time memory. alygg77's negative-results list (item 11) reports that 3 MB of extra scratch made unchanged decode-1 and verify-1 shapes 0.5 to 1.3 us slower with identical SASS. We decided not to carry that risk, and dropped the GEMV along with the two knobs, which had not helped anyway. The next version was built on the vjeux file, not on this one.

## Lessons

- In a PDL kernel, never read producer-written data through `const __restrict__` pointers or `__ldg`. Compare the SASS order of loads against the wait instruction (it shows up as `ACQBULK`) for every kernel that depends on the previous one.
- A standalone self-test cannot validate PDL ordering. The test has to run the real chain, ideally with the buffer pre-filled with NaN.
- More weight stages before the wait is a closed lever: alygg77 found that 2 is optimal, and that 4 and 6 slow attention by 1.7 us and verify 1 by 0.9 us. Our prepare also stayed at 2 at every T.
- Routing with 2 warps instead of 8 did not move the chain (68.7 against 68.5 us at T = 1, within noise).

## What other teams reported about this GEMV

- VkaZas reports the T = 1 GEMV working for qkv and the unpooler: in-layer qkv 9.0 us against our 11.7 us, unpooler 3.5 against about 4.8 us. A GEMV that reads the existing sk2 tile-unit stream instead of its own planes would avoid the memory cost; we did not build it.
- A third-party write-up measured the full-layer score: a complete GEMV port moved it from 2.1898 to 2.2007 (barely), a qkv-only port gave 2.1704 (-0.9%), and the GEMV fused with the router gave 2.1859 (below their own baseline).

Given those two reports and our failure, we treated the GEMV in any form as closed for this problem.

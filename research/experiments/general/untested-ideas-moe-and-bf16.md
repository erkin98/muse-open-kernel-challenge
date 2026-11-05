# Untested ideas for the MoE, bf16 and decode-layer problems

This file records what we found when we went through a list of ideas nobody on the team had tried yet for the MoE, bf16 and decode-layer problems, with the evidence for each verdict.

We asked of each idea whether it could move the whole-problem score by more than a judged check can see. A single run of a problem has about 0.8% of noise, and the offset between judge clusters is 1 to 2%, so an idea needs a plausible 2.5 to 3% to justify a check. Most of these fail that test before any GPU time is spent. One was tested on the judge (see [pdl-cap-and-other-checks.md](../decode-layer/pdl-cap-and-other-checks.md)). The attention ideas are in [untested-ideas-results.md](../attention-extra/untested-ideas-results.md).

|Idea|Problem|Verdict|
|-|-|-|
|Stream-K flush constant 3 to 4|MoE|Closed earlier as below noise, not re-measured|
|Per-unit overhead constants in the stream-K cost model|MoE|Untested, expected under 0.5%|
|8-op E2M1 dequant against the 9-op version|MoE|No-go on SASS evidence|
|Deferred normalization at T=1|MoE and decode|Untested, a deep redesign|
|5-bit scale code|MoE|No-go|
|Uniform wgmma descriptors|MoE|No-go|
|Column-major or split-expert reuse|MoE|Untested, upper bound +0.8%|
|Router K-split table, second entry 12 to 16|bf16|No-go|
|Validated route for non-scored m|bf16|Assessed, no change|
|Sub-12-bit codes|bf16|No-go|
|PDL cap 32 to 128|decode layer|No-go, judged (job 304f0ebb702346d2)|

## MoE

### Stream-K flush constant (3 to 4) and per-unit overhead constants

The flush constant was already closed by the MoE experiments as below noise. We did not measure it again.

The cost model that places stream-K work has no per-unit overhead term. The constants in the public entry that does have one differ by a factor of two between the standalone kernel and the in-layer kernel (keell0renz uses 80 and 70 in one, 160 and 140 in the other), so picking a value blind has no basis. Doing it properly needs new cost-model code in a 198 KB kernel and a prepare-time scan of the constant on the GPU. We expect under 0.5% of the score. Untested.

### The 8-op E2M1 dequant layout

The skinny MoE loop dequantizes NVFP4 (E2M1) codes with a 9-op sequence, which we call DQ7: seven ALU ops and two IMADs. A search over layouts found one that needs 8 ops: the word is used as is, rotated two different amounts, and one more copy is made by masking, multiplying by a constant with three set bits and masking again. That is seven ALU ops and one IMAD.

Whether it helps depends on which pipe limits the loop. We built the two skinny kernels (the tiny and mid variants) through clang and ptxas 12.9 for sm_90a and counted instructions in the main loop. Per HMMA, which is per 8-weight code word:

|Pipe|Instructions per HMMA|
|-|-|
|All|15.69|
|ALU|7.75 (LOP3 5.06, SHF 2.06, PRMT 0.50, other 0.13)|
|FMA|6.75 (HMUL2 4.00, IMAD 2.62, FADD 0.12)|
|HMMA|1.00|
|LSU|0.12|

In our model the ALU pipe issues a warp instruction every other cycle, so 7.75 ALU instructions occupy about 15.5 cycles, almost the same as the 15.69 issue slots. Dropping one IMAD takes issue slots from 15.69 to 14.69 but leaves the ALU pipe where it was, so the bound moves from 15.69 to 15.5 cycles at best, a gain of 1.2% of the loop. The kernel also runs close to the DRAM limit (about 4.6 TB/s), which hides most of that.

Caveats. The counts come from the clang route, which produced 166 registers against 149 and 153 in the nvcc build, so they are approximate. And the argument is a throughput model. It does not cover a latency-bound loop, which is possible with 2 warps per scheduler sub-partition. We still think the verdict stands, since the bound is 1.2% of the loop in either reading. The one lever left is an ALU-light dequant that moves the rotates onto the FMA pipe (IMAD.WIDE). We did not try it. A zero-cost-dequant ablation inside prepare would bound it cheaply. The same layout was found independently while the MoE kernel was being worked on; this note only covers the static bound.

Verdict: no-go on the SASS evidence.

### Deferred normalization at T=1

The idea is to postpone the normalization at T=1, which touches the o projection, pooler and router kernels. alygg77's write-up estimates roughly 8 us at decode 1. It is the only idea on this list that is above the noise on its own (about +1.7%). It is a multi-kernel redesign of a 570 KB file with no spare GPU and one shot at the judge, and the decode work deferred it for the same reason. Untested.

### 5-bit scale codes

No-go. The skinny loop is limited by the ALU pipe (7.75 instructions per code word above), and a 6-bit scale field already costs one more ALU op per scale. A 5-bit field costs the same or more.

### Uniform wgmma descriptors

The first reading of the SASS from the clang build showed the kernel's 1024 HGMMA with descriptors held in uniform registers but filled by R2UR from vector registers before each group, which suggested a saving. That reading was an artifact of the clang route. Rebuilt with NVRTC 13.2 plus ptxas, which reproduces the judge's 96 registers and no spill, the loop has one UIADD3.64 per HGMMA and no non-uniform descriptor arithmetic left. A public judge comparison by lee101 (his v53d against v52) also came out as noise.

Verdict: no-go. The lesson is to compile through the judge's toolchain before reading SASS.

### Column-major order or split-expert reuse

Stream-K work ranges sometimes cut one expert's weight tiles into two or more jobs, and the weights are then streamed more than once. From the tile table, 73% of routings at verify 32 split an expert into two or more jobs (6.2% extra weight streams), 30% at verify 16 and 35% at decode 16. We do not know how many of the repeated streams hit L2. The MoE work put the upper bound at +0.8% of the score. Untested. The next step would be a prepare-time diagnostic that times one expert shared by every token's top 8 against four experts at 64 tokens.

## bf16

### Router split constant

The router's K-split table (keys 1 to 4, values 16, 12, 10 and 8) came from a sweep. The idea was to set the second entry to 16 instead of 12. The sweep chose 12, and any change to the router split is worth at most about 0.1% of the score. No-go.

### Behaviour at m values that are not scored

We read how the entry treats m values outside the scored set (1, 8, 16, 32, 128, 256). Dispatch is contiguous buckets: 1, 2 to 8, 9 to 16, 17 to 32, 33 to 128 and 129 to 256. An untuned m takes the choice tuned for the scored m of its bucket. For m above 256 the entry goes to cuBLAS, as SGLang does. For m from 129 to 255 on any layer except the router it also falls back to cuBLAS, because the picks tuned at m=256 lose at partial tiles (pooler CUTLASS at m=129 was 14 us slower, unpooler torchT at m=240 was 11% slower, per comments in the file). The self-test covers two unscored values, 33 and 200.

So the entry already generalizes by buckets with a safe fallback, and nothing dispatches on a scored value alone. What it does not do is time unscored m values, so a pick tuned at the top of a bucket can be slower at the bottom, though never wrong. No change proposed.

### Sub-12-bit codes

A published scheme (ZipServ) uses a 3-bit code per weight over the seven most common exponents, with one code reserved for a fallback. At 96% coverage it averages 11.3 bits, so at most 5.8% fewer bytes than our 12-bit pack. Its decoder needs a popcount rank per element and then a data-dependent load from one of two streams, which is much heavier than our fixed-width decode. No-go.

## Decode layer

The PDL cap idea is covered, with all numbers, in [pdl-cap-and-other-checks.md](../decode-layer/pdl-cap-and-other-checks.md). In one line: raising the cap from 32 to 128 changed only the verify-16 rows, they moved no more than the rows it could not touch, and the in-process A/B at T=128 was 0.3 us.

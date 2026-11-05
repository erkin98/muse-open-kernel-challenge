# Method, per-problem notes and compliance

## How we measured

- `okc check` runs the same tests and timing as a submit but stays off the board. Identical bytes come back with status `duplicate` and cost no quota.
- The judge's single-run noise is about 1% on the score (2.582x to 2.627x for the same attention kernel) and 2 to 5% on the short sliding shapes. We read per-shape rows against the spread of identical-code runs, not the headline.
- No GPU was available locally for the H200 kernels (they target sm_90a). Code was compiled with NVRTC and ptxas for sm_90a, the SASS was read, and numerics were modelled on the CPU before any judge run.
- A file can carry a different launch-plan setting for every scored shape, because the shapes are timed independently. That made an 8-file sweep of all shapes possible in one round.

## Per problem

|problem|what we tried|outcome|
|-|-|-|
|muse_attention_decode|see `attention.md`|2.621x; small real gains from the flat 8-token path and BF16 vector reductions, everything else flat or worse|
|muse_moe_w4a16|K-reordered NVFP4 codes, 6-bit block scales, stream-K constants, scheduling and dequant variants|2.362x; an 8-op dequant variant regressed (2.222x against 2.36x); the other variants were inside the noise|
|muse_decode_layer|the MoE changes above, an FP32 router input up to 8 tokens, a PDL cap of 128 for the glue kernels|2.083x; the PDL cap was inside the noise|
|muse_bf16_proj|a Triton router candidate, a 128-row split variant|1.491x; the split variant gave nothing|
|muse_prefill_layer|a reproduction of SGLang's own layer|0.999x, parity|

## Compliance notes

- Approximate math accurate to BF16, FP8 probabilities, and BF16 or FP16 split-K partials are allowed by the organizers' guidance of Nov 3. Weights stay lossless, the KV cache stays FP8, every key is attended, nothing is memset in the timed call, and the kernels are graph-capturable.
- decode_layer: up to 8 tokens the router reads an unrounded FP32 input with FP32 accumulation, as SGLang does; a BF16-rounded input there is not equivalent.
- The organizers warned about very long files and per-shape variants. For transparency: the decode_layer file is 11,718 lines, and the attention file carries two exact-key launch overrides (`_TUNED`, for verify 32 sliding and decode 1 full) that we assessed and kept.

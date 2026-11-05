# Muse Spark Open Kernel Challenge: erkin98

The kernels we submitted to the Muse Spark Open Kernel Challenge, with notes on what worked and what did not. Everything was judged on the organizers' H200 against SGLang. Each problem has one folder under `kernels/` holding the submitted file byte for byte; `results/SHA256SUMS` lets you check that (`sha256sum -c results/SHA256SUMS`).

## Results

Best run per problem, as reported by the judge. Ranks drift as other teams improve, so none are listed here.

|problem|speedup vs SGLang|judge job|what the file is|
|-|-|-|-|
|muse_attention_decode|2.621x|8c9a79b3381c4707|One fused CUDA kernel per call for paged FP8 attention decode and 8-token verify. Based on the vjeux solution, plus a flat 8-token verify path and BF16 vector reductions.|
|muse_moe_w4a16|2.362x|c663b9dfc629461b|W4A16 routed experts. Based on the vjeux solution, plus a K-reordered weight copy with 6-bit block scales for the wgmma path at 17 or more tokens.|
|muse_decode_layer|2.083x|8aa6ec880e394012|Whole decoder layer. Based on the vjeux solution with the same MoE changes and an FP32 router input up to 8 tokens.|
|muse_bf16_proj|1.491x|6c7a0061944e4339|Dense projections. The vjeux solution plus a Triton router candidate that is used only where it measures faster.|
|muse_prefill_layer|0.999x|06001358a94b40db|A reproduction of SGLang's own decoder layer. Parity, no speedup.|

The judge's single-run noise is about 1%: the same attention kernel, run four times (twice with only a comment changed), scored 2.582x, 2.602x, 2.621x and 2.627x. The board keeps the best run, so a reported score sits slightly above the typical one.

## Layout

|path|contents|
|-|-|
|`kernels/<problem>/submission.py`|the file as submitted (headers keep the upstream attribution and list our changes)|
|`results/board-2026-11-05.md`|scores, job ids and file checksums|
|`results/SHA256SUMS`|checksums of the five kernel files|
|`notes/attention.md`|where the attention time goes, what changed and what did not help|
|`notes/method.md`|how we measured, what we tried per problem, and compliance notes|
|`research/`|the full set of research notes (cost models, experiments with numbers, negative results, method, literature and repo surveys), indexed in `research/README.md`|
|`NOTICE`|credits|

## Running a kernel

With the `okc` command line installed and logged in:

```
okc check muse_attention_decode kernels/muse_attention_decode/submission.py
```

`check` runs the same tests and timing as a submit but keeps the run off the board.

## License

Apache-2.0, see `LICENSE`. Most files are derived from the vjeux solutions (see `NOTICE`).

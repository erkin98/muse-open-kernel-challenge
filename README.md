# Muse Spark Open Kernel Challenge: team erkin98

Five H200 kernels we submitted to the Muse Spark Open Kernel Challenge, the notes behind them, and what we measured along the way. The kernel track closed on 2026-11-05. Every kernel here is the file we submitted, byte for byte.

## Results

Best run per problem, as the judge reported it. Ranks drift as other teams improve, so none are listed.

|problem|speedup vs SGLang|judge job|what the file is|
|-|-|-|-|
|muse_attention_decode|2.621x|8c9a79b3381c4707|One fused CUDA kernel per call for paged FP8 attention decode and 8-token verify. Based on the vjeux solution, plus a flat 8-token verify path and BF16 vector reductions.|
|muse_moe_w4a16|2.362x|c663b9dfc629461b|W4A16 routed experts. Based on the vjeux solution, plus a K-reordered weight copy with 6-bit block scales for the wgmma path at 17 or more tokens.|
|muse_decode_layer|2.083x|8aa6ec880e394012|The whole decoder layer. Based on the vjeux solution with the same MoE changes and an FP32 router input up to 8 tokens.|
|muse_bf16_proj|1.491x|6c7a0061944e4339|Dense projections. The vjeux solution plus a Triton router candidate. The file times it against the existing router kernels and on the judge it lost at every size, so the speed is the vjeux solution's.|
|muse_prefill_layer|0.999x|06001358a94b40db|A reproduction of SGLang's own decoder layer. Parity, no speedup.|

## What we learned

- Most of the speed is borrowed. The public vjeux solutions did the heavy lifting, and our changes added about 0.6% on attention and 1.4% on moe. We say so in every file header.
- The judge is noisy. The same attention kernel, run four times (twice with only a comment changed), scored 2.582x, 2.602x, 2.621x and 2.627x. The board keeps the best run, so a reported score sits a little above the typical one. We read per-shape rows against the spread of identical-code runs, not the headline.
- Attention is held up by fixed costs, not bandwidth alone. The heavy full-attention shapes stream at 3.6 to 3.8 TB/s on the margin and also pay about 7 us (decode) or 15.7 us (verify) per call. Reaching rank 10 would have taken roughly 9% off every full shape and 20% off every sliding shape at once.
- Most ideas lost. An eight-file sweep of launch plans, a one-CTA-per-SM grid, half as many sliding splits, an exp2 bit trick, packed half2 softmax and a bulk-copy gather were all neutral or worse. The negative results are written up with their numbers.

## Where to start reading

- `notes/attention.md` for the attention breakdown in one page.
- `research/method/measuring-on-the-judge.md` for how to read a score and its noise.
- `research/attention/cost-model-and-breakdown.md` for the cost model.
- `research/README.md` for the full index of cost models, experiments, literature and repo surveys.

## Layout

|path|contents|
|-|-|
|`kernels/<problem>/submission.py`|the file as submitted; the header keeps the upstream attribution and lists our changes|
|`results/board-2026-11-05.md`|scores, job ids and file checksums|
|`results/SHA256SUMS`|checksums of the five kernel files|
|`notes/`|short summaries: `attention.md` and `method.md`|
|`research/`|the research notes, indexed in `research/README.md`|
|`NOTICE`|credits|

## Running a kernel

With the `okc` command line installed and logged in:

```
okc check muse_attention_decode kernels/muse_attention_decode/submission.py
```

`check` runs the same tests and timing as a submit but keeps the run off the board. To confirm a kernel file is unchanged: `sha256sum -c results/SHA256SUMS`.

## Credits and license

Apache-2.0, see `LICENSE`. Most files derive from the vjeux solutions in `vjeux/okc-2026`; ideas we borrowed from other teams are credited in `NOTICE` and in the file headers. We only took ideas from other teams' write-ups, never their code.

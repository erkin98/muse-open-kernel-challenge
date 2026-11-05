# Two blog posts people pointed us at

Two public posts were sent to us during the challenge, and this file says what each claims and whether either helps a kernel score.

We did not reproduce text from either, and we could not verify any result in them.

## PyTorch: locality-aware MoE kernel design

URL: https://pytorch.org/blog/accelerating-moe-model/. A Triton fused-MoE GEMM adapted from an open-source inference server, tested on Mixtral 8x7B (8 experts, top-2), FP16, on A100 and H100 at decode-sized batches (M = 1 to 512). The linked code is in pytorch-labs/applied-ai under `kernels/triton/inference/col_major_moe_gemm`; we did not open it.

What it says:

- Split-K work decomposition raises parallelism from M x N tiles to M x N x SplitK, with atomics to combine. About +18 to 20% over a data-parallel kernel.
- Column-major tile scheduling orders the output tiles so that all M-tiles of one weight column run back to back, which keeps that column in L2. Up to 4x on A100 and 4.4x on H100 over row-major and grouped launch in their Triton kernel. Nsight shows L2 hit rate +254%, global loads -49%, and 2.7x at M = 2.
- Why it works: the activation matrix is skinny and fits in cache, the weight matrix is large and comes from DRAM, so the schedule that reuses each weight column the most wins.
- Limits they state: Triton `atomic_add` has no BF16, launch latency needs CUDA graphs, an expert-mapping inconsistency end to end, and the next step was a CUDA kernel with fused dequant for W4A16.

How it maps to our MoE kernel (our reading, then measured on a model of the task's routing):

- Our kernel is native CUDA, fused, stream-K over 132 SMs with NVFP4 weights. Split-K style parallelism over K is what stream-K already gives.
- The column-major idea only pays when several M-tiles share one weight tile. We first assumed the task's shapes give about 8 tokens per expert, one tile each, and so nothing to reuse.
- That assumption was wrong at the larger shapes. Expert popularity in the task's generator is heavy-tailed, so at 256 tokens the hottest local expert gets a median 98 tokens and 73% of routings split an expert into several jobs. At 128 tokens it is 30%. The table and the upper bound (about +1% on the score, a static estimate) are in `../moe/ideas-tried-and-closed.md`. Whether the re-streams already hit L2 is unmeasured.
- The article is a Triton article for FP16 Mixtral. It says nothing about W4A16 stream-K, scale layouts, or Hopper TMA and wgmma, which is where our remaining headroom was.

## Sparks of Weight-Space Programming

URL: https://rikiyat.github.io/blog/weight-space-programming/, by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL), published November 2. Their logs live at github.com/RikiyaT/lm_understand-logs, which we did not open. Everything below is the authors' own numbers.

The claim: a frontier language model can read raw network weights given as text and describe what the network does, and can write or edit weights, with no gradient descent and no training loop.

- Reading: it named the game a small policy network was trained on (4 of 4 right, one at 99% confidence), the shape a small diffusion model draws, the language a small character model was trained on (3 of 3, from the middle layers only), and which attention heads copy earlier text. It failed to tell which of two 270M-parameter copies was sycophantic from weights alone, and succeeded only when it could look at attention patterns on text it had written.
- Editing: changing 3.5% of a failing car-control policy's weights in one try took success from 0% to 100% over 1,024 episodes. Editing 0.016% of a 0.8B-parameter model (131,072 numbers) after about 50 forward passes lifted MATH-500 from 7.9% to 48.1%, GSM8K from 13.8% to 67.1% and IFEval from 21.0% to 62.3%. The edit removes one direction, written by two late-layer MLPs, that causes looping: answers hitting the token limit fell from 93% to 33%. With 10 or 25 forward passes the gains were small.
- Writing from scratch: networks for three MinAtar games beat PPO trained for 100M steps (Breakout 517 against 15.8). A 1.5M-parameter Go network written in Python beats small PPO nets about half the time with no forward passes, and with 50 forward passes beats an equal-size PPO net trained on 20M games.
- Their own limits: on a 270M model that rarely loops, the gain was about one point or less, so editing can remove an obstacle but not add an ability. Capability is limited by context length and tooling, and their benchmark is still being built.

Relevance to the challenge: none for scores. The challenge needs lossless weights and SGLang-matching outputs, so editing weights to change behavior is outside the rules, and the post is about model behavior, not kernel speed. The one loose parallel is process: their Go and 0.8B experiments interleave edits with a small budget of measured runs and keep a change only if it wins more, and more measured runs gave better results (10 and 25 passes gave small gains, 50 a large one). Our measurement budget was judge checks, about an hour each, which argues for spending them on well-prepared candidates instead of many small guesses.

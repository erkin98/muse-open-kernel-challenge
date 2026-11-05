# Challenge rules as we read them

A plain-words summary of the Muse Spark Open Kernel Challenge rules we worked under, with organizer guidance kept apart from our own interpretation.

Part 1 restates what the organizers published on their rules page, updates page and problem pages (openkernelchallenge.dev). It is a paraphrase in our words, current as of Nov 4. Part 2 is what we concluded from it. Where we are guessing, it says so. The live site is the authority, and the organizers say the limits can change.

## Part 1: what the organizers say

### The challenge

Meta runs it, with Paradigm and Andromeda (according to the CLI banner). The goal is to make the Muse Spark Open model run faster, in two tracks, with a $100,000 prize pool.

The kernel track ran from Oct 22, 11:00 am PT to Nov 4, 11:59 pm PT (that is Nov 5, 11:59 am in UTC+4). You write `kernel_fn(data)` in a single Python file. Each problem gets judged on one H200 with no network. Correctness is tested on unseen random inputs within a per-problem tolerance, and one error over the tolerance fails the run. Timing is against SGLang's own kernel, both captured as CUDA graphs, from a cold L2, on shapes taken from serving coding assistants at 1 to 32 users. The score is the speedup over SGLang.

In the serving track you submit a diff to SGLang's `python/sglang`, and it runs on a whole 8×H200 node against a hidden multi-turn coding workload at 1, 4, 16 and 32 concurrent users. The score is the throughput-per-GPU and per-user curve against SGLang. Output has to stay within stock SGLang's own run-to-run variation. This track was first announced as Nov 5 to Nov 19. The Nov 3 update moved it to open Nov 10 at 9 am PT and close Nov 19.

Prizes go to the serving track only: $50,000 and a signed H200 for first, $30,000 second, $10,000 third, $3,000 fourth, $2,000 fifth, $1,000 each for sixth to tenth. Top contestants may be offered a Meta interview. The kernel boards pay nothing. The top 50 on the overall kernel board when the kernel track closes get priority in the serving queue; teams without it can still submit serving runs.

### The problems

`okc init <problem>` gives you `submission.py` (plain PyTorch) and `baseline.py` (SGLang's own kernel). Shape counts and SGLang totals below are from the problem pages. Our notes record those totals without saying exactly what each one sums over.

|Problem|What it is|SGLang total|Scored shapes|
|-|-|-|-|
|muse_bf16_proj|Dense projections `x @ W.T`: qkv_gate [2304,8192], o [8192,1024], pooler [4096,8192], unpooler [8192,4096] in BF16; router [256,8192] in FP32|4.83 ms per decode step|30: m = 1, 16, 32 (decode) and 8, 128, 256 (verify) across 5 layers|
|muse_attention_decode|Gated GQA decode and 8-token verify over a paged FP8 (e4m3) KV cache; QK-norm, RoPE, 2048-key window or full attention, about 80K context; 16 query heads, 2 KV heads, dim 64 per GPU; no weights|3.77 ms|12: batch 1, 16, 32 × sliding/full × decode/verify|
|muse_moe_w4a16|Top-8 of 256 experts, 32 local (expert parallel 8), width 4096, SwiGLU, NVFP4 weights (E2M1 with FP8 scales), post-expert RMSNorm|13.45 ms|6: batch sizes 1, 16, 32, 8, 128, 256|
|muse_decode_layer|A whole decoder layer: projections, attention, router, experts, norms, residuals; communication is the identity|22.62 ms|12, as for attention|
|muse_prefill_layer|The same layer at prefill: 753 new tokens on 80K context (1 and 4 requests) and 16K-token first-turn chunks|387.40 ms per load|8|

When we read the page the best speedups per problem, in the order of the table, were 1.69×, 3.34×, 2.76×, 2.51× and 2.25×. The bf16 leader was at 2.85 ms against SGLang's 4.83 ms.

### Environment and limits

Submissions run in the SGLang dev image for Muse Spark Open with no network. PyTorch, Triton, FlashInfer, DeepGEMM, TileLang, CuTe DSL and CUTLASS are available. Native CUDA is built inside the submission file with `load_inline`, and compile time counts toward the run's time limit but not the score. Every problem except attention has weights and calls `prepare` once with them, untimed, for up to 300 s.

Each problem allows 10 submissions and 20 `okc check` runs a day, resetting at 00:00 UTC. The organizers say the limits may change with load. Leaderboard rows carry an auto-review mark, and final decisions come from manual review.

### What a kernel may and may not do

- It must be capturable in a CUDA graph: no host synchronization, and no allocation or control flow that depends on input values. Branching on shapes and `config` fields is fine.
- Weights may be repacked and compressed but never changed in value or precision. The rules page puts it this way: "BF16 weights stay BF16, or smaller, losslessly". NVFP4 stays NVFP4, and dequantizing inside a kernel is fine. The router is the one exception: its FP32 weights may run as a BF16 high part plus a BF16 low part with FP32 accumulation, as SGLang does above 8 tokens.
- Activations stay BF16 wherever SGLang keeps them BF16, and the FP8 KV cache stays FP8. Attention queries may drop to FP8 E4M3 but no further.
- Attention covers every key it should, with softmax and the weighted sum over all of them. Top-k, sparse and block-skipping schemes are out, and so is skipping values by attention weight.
- Source must be readable. Everything on the GPU is built during the run from source in the file. Precompiled or hand-edited GPU code and compressed or obfuscated source are out. A clarification on Nov 2 came with a short window to withdraw runs that broke this.

### Changes on Nov 3

- Kernels (12:15 pm PT). Newly allowed: approximate math accurate to BF16 precision (fast exp, tanh, rsqrt); attention probabilities P in FP8 E4M3 at any power-of-two scale, with exp2 approximated; split-K and split-KV partial sums in BF16 or FP16; FP16 accumulation for QKᵀ and P·V; activations in BF16 or FP32 where SGLang keeps BF16. The weight, KV, query, every-key and readable-source rules did not change.
- Serving (3:45 pm PT). The track opens Nov 10 and closes Nov 19, the board stays private until it closes, patches must be of a quality SGLang maintainers could merge, and the patch cap fell from 720 KB to 400 KB. The organizers noted that many kernel entries exceed 10,000 lines with shape-specific variants.

### Serving restrictions, briefly

Serving patches apply to SGLang at commit `a55ab7d5`, touch only source and JSON config files under `python/sglang` (not the model loader), and carry no binaries or renames. The model has to stay the same model, with its 8 routed experts, 2,048-token window, norms and parameter names. The DFlash drafter may be sped up but not changed. Each submission gets a whole node for up to 90 minutes, with 1 submission and 3 checks (`okc serve check`) a day. Every request must return exactly 220 tokens, outputs must agree with stock SGLang about as well as it agrees with itself, and time to first token can be at most twice SGLang's at each load.

### Validation

The organizers say they want real speedups that could be upstreamed. They concede the judge misses some gaming, so they will read top submissions and rerun them several times after the close, including on held-out data, and final positions will be the medians of those reruns. Until then every position is provisional. Entries that tamper with timing or the harness, do work outside the timed calls, cache or replay outputs, or compute below the allowed precision will be removed, at the organizers' discretion. A team that finds one of its own runs did that can withdraw it with `okc withdraw <job>`.

Submissions are licensed under Apache 2.0. The judge was adapted from GPU MODE's published reference kernels and kernelbot. The full terms are in a PDF at openkernelchallenge.dev/terms.pdf, which we did not read.

## Part 2: how we read it

These are our interpretations. None of them is organizer text.

### Lossless 12-bit weights are allowed

The public entries at the top of the bf16 board store BF16 weights in 12 bits: the low byte raw, the sign and exponent through a small window, and the rare out-of-window weights patched exactly from an exception list. It reads 25% fewer bytes. We read the lossless-compression wording as covering this, because the decoded weights match the originals exactly. For our own repacks we checked that with a bitwise oracle. We measured the entropy of the weights at 10.46 bits per value, so a 12-bit code sits about 1.5 bits above the floor and there is real headroom.

### Precision is checked against what SGLang keeps

The Nov 3 text lets activations be BF16 or FP32 where SGLang keeps BF16. We read it as one-directional: nothing may be rounded down where SGLang keeps FP32. That is how we found that the decode_layer board entry rounds the router's activation to BF16 at 8 tokens and below, where SGLang keeps FP32. See `lessons-learned.md`. We also checked that the borrowed attention file's FP8 probabilities match SGLang's own FA3 FP8 path.

### Keep it general

The organizers will rerun top entries on held-out data, and they flagged shape-specific variants in many entries. We took that to mean that any trick that works only at the exact scored sizes is a risk, and that nearby shapes are probably part of the final reruns (inference; no one told us that). So we avoided choosing by exact token count and preferred route families and token buckets.

### Tuning in `prepare()`

The organizers describe `prepare` as untimed, with up to 300 s, for repacking. We read timing a handful of candidate kernels there and keeping the winner as inside the rules, since several public entries do it. A heavy replay test of every route (see `replay-race-hazards.md`) is a different matter. Doing work outside the timed calls is on the organizers' list of removable offenses, and we judged a soak a gray area we did not want to test. A public write-up said one team lost judge runs because prepare plus build took 437 s. We have not verified that.

### The run time limit

The task files, not the rules page, say a run has 15 minutes in all, compiling included, and 10 minutes for muse_bf16_proj. A run past that is charged as a timeout, so a kernel that spins forever on a non-resident CTA hangs the run instead of failing a case (our reading).

### Limits come from the live CLI

A cached copy of the rules page said 10 submissions and 20 checks per problem per day. On Nov 4 `okc problems` printed 20 runs and 40 checks. The page does say limits move with load, so quote quotas from the live tool, with the time you read them.

### Rank bookkeeping

The overall board is described only as a ranking across the problems, and its API returns ranks without scores. In one snapshot it listed 1,946 teams with these `solved` counts: 1 problem, 580 teams; 2 problems, 139; 3 problems, 101; 4 problems, 1,126. Five kernel problems exist but the maximum is 4, so one is not counted, and we infer that it is prefill (which has its own board). Of the top 50, 48 had solved all four. One team at #21 had solved only one, so the formula is not a plain per-problem sum, and it is not documented. Our own overall rank went from #1,451 (1 solved), to #1,201 of 1,956 (3 solved), to #449 of 1,957 (4 solved).

### Still open

- Whether local testing is supported without an H200. Our checks all ran remotely.
- Whether there are team rules.
- The contents of the terms PDF.

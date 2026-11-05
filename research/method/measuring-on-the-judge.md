# Measuring on the judge

How the judge scores a run, how noisy that score is, and the habits that kept us from chasing noise.

Shape names follow the challenge's own scheme: decode and verify loads, each at batch 1, 16 or 32, on sliding-window or full-context layers. Verify batch n is 8n tokens, so verify 16 is 128 tokens and verify 32 is 256. Our notes also use short tags such as d1s, v1, d1f and d32. We read d as decode, v as verify, the number as the batch, and s/f as sliding/full. The notes never define them, so treat that reading as ours.

## What a score line means

A result line looks like `2.605×  …  faster than SGLang  95%: 2.565–2.644×`. The speedup is 2.605×. The "95%" labels the confidence interval next to it, 2.565 to 2.644. It is easy to read it as "95% faster".

How a run is timed, from the problem's task file:

- Each shape is captured as a CUDA graph. Our replays and SGLang's take turns on the same GPU with the same fresh inputs and a cold L2.
- A shape's speedup is the median of paired ratios, SGLang's time over ours, with the order swapped between repetitions so that going first or second cancels out.
- The judge subtracts a graph replay's fixed cost (about 4 µs) from both sides.
- A problem's speedup is the geometric mean over its loads of SGLang's load time over ours. Each load counts equally.
- Every run reports a 95% interval, never narrower than ±1.5%. The task file warns that gains inside the interval may be noise.

A headline speedup can look much bigger on a single shape than on the problem. A router at decode m=1 went from 10.4 to 5.4 µs (1.94×) in one of our checks, and the unpooler at decode m=1 read 1.950× in another. Those are per-shape cells. The problem score they sit in was 1.25× and 1.50×.

## How noisy it is

- The same unchanged file moves several percent between runs. A public write-up saw 1.526–1.550× for one file within a day. Our own bf16 entry scored 1.520× on a check and 1.491× on the board.
- Five runs of the same decode_layer runtime code spanned 2.4%.
- In the in-process comparison described below, the spread of paired ratios for identical code was 0.2–0.8% on the flat attention cases, 0–1% on moe d32, v16 and v32, and 2–15% on d1s, v1 and d1f. The small batch-1 cases are the noisy ones.

What we settled on:

1. Count a change only if it wins by more than about 3%, roughly twice the minimum interval.
2. For a 3 to 5% gain, get two independent samples before believing it.
3. Never recheck an unchanged file hoping for a lucky draw. Final ranks come from organizer reruns anyway.

## Use every identical-code run as a baseline

At one point we told ourselves that the shapes we had touched in decode_layer moved 2% the wrong way. That came from a single baseline. For 17 tokens and up (decode 32, verify 16 and 32), the board run and the later v3 file ran identical code, which gives two baselines besides the candidate. Against both, the change was zero: the three geometric means over those shapes were 2.0164, 2.0208 and 2.0168.

The rule we kept is simple. When two or more runs share identical code on a set of shapes, compare against all of them and report the spread before naming a direction. The sign of a difference against one baseline is inside the noise.

A second case of the same mistake: a PDL change in decode looked like +2%, and we put it down to the judge cluster that ran it. A baseline run on the same cluster showed the same shift with unchanged code. It was run-to-run noise, and the five-run span of 2.4% above is the evidence.

## Read what a field measures before using it as a cause

After one check, the unchanged shapes dipped slightly and we blamed 2.5 MB of extra layouts, citing a `prepare_pad` field in the output. A closer reading showed that this field is the judge's random layout padding, not memory our code allocated. The explanation was a guess dressed as a finding.

We have a separate observation that a few MB (about 3) kept allocated after `prepare()` came with 0.5 to 1.3 µs slowdowns on unchanged shapes. Given the above, we would not put weight on it as a cause. So any prepare-time work we add should free its buffers and empty the allocator cache when it finishes.

## Duplicate checks are free

Submitting a check for code that the judge has already run produces a new job id with status `duplicate` and a pointer to the earlier job. With `--no-wait` the CLI prints no hint at launch, and the status shows it within seconds. Our example: job 2edf85bcdab04a1e was a duplicate of 042deb1ca7a94df6.

Duplicates do not count against the daily check allowance and take no judge time. We confirmed this with the counters after the 00:00 UTC reset:

- moe had 5 non-duplicate checks plus 1 duplicate, and `okc problems` showed 35 of 40 left.
- decode had 7 non-duplicate checks (one of which failed) plus 1 duplicate, and showed 33 left.

So checking an unchanged file is a free way to read back its last result, for example to learn which file a job actually ran.

## The queue sets your real budget

On the morning of Nov 4, with a day of the track left, the judge backlog was 1,723 jobs (read from the queue endpoint of the judge's API). Checks that ran in about 4 minutes waited 1h10 to 1h20 in line, so a single run was already about an hour in the queue.

Two consequences. Read the queue before planning how many checks you can fit. And batch independent per-shape candidates into one checked file, so each trip through the queue answers several questions.

## Getting several answers from one check

Where a problem has a `prepare()` (bf16, moe, decode_layer), we compiled several candidates, timed each per shape inside `prepare()` as a CUDA graph from a cold L2, printed the table to the log, and kept a candidate only when it won by a clear margin:

- more than 3% in the bf16 and decode_layer files;
- at least 1% in the moe file, with an A/A control (the same code against itself) and a cap of 200 s so that prepare stays inside its time limit. The noise of that in-prepare A/B was about 0.5%.

A losing candidate then cannot slow the entry, and every check doubles as an A/B on the real H200. Attention has no `prepare()`, so there we split candidates by shape branch and compared across checks.

Timing-based picks have a cost: the choice can differ from run to run, which makes the judged code itself a little unstable. We count that as a pick-stability problem, separate from the replay hazards in `replay-race-hazards.md`.

Three habits made the results readable:

- Read the per-kernel profile after every check. Compare per-load ratios between runs, never only the headline.
- Make the prepare log print which builds actually loaded. One moe check silently ran the previous code because an extension failed to import (see `lessons-learned.md`).
- For levers under 1%, use an in-process twin comparison. Timing stamps go only into extra "twin" copies of the kernels, so the shipped machine code stays byte-identical. Both variants run in the same process, starting from the first eager call. That gave about 0.1 µs resolution on the judge H200 in a single check. Read other checks by the paired speedup ratio per case, not by microseconds.

## Gates before spending a check

Cheap static checks, in the order we ran them:

1. A clang syntax pass on the CUDA source.
2. `ptxas -arch=sm_90a` for register counts and spills.
3. A bitwise oracle for any lossless repack of weights.
4. For one moe candidate set, we also confirmed the baseline compiles to byte-identical PTX, the candidates use 96 registers, and a CPU emulation of the new layout passes.

Local GPU timing is a filter and nothing more. An RTX 3050 Ti saturates its memory system at about 6 to 7 KB in flight per SM, far below an H200, so a local sweep cannot rank H200 configurations. We used it for correctness only.

## What a check cannot show you

Replay-only bugs, such as a float atomic into a workspace that zeroes itself or a missing proxy fence, are reported by other teams at roughly 1 failure in 10,000 to 20,000 replays. A judged run replays each shape far fewer times than that, so a pass says little. `replay-race-hazards.md` has the detection arithmetic.

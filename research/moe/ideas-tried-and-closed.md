# muse_moe_w4a16: ideas we weighed and closed

Most of the MoE ideas we did not ship are listed here with the numbers or reasoning behind each, along with two open questions.

## Closed by other people's measurements

None of these were re-measured by us. The numbers come from alygg77's measured-facts and negative-results lists, keell0renz's notes and reports from other teams. Negative values are losses.

|idea|reported result|
|-|-|
|4th weight slot in the ring|within 1 us either way|
|x-ring geometry|closed, no number in our notes|
|dynamic phase-B tail|a unit takes 18.5 to 19.4 us, the spread it could recover is 2.7 to 8.2 us|
|moving the fences (NOFENCE)|+0.234%, below the A/A spread|
|WGS32 (32-token variant of the wgmma build)|-2.2 to -3.0 us|
|refit cost table|-0.8 to -4.6 us at decode 32|
|second read at decode 16|+1% (the notes do not say whether that is a gain or a cost)|
|compressing whole experts|at most 5% of bytes|
|Marlin or SIMT replacements|closed, no number|
|fused-slice MoE|-5 to -13%|
|L2 prefetch of the first weights|x0.99 to x1.00|
|dual-lane wgmma|-8 to -20%|
|compile-time ring-stage index|does not fit under the 96-register cap|
|flush-warp redesign|-9%|
|alygg77's decoder build for 17 to 32 tokens (B4FIN)|not attempted, large change|
|helper-warp-zero in the mid build|no number|

Two items stay in the "maybe later" column. VkaZas described publishing phase A's partial earlier, with a bubble of about 0.764 + 0.153 x (max tokens / mean tokens) us per expert. We did not try it. vjeux tuned the flush-count constant `OKC_WGSFCI` to 3 for +0.5 to 1%.

## Decisions along the way

- The second version starts fresh from vjeux's file and was written from the formulas in the other teams' notes. No code from alygg77 or keell0renz was reused.
- Every change sits behind a compile-time flag that defaults to off, so the original builds stay bit-identical.
- Candidate builds compile after the base builds, and only as lean one-kernel builds. alygg77 saw prepare take 273 to 312 s against the 300 s limit and VkaZas lost runs to build time.
- Free arms at 16 tokens or fewer (comparing the wgmma builds with mid and tiny2) cost no extra compile.
- We did not free the base wgmma weight copies even when no range picked them: alygg77 saw unchanged code slow down with more allocated memory, and the small builds index into the same weight list. Only unused candidate copies were freed.
- The state holds only tensors, ints, strings and define tuples. The judge may call the kernel on clones of the state, so no module objects, addresses or graphs live in it.
- No replay-race soak. The new builds do not touch the stream-K atomics or barriers; the protocol is the original's.

## 6-bit scales on the skinny builds (closed, inference)

The tiny, tiny2 and mid builds read one `uint4` of scales per lane per unit and decode each scale with one byte-permute and one multiply-add. A 6-bit field needs a shift plus a logic op plus the multiply-add, or a permute plus a logic op plus the multiply-add: one more ALU-pipe op per scale than now, 16 more per lane per unit.

The source comment in the skinny loop says the ALU pipe is the saturated one, since the FP4 decode already uses it. By count, the loop has about 240 ALU-pipe ops per lane per unit (8 chunks x 4 words x 7 for the decode, plus 16 permutes). 16 more is about +7% ALU work against -2.78% bytes. A multiply-high can extract a field only from the top of a word, so we found no layout that keeps all the extra work off the ALU pipe.

This is an op count and a source comment, not a measurement, so it is inference. A report from another team (bemlerlabs/okc-2026) later measured 6-bit scales on the skinny MoE builds at +0.77% worse in judge job 402045a4cf644295, noise level, and keell0renz ships them off too. Our expected sign was negative, so we spent no check on it. It reopens only with a decode that adds no ALU-pipe op, or a measurement showing the skinny loop is bandwidth-bound.

## Token tiles per expert (open)

The PyTorch blog on locality-aware MoE kernels (see `../other-teams/blog-posts.md`) orders tiles so every M-tile of one weight column runs back to back, and reports large L2 gains. It only helps when an expert has several M-tiles. The blog's premise is about 8 tokens per expert, which would mean one tile per expert and nothing to reuse. We checked that against the task's own input generator.

Job rules, read from the kernel: the wgmma build splits an expert with more than 8 routes into ceil(count / 64) jobs of at most 64 tokens. The mid build (9 to 16 tokens) splits into ceil(count / 8) jobs of at most 8. Tiny and tiny2 never split. Every job streams all of its expert's weights again. Stream-K order is job-major, so a split expert's jobs are adjacent in the schedule.

Results from the generator on CPU, 300 seeds per case:

|tokens|distribution|routes on this GPU|active experts|max routes on one expert (median / p95 / max)|routings with a split expert|extra weight streams (share of active experts)|
|-|-|-|-|-|-|-|
|1|normal|2.5|2.5|1 / 1 / 1|0%|0%|
|8|normal|13.6|8.5|3 / 8 / 8|0%|0%|
|16|normal|25.9|11.8|7 / 16 / 16|35.3%|3.71%|
|32|normal|50.1|16.0|12 / 32 / 32|0%|0%|
|128|normal|194.1|22.4|45 / 128 / 128|30.0%|1.65%|
|256|normal|391.7|24.8|98 / 256 / 256|72.7%|6.20%|
|128|skewed (test only)|299.5|3.7|125 / 128 / 128|100%|69.2%|
|256|skewed (no scored shape)|609.8|3.9|249 / 256 / 256|100%|196%|

The premise does not hold here. Expert popularity is heavy-tailed: at 256 tokens the hottest local expert gets a median 98 tokens, and at the 95th percentile every token. So the one-tile argument does not close the idea for verify 32 (6.2% extra weight streams on average), verify 16 (1.65%) or the mid build at decode 16 (3.71%). Decode 1, verify 1 and decode 32 never split.

Whether the extra streams cost DRAM bytes depends on L2 reuse between the split jobs. They run on different CTAs at a stage offset of (job length mod CTA range length): sometimes a few stages apart, which is an L2 hit, often tens of stages apart, which at about 15 us of L2 residency is a miss. We cannot measure that locally.

Upper bound, if every extra stream is a DRAM read today and a perfect schedule turned all of them into L2 hits at zero overhead: about -5% bytes on verify 32, worth about +0.8% on the score, plus +0.2% from verify 16, plus +0.6% for decode 16 in a different kernel. This is a static bound (inference), not a measurement. Applied to v3: 2.3442 x 1.010 = 2.368 at the very best, against the 2.364 bar. A real fix has to process one expert's weight unit for all of that expert's jobs back to back, which touches the stream-K stage-to-job map, the fixup slots and the cost prefix.

Suggested first step, not done: a prepare-time diagnostic that times one expert holding all 256 tokens against four experts holding 64 each. DRAM bytes are equal without reuse, so the time gap measures the reuse.

## Weight-ring release and the async proxy (open question)

The wgmma weight ring is read with ordinary shared-memory loads (codes and scales go to registers), released by an mbarrier arrive after the last load of the stage, and refilled by `cp.async.bulk`. There is no `fence.proxy.async` on that path. This is vjeux's design, so it is in the board entry too, and our changes only moved the scale loads earlier and changed byte counts.

The two readings we have:

- The PTX ISA says memory touched through both the generic and the async proxy needs a cross-proxy fence, and that completion of a bulk copy is followed by an implicit generic-async fence. That covers read-after-write: the generic proxy sees the copy's data. It does not obviously cover our case, write-after-read: generic reads finish, then the bulk copy overwrites the slot.
- CUTLASS's TMA-async pipeline docs describe the consumer's release as ordered through the mbarrier alone, and its producer waits on the empty barrier, then issues the bulk copy with no proxy fence. That is the same pattern as ours.

We could not settle this from the documents, and the ring has passed every judge run. We flagged it as an unmeasured exposure and did not change it. A fix would be its own device-code change with a long soak (a 2,000-replay soak catches a 1-in-10-20k bug only 10 to 18% of the time, so 30,000 or more replays are needed to say anything).

## Why we stopped

After closing the skinny 6-bit and flush-count ideas, only the token-tile reuse remained. Its best case lands at 2.368x, barely above the 2.364x bar, and its realistic share is a fraction of that (inference: L2 already catches the re-streams with a small stage offset, and the interleave adds work of its own). It would also need a diagnostic run, a deep stream-K change and a verification run. No open idea had an expected gain above one run's noise.

## What happened afterwards

The repo's method notes record two later variants on this board: an 8-op dequant variant that regressed (2.222x against 2.36x), and other variants that stayed inside the noise. See `notes/method.md` at the repo root. The submitted file is v3 (see `results.md`).

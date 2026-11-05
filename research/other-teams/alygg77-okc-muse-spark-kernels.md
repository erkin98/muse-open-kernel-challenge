# alygg77/okc-muse-spark-kernels: what the repo shows

Our K permutation and our FP32 router fix both started from alygg77's public repo (19 files, Apache-2.0, commit 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0), and this file covers what they changed on top of vjeux's solutions, the measurements in their docs and how we used them.

We read every prose file in full (README, RESULTS, NOTICE, the timing kit's README, `docs/H200_MEASURED_FACTS.md`, `docs/NEGATIVE_RESULTS.md`). The kernels were diffed against vjeux's files and described in prose, and none of their code is copied here. Links in the repo point to vjeux/okc-2026 (their base), NathenFernandes/muse-bf16-proj, openkernelchallenge.dev, posts on a second public code-sharing site and the Apache license page. The libraries named are CUTLASS, Marlin, SGLang, FA3 and FlashInfer.

## What they changed

|board|their score|ours at the time|their change on the vjeux base|
|-|-|-|-|
|attention decode|2.6530x|2.605x|none: vjeux's file unchanged, header only|
|prefill layer|1.8167x|not entered|none: vjeux's file unchanged|
|MoE W4A16|2.4015x|2.329x|K permutation of the weight codes (+1.25% same-cluster), per-thread CUDA context bind, cuBLASLt pins (code from that site)|
|decode layer|2.2058x|2.096x|exact FP32 router at T <= 8 fused into the pooler (pinned); K permutation at 17+ tokens; deepcopy-safe cuBLASLt holders; context bind|
|bf16 projections|1.5502x|1.491x|custom WGMMA O-projection kernels (FP32 accumulation), pooler tuning, nathenfernandes's o m8 and pooler m8/m16 routes (+0.60%), deepcopy fix|

On attention they scored 0.05 above us with a file that is vjeux's unchanged. That gap is judge noise (inference from the diff plus the noise numbers below).

Decode-layer speedup per case, theirs against ours at the time:

|case|theirs|ours|their lead|
|-|-|-|-|
|decode 1 sliding|2.497|2.220|+12%|
|decode 1 full|2.445|2.193|+11%|
|decode 16 sliding|2.060|1.975|+4%|
|decode 16 full|2.119|2.057|+3%|
|decode 32 sliding|1.750|1.668|+5%|
|decode 32 full|2.025|1.966|+3%|
|verify 1 sliding|2.443|2.378|+3%|
|verify 1 full|2.400|2.313|+4%|
|verify 16 sliding|2.160|2.062|+5%|
|verify 16 full|2.116|2.082|+2%|
|verify 32 sliding|2.319|2.198|+5.5%|
|verify 32 full|2.241|2.172|+3%|

Part of that is noise: their own facts file shows unchanged code swinging by up to about 6% per shape between runs (verify 1 sliding at 99.3 and 109.7 us). The decode 1 lead (+11 to 12%) is not explained by their listed changes, and their FP32 router at T <= 8 costs time rather than saving it.

## Compliance finding

Their facts file (section 5) states the rule we later fixed in our decode-layer file: at 8 tokens or fewer, SGLang gives the router an FP32 normalized activation and an FP32 router weight of shape [256, 8192]; above 8 it uses a BF16 activation with the BF16 hi + lo weight pair. They credit a poster on that site and say they checked it against the task's baseline. Kernels that feed a BF16 activation to the router at 8 tokens or fewer do not follow the reference.

The live rules say activations stay in BF16 or FP32 wherever SGLang's kernel keeps them in BF16, and give only the router weights the hi + lo exception. Tests still pass with a BF16-rounded input (expert picks within 0.05), but it is lower precision than the rules allow. The vjeux decode-layer file does that, so ours did too until we changed it. Details and costs are in `../decode-layer/fp32-router-input-rule.md`. The bf16_proj problem is unaffected, since its reference generates the router input in BF16.

## Facts we used

### Judge noise

- An unchanged bf16 file scored 1.4866 to 1.5201 over 4.5 hours and 1.5259 to 1.5502 over four submits.
- The two clusters they saw (sfo3 and sfo3c) differ by +0.9 to 1.8% per load. A cross-cluster "+4.19%" was really +2.4%.
- Decisions need at least two runs per arm on the same cluster, or a reading of changed shapes against unchanged ones. We read ours the second way.

### Prepare

- Extra scratch memory in prepare matters: +3 MB slowed unchanged shapes by 0.5 to 1.3 us (1.1 to 2.4 us at decode 32). Reuse buffers.
- One MoE file needed 273 to 312 s of prepare against the 300 s limit.
- The judge records a different layout pad on every run: `prepare_pad` ranged from 0.85 to 30.5 MB in 512-byte steps. That is why `prepare_pad` in a judge result is not our own memory use.

### Hardware

- HBM read ceiling depends on size: 4.60 TB/s at 2 GiB, 4.19 at 128 MiB, 3.63 to 3.81 at 66 MiB.
- L2 write is 40 to 50 GB/s per SM, so split-K partials are expensive at m >= 128. Distributed shared memory runs at 30 GB/s per SM.
- Integer ALU work does not overlap HGMMA, which is why in-register decode loses at m >= 128.

### Decode layer

- At T <= 8 the layer is one serial PDL chain with no gaps. Skipping all three glue launches would save only 4.0 to 4.8 us. Two pre-wait PDL stages are optimal.
- Budgets: decode 1 sliding 73.5 us against a 44.6 us floor. Decode 32 sliding 154 us = MoE 84.0 + dense 51.5 (floor 34.9) + glue 8.8 + attention 9.8. Verify 32 sliding 305.6 us = MoE 188.2 + dense 83.4 + attention 14.8 + glue 18.4. The MoE dominates.

### MoE

- The stream is boundary-bound. Each unit end costs 1 to 7 us and the stream phase runs at 3.87 to 4.05 TB/s.
- E4M3 scales carry 3.71 bits, so lossless compression is realistic only for scales (about 5% of bytes in total).
- Their K-permutation write-up says the cheaper scale decode accounts for the whole gain, and that an earlier variant spilled 32 bytes. See `../moe/kperm-and-6-bit-scales.md`.

### Attention

- Full verify is per-SM consumer-bound: 40.5% exp, 17.5% QK plus V transpose, 12.7% PV barrier, 12.5% waits. Decode 1 full and verify 1 full are fixed-latency bound.
- Not measured by them: cold-code warming, a multi-CTA tree merge, and prefetch inside one persistent kernel.

## Their timing method

The timing kit compares a base, a candidate and an A/A arm (the base again). It uses 8 captures with random 0 to 32 MiB pads and 48 ABBA rounds, with Student-t intervals over captures. A gain counts only if its interval is above zero and larger than the biggest A/A difference. We added an A/A arm to our MoE prepare-time comparison for this reason.

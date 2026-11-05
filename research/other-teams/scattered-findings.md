# Findings from other teams, by team

Other teams' public repos and write-ups told us the things in this table, each with a pointer to where it shows up in our notes.

These are second-hand: they come from our reading of public repos and posts, and we did not re-measure them except where a note says so. Job ids are the other teams' own judge jobs as they cited them.

|team|finding|where we used it|
|-|-|-|
|vjeux (vjeux/okc-2026, commit f35a41e)|The base for four of our five kernels. Overlaps the serial stream-K prologue with the x gather. Tuned the flush-count constant `OKC_WGSFCI` to 3 for +0.5 to 1%.|`../moe/kperm-and-6-bit-scales.md`|
|alygg77 (alygg77/okc-muse-spark-kernels)|K permutation of NVFP4 codes; FP32 router at 8 tokens or fewer; timing method with an A/A arm; notes on prepare memory, prepare time and judge noise.|`alygg77-okc-muse-spark-kernels.md`, both problem folders|
|keell0renz (keell0renz/muse-open-kernel-challenge)|6-bit block scales (about +2% at 32 to 256 tokens), on the wgmma builds only and off for the skinny builds; warp-parallel stream-K ranges (+0.9% at 32 tokens, +0.5% at 128); `OKC_WGSFCI` 4 after a 3 to 7 sweep; PDL cap of 128 tokens; PDL trigger placement (+0.9% on the layer).|`../moe/kperm-and-6-bit-scales.md`, `../decode-layer/moe-port-and-other-ideas.md`|
|a poster on a second public code-sharing site|The rule that the router input stays FP32 at 8 tokens or fewer; the cuBLASLt pin code alygg77 reused.|`../decode-layer/fp32-router-input-rule.md`|
|VkaZas|Publishing phase A's partial earlier (bubble of about 0.764 + 0.153 x max tokens / mean tokens us per expert); a working T = 1 GEMV (in-layer qkv 9.0 us against our 11.7, unpooler 3.5 against about 4.8); pinning the tuner's picks at 33 to 128 tokens was worth more than any kernel edit; lost runs to a 437 s build.|`../moe/ideas-tried-and-closed.md`, `../decode-layer/cgemv-failure-under-pdl.md`|
|gaidar0yegor|A local +6.6% became -5% on the judge, so a prepare-time A/B is only a filter.|`../moe/kperm-and-6-bit-scales.md`|
|bemlerlabs (bemlerlabs/okc-2026)|6-bit scales on the skinny MoE builds measured +0.77% worse (judge job 402045a4cf644295); a cluster and distributed-shared-memory boundary kernel for T >= 17 gave no gain; their 6-bit twin on the T >= 17 builds measured 1.021 to 1.029 at T = 32, 128 and 256 (judge log 0cf64d4a15434555); their 11-bit weight code is dead; no decode lever above about 1%.|`../decode-layer/moe-port-and-other-ideas.md`|
|NathenFernandes (muse-bf16-proj)|The o m8 and pooler m8/m16 routes for the bf16 problem, which alygg77 adopted for +0.60%.|not used by us; the bf16 problem is outside the MoE and decode-layer notes|
|trxphxle|Observed that the vjeux upstream skips BF16 rounding points, which is fine under the rule "BF16 or FP32", while rounding the router input down is not.|`../decode-layer/fp32-router-input-rule.md`|
|a third-party write-up on the decode-layer GEMV|Full port moved the score from 2.1898 to 2.2007; qkv-only 2.1704 (-0.9%); fused with the router 2.1859.|`../decode-layer/cgemv-failure-under-pdl.md`|
|lee101 (lee101/openkernels)|A from-scratch effort that stopped at scores below ours; the useful part is its measured negatives.|`lee101-openkernels.md`|

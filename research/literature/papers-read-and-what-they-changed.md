# Papers read and what they changed

This file lists the papers, vendor articles and library documents we read for the four kernel boards, how deeply we read each one, and whether it changed anything in our entries.

Short answer: no paper moved a score. When we started reading we had four entries on the boards: attention 2.605x, bf16 1.491x, moe 2.329x and decode_layer 2.096x over SGLang. Every usable mechanism in the papers was already in those entries (they descend from the public vjeux base), or had already been measured and dropped, or came from a different hardware class. The papers did earn their keep in two other ways. They gave us a way to price ideas before spending a judge check, and the NVIDIA documents gave us a correctness checklist for async copies and replays (see cuda-docs-and-replay-hazards.md).

Depth words used below:

- Full: the whole text was read.
- Method: abstract, introduction and the method or kernel sections were read, the rest skimmed by headings.
- Sections: only the named sections were read.
- Abstract: abstract only.
- Skim: abstract, section list, the opening of the method section and a count of GPU or kernel words. This does not mean the experiments were read.
- Not read.

Many headline numbers in these papers live in figures. The figures are SVG objects and we could not extract values from them, so a figure-only number is marked as such and not guessed.

## Attention and softmax

|Source|Depth|What it changed|
|-|-|-|
|FlashAttention, arXiv 2205.14135 (https://arxiv.org/abs/2205.14135)|Read for mechanism and headline numbers|Nothing. A fused decode kernel never writes the score block anyway.|
|FlashAttention-2, 2307.08691 (https://arxiv.org/abs/2307.08691)|Read|Nothing. The one transferable trick, keep the output un-normalised and divide once at the end, is already in the entry.|
|FlashAttention-3, 2407.08608 (https://arxiv.org/html/2407.08608)|Sections 4.1, C.1 and C.2 only|Nothing. The FP8 head-dim-64 results are figures at prefill shapes (16k tokens, sequence 512 to 16896, H100 SXM5 at 1830 MHz, lengths made divisible by 132 SMs). The main text gives FP8 numbers only for head dim 256. There are no decode or KV-cache numbers in the text. Making lengths divisible by the SM count is what our flat equal-share grid already does.|
|Online normalizer for softmax, 1805.02867 (https://arxiv.org/abs/1805.02867)|Read|Nothing. The (max, sum) combine being associative is the basis of every split-KV merge, ours included.|
|FlashInfer, 2501.01005 (https://arxiv.org/html/2501.01005)|Intro, section 4.2 and appendix B|Nothing. Fig. 8 (bandwidth and FLOPs utilisation at batch 16, lengths 1024 constant, uniform and Zipf, against FlashAttention 2 and 3) is figure only. The text credits the load-balanced scheduler and tile-size selection, both already in our grid. Appendix B, Fig. 12: page-size-1 (vector-sparse) decode is within 1% of contiguous KV at head_dim 128 with 32 heads, while prefill loses about 10% on the FA3 template because TMA cannot gather. That supports dropping the idea that row gather limits decode.|
|PagedAttention, 2309.06180 (https://arxiv.org/abs/2309.06180)|Read|Nothing. The judge fixes page size at 1.|
|Ring Attention, 2310.01889 (https://arxiv.org/abs/2310.01889)|Read|Nothing. It targets multi-device sequence sharding.|
|FlashDecoding++, 2311.01282 (https://ar5iv.labs.arxiv.org/html/2311.01282)|Headings and the opening of section 3|Nothing. Its asynchronized softmax with a unified maximum avoids roughly 20% of partial-softmax synchronisation. We already use a fixed reference plus a rare conditional rescale, which QK-norm makes safe.|
|Flash-Decoding blog (https://crfm.stanford.edu/2023/10/12/flashdecoding.html)|Full|Nothing. Split the KV sequence, run attention per split, write a log-sum-exp per split, reduce. That is the split-KV plus merge our entry has. The micro-benchmark is batch 1, 16 query heads over 2 KV heads of dim 128, A100, f16.|
|FlashAttention-4 blog (https://tridao.me/blog/2026/flash4/)|Read|Nothing on Hopper. It targets Blackwell. Its two ideas, splitting 2^x between the MUFU and an FMA-pipe polynomial, and conditional softmax rescaling, are covered by ideas we had already measured or closed (see attention-papers.md). The paper number 2603.05451 appeared only in search mirrors and we never fetched it.|
|2605.30571, batch-1 decode is memory-bound but not bandwidth-limited (https://arxiv.org/html/2605.30571)|Skim|Nothing new. Fixed per-step overhead keeps batch-1 decode far from peak bandwidth. The judge already times CUDA-graph replays, and the 1.4 us fixed cost and 3.0 us floor in our headroom model say the same thing.|
|2609.12923, dissecting GPU utilisation for inference on Hopper (https://arxiv.org/html/2609.12923)|Skim|Nothing new. bf16 GMMA runs fixed 64-row fragments, so small-batch decode fills a fraction of each fragment. Our swap-AB register-A wgmma puts the weight rows on the 64-row side and the tokens on N, which is the fix.|

## GEMM, MoE and weight compression

|Source|Depth|What it changed|
|-|-|-|
|Stream-K, 2301.03598 (https://arxiv.org/abs/2301.03598)|Read|Nothing new. Our moe kernel is already a grouped stream-K with a job queue. The unused ideas, a grid-size cost model and a cap on collaborators per tile, stayed in the ideas ledger as untested.|
|MARLIN, 2408.11743 (https://arxiv.org/abs/2408.11743)|Read|Nothing. It is an Ampere mma.sync pipeline. Evict-first weights, offline fragment reshuffle, bit-field dequant and striped partitioning are already in the moe entry.|
|Machete (https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)|Full prose|Nothing. Offline weight pre-shuffle, swapped operands so the weights are the register-sourced A of wgmma, TMA and warp specialisation are all in the moe entry. Its own future-work list says batches under 32 are still open, and it gives no numbers for them.|
|Marlin FP4 dequant header in the serving library's repo, dequant.h (https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/dequant.h)|Header located (22,312 characters, E2M1 functions found); body not re-read|Nothing. E2M1 bits are moved into the float's sign, exponent and top mantissa and the result carries a fixed power of two, which is what our entry does. A survey by lee101 counts Marlin's FP4 dequant at 15 operations per 8 values against our 9.|
|PyTorch locality-aware MoE blog (https://pytorch.org/blog/accelerating-moe-model/) and its repo|Blog full, repo scripts and kernels read|Nothing. Column-major tile order needs two or more M-tiles per expert, and at about M/32 tokens per local expert we have one. The published tile-index helper also looks wrong (a reading of the source, not a run).|
|DFloat11, 2504.11651 (https://arxiv.org/abs/2504.11651)|README and 50,000 of 56,614 characters of the paper|Nothing. It decodes into a full BF16 buffer before a normal GEMM, so it adds traffic. The Huffman family is closed.|
|lszxb/bf16_huffman_infer (https://github.com/lszxb/bf16_huffman_infer)|README and a skim of the kernel|Nothing. A fused CUDA-core GEMV for batch up to 8, reaching 80 to 90% of plain BF16 speed.|
|ZipServ, 2603.17435 (https://arxiv.org/html/2603.17435v1)|Abstract, then sections 4.2 (TCA-TBE) and 4.3.2 (decompressor)|Nothing now. Sub-12-bit codes stay closed.|
|Reddit thread on DFloat11 (user-supplied PDF)|All 11 pages|Verdict unchanged.|
|IBM split-K W4A16 Triton, 2402.00025 (https://arxiv.org/html/2402.00025v2)|Saved HTML copy read|Nothing. It compares split-K for skinny M on H100 against a data-parallel Triton baseline, not against cuBLAS, and our moe and bf16 entries already split K.|
|2603.29010 (https://arxiv.org/pdf/2603.29010)|Read|Nothing. Shape-specialised BF16 GEMM, up to 9% over cuBLAS at small M, on B200 only.|
|2606.09682 AutoMegaKernel (https://arxiv.org/pdf/2606.09682)|Abstract level|Nothing. A hand-written bf16 GEMV reached about 63% of peak bandwidth against about 90% for cuBLAS on L4, L40S and A10G, not Hopper.|
|2610.07334, Weight Oracles (https://arxiv.org/abs/2610.07334)|Not read|Cited only by the weight-space blog below, which we judged not applicable.|

The long notes on these are in gemm-moe-and-compression-papers.md.

## Automated tuning, search and thesis work

|Source|Depth|What it changed|
|-|-|-|
|Astra, 2509.07506 (https://arxiv.org/abs/2509.07506)|Abstract and a partial local HTML copy (excerpt without introduction and references); OpenReview not opened|Nothing for scores. It tunes three small SGLang kernels on H100.|
|Minimal Executable Programs, 2512.22147 (https://arxiv.org/abs/2512.22147)|See the conflict note below|Nothing for scores. Its own table shows standalone-to-integrated gaps, which supports our rule that standalone wins are not trusted until they survive the full layer.|
|Jansen, Analyzing Optimization Variables in GPU Kernel Tuning, Leiden bachelor thesis (https://theses.liacs.nl/3374)|See the conflict note below|Nothing. Its parameter ranking comes from A100, A4000, A6000 and AMD cards.|
|Tuning the tuner, Leiden PhD thesis (https://scholarlypublications.universiteitleiden.nl/handle/1887/4301430)|Abstract page only|Nothing.|
|Kernel Tuner optimization docs (https://kerneltuner.github.io/kernel_tuner/stable/optimization.html)|Read|Nothing. Our per-range knob spaces have a handful of values, so brute force beats any search strategy.|
|Weight-space programming blog (https://rikiyat.github.io/blog/weight-space-programming/)|Read in full|Not applicable. It is about editing network weights directly, and the name Astra there is unrelated to the kernel paper above.|

An earlier pass reports reading the Leiden thesis in full and sections 3 to 5 of the Minimal Executable Programs preprint. A later coverage ledger still lists both as not read, and says the Minimal Executable Programs body was behind a bot check. We could not reconcile the two. The numbers in tuning-and-automation-papers.md come from the earlier pass.

## NVIDIA documentation

|Source|Depth|What it changed|
|-|-|-|
|PTX ISA 9.4 (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html): griddepcontrol, async proxy, fences, cp.async.bulk, mbarrier arrive/inval/expect_tx/complete_tx, setmaxnreg, red, ld.global.nc, wgmma async proxy|Instruction pages and the async-proxy subsection; memory-consistency chapters not read|Produced the correctness checklist in cuda-docs-and-replay-hazards.md.|
|CUDA Programming Guide chapters 4.5 (programmatic dependent launch), 4.12 (async copies), 4.2 (graphs), 5.7 (memory model) (https://docs.nvidia.com/cuda/cuda-programming-guide/)|Read|Same.|
|CUTLASS pipeline synchronisation page (https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html)|Read|Nothing beyond the above.|
|threadFenceReduction sample (https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh) and NVIDIA forum thread 244413 (https://forums.developer.nvidia.com/t/is-threadfence-useful-at-all/244413)|Read|Reference pattern for the last-block ticket merge.|
|GTC 2010 session 2238 (better performance at lower occupancy)|Saved copy; slides 1 to 40 read|Little's law, used to explain why a first weight tile lands several microseconds after launch.|

## Generic tuning material we were sent

|Source|What it changed|
|-|-|
|CUDA performance checklist, fused softmax part 2|Generic advice. Nothing for scores.|
|Triton guide paste|Judged unreliable. It calls tl.cache_read, tl.shared_memory and tl.barrier, which do not exist in Triton 3.9.|
|Modular GPU kernel-optimization handbook, ENCCS parallel reduction tutorial|Generic. Nothing for scores.|
|Intel KernelFoundry README|Generic. Nothing for scores.|
|Cursor's kernel-optimization blog and results repository (anysphere/kernel-optimization-results)|Its kernel is Blackwell-only. Its per-shape tuning would meet the organisers' nearby-shape rerun, so we kept general fallbacks.|

## What we did not read

- Figure-only numbers: FlashInfer Fig. 8 and Fig. 12, FlashAttention-3 appendix C.2, the DFloat11 latency-by-batch figure.
- arXiv 2610.07334 and arXiv 2603.05451.
- The ZipServ decode loop and repository (github.com/HPMLL/ZipServ_ASPLOS26 was only named in a search result).
- MARLIN and Machete measurements on Hopper. Machete's figures are images, and MARLIN reports only A10, 3090, A6000 and A100.
- About 25 other papers on workflow and memory topics were skimmed at abstract or section-list depth. None has kernel content, none moved a board, and we left them out of this list.

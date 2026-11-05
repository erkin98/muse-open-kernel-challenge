# Automated-tuning papers, a tuning thesis, and a plan for searching per-range settings

This file covers the Astra paper, the Minimal Executable Programs preprint, a Leiden thesis on tuning parameters, the Kernel Tuner documentation, and a plan for testing several shape ranges in one judge check.

None of these sources contains a lever for our four boards. They are off-target on hardware, kernel class or both. The useful outputs are arithmetic about what a single judge check can resolve, and independent support for two rules we already followed: re-validate in the full pipeline, and do not trust standalone wins.

## Astra (arXiv 2509.07506)

We read a local copy of the arXiv HTML (abstract, roles, setup, tables 2 to 4, case studies, limits; the introduction and references were not in our copy). We did not open the OpenReview version, because a bot check blocked it and the arXiv text is the same paper.

- Method. Four roles run in a loop of five rounds: testing (builds test cases from the baseline kernel and checks candidates), profiling (times candidates), planning (combines correctness and speed signals into edits) and coding (applies them). SGLang CUDA kernels are cut out into stand-alone versions by hand and the result is put back into SGLang by hand and re-validated there. The authors list the manual extraction as the main limitation. Timing is on an H100, 100 repetitions after 20 warm-ups, averaged over several shapes derived from 7B, 13B and 70B models.
- Results against the original SGLang kernels:

|Kernel|Baseline us|Optimised us|Speedup|Single-role speedup|
|-|-|-|-|-|
|merge_attn_states_lse|31.4|24.9|1.26x|0.73x|
|fused_add_rmsnorm|41.3|33.1|1.25x|1.18x|
|silu_and_mul|20.1|13.8|1.46x|1.48x|
|Average|30.9|23.9|1.32x|1.08x|

- Per-shape spread (their table 4): the merge kernel 1.46x, 1.57x, 1.00x and 1.14x on four shapes; rmsnorm 1.07x to 1.33x; silu 1.47x to 1.50x. The authors say nothing told the system to tune a particular shape.
- The single-role run lost on the merge kernel (0.73x) because the generated test inputs were unrepresentative and biased the profiling. The lesson we kept: the test inputs steer what gets kept.
- Optimisations in the case studies: hoist the per-element exp and division out of the inner loop (merge kernel), an intra-warp shuffle reduction followed by a small shared-memory reduction instead of a shared-memory tree (rmsnorm), and half2 vectorised loads with fast-math exp and reciprocal (SiLU).

What it means for us. All three kernels are tiny elementwise or reduction kernels, 20 to 40 us before tuning, on an H100. Our boards are GEMV, attention decode, W4A16 MoE and a fused layer, which are bandwidth or latency bound at much larger sizes. No SGLang attention-decode, MoE or GEMM kernel was tried. merge_attn_states_lse is the same operation as our split-KV merge, and the hoisting idea (compute per-vector weights once, not per element) helps only if our merge recomputes exp and a division per element. We did not check the merge code for that, so it is untested and possibly already in the entry. Fast-math SiLU is the approximate-SiLU row in our importance table, rated small. The comparison of one role against several (1.08x against 1.32x on three kernels) is weak evidence.

## Minimal Executable Programs (arXiv 2512.22147)

Preprint submitted 15 Dec 2025. The authors and title match a PAKDD 2026 chapter (DOI 10.1007/978-981-92-1468-6_19), and we treated the preprint as the same work without diffing them. Source: https://arxiv.org/html/2512.22147v1, sections 3 to 5. The conflict about how deeply it was read is noted in papers-read-and-what-they-changed.md.

- Method. Extract the hot kernel and wrap it into a minimal executable program that allocates inputs, calls the kernel and checks results, under two time constraints: the average kernel time must exceed a minimum significance threshold, and the whole program must run under an upper bound. Each round proposes a few candidates guided by profiler feedback (cache hit rate, occupancy), runs each several times, scores a trimmed mean (drop the lowest and highest runs), and requires an output-equivalence check. The best candidate becomes the next baseline, and the loop stops on a threshold or when gains fall below a small threshold. Two helpers: automatic error repair, and inheriting good patterns from earlier rounds (tiling choice, coalescing) as hints.
- Setup. An RTX 2080 Ti with CUDA 11.5, and a Haiguang DCU with HIP. PolyBench (20 kernels), the AMD APP SDK (12 programs, DCU only) and three HPC hotspot kernels (LAMMPS, MISA-MD, xblue, DCU only).
- Numbers. PolyBench on NVIDIA: average 5.29x standalone, 5.05x after putting the kernel back into the application, against 2.10x for asking directly with no feedback loop. Per-kernel spread is large (CORR 14.6x, COVAR 14.0x, SYRK 1.08x). HPC applications on DCU: average 1.78x standalone and 1.25x integrated, against about 1.00x direct. LAMMPS 1.85 to 1.60, MISA-MD 1.01 to 1.01, xblue 2.48 to 1.13. Abstract-level figures of 7.77x (PolyBench on DCU) and 1.77x (AMD APP SDK) were read in a listing, not checked in the body.
- The authors say the standalone result mostly predicts the integrated one, with occasional larger gaps from compiler, runtime and dataflow differences. Their own table shows such gaps (2MM 3.39 to 1.45, ATAX 2.19 to 1.06, xblue 2.48 to 1.13). They report no standalone gain that turned into an integrated regression.

What it means for us. The isolate-check-time loop is what `local_test.py` plus `okc check` already gives us. The standalone-to-integrated gaps in their own table are independent support for our rule that dense and bf16-board GEMM wins do not carry into the decode layer, so the gap is common and not specific to us. Trimmed-mean timing would only matter for a local A/B filter, since the judge times itself. The hardware and tasks are far from Hopper decode kernels.

## Leiden thesis on tuning parameters

Analyzing Optimization Variables in GPU Kernel Tuning, a bachelor thesis at LIACS, Leiden University, 22 July 2025, 32 pages (https://theses.liacs.nl/3374, PDF https://theses.liacs.nl/pdf/2024-2025-JansenTThomas.pdf). An earlier pass reports reading the full text; a later checklist lists the thesis as never read, so the findings below are that earlier pass's reading.

- It runs no new tuning. It analyses existing Kernel Tuner brute-force cache files of the BAT benchmark kernels (2D convolution, dedispersion, a CLBlast-style GEMM, Hotspot) on A100, A4000, A6000, W6600, W7800 and MI250X.
- Two metrics: Pearson correlation of each parameter with run time, and Sobol' first-order variance-based sensitivity, including pairwise subsets. Total-order indices are left as future work.
- Findings. Block size and loop-unroll or temporal-tiling factors had consistently strong influence. In convolution, use_shmem had a big beneficial effect on average and larger tile_size_x hurt. In Hotspot, temporal tiling came first and unrolling second. In the GEMM, NWG was the most negatively correlated (best) and most sensitive parameter on all Nvidia cards, MWG second, NDIMC third; with clock speed included, the clock dominated. Importance changes by kernel and by GPU architecture. read_only and use_shmem showed odd, possibly artifact-driven significance on A100 and A4000 only, which the author flags as unresolved.
- It quotes related work on a shrinking-sample search reaching around 99% of exhaustive-search performance with far less tuning time, but gives no new cheap-search method itself.

For us: narrowing the search by sensitivity is already how we tune (only bytes-in-flight parameters: tile widths, stages, splits). The ranking comes from A100-class and AMD cards and a CLBlast-style GEMM, and the author says it moves with GPU generation, so we would not import it to an H200 Hopper kernel with TMA, wgmma and stream-K. The better source for cheap search methods is a Leiden PhD thesis, Tuning the tuner (2026, https://scholarlypublications.universiteitleiden.nl/handle/1887/4301430): Bayesian optimisation for GPU tuning, a fast solver for constrained search spaces, and tuner hyperparameter tuning with gains the abstract puts above 200%. We read only its abstract page; the full-text download links returned invalid PDFs twice.

## Kernel Tuner documentation

The default strategy is brute force, and there are 15 optional strategies (Bayesian optimisation, genetic algorithm, differential evolution, simulated annealing, several local-search variants, random sample and others). Random sample covers 10% of the space by default, every strategy can be capped by a time limit or a maximum number of unique evaluations, and local-search strategies accept a start point (https://kerneltuner.github.io/kernel_tuner/stable/optimization.html). Our per-range knob spaces have a handful of values, so exhaustive enumeration beats a clever strategy. The scarce resource is judge checks, not kernel runs.

## What a judge check can resolve

These numbers decided how we planned experiments.

- A gain g on one of the six moe loads moves the score by (1+g)^(1/6). Five percent on one load is about +0.8%, the size of one judge run's own noise (about 0.8%). The headline score therefore cannot decide per-range picks; the per-case table has to.
- Per-shape repeat noise, from two judge runs of the same attention file: about 0.2 to 0.6 us (up to 5%) on 5 to 15 us shapes, and 0.1 to 0.8% on 100 to 190 us shapes. A change under about 0.5 us on a short shape, or under about 1% on a long one, is invisible in one run. A headline score carries a 95% interval of at least plus or minus 1.5%, so an idea needs roughly 2.5 to 3% expected whole-score gain to justify a check.
- Our bounds used 4.1 TB/s as the practical read rate (85% of 4.8) plus 1.4 us fixed cost. Both are assumptions, flagged as such in the headroom table.
- The live rules say the final review reruns top entries on the scored shapes and on nearby ones, with positions coming from the scored shapes, and that an entry much slower on nearby shapes gets a closer look. Nearby is not defined anywhere we found. The updates page notes that many submissions include separate variants for the tested shapes. We took the strict reading: dispatch on contiguous ranges and treat boundary behaviour as real.

## A plan for testing several ranges in one check

Untested design, written down because it saves checks.

1. Pick candidates with a bound filter first. Touch a range only if its gap to the practical bound times the board weight exceeds the detection limit above. By our headroom table, moe and decode_layer are stream-bound (any single idea moved a board 0 to 1.4%), bf16 has room in qkv_gate and the router (about 3 to 4% each if half the gap closes), and attention gaps are mostly latency chains with no first-principles bound.
2. Where a prepare-time harness already exists, brute-force the few settings per range on the judge's GPU with an A/A arm and a margin larger than the A/A deviation seen so far (the in-process A/A at moe M=32 was 0.9853, so 1.5% off; margin 2% or more). Do not add a new harness to a kernel that deliberately removed one: extra prepare memory slowed unchanged shapes.
3. One check applies a different candidate in each contiguous range, leaving at least two ranges on the default as controls. If a control moves by more than its own repeat noise, discard the check's conclusions for the other ranges. Never change two knobs in one range in one check, because chains interact (our stacking failures).
4. Read the answer per range from the per-case table: a candidate wins if the shape time drops by more than max(2 x that shape's repeat noise, 1%) and the controls did not move.
5. Confirm without resubmitting an unchanged file: the next check keeps every earlier winner in its range and changes only the other ranges. A winner that repeats in direction and size is confirmed.
6. Budget: six ranges with three candidates each need 3 checks plus one confirmation, against 18 checks if each range were tested alone. It pays only when the summed expected gain over all ranges is at least 1.5 to 2%.
7. Boundaries are the honest tension. The judge check times only scored shapes, yet nearby shapes will be rerun. What can be done: choose each boundary at a structural breakpoint of the kernel (tile count, split count, tokens per expert) and not between measured points; rely on the judge's own non-scored correctness cases (the decode_layer suite includes 13 tests such as batch 1 with context 10) for correctness, though not for speed; where a prepare-time harness exists, time the two off-grid sizes beside each boundary as a veto, within the 300 s prepare limit (one team was killed at 437 s); and keep the default route as the explicit fallback for anything outside the measured range.

Not worth doing for us: a 2,000-experiment genetic search with a surrogate model, because it needs a GPU fleet and no public detail exists (see social-post-claims.md).

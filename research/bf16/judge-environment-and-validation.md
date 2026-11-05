# Judge environment and validation risk

This file covers what we know about how muse_bf16_proj is timed, what the judge image should contain, which ready-made GPU libraries and DSLs are worth trying, and how to keep a kernel from overfitting the exact shapes that get scored.

Labels: "verified" means we read the primary source, "third-party" means another team reported it, "inference" is our own reading.

## How a run is timed

From the task statement (verified): each shape is timed as a CUDA graph replay, with a cold L2 produced by reading a buffer larger than L2, as a median of paired ratios with the order swapped. About 4 µs of fixed graph-replay cost is subtracted from both sides. The local reference files hold only the reference, the checks and the speed-of-light function, so the timing loop itself is not visible to us.

thethings rebuilt the judge as a local benchmark script and it predicted 1.221x, identical to the official check (third-party). It uses graph replay, a cold L2 from a 256 MB read, and paired alternation. Their first version flushed with a write and mis-measured, so use a read flush in any local script. They also found that freeing a captured graph's memory pool left cuBLAS with a dangling workspace and an illegal address, so keep every captured graph alive.

The gaps between our own kernels count in full, since the scored time is the whole replay. Per kernel in a graph, thethings measured about 0.65 µs before the first CTA and about 0.9 µs after the last, and an SM clock of 1,980 MHz under load. Hardware counters were blocked on their rented pod (ERR_NVGPUCTRPERM).

`prepare()` is untimed for up to 300 s and receives the real weights on the real GPU. Triton and nvcc compile time counts toward a 10-minute run limit. Both public full builds took about 53 to 54 s cold in `prepare()` (nathenfernandes's cold run 53.8 s), and thethings' official run took 1:41 in total, so a few CUDA extensions fit easily. Each `load_inline` module is a separate nvcc and ninja build, so keep their number small. A hard-coded config table filled from check results is the safe first version of any dispatch; a small autotune in `prepare()` with graph replays and a cold-L2 flush is the next step. The submitted file does the latter.

## What the judge image should contain

The baseline names SGLang commit a55ab7d5. Its `python/pyproject.toml` and `docker/Dockerfile` (https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/pyproject.toml and the same path with `docker/Dockerfile`) are build inputs, so the versions below are what the image should contain (verified from those files). Whether the published image matches is unverified, and its tag is mutable, as nathenfernandes's README warns. thethings ran on a rented pod from the same image and reports torch 2.14+cu130, Triton 3.8 and CUDA 13, and they compiled CUDA with nvcc inside `prepare()` in official runs, which independently confirms that nvcc works on the judge.

|Component|Version|Evidence|
|-|-|-|
|CUDA toolkit|13.0.3, base image `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, a devel image so nvcc is present|Dockerfile; its comment says it uses the devel base for the full NVCC toolchain DeepGEMM's JIT needs|
|nvcc, ninja|present; `ninja-build` apt-installed, `ninja` a pip dependency|Dockerfile|
|Python|3.12, venv at `/opt/sglang`|Dockerfile|
|torch|2.14.1 (cu130)|pyproject; also printed at the top of our judge run logs (`torch 2.14.1+cu130 triton 3.8.0 gpu NVIDIA H200`)|
|Triton|3.8.0, bundled with torch 2.14.1|our judge run logs, thethings|
|sgl-kernel|0.4.9|pyproject, Dockerfile|
|DeepGEMM (`sgl-deep-gemm`)|0.2.1.post1|pyproject|
|FlashInfer|0.7.0.post1 plus `flashinfer-cubin`|pyproject, Dockerfile|
|CuTe DSL (`nvidia-cutlass-dsl`)|4.8.0|pyproject|
|quack-kernels|0.6.5|pyproject|
|TileLang|0.1.12|pyproject|
|others|`cuda-tile` 1.6.0rc5, `nvidia-mathdx` 25.6.0, `humming-kernels` 0.1.12, `flash-attn-4`, `apache-tvm-ffi` 0.1.11, `cuda-python` >= 13, `numba` 0.65.1, Hugging Face `kernels` 0.14, Tencent `hpc-ops` built for sm90a|pyproject, Dockerfile|
|CUTLASS C++ headers|not pinned in these files; thethings used 4.5|unverified|

## Ready-made kernels and DSLs

- DeepGEMM `bf16_gemm_nt`: identical to cuBLAS at m = 128/256, since it dispatches there (third-party). No gain.
- CUTLASS sm90 GEMMs (TMA, wgmma, swap-AB, tiles 64 to 256, cooperative, stream-K, split-K; 12 configurations): slower than cuBLAS at m = 128/256 (third-party). The vjeux file still uses a CUTLASS kernel for the pooler at m = 256, where it beats cuBLAS (31.1 against 35.6 µs raw replay).
- cuBLASLt algorithm choice: the best of 8 heuristic algorithms gives about 2% summed over the dense layers, and nathenfernandes ships pinned algorithms for some large shapes. On our H200 checks every configuration was algorithm id 66 and none beat the existing custom pick on any shape; against the plain cuBLAS call alone the best configuration was faster on a few shapes (see large-m-plan-and-results.md). From torch the algorithm cannot be pinned directly; one has to call cuBLASLt through `load_inline`. Whether `torch.backends.cuda.preferred_blas_library("cublaslt")` changes the choice is unverified.
- FlashInfer `mm_bf16` and its TGV GEMM: not usable on H200. The version-pinned docs (v0.7.0.post1, `examples/pytorch/README.md` GEMM table) list backends cudnn, cutlass, tgv and auto with SM support listed as SM100 and later (Blackwell), and `bmm_bf16` is the same. The only SM90 GEMM in FlashInfer is FP8 W8A8, which these rules forbid for BF16 weights (verified).
- quack-kernels 0.6.5 (CuTe-DSL GEMMs from the FlashAttention authors' group) is in the image. Its sm90 small-M performance against cuBLAS is unverified. A public team's CUTLASS sweep lost to cuBLAS, so we expect little.
- CuTe DSL 4.8.0: `cute.compile(jit_fn, *args, options="--enable-tvm-ffi")` returns a callable that accepts torch tensors and a stream, and CUTLASS's own CUDA-graph notebook says compilation must happen outside the graph. So the pattern is compile in `prepare()`, call in `kernel_fn`. Compile time per kernel is undocumented. No team reported a CuTe DSL win on these shapes.
- TileLang 0.1.12: `tilelang.compile(func, out_idx=..., target={"kind": "cuda", "arch": "sm_90"}, execution_backend=...)`. The default `tvm_ffi` backend pre-compiles device code; `nvrtc`, `cython` and `cutedsl` compile at run time. CUDA-graph capture is undocumented. Treat it like any JIT.
- SGLang's `dsv3_router_gemm` only fits K = 7168 BF16 weights; see router-and-small-m-kernels.md.

Our verdict: neither DSL has public evidence of beating cuBLAS on these shapes, while `load_inline` CUDA has evidence at m <= 32 from two teams.

## Triton 3.8 features

Documentation checked is the Triton main branch, which can be ahead of 3.8, so each item needs a probe on the judge.

|Feature|API|Notes|
|-|-|-|
|TMA|`tl.make_tensor_descriptor(base, shape, strides, block_shape, padding_option="zero")`, then `desc.load` / `desc.store`|2 to 5 dimensions, 16-byte aligned base, contiguous last dimension. Needs a global scratch allocator set with `triton.set_allocator` (returns an int8 `torch.empty`); allocate it once in `prepare()` (unverified). thethings: TMA helped their Triton GEMV, 12.6 against 13.9 µs, though not enough to change the picture.|
|Warp specialization|`tl.range(..., warp_specialize=True)` (persistent matmul tutorial)|Hopper and Blackwell persistent GEMMs; no evidence that it helps a GEMV|
|Gluon|`triton.experimental.gluon`|Experimental; too risky on our schedule|
|PDL|`gdc_wait()`, `gdc_launch_dependents()`, `launch_pdl=True`|See the router file|
|`num_ctas`|launch and compile option for clusters|Clusters gave nothing measurable on these shapes|
|`maxnreg`|a CUDA backend compile option|Spelling unverified. Register caps matter: thethings saw a 25% swing from register allocation|
|`tl.atomic_add`|`atomic_add(pointer, val, mask=None, sem=None, scope=None)`, returns the old value|A scalar pointer gives one atomic per program, and the returned ticket can be branched on. Needs a local stress test.|

## Validation risk and screening

We could not read the organizers' screening document. Its public URL returns the site's HTML shell, and the API route answers 401 without credentials. What we do know, from the published rules text: a leaderboard row carries `screening` once its run is judged (screened, pending, under_review or final), and `screened_speedup` on a row whose own run is not screened but whose team has one that is. A job ends as done, failed, rejected or duplicate, and `duplicate` means byte-identical code was resubmitted. On the Nov 4 board we saw only those three values: 12 screened, 10 pending and 1 under review in the snapshot just after midnight, and 15, 3 and 2 in the one taken around 11:55.

As we read the rules text, the final review also tests held-out data and shapes near the scored ones. What we changed or would change to avoid overfitting:

1. Dispatch on ranges of m, not exact m. For example 1 to 8, 9 to 16, 17 to 32, 33 to 64, 65 to 128, 129 to 256, and anything above 256 to cuBLAS. Use runtime-m kernels with masking, with one compiled variant per range, instead of compile-time constants or templates per exact m. The judge's own tests already hit m = 5, 7, 9, 33, 130, 512 and 1000, and m = 2, 3, 15, 17, 31, 33, 100, 200 and 250 should land within a few percent of their neighbours.
2. Respect `cfg["n"]`. The unpooler is 1024 rows at m <= 8 and 8192 above, so do not hard-code the scored n.
3. Make the compressed format tolerant of held-out weights: window chosen per matrix in `prepare()`, an exact exception list, and a fallback to plain BF16 if the exception rate is too high (details in lossless-weight-packing-and-formats.md).
4. Run a correctness sweep before every check. thethings ran 420 cases. We can run a cheap version locally on the RTX 3050 Ti (sm_86), which has no nvcc, but a Python environment with NVRTC 13.2 and cuda-bindings 13.4.4 compiles CUDA C++ for correctness testing. `mma.sync.m16n8k16` BF16, `ldg.128`, `prmt` and `lop3` all exist on sm_86; only wgmma, TMA and clusters need the H200. The rules explicitly allow CUDA or C++ through `load_inline` or NVRTC, and NVRTC compiles faster than nvcc + ninja + g++. Whether an NVRTC-launched kernel captures inside `torch.cuda.graph` on the judge is expected (it is an ordinary driver launch) but unverified.
5. Keep a margin rule: ship a custom path only where it beats SGLang by 3% or more. Per-shape run noise is 1 to 2%.
6. Source rules: readable source only, no compressed or base64 CUDA strings, generating code from readable templates is fine. Keep the file well under 1 MB.
7. Submissions that start processes are rejected (for example `subprocess`, or calling `nvcc --version`). Use the Python APIs that report versions instead.
8. Never use L2 persistence, `evict_last` or similar tricks to survive the cold-L2 flush. That games the measurement. thethings refused it for that reason and so do we.

## Reverse engineering: what we did and did not do

- cuBLAS and nvjet are SGLang's baseline. The NVIDIA SDK EULA (section 1.2) forbids reverse engineering, decompiling or disassembling any portion of the SDK, so we did not. What is fair game: the kernel names in the judge profile (which encode tile and stages), the per-kernel GPU times, and the public cuBLASLt API (`cublasLtMatmulAlgoGetHeuristic` and `cublasLtMatmulAlgoConfigGetAttribute` report tile, stages and split-K), which `prepare()` can print.
- Other teams' leaderboard kernels stay on the judge and no API returns them. Extracting them would interfere with the judge's integrity (the terms' section 11b), so we did not try.
- Public repositories with Apache-2.0 or MIT licences are plain source. Reading them for ideas is allowed; building on one, with attribution, is what the vjeux-based file does.
- Our own kernels can be inspected offline: we compiled v3's partial kernel for sm_90 with Triton's bundled ptxas and cuobjdump and read the SASS, no H200 needed.

## Open questions

|Unknown|Why it matters|Cheapest resolution|
|-|-|-|
|Actual package versions in the published image|API names for DeepGEMM, CuTe DSL, Triton PDL|print versions from `prepare()` and read the run log|
|Does Triton 3.8 have `gdc_wait` and `launch_pdl`?|PDL for two-kernel designs|check with `hasattr` in the same probe|
|Does NVRTC via cuda-bindings work on the judge, and does an NVRTC-launched kernel capture in `torch.cuda.graph`?|faster compile; same path as local tests|compile and launch a trivial kernel in `prepare()`, capture it once, replay it, compare|
|Is there a 1 MB submission size limit?|large inline CUDA strings|stated only in nathenfernandes's README; keep the file well under 1 MB|
|What does the organizers' screening check?|criteria for review|the screening document, which we could not read|
|Third-party numbers (thethings, nathenfernandes)|every µs estimate in these notes|treat as an upper bound on what an expert gets; our own checks measure our own|
|Exception rate of the 12-bit format on held-out weights|speed under final review|choose the window per matrix in `prepare()` and log the rate|

## Sources

- SGLang pinned files: https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/docker/Dockerfile
- thethings: https://thethings.ai/p/okc/muse-bf16-proj-report and https://thethings.ai/p/okc/muse-bf16-proj-guide/03-strategy
- nathenfernandes: https://github.com/NathenFernandes/muse-bf16-proj
- Challenge site: https://openkernelchallenge.dev/docs/SCREENING.md (returned the HTML shell when we fetched it)
- GPU MODE reference kernels, public problem definitions the judge looks derived from (our reading): https://github.com/gpu-mode/reference-kernels

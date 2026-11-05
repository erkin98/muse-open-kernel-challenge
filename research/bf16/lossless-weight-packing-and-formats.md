# Lossless weight packing for the BF16 projections

This file covers why a lossless 12-bit weight format pays on the decode shapes, how the public format works, where it loses time, and how to keep it exact on weights we have not seen.

Labels: "verified" means we measured it or read the primary source, "third-party" means another team reported it, "inference" is our own reading.

## Why read fewer bytes

At m <= 32 every BF16 layer is bandwidth-bound, so the task's floor (bytes over 4.8 TB/s) can only be beaten by reading fewer bytes. The challenge rules allow this: the published rules text says BF16 weights "stay BF16, or smaller, losslessly", and dequantizing inside your own kernels is fine.

The judge shows it clearly. nathenfernandes Fernandes' v18 has four cells under the 16-bit floor (pooler at decode 1, decode 16 and verify m=8, unpooler at decode 16; numbers in where-the-score-comes-from.md). In the file we submitted, which uses the same kind of format, pooler at decode 1 ran in 12.9 to 13.6 µs across four checks against a floor of 13.99 µs, and 13.68 µs in the profile of the scored run. Unpooler at decode 16 ran in 13.7 to 14.0 µs against 14.06. At decode 32 it did not hold: pooler 14.7 to 15.2 µs and unpooler 14.3 to 15.2 µs, back above the roughly 14 µs floor, so the saving shrinks as m grows and the unpacking starts to compete with the math.

How much is compression worth by itself? thethings measured an uncompressed CUDA GEMV stack at 1.26x and the same stack with 12-bit weights at 1.33 to 1.34x (third-party; 1.331x scored, 1.343x on a check). So roughly +0.07x. Our own estimate on top of an uncompressed GEMV was about +0.08x.

## What the weights look like

The task generates weights as `randn / sqrt(K)` cast to BF16. We measured the entropy on the CPU over all four BF16 shapes (verified):

|Field|Entropy (bits)|
|-|-|
|exponent|2.54|
|high byte (sign plus top 7 exponent bits)|2.62 to 2.71|
|low byte|7.97|
|whole BF16 value|10.46, which is 65.4% of 16|

The mantissa is noise, so only the sign and exponent bits compress. A fixed 12-bit code is about 1.5 bits above the entropy. We did not look at variable-length codes; decoding those inside a GEMV inner loop is a different cost question.

## The 12-bit layout

This is how nathenfernandes's README and the thethings report describe it (third-party; the file we submitted carries vjeux's version of the idea).

- The low byte of each weight is kept raw. That byte holds the exponent's lowest bit and the 7 mantissa bits.
- A 4-bit field holds the sign plus a 3-bit index into a window of 8 consecutive values of the 7 high exponent bits. By our reading (inference) that covers 16 consecutive exponent values, which is why a 3-bit index is enough.
- Weights outside the window, about 0.014%, are stored clamped and corrected exactly from a per-row exception list by one extra warp.
- Unpacking costs 12 `prmt` / `lop3` instructions per 8 weights.

Our CPU numbers on how many weights fall outside the window (verified): a code with sign, a 4-bit index over the 16 most common exponents and 7 mantissa bits leaves 0.7e-4 to 1.0e-4 of the weights as escapes, about 1,300 to 1,900 in qkv_gate. A 3-bit code over only 8 exponents leaves about 1.3%, which is too many to patch cheaply. The later rs128 variant of our file logged its escape counts on the judge: pooler 4,659 and 4,583, unpooler 3,258 and 3,168, o 930 and 990 in two runs. Those are all around 1e-4 of the weights. The counts differ between runs, which suggests the weights are regenerated each run (inference).

## Where it helps and where it loses

|Shape and m|Effect of 12-bit weights|
|-|-|
|pooler, qkv_gate, o at m <= 16; unpooler at m = 16|13 to 20% faster (third-party)|
|unpooler slice at m <= 8|about 2% slower, the shape is too small (third-party)|
|qkv_gate and o at m = 32|slower (third-party)|
|router|thethings found compressing its hi/lo pair slower; the router is too small and latency-bound|
|m = 256 anywhere|no help, the shapes are compute-bound|

Our arithmetic for the saving per shape at m = 1, on top of an uncompressed CUDA GEMV (estimates): pooler 17.1 to 13.6 µs, qkv_gate 11.2 to 9.5, o 6.1 to 5.3, and unpooler at m = 16 from 17.9 to 14.9.

## Inside the m = 128 GEMM

Putting the format inside a tensor-core GEMM at m = 128 lost every time we have evidence for. alygg77 reported losing 2.4 to 4.1 µs on o and unpooler and 12 µs on the pooler, and noted that integer unpacking does not overlap HGMMA. We then built our own version of this (the "rs128" design, described in large-m-plan-and-results.md) and it was far slower than the existing picks at m = 128.

Before building it we did an arithmetic check on whether the unpacking could hide behind the tensor work (inference). From 989 TFLOP/s over 132 SMs at 1.83 GHz we get about 2,047 BF16 multiply-accumulates per clock per SM. One wgmma m64n128k16 is 131,072 MACs, roughly 64 tensor-core clocks. Unpacking the 8 weights a thread holds per k16 step is 12 integer instructions plus about 3 shared-memory loads, so around 15 issue slots per 64 clocks on each sub-partition, about 25%. That fits only if the unpacking of the next fragment overlaps the asynchronous wgmma through double-buffered A registers, and that overlap is what alygg77 says did not happen in his version. Our build bore him out in one way: the escape-correction phase was small (about 2,100 to 2,900 cycles per CTA on pooler and unpooler, under 1,000 on o) and the main loop was where the time went. See the large-m file.

## Staying exact on weights we have not seen

The final review may use held-out data, possibly with different weights. What helps:

- Choose the exponent window per matrix in `prepare()` from the weights actually passed in, as thethings does, and log the exception rate.
- Keep the exception list exact. Test it bit for bit with one-hot inputs at the exception columns.
- Fall back to plain BF16 for a layer if the exception rate passes a threshold. We had planned 0.1%; the vjeux file corrects escapes from a per-CTA table and rejects a layer above 1 in 200.
- Run a correctness sweep before every check. thethings ran 420 cases: 2 weight seeds, 5 layers, 21 values of m, normal and outlier inputs. The vjeux file also checks each custom path against an FP64 reference at start-up and disables any path that fails, so a wrong kernel costs a check run but does not sink the entry.

## Rules that border this technique

- A TF32 router is not allowed. The rules allow the router's FP32 weights to run as a BF16 hi + lo pair with FP32 accumulation, and nothing else. TF32 keeps 10 mantissa bits, a relative weight error of about 2^-11 per element, around 2.8e-4 relative L2 by our arithmetic, which would also miss the router's 1e-4 tolerance.
- FP8 anything is out for BF16 weights.
- Compressed, encoded or obfuscated source (base64, zlib, xz and the like) has been forbidden since the Nov 2 rule update. One public README says its CUDA source strings were compressed to fit a 1 MB submission limit; we have not verified that such a limit exists. Keep source readable and well under 1 MB.

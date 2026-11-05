# muse_decode_layer: judged runs

The decode-layer kernel scored between 2.072x and 2.105x in the passing runs below, which also show how much of the per-shape movement is noise.

## Shapes

Twelve scored shapes: decode and verify at 1, 16 and 32 requests, each on a sliding-window layer and a full-attention layer. The token count T that reaches the layer's projections and MoE is 1, 16 and 32 for decode, and 8, 128 and 256 for verify (8 tokens per request).

|case|T|
|-|-|
|decode 1|1|
|decode 16|16|
|decode 32|32|
|verify 1|8|
|verify 16|128|
|verify 32|256|

The MoE uses its small builds up to T = 16 and its wgmma build from T = 17. That matters for reading the controls later.

## Runs

|run|job|result|95% interval|time|
|-|-|-|-|-|
|vjeux file, check|7d885f57d219428f|2.105x, 13/13|2.074-2.137x|10.74 ms|
|vjeux file, board entry|9b0d0973f11a4758|2.096x, 13/13|2.065-2.127x|10.79 ms|
|first attempt (compressed GEMV, "v2")|a44e46117b9449f8|failed, 10/13|-|-|
|v3, FP32 router input up to 8 tokens|f6918aa1415a469a|2.072x, 13/13|2.041-2.103x|10.92 ms|
|v4moe, v3 plus the MoE byte cuts|042deb1ca7a94df6|2.090x, 13/13|2.059-2.121x|10.82 ms|
|v4moe, submitted|8aa6ec880e394012|2.083x|not recorded|not recorded|

The submitted file is v4moe (sha256 prefix 706088bb, from the repo's results table). It scored 2.083x as a fresh judge draw. That is below the old board entry's 2.096x. The old entry fed the router a BF16-rounded input at 8 tokens or fewer where SGLang keeps FP32 (see `fp32-router-input-rule.md`), and the submitted file does not.

We did not record why v4moe was chosen over v3 for the final submit. The two checks scored 2.090x and 2.072x, and the MoE port in v4moe measured as net zero within noise (see `moe-port-and-other-ideas.md`).

## Speedup per case

Shapes at T = 16 or below run identical code in v3 and v4moe.

|case|vjeux check|vjeux board|v3|v4moe|
|-|-|-|-|-|
|decode 1 sliding|2.239|2.220|2.196|2.219|
|decode 1 full|2.215|2.193|2.159|2.162|
|decode 16 sliding|1.963|1.975|1.928|1.975|
|decode 16 full|2.063|2.057|2.051|2.073|
|decode 32 sliding|1.682|1.668|1.698|1.689|
|decode 32 full|1.989|1.966|2.010|1.982|
|verify 1 sliding|2.394|2.378|2.255|2.361|
|verify 1 full|2.274|2.313|2.239|2.202|
|verify 16 sliding|2.084|2.062|2.051|2.021|
|verify 16 full|2.067|2.082|2.061|2.031|
|verify 32 sliding|2.213|2.198|2.179|2.235|
|verify 32 full|2.186|2.172|2.166|2.191|

## Times in microseconds, SGLang / ours

|case|vjeux check|v3|v4moe|
|-|-|-|-|
|decode 1 sliding|173.6 / 77.5|171.1 / 77.9|171.3 / 77.2|
|decode 1 full|182.6 / 82.4|182.0 / 84.3|180.5 / 83.5|
|decode 16 sliding|260.0 / 132.5|274.6 / 142.4|250.6 / 126.9|
|decode 16 full|437.6 / 212.2|444.0 / 216.5|433.1 / 208.9|
|decode 32 sliding|264.3 / 157.2|269.4 / 158.7|263.3 / 155.9|
|decode 32 full|613.2 / 308.3|634.4 / 315.6|611.5 / 308.6|
|verify 1 sliding|260.6 / 108.8|267.5 / 118.6|259.4 / 109.9|
|verify 1 full|282.7 / 124.3|281.2 / 125.6|280.8 / 127.5|
|verify 16 sliding|539.7 / 259.0|536.3 / 261.5|506.7 / 250.7|
|verify 16 full|714.7 / 345.7|699.6 / 339.5|672.7 / 331.3|
|verify 32 sliding|747.0 / 337.6|714.6 / 328.0|726.0 / 324.8|
|verify 32 full|1,093.1 / 500.0|1,069.8 / 493.8|1,080.4 / 493.0|

## Noise

The first thing to take from the tables is how much moves on code that did not change. Decode 16 sliding runs identical code in v3 and v4moe, yet our time went from 142.4 to 126.9 us (11%) while SGLang's went from 274.6 to 250.6 us. Verify 1 sliding went 118.6 to 109.9 us, also on identical code. Across the 12 shapes and these runs, per-shape speedups on unchanged code swing by roughly 3 to 5%, and up to about 6% at the extreme. alygg77 reports the same effect; one of their examples is verify 1 sliding at 99.3 and 109.7 us on unchanged code, a 10% gap in absolute time.

The headline score is steadier: single-run noise is about 0.8%, and the interval half-width is about 1.5%. alygg77 also found an offset of 0.9 to 1.8% per load between the two judge clusters they saw, so runs on different clusters are not comparable without a control.

What we did about it:

- We read each changed shape against the shapes that run unchanged code in the same run, instead of against the headline. For v3 against the board run, the eight T > 8 shapes (identical code, identical kernel lists in all twelve profiles, so no tuner pick flipped) had a median speedup ratio of 0.9959, range 0.9762 to 1.0224.
- For the v4moe comparison we used the two decode-16 shapes (T = 16) as the offset reference. Per-run offsets against the board run were 0.9866 (v3) and 1.0039 (v4moe).
- Where two runs share identical code at a given T (the board run and v3 at T >= 17, v3 and v4moe at T <= 16), they count as two samples of the same thing. That is how we saw that the MoE port was net zero (see `moe-port-and-other-ideas.md`).

Two more judge facts that shaped decisions: the judge calls the kernel on two to four clones of the prepared state, so prepare may not bake raw addresses or captured graphs into it; and prepare plus compilation counts toward the run's time limit (VkaZas lost runs at 437 s of build and prepare).

## Files and checksums

|role|sha256|
|-|-|
|vjeux's decode-layer file, the board entry|db98cafe93010556f29a41c78bf5ebde6b7ac470976689803462fa98a1ef1931|
|first attempt (compressed GEMV)|ec6f6bedc9ce1b104deaf973b2db072a58c5cc82afc0109ce398f340b91a1697|
|v3 (FP32 router input up to 8 tokens)|b76ee5f0320ec20f879a7bb288b1c9ba5e14f31c2af66f8ef850b90212854361|
|v4moe (v3 plus the MoE byte cuts), the submitted file|706088bbd8a57f476842a51888a596ca320156bf72c94c2606ea7fc9592ac126|

Integrity of the base: vjeux's upstream decode-layer file (vjeux/okc-2026) is 568,655 bytes with sha256 cf4a75ee6e4a5cd9e33b82b5347c47abcdc7b81d80dc6a8b5262dea7388f849c and git blob 936fefffb493c4a836e8c08fc58dff50ce338dd5. The board-entry file is that file plus a 15-line attribution header and nothing else (verified with `diff`). The upstream LICENSE is 11,357 bytes with sha256 c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4. Both match the raw files on GitHub.

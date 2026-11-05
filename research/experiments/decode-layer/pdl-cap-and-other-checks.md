# Raising the PDL cap in the decode layer, and how much identical runs differ

This file records the judged checks of a higher programmatic-dependent-launch (PDL) token cap on the decode-layer problem, the follow-up that separated the change from run-to-run noise, and the earlier test of porting the MoE byte cuts into the decode entry.

The cap change did nothing we could measure. The first check made it look like a 2% win because the whole run happened to be fast, and the follow-up check showed that. It also gave us a useful number for how far apart two runs of the same code land.

## Setup

The decode-layer problem (muse_decode_layer) has 12 scored rows: decode 1, 16 and 32, and verify 1, 16 and 32, each with a sliding-window layer and a full layer. The token count T per step is the load size for decode (1, 16, 32) and eight times the request count for verify (8, 128, 256).

The entry launches its glue kernels (Triton) and its CUDA kernels (the fused attention launcher, the MoE extension, the sk2 GEMV) with PDL when T is at most `PDL_MAX_T`, which was 32. PDL lets a kernel start launching while its predecessor is still running. The kernel then blocks on a grid-dependency wait before it reads anything the predecessor writes, so only loads of weights and other early inputs can happen before the wait. With the cap at 32, rows with T above 32 get plain launches.

Only verify 16 (T = 128, both the sliding and the full row) lies between 32 and 128. Verify 32 (T = 256) is above both caps. So raising the cap to 128 can change 2 of the 12 rows.

## Hypothesis and prediction

Raising the cap to 128 should hide some launch latency at T = 128. We expected verify 16 to drop by 1 to 3 us on rows of about 260 us (sliding) and 340 us (full), under 1%. That is small, and we said so before launching: the change could not beat the board's interval on its own, so the check had to be read from the two target rows against the ten rows it could not touch.

The decision rule we wrote down beforehand: call it a win only if both verify-16 rows come in below the lower of the two identical-code baselines by more than the spread between those baselines (0.4 us on the sliding row, 2.5 us on the full row), and the prepare-time A/B at T = 64 and 128 agrees in sign. Otherwise it is a no-go.

## Safety audit before the check

We scripted a scan of every Triton kernel that takes the PDL flag. The script reported nine, all of them calling the dependency wait (our written notes say ten; we did not reconcile the difference). Four of the nine load something before the wait: the first boundary kernel (stream input and a norm weight), the router and its split variant (bias, a norm weight and the local-start value, all independent of earlier kernels), and the second boundary kernel (weights, plus an input written by the first boundary kernel two kernels back). Every kernel between those two boundary kernels waits before it triggers its own successor, so the early load of that input is covered. At T = 128 some of the dense matmuls between them are plain cuBLAS launches with no PDL edge (the tuner's picks vary from run to run), which makes the chain stricter than at T = 32.

The CUDA kernels involved (attention, the MoE main build, sk2) are the same code that already runs under PDL at T = 32. One gap: the prepare-time A/B did not exercise the qkv-to-attention edge, and 13 of 13 tests passing is not proof that no race exists. The safety argument rests on the structural audit, not on the tests.

We also added a log-only A/B in prepare. It times the post-attention chain as a CUDA graph with cold L2 at T = 32, 64, 128 and 256, with the cap at 32 and at 128, three alternating rounds, median. T = 32 and T = 256 are the noise floor, because neither cap changes them.

## Check 1: cap 32 to 128

Job 304f0ebb702346d2, file sha256 prefix 080f87c7, cluster nyc3b. Passed 13/13, 2.106x (95% interval 2.075 to 2.138), score time 10740.35 us.

The two identical-code baselines were the board run a006fb26786d4c98 (cluster sfo3, headline 2.059x) and an earlier check f6918aa1415a469a (cluster nyc3b, headline 2.072x). Speaking only of headlines, 2.106x is above the upper end of the board's interval (2.090). But the rows we changed did not move more than the rows we did not touch:

- The ten control rows (every row with T other than 128) came out 2.40% faster than the sfo3 baseline and 2.03% faster than the nyc3b baseline, in speedup geomean.
- The two target rows, relative to the controls: x1.0014 against a006fb26786d4c98 and x0.9824 against f6918aa1415a469a. Both are inside the noise.
- Rows we never touched moved as much or more: decode 32 full went from 311.1 and 315.6 us to 290.1 us, decode 16 full from 213.3 and 216.5 to 207.7, verify 1 sliding from 119.0 and 118.6 to 111.3.
- SGLang's own time on the same row was lower in this run too (verify 16 sliding: 497.9 us against 524.3 and 536.3 us). That points at a machine-level shift in the run, not at our change.

The prepare-time A/B said the same thing (cap 32 against cap 128, us):

|T|Cap 32|Cap 128|Delta|
|-|-|-|-|
|32|166.0|165.8|-0.3 (floor)|
|64|223.9|221.6|-2.3|
|128|263.5|263.2|-0.3|
|256|294.6|295.2|+0.6 (floor)|

The deltas are medians over the three alternating rounds, so they need not equal the difference of the two rounded columns. At T = 128, the one scored size that changes, the gain is 0.3 us, the same size as the floor. Only T = 64, which is not scored, shows a small gain.

Verdict on the change: no-go, keep the cap at 32.

## The open confound and the follow-up check

The control rows were 2 to 2.4% faster than the baselines in a run whose code did not touch them. Two explanations: ordinary run-to-run noise, or something in the candidate file. The candidate carried the log-only A/B block in prepare (24 graph captures, a cache-flush tensor, an `empty_cache` call at the end), and that tail could change the state the timed run starts from. The cluster alone did not explain it, because the nyc3b baseline (f6918aa1415a469a) also sat 2.0% below the controls of the cap-128 run.

We ran one more check to separate the two. Job c918dc92e4b949f7, file sha256 prefix a32fd969, cluster nyc3b: the base decode file plus only the A/B block, cap left at 32. We wrote the prediction first: if the tail is harmless, the control rows land within 1% of the baselines; if it matters, they land near the cap-128 run. The rule: a control shift of +0.8% or less against the same-cluster baseline means noise and we stop.

Result: 13/13, 2.0856x (95% interval 2.054 to 2.117), score time 10845.81 us. Control geomean 2.0920, against 2.0812 for the same-cluster baseline (+0.52%) and 2.0737 for the sfo3 baseline (+0.88%). Against the cap-128 run, whose controls run identical code, the follow-up was 1.51% lower on the controls and 0.11% lower on the verify-16 rows. The A/B repeated at T = 32 +0.2, T = 64 -1.2, T = 128 +0.4 and T = 256 +0.1 us.

So the tail shifts the controls by at most half a percent, and the rest of the cap-128 run's gain was spread between runs. Verdict: noise, no bisect.

Yours-time per row (us) in the four runs that execute the same code on the control rows. The last column (cap 128) differs in code only on the two verify-16 rows, where the cap acts.

|Row|a006fb26786d4c98 (sfo3)|f6918aa1415a469a (nyc3b)|c918dc92e4b949f7 (nyc3b)|304f0ebb702346d2 (nyc3b)|
|-|-|-|-|-|
|decode 1 sliding|77.9|77.9|77.4|76.0|
|decode 1 full|84.1|84.3|83.3|82.4|
|decode 16 sliding|134.8|142.4|134.6|130.3|
|decode 16 full|213.3|216.5|213.2|207.7|
|decode 32 sliding|159.0|158.7|156.3|150.0|
|decode 32 full|311.1|315.6|308.7|290.1|
|verify 1 sliding|119.0|118.6|111.2|111.3|
|verify 1 full|125.3|125.6|121.2|119.5|
|verify 16 sliding|261.1|261.5|260.6|245.2|
|verify 16 full|342.0|339.5|345.9|334.2|
|verify 32 sliding|339.7|328.0|329.9|325.5|
|verify 32 full|505.3|493.8|495.8|485.4|

Control-row speedup geomeans: 2.0737, 2.0812, 2.0920 and 2.1235 in that order. A fifth run (job prefix 042deb, a decode file with a different MoE kernel) gave 2.1008. These runs span 2.4% on the control geomean. On the ten control rows, the per-row ratio between the cap-128 run and the follow-up runs from 0.980 to 1.067 (decode 32 full 1.067, decode 16 full 1.037, decode 32 sliding 0.980), again with identical code on those rows. The 95% interval on one run is about plus or minus 1.5%.

## What we take from this

- Two runs of the same decode-layer code differ by about 1% on the headline and by up to about 9% on single rows (decode 16 sliding: 130.3 to 142.4 us). A judged check cannot resolve a sub-1% idea on this problem. We reserve checks for ideas that predict at least 2.5 to 3%.
- Read a change from the rows it touches against the rows it cannot touch, and write the decision rule down before launching. A headline above the board's interval is not a result by itself.
- The prepare-time A/B was the more useful instrument here. It runs in one process on one machine, so it avoids the spread between runs. Its own floor (the T = 32 and 256 deltas in the two runs) was 0.1 to 0.6 us.
- The tuner picks different matmul paths from run to run at the same T (at T = 64, two runs differ in the pooler and qkv choices), so some of the spread is the tuner.

## Earlier check: porting the MoE byte cuts into the decode entry

Job 042deb1ca7a94df6. The MoE work had found several byte-saving changes in the standalone MoE kernel: a K reorder, 6-bit scale codes, stream-K work ranges and a variant the code calls ONEK. We ported them into the MoE inside the decode entry. This is also the one concrete lever a public post suggested (combine the component optimizations inside the full decode kernels), see [claims-in-a-public-post.md](../general/claims-in-a-public-post.md).

Result: 2.090x (95% interval 2.059 to 2.121), against 2.072x for the base file (f6918aa1415a469a) and 2.059x for the board run (a006fb26786d4c98). On the six shapes the port was aimed at, the geomeans of the three runs were 2.0164, 2.0208 and 2.0168. Our notes do not record which is which, but the spread is 0.2%. Net zero within noise. We treated the port as closed.

A caveat: the headline is 0.9% above the base, which is well inside the interval width, and the three-run geomeans show nothing at the target shapes. We did not run it twice.

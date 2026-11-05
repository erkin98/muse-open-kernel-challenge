# In-process timing lab for the flat attention kernels

This file describes the timing lab we built into an attention submission, what it measured on the judge's H200, and how the paired A/B arms it carried compared with the judge runs that followed.

## Why we built it

A judge run of the attention problem gives a headline score with a 95% interval of about plus or minus 1.5%. Four runs of machine-identical kernels scored 2.589 to 2.676. The per-shape spread is worse for the small rows (2% to 10% between identical-code runs) and better for the big flat rows (0.1% to 0.8%). Every lever left on our list was worth 0.1% to 1% of the score, so a judge run could not tell us whether it worked.

The way around it was to measure inside one judge job. The submission file is allowed to run code on its first call, and the judge returns the tail of the log (about 16,000 characters). We put a measurement routine there. It times several variants of a kernel back to back on the judge's own GPU, interleaved, and prints paired differences at a resolution of roughly 0.1 to 0.5 us. We used the same routine first to take the kernel apart phase by phase and then to compare candidate changes.

## Shape names

The scored shapes all use an 80,000-token context. We write them as a letter for decode (q_len 1) or verify (q_len 8), the batch size, and s or f for a sliding-window or full-attention layer. So v16f is verify, batch 16, full layer.

Our model of the score, which reproduces the 2.6045 of the board run exactly, is a geometric mean over six loads, with each load costing 46 times the sliding shape plus 16 times the full shape. Percent of score per 1% of time saved in one shape:

|shape|d32f|v32f|d16f|v16f|v1s|d1s|d1f|v1f|v16s|d16s|v32s|d32s|
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|score %|0.142|0.135|0.130|0.126|0.096|0.093|0.074|0.070|0.041|0.036|0.031|0.025|

The four flat full shapes together are worth 0.533% per 1%. Getting +1% of score from them takes about 1.9% off all four.

## Starting point and kernel layout

The starting point was vjeux's board entry (sha256 prefix cf04259b), board run c20df44ab71d4005 at 2.605x (CI 2.565-2.644). A rerun of the same file, 705c7dd9f34446dd, scored 2.589x.

The file is one kernel template, `attn_kernel<L, FLAT, MODE>`, with L the q_len. The four full-context shapes with batch 2 or more (d16f, d32f, v16f, v32f) use the flat variant: 264 CTAs, two per SM, each CTA walking a share of the key range. Every other shape uses a per-request variant selected by MODE bits. All variants merge split-KV partials with `red.global.add` into a self-zeroing fp32 workspace, count arrivals with an `atom.acq_rel` counter, and finish in a last-CTA or cooperative finalize that normalises, applies the gate, stores, and re-zeroes the workspace.

## How the lab works

- The lab file is a copy of the base. The 20 shipped kernel instantiations compile to SASS identical to the base build (compared on 128-bit instruction words after whitespace normalisation, NVRTC 13.2 plus ptxas). Instrumented copies ("twins") exist only as extra instantiations and launch only when a debug pointer is non-null, which the timed path never passes.
- The routine runs once, on the first eager call. It is skipped under graph capture, wrapped in try/except, and uses its own workspace and synthetic inputs built to mimic the reference input generator. The extension build happens before the lab's own clock starts (43.5 s in lab 1, 49.0 s in lab 2), and the lab itself took 0.2 s and 16.5 s.
- Every launch is preceded by a 256 MB read so L2 is cold, as in the judge's timing.
- Stamps: thread 0 of each CTA writes `%globaltimer` at twelve points, and `clock64` beside each one as a hedge against a coarse timer. The timer step turned out to be 32 ns.
- Validity check: the span from the first CTA entry to the last CTA end matched the judge's profiled kernel time within 1% to 3% on the four flat shapes (d16f 85.2 vs 85.8 us, v16f 101.0 vs 101.5, d32f 165.5 vs 164.8, v32f 184.0 vs 184.2). The SM clock read 1.64 to 1.74 GHz depending on the shape.

The stamp points:

|stamp|meaning|
|-|-|
|entry|first line of the kernel|
|part|partition of keys into segments finished|
|idxwait|page indices landed|
|qready|Q group landed and processed|
|tile0|first K/V tile landed|
|loopend|tile loop finished|
|reds|merge reductions issued|
|arrreq / arrret|arrival atomic about to issue / returned|
|release|cooperative barrier released, or last-CTA finalize started|
|finload|finalize loads returned and math done|
|end|finalize stores issued|

## Lab 1: anatomy of the flat kernels (job 56793dfd379e4741)

The lab file scored 2.631x (CI 2.592-2.671), 16/16 tests, which counts as one more sample of identical shipped code. Times below are microseconds after the earliest CTA entry; the entry spread of all 264 CTAs was only 0.2 us. Medians over five launches.

|shape|span|part|idxwait|qready|tile0 med/max|loopend min/med/max|tail after last loopend|SM-pair skew med/max|SM clock GHz|
|-|-|-|-|-|-|-|-|-|-|
|d16f|85.2|1.4|2.4|3.2|4.0/5.5|74.8/80.4/83.3|1.9|0.9/4.8|1.74|
|v16f|101.0|2.1|3.3|4.9|5.3/6.8|74.7/87.4/97.4|3.6|12.4/18.5|1.66|
|d32f|165.5|1.4|2.6|3.7|4.2/5.6|151.5/158.8/163.4|2.1|1.8/7.0|1.71|
|v32f|184.0|2.4|3.6|5.4|5.7/9.5|143.5/170.1/180.1|3.9|25.5/31.7|1.64|

What the table says about the q_len 8 flat kernels:

1. The q8 minus q1 gap is 15.8 us at batch 16 and 18.5 us at batch 32. Roughly 1.3 to 1.5 us of that is prologue (the q8 partition step takes 2.4 to 2.7 us against 1.5 us on the q1 path, and waiting for the page indices and then processing the Q group takes 3.2 us in total, which is the q8 critical path). Another 1.7 to 1.8 us is the tail: the grid-wide finalize slice takes 1.9 us from release to loads returned, against 0.7 us for the q1 last-CTA finalize and 0.7 to 1.0 us in the per-request cooperative kernels. The remaining 12.8 us (batch 16) and 15 us (batch 32) is the slowest CTA's loop.
2. The median CTA loop is only 5.2 us (batch 16) and 10.3 us (batch 32) longer than the q1 loop, about 6.8%. The q8 kernels also run at a 4% to 4.6% lower SM clock. Most of the median gap is the clock.
3. The q8 kernels have a large imbalance inside an SM pair (median skew 12.4 and 25.5 us). In all 20 launches of the four flat shapes the last CTA to arrive had index 132 or higher, meaning it was dispatched second. alygg77 reported that equalising the pair does not move SM completion time. That fits a picture where an SM's throughput is conserved and the pair split does not matter.
4. What can still move is the spread of completion times across SMs: max minus mean is about 3.5 (d16f), 5.1 (d32f), 5.0 (v16f) and 5.0 us (v32f). Recovering it needs dynamic work claiming, which means a new producer/consumer sentinel protocol and new counters. We judged it a race-class change that we could not audit or test well enough without a GPU, and left it alone.

Options after lab 1 were a micro-bundle on the prologue and tail (about 2 us on v16f and v32f), the dynamic tail pool (up to 3.5 us but a large and risky change), or stopping. We took the micro-bundle, to be measured with twin A/B inside a second lab job. The per-request rows of lab 1 are in the latency chain study.

## Lab 2: paired arms (job 7c7e80b4e68f4850)

Lab 2 ran as a file with the same shipped kernels plus 17 twins. The file scored 2.676x (CI 2.636-2.716) on identical shipped code, the fast end of the identical-code range. Gates before launch: the 20 shipped kernels SASS-identical to the base, all twins without spills, flat q8 twins at 80 registers (the base's count), synchronisation instruction counts per kernel compared (the fpart arms drop one `__syncthreads` by design, everything else matches), and a CPU dry run of the routine with a fake launcher.

The arms for the flat q_len 8 kernel:

- fpart: warp 0 computes the CTA's key range and segment list with ballots, as the q_len 1 flat kernel already does, including an L2 prefetch of the first page-index lines.
- qdir: four threads per query row write the e4m3 query tile straight from the raw bf16 rows, skipping the f16 staging and the consumer-side repack.
- warm: the grid-wide finalize slice runs twice. Pass 0 has every global access predicated off by inline PTX, so it only pulls code into the instruction cache, while thread 0 does the arrival atomic and the acquire spin. Pass 1 is the original slice. The release/acquire chain is textually unchanged.
- combinations: fpart+warm, fpart+qdir, and all three ("all").
- R1, for the non-flat q_len 8 kernels: bf16 reds. This is covered in the latency chain study.

Each arm ran in 31 interleaved launches on the flat shapes and 41 on the others, with the baseline arm run twice (old and old2) so that old2's paired difference measures the noise of d. The table gives the paired median span difference in us (arm minus old, negative is faster) with the 25th and 75th percentile. Reference spans for old: v16f 96.74, v32f 178.94, v12f 76.67, v40f 219.90. The lab's spans are kernel spans and sit a few microseconds below the judged per-case times, so only the differences are used.

|arm|v16f d [p25,p75]|v32f d [p25,p75]|v12f d|v40f d|rel L2 vs old (v16f / v32f)|
|-|-|-|-|-|-|
|fpart|+0.45 [-0.29,+1.02]|-1.09 [-2.08,-0.45]|-0.03|-0.19|1.2e-5 / 1.4e-5|
|qdir|-0.74 [-1.12,-0.22]|-2.05 [-2.56,-1.38]|not run|not run|1.7e-2 / 1.9e-2|
|warm|-0.06 [-0.70,+0.45]|+0.29 [-0.22,+0.83]|-0.03|+0.35|1.0e-5 / 2.5e-7|
|fpart+warm|-0.80 [-1.41,-0.29]|-1.34 [-2.37,-0.67]|-0.93|-1.06|1.1e-5 / 1.4e-5|
|fpart+qdir|-0.99 [-1.95,-0.74]|-2.37 [-3.14,-1.54]|not run|not run|1.7e-2 / 1.9e-2|
|all|-1.54 [-2.30,-1.22]|-2.75 [-3.17,-1.92]|-1.31|-1.44|1.7e-2 / 1.9e-2|
|old2 (noise)|+0.45 [-0.42,+1.12]|+0.06 [-0.42,+1.09]|0.00|-0.26|1.1e-5 / 1.3e-6|

Step medians explain part of it. fpart cut the partition step from 2.1 to 2.4 us down to 1.47 us, and warm cut the finalize load step from 2.0 to 2.4 us down to 1.1 to 1.7 us. Both gave part of the saving back somewhere else, so the span moved less than the step did (fpart alone on v16f moved +0.45 us even though its partition step fell by 0.64 us; that +0.45 is inside old2's noise). We did not find where the time went.

### Numerics

qdir changes how q is rounded. The old path rounds fp32 to f16 to e4m3, two roundings. qdir rounds once. The outputs therefore differ from the old arm by relative L2 1.7e-2 (v16f) and 1.9e-2 (v32f), with an elementwise max/mean of 0.83 and 0.78, but the distance to the PyTorch reference is the same or slightly smaller: 0.0757 for old against 0.0757 (v16f) and 0.0762 against 0.0761 (v32f). Correctness-only cases at other sizes agreed: batch 2 0.0744 old vs 0.0740, batch 4 0.0756 vs 0.0748, batch 8 0.0752 vs 0.0747, batch 48 0.0746 vs 0.0745. The judge's tolerance is 0.15 relative L2, and the kernels' own error is about 0.07 to 0.09 because everything is quantised to e4m3.

Soak: 800 launches of the old arm and 4,000 each for warm, fpart+warm and all, no flush, every output compared on the GPU with the old arm's first output. For warm and fpart+warm the launch-to-launch distribution matched the old arm (median about 1e-5, max about 2.5e-5). For "all" the distance to old was a constant 1.7e-2 / 1.9e-2 with max equal to median, so no tail.

### Reading the result

We wrote the decision rules before launch. An arm would go forward to a real check only if it gained at least 1.5 us on both v16f and v32f, stayed within 5e-3 relative L2 of the old arm, matched the old arm's distance to PyTorch within 1e-3, and had a clean soak. It also could not lose more than 0.5 us at v12f and v40f. old2's d was printed next to every d, and a shape whose old2 moved more than 0.5 us counted as inconclusive.

By that rule the answer was no. Only "all" cleared 1.5 us on both shapes, and it failed the 5e-3 bar through the qdir rounding offset. fpart+warm had identical numerics but gained 0.80 and 1.34 us.

We shipped "all" anyway. The reasoning was that the only criterion it failed was distance to our own earlier output, which comes from rounding q once instead of twice, while the distance to the reference was equal or better. That is a judgement call, not a pass of the rule, and we say so here. The soak rule also needed a deviation: a quantile-over-median bound failed for the old arm itself, because fp32 reduction-order noise has a wide relative spread (median 1.1e-5, max 2.5e-5). We compared each arm's soak distribution with the old arm's instead.

## Check 3: the bundle on the judge (job 04aae1ba8df8488f)

The candidate was rebuilt from the base file, not from the lab file: three device edits for the flat q_len 8 kernel only (`fpart = L == 1 || FLAT`, `QDIR = L == 8`, the two-pass warm finalize), a host check that a flat q8 plan has nc equal to 1, a new extension name, and a header line. No stamps, no lab code. 19 of 20 kernels came out SASS-identical to the base, and the flat q8 kernel (6424 to 6072 instructions, 80 registers, no spills) matched the lab's non-stamp "all" instantiation exactly, so the shipped code is the code the lab measured minus the stamps. The candidate file has sha256 prefix 5cc7ec84.

The ordering argument, per piece (valid for any subset of them):

- fpart touches only the prologue. Warp 0 publishes the segment list in shared memory and every thread reads it after one `__syncthreads` where the old route used two. Nothing global is added, and the CTAs with an empty share still run the finalize, so the grid counter still sees 264 arrivals.
- qdir writes only shared memory (followed by the same `fence.proxy.async` the old path issues) and is read before the loop's first barrier.
- Warm finalize pass 0 touches no global memory, so it cannot interact with any other CTA. Thread 0's arrival and spin are the same atoms in the same position relative to the barrier.

Judge result: 16/16, 2.620x (CI 2.581-2.676), 1,437.4 us. Per case in us and speedup: d1s 4.87 / 4.275, d1f 10.94 / 2.839, d16s 7.92 / 3.361, d16f 87.46 / 2.277, d32s 9.65 / 3.253, d32f 167.50 / 2.287, v1s 6.59 / 3.495, v1f 13.67 / 2.601, v16s 11.18 / 2.600, v16f 100.28 / 2.033, v32s 14.64 / 2.354, v32f 183.01 / 2.117.

The headline tells us nothing: identical-code runs span 2.589 to 2.676. The raw v16f and v32f speedups sit inside the identical-code envelope (v16f 2.022-2.122, v32f 2.104-2.197). Because whole-run shifts move all rows together (a faster cluster raised every full row by about 4% in lab 2), we normalised each flat verify row by the flat decode row from the same run, which is unchanged code:

|ratio|this run|identical-code range (mean)|difference to mean|
|-|-|-|-|
|v16f / d16f|0.8928|0.8746-0.8827 (0.8782)|+1.67%|
|v32f / d32f|0.9257|0.9060-0.9149 (0.9087)|+1.86%|

Both sit 1.1% to 1.2% above the top of the range, and they match the lab's prediction of about 1.5% in time. Against lab 1 in absolute microseconds, v16f went 100.47 to 100.28 and v32f 183.88 to 183.01 while the unchanged d16f and d32f went up 0.8% and 0.9%, a net of about 1.1% and 1.4%. No unchanged row regressed: d16s and d32s were 4.6% and 4.7% better than before, and the d16f and d32f speedups were 1.1% and 0.7% below the lowest earlier values, which is run noise on an unchanged kernel. The expected score effect is +0.3% to +0.4%, which one judge draw cannot resolve. The judge's single-launch profile times for the same kernel vary 4% to 5% between identical-code runs (v16f 97.2 here, 97.9 in lab 1, 97.0 in lab 2), so a 1.5% change cannot be read from them.

## Things we would do the same way, and mistakes to avoid

- Compare against in-run controls. A 1.5% change is readable only through unchanged rows of the same run and the paired ratio.
- Trace the launcher's routing for every case the lab runs. One correctness case ("sliding batch 3, q8") routed to a MODE with no twin, so nothing launched and the lab compared uninitialised output memory. We caught it by reading the routing table, replaced the case with batch 20 (which routes to MODE 3), and added a model of the launcher that reproduces the judge profile's kernel names for all twelve shapes.
- A twin is not the shipped SASS. Stamp twin versus stamp twin is a fair A/B, but before shipping we rebuilt the winner from the base with only the needed edits and proved it SASS-identical to the measured twin's non-stamp instantiation.
- Run the baseline arm twice and look at old2's d before reading any other d.
- The lab 2 arms are selected through a launch override. The base launcher applies its tuned plan only when no override is passed, so the override is built from the file's own tuned plan, otherwise the arms would run a plan the judge never runs.
- Keep the printout small. The judge returns about 16,000 characters of log and lab 1 printed 13.4 KB.

## Open questions

- Where did the fpart and warm savings go? Each cut its own step by 0.6 to 1 us and the span moved by less. We have no stamp that explains the leak.
- The q8 flat kernels carry about 10 us of batch-independent cost beyond the q1 flat kernels (fit of time against bytes: intercept 7.6 us for q1 and 18.0 us for q8). Lab 1 accounts for 1.3 to 1.5 us of prologue and 1.7 to 1.8 us of tail, plus a lower clock. The rest is the slowest CTA's loop, and we found no static change that removes work from it.
- The dynamic tail pool is untested. Its upper bound is about 3.5 us on each heavy shape, 1% to 1.4% of score if perfect, and we put its odds of working at roughly 30%.

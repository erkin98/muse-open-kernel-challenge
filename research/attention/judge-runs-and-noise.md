# Judge runs and noise

Identical code does not produce identical judge numbers, and this file records how far they move, what that hides from us, and how we read results because of it.

## Headline spread

Same bytes, or the same kernel with only a comment line changed, scored like this:

|job|file|score|95% interval reported|judge time line|
|-|-|-|-|-|
|c20df44ab71d4005|vjeux's entry, unchanged|2.605x|2.565 to 2.644|1,446.1 us|
|705c7dd9f34446dd|vjeux's entry, second run|2.589x|2.550 to 2.637|1,454.9 us|
|8c9a79b3381c4707|our check-4 file (the board run)|2.6211x|||
|1c1107b8a97d4e45|check-4 file, an earlier check run of the same bytes|2.627x|||
|9163402d01c54f9b|check-4 file, comment line changed|2.582x|||
|53d818c95349439a|check-4 file, comment line changed again|2.602x|||
|800c1940cf774991|check-4 file with v16s at 16 splits|2.627x|||

The last row is not identical code, but v16s did not improve (11.8 us against 11.2 to 11.6), so the other shapes' noise accounts for the headline. For the same kernel the score therefore ran from 2.582x to 2.627x, about 1.7% peak to peak. The judge's own 95% interval is about plus or minus 1.5% around a single run. We treat roughly 1% as the noise on the score from one run.

Our 2.6211x sits in the upper half of that range. If we had to guess the kernel's true level we would say somewhere around 2.60x to 2.61x, but that is a guess from a handful of points.

## Per-shape spread

Two runs of vjeux's entry, in microseconds (jobs c20df44ab71d4005 and 705c7dd9f34446dd), with the difference between the two as a percentage of the smaller:

|shape|run 1|run 2|difference|
|-|-|-|-|
|d1s|4.94|5.17|4.7%|
|d1f|11.27|10.66|5.7%|
|d16s|8.37|8.54|2.0%|
|d16f|86.6|87.0|0.5%|
|d32s|10.00|10.25|2.5%|
|d32f|165.6|165.8|0.1%|
|v1s|6.80|6.59|3.2%|
|v1f|14.31|14.48|1.2%|
|v16s|11.39|11.63|2.1%|
|v16f|101.0|101.6|0.6%|
|v32s|14.85|14.66|1.3%|
|v32f|184.1|185.6|0.8%|

The heavy full shapes repeat to within about 1%. The short shapes (batch-1 and the sliding layers) vary by 1 to 6% from one run to the next. Across the check-4 reruns the unchanged sliding rows moved by up to 5%. A single run can therefore only show a change of more than about 6% on a short shape, or about 2% on a heavy full shape (the full rows in one of our concurrency runs drifted +1 to 2% and -1% while the code was unchanged). This is our reading of the table and not a statistical bound; we only had two or three points per shape.

## The denominator moves too

The judge reports each case's speedup against SGLang timed in the same run, and that reference drifts as well. SGLang's decode 1 sliding time was 20.5 us in one run, 21.1 us in another and 20.32 us in the check-4 profile; decode 32 full was 384.5, 385.4 and 385.95 us. So a headline can shift without our kernel changing at all, which is one more reason to read our own microsecond rows against identical-code runs rather than the speedup column.

## Habits that came out of this

- Compare against the spread of identical-code runs, never against a single earlier run, and never rely on the headline for a gain below the noise level. Both small gains we kept (about +0.3 to 0.4% and about +0.15%) are well below the 1% score noise, so on the headline alone they cannot be distinguished from a lucky draw. They were judged from the per-shape rows they were supposed to move and from the controls.
- Put byte-identical kernels in the same file as controls. When a candidate leaves some kernels' machine code unchanged (we checked in the SASS), those shapes show the offset of that particular run, and dividing it out cleans up the shapes that did change. In check 2 the four heavy full shapes served this way and showed an offset between 0 and about +1%.
- Use the independence of the shapes. The judge times the twelve shapes separately, so one file can test a different variant on each shape. That is what the [sweep](launch-plan-sweep.md) did.
- Rerunning identical bytes can come back from the judge as a duplicate of the earlier job and then produces no new draw. A new draw needs different bytes, which is why our extra draws changed a comment line. We do not count extra draws as progress. Picking the best of several draws moves the reported number by perhaps 0.5 to 0.8% without changing the kernel.
- Run time of a check was about 6 to 7 minutes, and queue time varied from about 10 minutes to over an hour.

## Open point

One sweep file with the unchanged plan was reported by the judge as a duplicate of job 1c1107b8a97d4e45, an earlier check of the same bytes as the submitted file, which scored 2.627x. So that number is a re-read of an existing draw, not a new one. It is also a coincidence that the 16-split experiment headlined the same 2.627x; that run used different code. That earlier check is the 2.627x row in the table above, so it is counted once.

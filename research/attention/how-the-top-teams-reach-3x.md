# How the top teams reach 3x on attention

This file sets out what the public record and our own numbers can say about the teams above 3.0x on the attention board, with a clear line between what we know and what we are guessing.

The short version: nobody at the top has published a mechanism. No handle in the top ten has a public repository, write-up or gist for this problem that we could find. What follows is a record plus a consistency argument, not a reverse-engineering result.

## The board on the morning of Nov 5

1,370 entries were ranked and we were #276 at 2.6211x.

|rank|team|score|attempts|best entry submitted|
|-|-|-|-|-|
|1|pjmullin3|3.339x|19|Nov 1, 02:05|
|2|tjiang-fair|3.319x|49|Nov 4, 21:04|
|3|LeshengJin|3.219x|19|Nov 5, 09:30|
|4|olety|3.173x|131|Nov 5, 04:26|
|5|jt914|3.091x|30|Nov 4, 14:48|
|6|gauravjain14|3.058x|28|Nov 4, 08:12|
|7|Fangzhou66|3.032x|11|Nov 4, 03:41|
|8|mgpai22|3.032x|111|Nov 5, 03:29|
|9|abhishekgahlot2|3.028x|92|Nov 4, 11:08|
|10|zzz0906|3.008x|25|Oct 28, 16:44|

Times are local (+04). Eleven entries are at or above 3.0x.

## What the record supports

- The top is still moving. Three of the top ten improved in the 12 hours before we looked (LeshengJin at 09:30 that morning, olety at 04:26, mgpai22 at 03:29).
- Attempt counts do not explain the scores. Fangzhou66 reached 3.032x in 11 attempts and LeshengJin 3.219x in 19, while olety needed 131 for 3.173x. We had 3 scored attempts on this board, and the board does not say whether other teams' counts include off-board checks.
- The #1 entry (Nov 1) and the #10 entry (Oct 28) both predate the organizers' Nov 3 guidance that approximate BF16-accurate math, BF16 or FP16 partial sums and FP16 accumulation in attention are allowed. So 3.0x to 3.3x was reachable without that latitude.
- Two teams describe how they work, in general terms: gauravjain14 (rank 6) says it profiles with Nsight on rented GPUs, and lxe, working on the decode_layer problem, says it iterates on two rented spot GPUs. Both measure on real hardware.

## What is inference

- Our cost model says the heavy full shapes stream at 3.6 to 3.8 TB/s on the margin with a 7 us (decode) or 15.7 us (verify) fixed part per call, and the sliding chains sit at 1.6 to 3.2 times the assumed floor (details in [cost-model-and-breakdown.md](cost-model-and-breakdown.md)). A 3.3x entry needs about 21.5% less time on every shape, so by arithmetic they have cut both the fixed parts and the chains. The arithmetic is in [gap-to-the-leaders.md](gap-to-the-leaders.md).
- One way to get most of the way there without any exotic trick: cut the fixed part per call and lift the marginal rate from the 75 to 79% of peak we see toward 85%. We did not check that this closes the whole gap, and we have no evidence anyone did exactly that.
- The difference we can name with some confidence is measurement. We had the judge's per-shape rows and static analysis of the compiled code; the teams that describe their setup profile on a real GPU. Every unexplained term in our model (the fixed parts, the true read ceiling, where the chain time goes) is the kind of thing a profiler can answer and a judge row cannot. This is inference, not something any team has told us.

## Approaches other public entries tried, which lost

From the public repositories and posts we read (ideas only, no code taken): attempts to shorten the sliding chain by fusing its stages, using thread-block clusters, a one-tile design or a cooperative grid barrier, show up in other teams' work, and they lost or tied. Our own checks of similar ideas went the same way; see [ideas-tried-and-results.md](ideas-tried-and-results.md).

The one technique from another team's notes that looked promising enough to spend a judge run on was warming the finalize code before it is needed (Kushagrabainsla's notes). It measured as no effect on our kernel. That experiment is written up in [linear-exp2-and-cold-code-warming.md](linear-exp2-and-cold-code-warming.md).

## What we did not do

We did not query any other team's job (the documented API only serves your own jobs) and we did not read any other team's submission for this note.

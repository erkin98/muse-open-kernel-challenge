# What the top attention scores imply

This file works through the attention score arithmetic to see what the leading 3.1x to 3.3x entries must be doing differently from our 2.605x, and whether anything public lets us close that gap.

The short answer is that we could not tell how they do it, and nothing public gets us anywhere near. The arithmetic does show where the payoff is.

## Facts

- The attention board in the late afternoon of Nov 4: pjmullin3 at 1128.0 us (3.339x, flagged pending the daily auto-review, 19 attempts), tjiang-fair 1155.8 us (3.259x, 27 attempts), LeshengJin 1193.5 us (3.156x, only 5 attempts), olety 1203.6 us (3.129x, 99 attempts), jt914 1210.0 us. Ours: 1446.1 us, 2.605x. A bemlerlabs snapshot from Oct 28 lists pjmullin3 at 3.03x with 5 attempts, and tjiang-fair reached 3.22x by Nov 2, so the top moved up about 10% in six days.
- The score is a weighted sum. Total time is the sum over cases of (weight x case time) / 8, with weight 46 for each of the six sliding cases and 16 for each of the six full cases (bemlerlabs model, which we re-checked: 1449.6 us against 1446 us on the board).
- Our per-case times in us, taken from an identical-kernel judge run. Sliding: d1 4.9, d16 8.4, d32 10.0, v1 6.8, v16 11.4, v32 14.9 (sum 56.4, contributing 324 us). Full: d1 11.3, d16 86.6, d32 165.6, v1 14.3, v16 101.0, v32 184.1 (sum 562.8, contributing 1126 us).
- The HBM floor of the full cases (judge speed-of-light at 4.8 TB/s): 4.3, 63.2, 126.4, 4.3, 63.4, 126.8 us, 777 us in score units. So our full cases run at 1.45x the floor on average: decode-full about 1.31x to 1.37x, verify-full 1.45x to 1.59x, batch-1 full 2.6x and 3.3x.

## What 3.0x needs

Total time must fall to 1255 us, which is 191 us less than now.

- Closing the whole verify-full gap to the decode-full ratio saves about 60 us and gets us to about 2.72x. This is the same 14% and 10% gap that our headroom note already had.
- Cutting every sliding case by 35% (d32 from 10.0 to 6.5 us, v32 from 14.9 to 9.7 us) saves 113 us.
- Both together give 1446 - 60 - 113 = 1273 us, which is 2.96x. So 3.0x needs slightly more than both cuts, and the sliding cut is the harder half.

## What 3.34x needs

Total time 1128 us, which is 318 us less than now. With the sliding cases 60% lower (about 130 us in total), the full cases must drop from 1126 to about 998 us, which is 1.28x the floor on average. That means v16f and v32f near 1.3x and batch-1 full near 2x. With sliding only 35% lower, full would need 917 us, or 1.18x the floor, about equal to the practical 4.1 TB/s ceiling in our headroom note.

Our reading: the leader needs either a structurally different fused design on the sliding cases, or full-case efficiency at the HBM wall. We cannot tell which from public data. The pending-review flag on the leading row means the organisers' screener was unsure about it, and we cannot say why.

## Payoff per microsecond

One microsecond saved on all six sliding cases is 34.5 us of the total (2.4%). One microsecond on v32f alone is 2 us (0.14%). So sliding-case first-launch costs matter far more than full-case tweaks. They are also exactly what our closed list and the public measurements block: every public attempt at a large sliding cut (clusters, a one-tile split, a cooperative barrier) lost on the judge, and cold-code warming lost on our own check.

## Can we get there

No, not in the time we had. The only public route is the keell0renz file with ZhimaoL's CUDA inside it. That file is worth about 1.3% in median (2.638x against our 2.605x, with a judge-image-day offset mixed in), its source cannot be copied under our rules, and the parts that would matter most (merge, balance, sliding cuts) are on our closed list or measured negative by bemlerlabs.

What is untested and cheap is small. The packed half2 softmax at q_len 8 was the only item aimed at the 39% of our time that sits in verify-full (570 us of 1446), and its realistic gain was 0 to 1%. It later died on SASS evidence (see attention-papers.md). Nothing new came from GitHub.

A related caution from the same data: bemlerlabs saw a single check on identical code spread up to 5% per case and 2% in score, and their 24-draw family medians show a 1.6% gap between two files. A single check of our 2.605x file cannot adjudicate a 1% idea.

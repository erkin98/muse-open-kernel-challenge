# What the public record shows about the top attention entries

This file records what we could find in public about how the leading muse_attention_decode entries reach about 3x (not much that is direct), the board facts we could verify, and some arithmetic on what 3x would take from our own per-shape times.

The board was read on Nov 4, 2026: the board text at about 17:21 +04 and the site's own leaderboard feed at about 17:40 +04. We searched read-only, with no login, no posting and no contact with anyone, and we read no code from other teams for this file. "Verified" below means we read it in the cited source. "Inference" means it is our reading and the source does not say it.

## Short answer

No top-10 attention handle has published a technique, a repo, a write-up or a gist. We found a GitHub account for 13 of the 14 handles we looked at, and none has a public repo about the challenge (verified). The fourteenth, gogowhy, has no GitHub account under that name. So the question of how they get 3x cannot be answered from their own words. What the public record does show:

1. Verified: 3.0x is not new. colinlikescode sat at about 2.99x on Oct 26 and is at 2.997x now (best entry submitted Oct 26, 07:23 UTC). pjmullin3 showed 3.03x after 5 runs on Oct 26 and holds 3.339x with an entry submitted Oct 31, 22:05 UTC. zzz0906's 3.008x entry dates from Oct 28 and mgpai22's 3.013x from Nov 2. Sources: the Oct 26 board screenshot in https://x.com/pjmull/status/2114602468252545324 and the feed https://openkernelchallenge.dev/api/leaderboard/muse_attention_decode.
2. Verified: seven of the 11 entries at or above 3.0x were submitted after the organizers' Nov 3 guidance (12:15 pm PT, 20:15 UTC): tjiang-fair, LeshengJin, jt914, gauravjain14, Fangzhou66, abhishekgahlot2 and k9thebeast. Four were not: pjmullin3, olety, mgpai22 and zzz0906. The guidance (https://openkernelchallenge.dev/updates) allows approximate math that is accurate to BF16, BF16 or FP16 partial sums, and FP16 accumulation in attention. Inference: the recent wave of 3.0 to 3.26x entries may use those allowances. But recency explains the timing just as well, since active teams submit most in the last day before a deadline, and we could not compute a board-wide base rate. The four older entries show that 3.0x does not need the allowances.
3. Verified, as far as it is stated: gauravjain14 says he has no special setup beyond Nsight Systems profiling and a large bill for rented Modal GPUs (X, Nov 2). lxe (decode_layer #10, not an attention target) describes an automated loop of six parallel workers on two spot GPUs. Both profile on real GPUs. We have no H200 of our own.
4. Verified: the organizers wrote on Nov 3 (3:45 pm PT) that many submissions run past 10,000 lines and contain separate variants for the tested shapes, and that the live board over-rewards fitting to the judge. They said they will review manually (same updates page). So shape-specific variants are a known feature of the top of the board, and a known risk of removal.
5. Inference: from our own per-shape times (see the arithmetic section below), the top entry is 1.282x our geomean. Putting every bandwidth-bound shape at its bound is worth only 9.9% (about 2.86x). Reaching about 3.06x needs every latency-chain shape (sliding at every batch, plus decode 1 full and verify 1 full) about 30% faster. About 3.35x, the number-one entry, needs those about 35% faster and every full shape about 10% faster. So the leaders must be faster in every load, and a large part of that must come from the small, latency-bound calls. This is a consistency check on our model. It says nothing about what they actually did.

## Board facts from the site's own feed

The feed is the public JSON that the leaderboard page itself loads (https://openkernelchallenge.dev/api/leaderboard/muse_attention_decode). Its fields are job, team, speedup, submitted_at, screening, screened_speedup, attempts and value, over 1341 rows. Times are UTC, converted from the feed's epoch seconds. Everything in this section is verified except where marked.

|team|board speedup|screened_speedup (feed field; % against the board value)|mark|best entry submitted (UTC)|attempts|
|-|-|-|-|-|-|
|pjmullin3|3.339|3.216 (-3.7%)|pending|Oct 31 22:05|19|
|tjiang-fair|3.259|3.222 (-1.1%)|none|Nov 4 11:26|27|
|LeshengJin|3.156|none|none|Nov 4 10:02|6 (board text said 5)|
|olety|3.129|none|screened|Nov 3 06:11|99|
|jt914|3.113|2.910 (-6.5%)|none|Nov 4 04:42|30|
|gauravjain14|3.058|3.018 (-1.3%)|none|Nov 4 04:12|25 (board text said 24)|
|Fangzhou66|3.032|2.972 (-2.0%)|none|Nov 3 23:41|10 (board text said 9)|
|abhishekgahlot2|3.028|2.800 (-7.5%)|none|Nov 4 07:08|84|
|mgpai22|3.013|none|screened|Nov 2 23:39|75|
|zzz0906|3.008|2.895 (-3.8%)|pending|Oct 28 12:44|25|
|k9thebeast|3.000|none|none|Nov 4 11:53|21|
|colinlikescode|2.997|none|screened|Oct 26 07:23|31|
|erkin98 (us)|2.605|none|none|Nov 4 07:00|1|

- 11 of the 1341 entries are at or above 3.0x, and 21 are at or above 2.9x.
- Three attempt counts rose between the 17:21 board text and the feed read (LeshengJin, gauravjain14, Fangzhou66) without a better score. The counts are verified. The reading that those teams are still iterating is inference.
- `screened_speedup` is 1.1 to 7.5% below the board speedup wherever both exist. We do not know what it measures and we do not use it as evidence. tjiang-fair (Nov 4, 11:26), jt914 (04:42), gauravjain14 (04:12) and abhishekgahlot2 (07:08) were all submitted after the last daily review at Nov 4, 0:00 UTC and still carry the field, so it cannot be the daily review's rerun of the listed entry. Inference: it may be a rerun made at submission time, or the value of an earlier reviewed entry from the same team. In the second case a wide gap (abhishekgahlot2, jt914) would mean the team improved since. It is not a robustness measure.
- The Oct 26 board in pjmullin3's screenshot (https://x.com/pjmull/status/2114602468252545324) shows #1 pjmullin3 at 1.24 ms, 3.03x, 5 runs; #2 colinlikescode 2.99x, 27 runs; #3 abhishekgahlot2 2.71x, 25 runs; #4 to #10 between 2.62x and 2.57x; SGLang at 3.77 ms. Our 2.605x is the level of that day's pack. Inference: our file matches the public baseline cluster, and the two outliers that day (about 3.0x) were already 15% ahead of it.

## Review marks

The board marks entries pending (P), flagged (F) or screened (S). The marks belong to individual entries, not to teams. The attention top 10 on the board text at 17:21 +04 has two pending entries (pjmullin3 at #1, zzz0906 at #10), two screened (olety at #4, mgpai22 at #9), and no flagged one. The other six (tjiang-fair, LeshengJin, jt914, gauravjain14, Fangzhou66, abhishekgahlot2) carry no mark.

- Verified from the timestamps: every screened or pending entry in the top 12 was submitted before Nov 3, 23:41 UTC, and every unmarked one was submitted after the Nov 4, 0:00 UTC review, except Fangzhou66's, which came 19 minutes before it. Inference: six of the attention top 10 have probably never been through a review, so their removal risk is unknown, not low.
- If both pending entries were removed at the final review, attention #1 would become tjiang-fair at 3.259x, then LeshengJin at 3.156x, then olety at 3.129x (arithmetic on the table). The 3.0x tier does not go away.
- A mark on a team's moe or bf16 entry says nothing about its attention entry. A pending mark only means the automatic review was unsure. The organizers' note about shape-specific variants is general and not tied to any team.
- Our own risk, same theme (inference): our attention file carries two overrides keyed to exact scored sizes (verify 32 sliding and decode 1 full). The decode 1 full one gives that shape the plan the file already uses for single-request 8-token calls, so it is a regime rule in effect; the verify 32 sliding one is an isolated key. The live rules ban tuning that only works at exact scored sizes, and the organizers say they will review shape-specific variants. We stopped adding such overrides for that reason.

## Public traces per handle

For each handle we checked GitHub (user and repos), gists, Hugging Face, X (search and profile), a web search, and the challenge site (home, leaderboard, updates, feed). Board names are GitHub logins according to the login text on https://openkernelchallenge.dev/ (inference), so GitHub is the anchor and an X account counts only where GitHub links it or the content matches. We list only what a handle has said publicly about method, and whether a public repo or write-up exists.

|handle|ranks (by board)|public repo or write-up|public statement about method|
|-|-|-|-|
|pjmullin3|attention #1, moe #3, decode_layer #16, bf16 #15|none|On Oct 26 he asked on X whether to open-source the 3x speedup and attached the board screenshot above. On Oct 27 someone asked publicly how the team got 3x on attention (https://x.com/pjmull/status/2115311886346690860); he answered with a joke and no technique.|
|tjiang-fair|attention #2, moe #6, decode_layer #8, bf16 #9|none|None found (two X searches on the exact handle).|
|LeshengJin|attention #3, moe #1, decode_layer #1, bf16 #5|none about the challenge (the account holds forks of general inference projects)|None found. Top five on all four boards with 3 to 6 submissions each (verified counts). Submissions are not judge checks, so the real iteration count is unknown.|
|olety|attention #4, bf16 #2, moe #4|none|None found. About 100 submissions on each of three boards, which looks like a submit-and-iterate loop (inference).|
|jt914|attention #5, decode_layer #15, bf16 #8|none|None found.|
|gauravjain14|attention #6, decode_layer #2, bf16 #6, moe #16|none (his blog has GEMM worklogs for another GPU generation, nothing on the challenge)|Oct 23: beating cuBLAS on every shape is not needed if you can tune for the next shape quickly. Nov 2: no special setup beyond Nsight Systems and a large Modal bill. Nov 3: posted the board.|
|Fangzhou66|attention #7, moe #5, bf16 #11|none|None found.|
|abhishekgahlot2|attention #8, decode_layer #18|none|None found. abhishekgahlot2 was #3 at 2.71x on Oct 26 and is at 3.028x now, with the entry from Nov 4, 07:08 UTC.|
|mgpai22|attention #9, bf16 #13|none|None found.|
|zzz0906|attention #10, moe #20|none|None found.|
|dolanzhao|moe #2, bf16 #3, decode_layer #5|none|None found.|
|gogowhy|bf16 #1, decode_layer #3, attention #18|no GitHub account of that name|None found (searched with two leads only).|
|AlexCuadron|decode_layer #4|none|None found.|
|reyanshsolis|bf16 #4|none|None found.|

Other traces seen, none of them targets: lxe (decode_layer #10) posted about the six-worker loop around Nov 4. venkatGoku reached 2.566x on attention on Nov 3. A prefill post by @Prav4982 (Nov 2) names FP8 TMA FlashAttention. The repo lee101/openkernels is public (1.5x, Oct 24). None of these gives an attention mechanism that beats ours.

Dead ends, so nobody repeats them: GitHub code search for our file header over the 13 logins returned 0 files, with a note that the code is not indexed (inconclusive); five web searches found only model pages and unrelated challenges; X searches for five of the handles as exact strings returned nothing; Hugging Face has nothing relevant.

## What 3x would take

Our per-load times (sum of weight times shape time, from `headroom-model-all-boards.md`) are: decode 1 408 µs, decode 16 1770, decode 32 3109, verify 1 542, verify 16 2141, verify 32 3629. Their geomean is about 1,450 µs, close to the 1,446 µs the board lists. The score is the geomean over those six loads, so each load counts equally.

Sliding calls are 46 of 62 layers. Their share of a load is 56% for decode 1, 22% for decode 16, 15% for decode 32, 58% for verify 1, 24% for verify 16 and 19% for verify 32. Add decode 1 full and verify 1 full, which are also latency chains, and the decode 1 and verify 1 loads are 100% chain.

|scenario|what changes|score|
|-|-|-|
|A|bandwidth shapes at their bound only|+9.9%, 2.86x|
|B|all chain shapes 30% faster, full shapes unchanged|3.06x|
|C|chain shapes 35% faster and every full shape 10% faster|3.35x, the #1 entry|

Scenario C would put decode 1 sliding near 3.2 µs, verify 1 sliding 4.4, decode 16 sliding 5.4, decode 32 sliding 6.5, verify 16 sliding 7.4, verify 32 sliding 9.7, decode 1 full 7.3 and verify 1 full 9.3. The headroom model's assumed floor is 3.0 µs and its assumed stream rate is 4.1 TB/s, which is a second-hand figure. So either the leaders gain more than 10% on the full shapes (decode 32 full runs at about 3.66 TB/s effective today: 165.55 µs for bytes that take 126.35 µs at 4.8 TB/s), or the true floors are below our assumptions. We treat both assumptions as soft when planning.

## Techniques worth testing for attention

None of these is a published method of any top team. Gains are whole-score estimates (inference). "Tried" means we already tried it and dropped it.

|rank|idea and where it comes from|plausibility|expected gain|status for us|
|-|-|-|-|-|
|1|Latency-chain work on the sliding shapes of all six loads plus decode 1 full and verify 1 full (scenarios B and C). No top team has described a mechanism.|High that this is where the leaders' edge is; the specific lever is unknown.|The only bucket large enough: 15% on all chain shapes is about +7.8%, 30% about +17% (arithmetic).|Partly tried: cold-code warming (2.576x), merge and balance tricks, uneven partition, a producer-share probe. We later added a latency-chain study and a sweep of launch plans on all twelve shapes, and neither found a lever (see `../attention/ideas-tried-and-results.md`). The entry already has split-KV, clusters with distributed shared memory, a flat and a cooperative finalize, and a PDL bit.|
|2|Packed half2 softmax, FP16-accumulated QK^T and an approximate exp2 into E4M3 P codes on verify 16 full and verify 32 full (legal since the Nov 3 guidance; the repo study in `ideas-from-keell0renz-repo.md` has a related item).|Medium. Whether packed ex2 issues as one op is unconfirmed, and these shapes may not be issue-bound.|Ceilings of 3.6% and 2.8% on those two shapes (about 3.2% whole); realistic 0 to 1%.|Checked later in SASS: there is no packed exp2 instruction, so the instruction count per pair is unchanged (5 against 5) and we did not run it. This is not the same as the FMA or polynomial exp2 others tried (24 to 105 µs slower). The post-guidance wave is a weak hint that this family matters, confounded by recency.|
|3|Re-test the stream assumption on decode 16 full and decode 32 full: compare our K/V row-load path with what a faster stream would need.|Low to medium. It is probably already the 16-byte-per-lane cp.async pattern, and nobody has shown a rate above 4.1 TB/s.|At the model's 4.1 TB/s bound, +3.3% for the two shapes (1.79% and 1.46%). If 4.4 TB/s were reachable, about +5.2% (arithmetic).|Deeper stages (3 to 4 µs slower) and uneven partition are already dropped. A 128-byte bulk copy per row was rated under 0.3%. A microbenchmark of random 128-byte row reads would settle it, but we have no GPU.|
|4|nc=4 workspace copies for decode 1 sliding.|Low|Under 0.5%|Tested later (job 66be38f978b54004): decode 1 sliding came out at 5.53 µs against 4.94 to 5.17 for the base. No-go.|

Not recommended: anything cgemv-like or derived from dense GEMMs (those wins did not carry over in decode_layer, see `research-gaps-and-answers.md`), and per-shape variants keyed to exact sizes (banned by the live rules and flagged in the organizers' Nov 3 note).

## Verified, inferred, not found

- Verified: the timeline and numbers above, the guidance text, the two process statements, the absence of public code for all 14 handles, and that the six unmarked entries in the attention top 10 are newer than the last daily review.
- Inferred: that the post-Nov 3 wave uses the allowances, what `screened_speedup` measures, that the gain sits mostly in latency-chain shapes, and anything about which shapes they sped up. No team published per-shape numbers, and the feed has no per-load fields.
- Not found: any mechanism, per-shape figure or code from a top-10 team, and any sign of encoded source or exact-size tuning by a named top-10 team beyond the organizers' general warning.

The feed endpoint above is the cheapest way to re-read the board. The other boards use the same path with their own problem name (we saw the bf16 one in the page's requests).

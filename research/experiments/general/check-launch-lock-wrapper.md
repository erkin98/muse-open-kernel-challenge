# A locked, logged launcher for `okc check`

This file describes the small shell wrapper we put around `okc check` so that several people could launch judged checks at the same time without stepping on each other, what it enforces, and what we saw when we used it.

A check is a judged run that stays off the board. The wrapper is about 235 lines of bash and needs nothing beyond `flock` and the usual coreutils. It decides nothing about whether a check is worth running. It only makes launching one orderly and leaves a record.

## The problem

`okc check` blocks until the judge returns. In the runs we followed, that was 118 to 611 seconds from launch to result, and we have seen queues longer than an hour. With a few people sharing one account and one set of daily allowances, three things go wrong:

- Limits such as "at most two checks on this problem" are enforced by counting earlier launches. Counting and launching are two steps, so two simultaneous launchers can both pass the count.
- Two launchers can send byte-identical files without knowing about each other.
- Nobody can see what is in flight, who started it, or which file hash it carried.

## What it does

Each run follows the same sequence:

1. Check the arguments and the file (details below), compute the file's sha256, and print a dry-run summary if asked.
2. Take a lock for that problem. Under it, take a second, much shorter lock on the ledger and, while holding both, read the ledger for caps and for earlier launches of the same bytes and append a START row. Release the ledger lock, run `okc check <problem> <file> --no-wait`, pull the 16-hex-digit job id out of its output, append a LAUNCH row, and release the problem lock. This takes a few seconds.
3. Outside the lock, wait for the result with the read-only `okc status <job> --wait`, then parse the verdict, speedup, interval and test count from the final text and append an END row. The full output goes to a per-run log file.

`okc check --no-wait` returns as soon as the job is accepted, which is what makes the short critical section possible. The first live run confirmed its output has three lines: a "checking <file> on <problem>" line ending in the job id, a sentence saying this is a check run kept off the board and out of attempts, and a "next okc status <job> --wait" line.

If no 16-hex job id appears, the wrapper logs `nojobid` and exits with code 4, and the person falls back to `okc jobs`.

## What it refuses

The wrapper accepts the word `check` and nothing else. `submit`, `withdraw`, `login`, `logout`, `status`, `jobs`, `whoami` and every other word are refused before any call to okc, and the refusal is logged. No extra flags are passed through to okc. We wanted a tool that could not publish or withdraw an entry by accident.

It also refuses when:

- the problem name does not look like `muse_` plus a word;
- the file is missing, unreadable or not a `.py` file;
- the first line of the file is not the header `# okc: problem=<name>` with the same name as the argument (this catches a file aimed at the wrong problem);
- the caller did not say who is launching;
- an optional cap, total or per problem, is already used up. Caps count non-duplicate launches in the ledger.

A repeat of a file the ledger already knows is a warning, not a refusal, and does not count against the caps. A repeat that only the platform recognizes counts until the END row shows it was a duplicate. This is conservative.

The wrapper never reads the okc credentials file, starts no background process and leaves nothing running after it exits. If it is killed, the kernel releases the lock because `flock` goes away with the process. A `--dry-run` option prints the command, the sha256, the header check, whether the lock is free (and who held it last), duplicate warnings and cap use, and writes nothing.

Exit codes: 0 ok, 2 refused, 3 lock busy, 4 launch failed or no job id, 5 launched but the status follow-up failed, 130 interrupted (the judge job keeps running).

## The ledger

One tab-separated row per event, with the columns time, run id, event, who, problem, sha256, file, job, return code and a free-text note. Events are START (written under the lock, before launching), LAUNCH (job id known), END (result or failure), REFUSED, and NOTE for manual annotations. A run with a START and no END is in flight, or the wrapper was killed hard, in which case `okc jobs` tells the truth. The caller can put a one-line description of the change and the incumbent job id in the note, so the ledger doubles as a promotion log: incumbent, challenger hash, job, verdict.

## Why a lock per problem, and why only around the launch

Our reasoning, which is partly inference:

- The count, duplicate check and START row have to happen as one step. That is what the short ledger lock does, and because it is shared across problems, a total cap (all problems) stays correct as well as a per-problem cap.
- The per-problem lock is for our own bookkeeping more than for the platform. We have no evidence that two simultaneous launches on one problem break anything on the judge side: three bf16 checks had run concurrently before the wrapper existed. What the lock gives us is order. START, the job id and the holder name are written one launch at a time, and the person who loses the race sees who got there first.
- The lock must not cover the wait. A judge run takes minutes and sometimes an hour. If the lock were held through the wait, one slow check would block everyone on that problem. With the lock around the launch only, a second launcher waits a few seconds, not an hour. Launches on different problems never wait for each other.
- When a lock is busy, the message names the last holder, so the second person retries in a minute (the default wait is 120 seconds, adjustable). Busy for more than a few minutes would mean a stuck launch.
- The locks are voluntary. They work only if everyone launches through the wrapper.

## Tests

The test script makes 62 assertions against a stand-in `okc` script and one verbatim real result text (2.589x, interval 2.550 to 2.637, 16/16 tests), and none failed. They cover:

- static properties: the script parses, never mentions the credentials file, and contains no okc call other than `check` and `status`;
- a dry run writes nothing and runs nothing;
- refusal of eleven other commands and several malformed option sets, with no okc call made;
- bad inputs: wrong problem name, missing file, wrong extension, wrong header, missing launcher name;
- a full run through the stand-in, with START, LAUNCH and END sharing one run id, ten fields on every row and a 16-hex job id;
- duplicates warn and do not count; caps refuse the right launch;
- three concurrent runs, two on one problem and one on another. The second launch on the first problem began at least 2.9 seconds after the first (the stand-in's launch takes about 3 seconds). It started its own wait before the first run's 6-second status wait had finished, so the lock was not held during the wait. The launch on the other problem was not delayed (under 1.5 seconds).

## What the live platform does with a repeat

We did one live launch, on a file byte-identical to an earlier decode-layer check (job 042deb1ca7a94df6, sha256 prefix 706088bb). We expected either a duplicate or a new queued job, and we meant to stop at once if it queued.

- `okc check --no-wait` printed a new job id, 2edf85bcdab04a1e, and nothing in the launch text said duplicate.
- A status call right after the launch still showed only `received` for 16 seconds. We interrupted the wrapper there (exit 130, as intended).
- `okc status <job> --json` then showed status `duplicate`, a `duplicate_of` field pointing at 042deb1ca7a94df6 and a null result. `okc status <job> --wait` returned at once with the line "the same code as job 042deb1ca7a94df6" and exit 0. No judge run happened.

After this we changed the wrapper to follow that pointer and read the earlier job's result. The END row then carries `dup-of=<earlier job>`.

A duplicate does not seem to use up one of the day's check allowances. This is an inference from counting. At the time, the moe problem had five non-duplicate checks and one duplicate since the daily reset and showed 35 of 40 left. The decode problem had seven non-duplicate checks (one failed) and one duplicate and showed 33 left. The arithmetic only works if duplicates are free.

## What happened in use

The ledger ended with 22 launches over about 12 hours, one of them the duplicate test. No launch was refused and none timed out waiting for the lock. In the last batch, nine checks on the attention problem were started within 12 seconds and the ledger shows them leaving one at a time, 1 to 2 seconds apart, which is what the lock should produce. That is consistent with the lock working, not proof.

Eight runs were followed to the end through the wrapper, and all eight produced a parsed verdict, speedup, interval and test count. One quirk: in those rows the speedup field holds two numbers (for example 2.363 followed by 2.398, with the interval 2.328 to 2.398). The parser picks every number followed by a multiplication sign on the headline line, so the second is the upper end of the interval. The first is the headline.

The other 13 launches used the no-follow option and collected the result later.

## Limits

- It sees only launches made through it. A direct `okc check` is not recorded, and caps and duplicate detection do not know about it. `okc jobs` stays the source of truth.
- Caps are only as good as the number the caller passes. The ledger counts its own launches, so a person who had already launched checks by hand has to subtract them.
- It does not read allowances, retry, queue behind a busy lock or judge whether a check is worth running.
- Result parsing was tested on a stand-in and one real text before the first live result. It worked on the eight live runs that followed, but the output format belongs to the okc CLI and could change.

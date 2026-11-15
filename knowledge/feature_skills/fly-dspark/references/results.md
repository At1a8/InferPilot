# FLy on DSPARK — measured results (DeepSeek-V4.1-Flash, gfx950)

Setup common to every table: DeepSeek-V4.1-Flash with its DSPARK drafter at gamma 5, TP4/EP4 on
gfx950 (`../scripts/serve_dspark_fly.sh`), entropy top-k 3, sglang `ed3831842a` (the FLy branch
merged with main `410e64bb84`). Two lanes on one 8-GPU node benched at the same time
(`../scripts/lockstep_ab.sh`), arms rotated between the lanes. "Wn/t" = FLy window n, entropy
threshold t. Measured 2026-11-13 to 2026-11-15.

## Accuracy

evalscope, full sets: gsm8k (1319), gpqa_diamond (198), simple_qa (4326, LLM judge); thinking on,
`reasoning_effort` max, `max_tokens` 102400, the server's default sampling (T = 1.0, top_p = 1.0).
Accept length per dataset from `/server_info` after a cache flush. Scores are the mean of the runs;
changes are against the mean of 5 baseline runs (gsm8k 0.9713, gpqa_diamond 0.9101, simple_qa 0.4452;
accept length 3.79 / 3.15 / 3.06). Accept length is the mean change over the three datasets.

| Setting | runs | gsm8k | gpqa_diamond | simple_qa | accept length |
|---|---|---|---|---|---|
| W1/0.3 | 1 | 0.9689 (−0.2pp) | 0.9242 (+1.4pp) | 0.4152 (−3.0pp) | +21.7% |
| W1/0.5 | 1 | 0.9750 (+0.4pp) | 0.9141 (+0.4pp) | 0.4154 (−3.0pp) | +17.3% |
| W2/0.2 | 2 | 0.9711 (−0.0pp) | 0.9015 (−0.9pp) | 0.4258 (−1.9pp) | +12.3% |
| W2/0.3 | 2 | 0.9705 (−0.1pp) | 0.8939 (−1.6pp) | 0.4258 (−1.9pp) | +10.9% |
| W2/0.4 | 1 | 0.9727 (+0.1pp) | 0.8939 (−1.6pp) | 0.4295 (−1.6pp) | +10.5% |
| W2/0.5 | 2 | 0.9712 (−0.0pp) | 0.9166 (+0.7pp) | 0.4360 (−0.9pp) | +9.7% |
| W2/0.6 | 1 | 0.9719 (+0.1pp) | 0.9040 (−0.6pp) | 0.4360 (−0.9pp) | +8.1% |
| W2/0.7 | 2 | 0.9720 (+0.1pp) | 0.9192 (+0.9pp) | 0.4407 (−0.4pp) | +6.0% |
| W3/0.2 | 1 | 0.9727 (+0.1pp) | 0.9192 (+0.9pp) | 0.4304 (−1.5pp) | +6.2% |
| W3/0.3 | 2 | 0.9715 (+0.0pp) | 0.9141 (+0.4pp) | 0.4363 (−0.9pp) | +6.1% |
| W3/0.4 | 1 | 0.9704 (−0.1pp) | 0.8939 (−1.6pp) | 0.4376 (−0.8pp) | +4.9% |
| W3/0.5 | 2 | 0.9750 (+0.4pp) | 0.9065 (−0.4pp) | 0.4334 (−1.2pp) | +4.8% |
| W3/0.6 | 1 | 0.9727 (+0.1pp) | 0.8939 (−1.6pp) | 0.4464 (+0.1pp) | +3.8% |
| W3/0.7 | 1 | 0.9682 (−0.3pp) | 0.9343 (+2.4pp) | 0.4452 (+0.0pp) | +2.4% |
| W4/0.2 | 1 | 0.9727 (+0.1pp) | 0.9141 (+0.4pp) | 0.4413 (−0.4pp) | +2.2% |
| W4/0.3 | 1 | 0.9757 (+0.4pp) | 0.8939 (−1.6pp) | 0.4526 (+0.7pp) | +2.1% |
| W4/0.5 | 1 | 0.9727 (+0.1pp) | 0.9040 (−0.6pp) | 0.4496 (+0.4pp) | −0.0% |

Noise: the 5 baseline runs span simple_qa 0.4434–0.4475 over two days; two runs of the same FLy
setting differed by up to 0.9pp (W3/0.5: 0.4378 / 0.4290), so a single run is good to about ±0.5pp.
gpqa_diamond has 198 questions: ±1.7pp is three questions.

## Throughput, 8k/1k

Random prompts of exactly 8000 input / 1000 output tokens (`ignore_eos`), T = 1.0, concurrency 32 / 64 /
128 with 4x prompts and 2x warm-up requests (`../scripts/bench_random.sh`). Each cell: the mean over the
runs and the three concurrencies of the change against the mean of all baseline runs, with the range of
the single runs. W2 / W3: 3 runs each against 6 baseline runs (34,727 / 45,553 / 54,169 total tok/s and
6.99 / 10.86 / 18.18 ms TPOT mean at c32 / c64 / c128; the 6 runs span −1.3 .. +0.5% of their mean).
W1 / W4: one run each against two baseline runs on the same lane.

| Setting | total tok/s | TPOT mean | TPOT median | accept length (bench) |
|---|---|---|---|---|
| W1/0.3 | +8.1% | −5.6% | −2.4% | +11.9% |
| W1/0.5 | +7.5% | −5.5% | −3.0% | +10.9% |
| W2/0.2 | +4.1% (+3.6 .. +4.5) | −3.1% | −0.6% | +6.3% |
| W2/0.3 | +3.4% (+2.4 .. +5.1) | −2.7% | −0.1% | +6.0% |
| W2/0.4 | +3.5% (+3.0 .. +4.1) | −3.0% | −0.2% | +6.4% |
| W2/0.5 | +2.5% (+2.0 .. +3.0) | −1.8% | −0.4% | +4.4% |
| W2/0.6 | +3.1% (+2.9 .. +3.2) | −2.5% | −0.7% | +5.0% |
| W2/0.7 | +2.8% (+2.5 .. +3.1) | −1.8% | −0.9% | +4.2% |
| W3/0.2 | +2.1% (+1.3 .. +2.6) | −1.3% | −0.2% | +2.9% |
| W3/0.3 | +1.8% (+1.7 .. +2.0) | −1.6% | −0.3% | +3.4% |
| W3/0.4 | +1.7% (+0.9 .. +2.2) | −1.0% | +0.5% | +2.6% |
| W3/0.5 | +1.4% (+0.6 .. +1.8) | −0.6% | +1.3% | +1.6% |
| W3/0.6 | +1.7% (+1.0 .. +2.7) | −1.0% | +0.4% | +2.5% |
| W3/0.7 | −0.2% (−1.2 .. +1.1) | +0.0% | +0.6% | +0.8% |
| W4/0.2 | +0.8% | −0.2% | +0.1% | +0.4% |
| W4/0.3 | +0.1% | +0.8% | −1.2% | −1.0% |
| W4/0.5 | +1.1% | −0.5% | −0.5% | +0.2% |

- The random-prompt benchmark already accepts about 4.4 tokens per step at baseline (out of 6), so it
  shows less of FLy's effect than the accuracy datasets (3.1–3.8 at baseline).
- TPOT median moved by less than its noise (±2%) for every setting; TPOT mean follows total throughput.

## Throughput, 100k/1k

Same benchmark with 100,000 input tokens; one run per setting (W2/0.5: two) against two baseline runs on
the same lane (lane A: 83,922 / 87,458 / 90,519 total tok/s at c32 / c64 / c128).

| Setting | total tok/s | TPOT mean |
|---|---|---|
| W1/0.3 | +0.6% | −0.7% |
| W1/0.5 | +0.5% | −0.4% |
| W2/0.2 | +0.2% | −0.2% |
| W2/0.3 | +0.1% | −0.5% |
| W2/0.5 | +0.2% | −0.1% |
| W2/0.7 | +0.0% | −0.1% |
| W3/0.3 | +0.2% | −0.4% |
| W3/0.5 | +0.0% | +0.4% |
| W4/0.2 | +0.2% | −0.1% |
| W4/0.3 | +0.1% | +0.1% |
| W4/0.5 | −0.0% | −0.4% |

At 100k/1k about 99% of the processed tokens are prompt tokens: the total is set by prefill, and a
decode-side change does not show in it.

## W2/0.7 repeat

Pending.

# Evaluation report (SYNTHETIC benchmark)

**Read first.** These numbers come from generated signals that carry the *real* NASA SMAP/MSL label geometry (channel ids, test lengths, anomaly index ranges, classes). The signal values and the injected anomalies are mine. They show the harness works and compare two methods fairly *on this benchmark*. They say nothing certain about real spacecraft data. Reproduce on real data with the commands in the README and replace this section with those numbers.

## Protocol

1. **Tuning** (`scripts/tune_hyperparameters.py`, results in `results/tuning_seed100.json`): synthetic seed 100, 16 channels. Grid-searched z-score window / threshold / level-vs-difference mode, and LSTM window length / hidden size / epochs / smoothing span / pruning threshold, by sequence-level F1. Chosen: z-score window 400, threshold 4, level mode; LSTM length 32, hidden 32, 16 epochs, smoothing span 15, pruning 0.13. Note that these choices used the labels of the tuning set (the *threshold itself* inside the LSTM detector is label-free).
2. **Final evaluation** (single run, no retries): synthetic seed 0, 12 channels disjoint from the tuning channels (A-1, A-8, D-11, D-2, D-8, E-2, E-8, F-5, G-4, M-5, P-14, R-1), 59.2 channel-days at 60 s cadence. Command: `python -m telemetry.anomaly.run_benchmark --synthetic-seed 0 --channels <those 12> --out results/synthetic_seed0`. Dataset fingerprint (sha256 of all arrays) is in `results.json`.
3. **Scoring** (`anomaly/evaluate.py`): sequence-level, as in Hundman et al. A true sequence is *detected* if any alert overlaps it; each alert overlapping no true sequence is a *false alarm*; counts are micro-averaged across channels. Latency = first overlapping alert start minus true start.

## Results

| Detector | Precision | Recall | F1 | False alarms / channel-day | Detected / true | Point | Contextual |
|---|---|---|---|---|---|---|---|
| Rolling z-score | 0.250 | 0.167 | 0.200 | 0.10 | 2 / 12 | 2 / 10 | 0 / 2 |
| LSTM + dynamic threshold | 0.600 | 0.500 | 0.545 | 0.07 | 6 / 12 | 6 / 10 | 0 / 2 |

Precision = detected / (detected + false alarms); recall = detected / true sequences. Alerts raised: LSTM 10 (6 detections, 4 false alarms); z-score 11 (2 detections, 6 false alarms; the other alerts fall on already-detected sequences). Exact counts per channel are in `results/synthetic_seed0/results.json`.

Training time: about 3-4 s per channel on a 2-core CPU (the first channel took 52 s, probably one-time start-up cost; I did not isolate the cause).

## How much to trust this

- **Tiny sample.** 12 true sequences. 6/12 vs 2/12 is suggestive, not significant; a different seed could move it noticeably. I did not run multiple seeds.
- **Lenient overlap rule.** Several labelled "point" ranges are long (D-2: 4,218 samples, E-2: 1,398, E-8: 623) while the generator puts only a few short spikes inside them. An alert on any one spike counts the whole range as detected. This is the same overlap logic the original paper uses, but it flatters recall. Likewise "mean latency" (LSTM 362 min, z-score 1,452 min) is dominated by those long ranges - the spike simply sits late in the range - so it is *not* a real detection delay. Short ranges (D-11, D-8, G-4, R-1) were caught 18-44 samples after the range start.
- **Self-made anomalies.** The point anomalies are spikes, which suit a forecaster; real anomalies are messier.
- **Hyper-parameters were tuned with labels** on seed 100; the generator is the same, so tuning may transfer better here than to real data.

## Failure cases (`docs/figures/`)

- **`A-8.png` - contextual anomaly missed by both.** The signal switches from a noisy stepwise pattern to a clean sine. That clean sine is *easier* for the LSTM to predict than the normal data, so its prediction error goes **down**, and an error-above-threshold rule cannot see it. The z-score also misses it and raises 3 false alarms on the stepwise jumps (every step looks like an outlier to a rolling mean/std). Lesson: forecasting error detects "harder to predict", not "different".
- **`M-5.png` - contextual anomaly missed** (301-sample range) with one false alarm just after the range ends.
- **`D-8.png` - one detection plus one false alarm.**
- **`D-11.png` - the clean win**: a short spike gives a clear LSTM error peak; the z-score on values cannot see it because the sine's own swing dominates the rolling standard deviation (see the docstring of `ZScoreDetector`; a difference-mode z-score was tested during tuning and traded recall for ~100 false alarms).
- **0 of 2 contextual anomalies detected by either method.** That is the honest headline weakness.

## Differences from Hundman et al. (KDD 2018)

- One-step-ahead, single-layer LSTM per channel in PyTorch, trained on the train split only (the paper uses a stacked LSTM in Keras and predicts multiple steps ahead).
- Simplified dynamic threshold: segments of 2,000 points, EWMA smoothing, the `(d_mean/mean + d_std/std)/(n_flagged + n_seq^2)` criterion, pruning with a 13% minimum drop. Written from the paper's description, not the authors' code.
- Warm-up errors (first `length` points of each test array) are set to the median so the train/test junction cannot create a fake spike.
- Multivariate telemetry inputs (extra command features present in the real arrays) are not used.
- Timestamps: SMAP/MSL have none; a fixed 60 s cadence is assumed, so "days" and "false alarms per channel-day" are assumptions of this project.

## Next experiments I would run (not done)

Multiple seeds with confidence intervals; the real SMAP/MSL data; per-channel threshold sanity checks; a model that scores *distribution* change (so smooth contextual anomalies are visible); comparing against an isolation forest.

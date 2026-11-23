# Data

Nothing in `data/raw/` is committed. `labeled_anomalies.csv` is committed (small).

## 1. Labels - committed
`labeled_anomalies.csv`: channel id, spacecraft (SMAP / MSL), anomaly index ranges, class (point / contextual), number of test values.
Source: NASA/JPL `telemanom` repository (https://github.com/khundman/telemanom), Apache-2.0, Copyright California Institute of Technology. Used here with attribution.

## 2. Real arrays - you download them
Source used for the scripts: Kaggle dataset `patrickfleith/nasa-anomaly-detection-dataset-smap-msl` (the telemanom README also describes the original archive). Check the dataset page for its current licence before redistributing anything.

```bash
bash scripts/fetch_smap_msl.sh      # needs a free Kaggle API token
```
Expected layout: `data/raw/train/<channel>.npy`, `data/raw/test/<channel>.npy`, values in column 0 (other columns are command features and are ignored).
The download was **not** possible in the build sandbox (blocked), so the script and layout are written from the dataset's documentation and are untested. After downloading, record `sha256sum data/raw/test/*.npy | sha256sum` below.

| Date downloaded | Source | Fingerprint |
|---|---|---|
| (fill in) | (fill in) | (fill in) |

## 3. Synthetic benchmark - generated, not committed
`python -m telemetry.anomaly.run_benchmark --synthetic-seed N` writes arrays in the same layout under the output directory. Real label geometry, generated values. Fingerprint of seed 0 / the 12 evaluation channels is recorded in `results/synthetic_seed0/results.json`.

## Timestamp assumption
The arrays carry no timestamps. The project stamps them at a fixed 60 s cadence from `STREAM_START` (default 2026-01-01T00:00:00Z). "Days" in the metrics are therefore `samples * 60 s / 86400`.

## Optional: orbit data
The pass-prediction stretch (`orbit/passes.py`) uses a synthetic ISS-like TLE generated offline. `fetch_tle` can download a real TLE from CelesTrak (not run in the sandbox).

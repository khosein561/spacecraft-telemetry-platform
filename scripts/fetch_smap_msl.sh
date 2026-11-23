#!/usr/bin/env bash
# Download the REAL NASA SMAP/MSL anomaly dataset into data/raw/ (needs a free Kaggle account).
# NOT run by the author of this repo: the build sandbox had no access to Kaggle or the S3 archive.
# Check the layout after downloading (see data/README.md) before trusting any results.
#
# 1. Create an API token at https://www.kaggle.com/settings  -> saves kaggle.json
# 2. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# 3. pip install kaggle
set -euo pipefail
mkdir -p data/raw
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl -p data/raw --unzip
echo "Layout expected: data/raw/train/<channel>.npy and data/raw/test/<channel>.npy"
echo "If the files landed in a sub-folder, move them so that layout holds, then run:"
echo "  sha256sum data/raw/test/*.npy | sha256sum    # record the result in data/README.md"

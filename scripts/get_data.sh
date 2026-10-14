#!/usr/bin/env bash
# Download the ThinkGuard critique training data at a pinned upstream commit.
# Run from the repo root: bash scripts/get_data.sh
set -euo pipefail

SHA=a7d0c8924a343009e9fd4b31aa84ac845b1ff454
URL="https://raw.githubusercontent.com/luka-group/ThinkGuard/${SHA}/data/beavertails_critiques_aug.json"
OUT=data/beavertails_critiques_aug.json

[ -f scripts/get_data.sh ] || { echo "Run from the repo root: bash scripts/get_data.sh" >&2; exit 1; }

mkdir -p data
curl -fL "$URL" -o "$OUT"
echo "Saved $OUT"

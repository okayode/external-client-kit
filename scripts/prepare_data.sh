#!/bin/bash
set -e

KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-$KIT_ROOT/data/cifar10}"

export PYTHONPATH="$KIT_ROOT/src:$KIT_ROOT:${PYTHONPATH:-}"

echo "Preparing CIFAR-10 data"
echo "Data root: $DATA_ROOT"

mkdir -p "$DATA_ROOT"

python - <<PY
from data.cifar10_data_utils import load_cifar10_data

data_root = r"$DATA_ROOT"

print(f"Loading CIFAR-10 data into: {data_root}")
load_cifar10_data(data_root)
print("CIFAR-10 data preparation complete.")
PY

#!/bin/bash
set -e

KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$KIT_ROOT/src:$KIT_ROOT:${PYTHONPATH:-}"

SITE_NAME="${1:?Usage: $0 <site_name> <site_index> <num_sites> [alpha] [seed] [data_root] [split_root]}"
SITE_INDEX="${2:?Usage: $0 <site_name> <site_index> <num_sites> [alpha] [seed] [data_root] [split_root]}"
NUM_SITES="${3:?Usage: $0 <site_name> <site_index> <num_sites> [alpha] [seed] [data_root] [split_root]}"
ALPHA="${4:-0.5}"
SEED="${5:-0}"
DATA_ROOT="${6:-$KIT_ROOT/data/cifar10}"
SPLIT_ROOT="${7:-$KIT_ROOT/splits}"

if ! [[ "$SITE_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: site_index must be an integer."
    exit 1
fi

if ! [[ "$NUM_SITES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: num_sites must be an integer."
    exit 1
fi

if [ "$SITE_INDEX" -lt 1 ] || [ "$SITE_INDEX" -gt "$NUM_SITES" ]; then
    echo "ERROR: site_index must be between 1 and num_sites."
    exit 1
fi

echo "Preparing federated split"
echo "  Site name:   $SITE_NAME"
echo "  Site index:  $SITE_INDEX"
echo "  Num sites:   $NUM_SITES"
echo "  Alpha:       $ALPHA"
echo "  Seed:        $SEED"
echo "  Data root:   $DATA_ROOT"
echo "  Split root:  $SPLIT_ROOT"

mkdir -p "$SPLIT_ROOT"

python - <<PY
from src.data.cifar10_data_split import split_and_save
import os
import shutil

site_name = r"$SITE_NAME"
site_index = int("$SITE_INDEX")
num_sites = int("$NUM_SITES")
alpha = float("$ALPHA")
seed = int("$SEED")
data_root = r"$DATA_ROOT"
split_root = r"$SPLIT_ROOT"

split_and_save(
    split_dir_prefix=split_root,
    num_sites=num_sites,
    alpha=alpha,
    seed=seed,
    data_root=data_root,
)

generated_dir = (
    split_root
    + f"_{num_sites}sites_alpha{alpha:.2f}_seed{seed}"
)

source_file = os.path.join(
    generated_dir,
    f"site-{site_index}.npy",
)

target_file = os.path.join(
    split_root,
    f"{site_name}.npy",
)

if not os.path.exists(source_file):
    raise FileNotFoundError(
        f"Generated split not found: {source_file}"
    )

shutil.copy2(source_file, target_file)

print(f"Created NVFlare site split: {target_file}")
PY

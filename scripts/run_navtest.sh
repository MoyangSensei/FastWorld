#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT to the NAVSIM v1.1 checkout}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM dataset root}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to the directory containing metric_cache}"

export FASTWORLD_CHECKPOINT="${FASTWORLD_CHECKPOINT:-$repo_root/checkpoints/fastworld_navsim_v1_92p2.pt}"
if [[ ! -f "$FASTWORLD_CHECKPOINT" ]]; then
    echo "Checkpoint not found at $FASTWORLD_CHECKPOINT" >&2
    exit 1
fi

export PYTHONPATH="$repo_root/src:$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

python -O -u "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    "hydra.searchpath=[file://$repo_root/configs,pkg://navsim.planning.script.config.common]" \
    agent=fastworld_navsim_v1 train_test_split=navtest worker=sequential \
    experiment_name=fastworld_navsim_v1 "$@"

#!/bin/bash
set -euo pipefail

RUNS_DIR="/workdir/diploma/masters/runs"
MERGED_DIR="/workdir/diploma/masters/merged_models"
SWIFT="/workdir/conda/miniconda3/envs/grpo-train/bin/swift"

export CUDA_VISIBLE_DEVICES=0

for run_name in \
    grpo_nerel_qwen3_4b_20260521_200100
do
    run_dir="$RUNS_DIR/$run_name"
    out_dir="$MERGED_DIR/$run_name"

    ckpt=$(find "$run_dir/model_output" -type d -name "checkpoint-*" | sort -V | tail -1)

    if [ -z "$ckpt" ]; then
        echo "SKIP $run_name — no checkpoint found"
        continue
    fi

    echo "=========================================="
    echo "Merging: $run_name"
    echo "Checkpoint: $ckpt"
    echo "Output: $out_dir"
    echo "=========================================="

    "$SWIFT" export \
        --adapters "$ckpt" \
        --merge_lora true \
        --output_dir "$out_dir"

    echo "Done: $run_name"
done
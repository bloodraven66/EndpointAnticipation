#!/bin/bash
# Offline evaluation on Full Duplex Bench v1 (https://github.com/DanielLin94144/Full-Duplex-Bench).
#
# Requires all services running (see dockerless/start_*.sh).
#
# Usage:
#   FDB_DATA=/path/to/Full-Duplex-Bench-Data bash infer_fdb.sh
#
# Output is written to ./results/<model_name>/<task>/

set -eo pipefail

# ── User-configurable ──────────────────────────────────────────────────────
# Path to the Full Duplex Bench dataset root (must contain v1.0/candor_turn_taking/)
FDB_DATA="${FDB_DATA:-/path/to/Full-Duplex-Bench-Data}"

# anticipate = speculative endpoint anticipation (main paper result)
# smalltalk_no_starter = standard VAD baseline
instruction_type=anticipate

version="v1.0"
task="candor_turn_taking"

skip_if_present=true
# ──────────────────────────────────────────────────────────────────────────

root_path="$(cd "$(dirname "$0")" && pwd)"

m=gemma3_1b
model_name="unmute_${instruction_type}_${m}"
output_path="${root_path}/results/${model_name}/${task}"
input_dir="${FDB_DATA}/${version}/${task}"

echo "Input dir:   ${input_dir}"
echo "Output path: ${output_path}"
mkdir -p "${output_path}"

export PYTHONPATH="${root_path}"

current_id=0
total_files=$(ls "${input_dir}" | wc -l)

for id in $(ls "${input_dir}"); do
    current_id=$((current_id + 1))
    audio_path="${input_dir}/${id}/input.wav"

    if [ ! -f "${audio_path}" ]; then
        echo "File not found: ${audio_path}"
        continue
    fi

    item_out="${output_path}/${id}"
    mkdir -p "${item_out}"
    save_path="${item_out}/output.wav"

    if [ "${skip_if_present}" = true ] && [ -f "${save_path}" ]; then
        echo "[${current_id}/${total_files}] Skipping ${id} (output exists)"
        continue
    fi

    echo ""
    echo "[${current_id}/${total_files}] Processing: ${id}"

    if [ "${instruction_type}" = "smalltalk_no_starter" ]; then
        python3.12 unmute/scripts/evaluate_recording.py "${audio_path}" "${save_path}"
    elif [ "${instruction_type}" = "anticipate" ]; then
        python3.12 unmute/scripts/evaluate_recording_speculative.py "${audio_path}" "${save_path}"
    else
        echo "Unknown instruction_type: ${instruction_type}"
        exit 1
    fi

done

echo ""
echo "Done. Processed ${current_id}/${total_files} items."
echo "Results in: ${output_path}"

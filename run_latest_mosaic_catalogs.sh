#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
PIPELINE_SCRIPT="scripts/run_cog_rf_pipeline.py"
TILE_MAX="${TILE_MAX:-48}"
TILE_WORKERS="${TILE_WORKERS:-0}"
MFMTK_WORKERS="${MFMTK_WORKERS:-1}"
MFMTK_TIMEOUT="${MFMTK_TIMEOUT:-10}"
FORCE="${FORCE:-0}"
DEBUG="${DEBUG:-0}"
DEBUG_INTERVAL="${DEBUG_INTERVAL:-60}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found at $PYTHON_BIN"
  exit 1
fi

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
  echo "ERROR: Missing pipeline script $PIPELINE_SCRIPT"
  exit 1
fi

mapfile -t FIELDS < <(
  find . -mindepth 1 -maxdepth 1 -type d -printf "%f\n" \
    | rg "^[0-9][AB]$" \
    | sort
)

if [[ "${#FIELDS[@]}" -eq 0 ]]; then
  echo "ERROR: No field directories found (expected names like 0A, 1B, ...)."
  exit 1
fi

declare -a FAILED=()
declare -a SKIPPED=()
declare -a DONE=()

run_one_field() {
  local field="$1"
  local latest_dir input_fits mosaic_name output_csv run_log err_log debug_log
  local -a cmd
  local -a candidates

  mapfile -t candidates < <(find "$field" -mindepth 1 -maxdepth 1 -type d -name "mosaic_${field}*" | sort -V)
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "[$field] WARN: no mosaic directory found"
    FAILED+=("$field:no_mosaic_dir")
    return
  fi

  latest_dir="${candidates[${#candidates[@]}-1]}"
  input_fits="$(find "$latest_dir" -mindepth 1 -maxdepth 1 -type f -name "*F444W*fits" | sort | head -n 1 || true)"
  if [[ -z "$input_fits" ]]; then
    echo "[$field] WARN: no F444W FITS in $latest_dir"
    FAILED+=("$field:no_f444w")
    return
  fi

  mosaic_name="$(basename "$latest_dir" | sed 's/^mosaic_//')"
  output_csv="$latest_dir/catalog_cog_rf_${mosaic_name}.csv"
  run_log="$latest_dir/run_catalog_${mosaic_name}.log"
  err_log="$latest_dir/err_catalog_${mosaic_name}.log"
  debug_log="debug_catalog_${mosaic_name}.log"

  if [[ "$FORCE" != "1" && -s "$output_csv" ]]; then
    echo "[$field] SKIP: $output_csv already exists"
    SKIPPED+=("$field:$output_csv")
    return
  fi

  rm -f "$latest_dir/err.log"
  if [[ "$DEBUG" == "1" ]]; then
    rm -f "$latest_dir/$debug_log"
  fi
  echo "[$field] RUN: $input_fits -> $output_csv"
  cmd=(
    "$PYTHON_BIN" "$PIPELINE_SCRIPT"
    --input "$input_fits"
    --output "$output_csv"
    --tile-max "$TILE_MAX"
    --tile-workers "$TILE_WORKERS"
    --mfmtk-workers "$MFMTK_WORKERS"
    --mfmtk-timeout "$MFMTK_TIMEOUT"
    --npixels 50
  )
  if [[ "$DEBUG" == "1" ]]; then
    cmd+=(
      --debug
      --debug-interval "$DEBUG_INTERVAL"
      --debug-log "$debug_log"
    )
  fi
  if "${cmd[@]}" >"$run_log" 2>&1; then
    if [[ -f "$latest_dir/err.log" ]]; then
      mv "$latest_dir/err.log" "$err_log"
    fi
    echo "[$field] OK: $output_csv"
    DONE+=("$field:$output_csv")
  else
    if [[ -f "$latest_dir/err.log" ]]; then
      mv "$latest_dir/err.log" "$err_log"
    fi
    echo "[$field] FAIL: see $run_log"
    FAILED+=("$field:$run_log")
  fi
}

echo "Running latest mosaics with TILE_MAX=$TILE_MAX TILE_WORKERS=$TILE_WORKERS MFMTK_WORKERS=$MFMTK_WORKERS MFMTK_TIMEOUT=$MFMTK_TIMEOUT FORCE=$FORCE DEBUG=$DEBUG DEBUG_INTERVAL=$DEBUG_INTERVAL"
for field in "${FIELDS[@]}"; do
  run_one_field "$field"
done

echo
echo "Summary:"
echo "  Done:    ${#DONE[@]}"
echo "  Skipped: ${#SKIPPED[@]}"
echo "  Failed:  ${#FAILED[@]}"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Failed entries:"
  printf '  %s\n' "${FAILED[@]}"
  exit 2
fi

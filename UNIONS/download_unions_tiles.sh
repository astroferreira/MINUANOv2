#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

LIST_FILE="${1:-$ROOT_DIR/tiles/unions_tiles_needed.txt}"
OUT_DIR="${2:-$ROOT_DIR/tiles}"

mkdir -p "$OUT_DIR"

while IFS= read -r tile_id; do
  [[ -z "$tile_id" ]] && continue
  out_file="$OUT_DIR/CFIS_LSB.${tile_id}.r.fits"
  if [[ -f "$out_file" ]]; then
    continue
  fi
  vcp "vos:/cfis/tiles_LSB_DR5/CFIS_LSB.${tile_id}.r.fits" "$out_file"
done < "$LIST_FILE"

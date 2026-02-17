#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <tile-id>"
    echo "Example: $0 042.345"
    exit 1
fi

tile_id="$1"
mkdir -p "$ROOT_DIR/cats" "$ROOT_DIR/tiles"

catalog_path="$ROOT_DIR/cats/CFIS.${tile_id}.r.cog.cat"
filtered_catalog_path="$ROOT_DIR/cats/CFIS.${tile_id}.r.cog.nogaia.cat"
clean_catalog_path="$ROOT_DIR/cats/CFIS.${tile_id}.r.cog.clean.cat"
full_region_path="${catalog_path}.reg"
filtered_region_path="${filtered_catalog_path}.reg"
clean_region_path="${clean_catalog_path}.reg"
stars_region_path="$ROOT_DIR/cats/CFIS.${tile_id}.r.cog.gaia_stars.reg"

# Use the project venv if available, otherwise fall back to python3.
python_bin="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    python_bin="$ROOT_DIR/.venv/bin/python"
fi

gaia_radius_r0_arcsec="${GAIA_RADIUS_R0_ARCSEC:-8.0}"
gaia_radius_mag0="${GAIA_RADIUS_MAG0:-15.0}"
gaia_radius_min_arcsec="${GAIA_RADIUS_MIN_ARCSEC:-1.0}"
gaia_radius_max_arcsec="${GAIA_RADIUS_MAX_ARCSEC:-40.0}"

quality_flags_max="${QUALITY_FLAGS_MAX:-3.0}"
point_min_criteria="${POINT_MIN_CRITERIA:-3}"
point_delta_mag_max="${POINT_DELTA_MAG_MAX:-0.3}"
point_fwhm_ratio_max="${POINT_FWHM_RATIO_MAX:-1.35}"
point_axis_ratio_min="${POINT_AXIS_RATIO_MIN:-0.75}"
point_kron_max="${POINT_KRON_MAX:-3.8}"
faint_margin_mag="${FAINT_MARGIN_MAG:-0.5}"
faint_magerr_cog_max="${FAINT_MAGERR_COG_MAX:-0.2}"
faint_magerr_auto_max="${FAINT_MAGERR_AUTO_MAX:-0.2}"

cutout_kron_multiplier="${CUTOUT_KRON_MULTIPLIER:-12.0}"
cutout_min_half_size="${CUTOUT_MIN_HALF_SIZE:-12}"
cutout_max_half_size="${CUTOUT_MAX_HALF_SIZE:-1024}"
cutout_default_kron_radius="${CUTOUT_DEFAULT_KRON_RADIUS:-3.5}"
cutout_output_dir="${CUTOUT_OUTPUT_DIR:-$ROOT_DIR/cutouts}"
cutout_min_snr_sb_cog="${CUTOUT_MIN_SNR_SB_COG:-}"
cutout_min_kron_radius="${CUTOUT_MIN_KRON_RADIUS:-12}"
if [[ "$cutout_output_dir" != /* ]]; then
    cutout_output_dir="$ROOT_DIR/$cutout_output_dir"
fi

run_mfmtk="${RUN_MFMTK:-1}"
mfmtk_output_path="${MFMTK_OUTPUT_PATH:-$ROOT_DIR/cats/CFIS.${tile_id}.r.cog.clean.morpho.csv}"
mfmtk_psf_path="${MFMTK_PSF_PATH:-}"
mfmtk_psf_fwhm_pix="${MFMTK_PSF_FWHM_PIX:-3.0}"
mfmtk_psf_size="${MFMTK_PSF_SIZE:-33}"
mfmtk_timeout="${MFMTK_TIMEOUT:-0}"
mfmtk_max_sources="${MFMTK_MAX_SOURCES:-}"
mfmtk_catalog_min_snr_sb_cog="${MFMTK_CATALOG_MIN_SNR_SB_COG:-}"
mfmtk_catalog_min_snr_sb_2arc="${MFMTK_CATALOG_MIN_SNR_SB_2ARC:-}"
mfmtk_catalog_min_kron_radius_pix="${MFMTK_CATALOG_MIN_KRON_RADIUS_PIX:-}"

if [[ "$mfmtk_output_path" != /* ]]; then
    mfmtk_output_path="$ROOT_DIR/$mfmtk_output_path"
fi
if [[ -n "$mfmtk_psf_path" && "$mfmtk_psf_path" != /* ]]; then
    mfmtk_psf_path="$ROOT_DIR/$mfmtk_psf_path"
fi

echo "$tile_id"

echo "vcp vos:/cfis/tiles_DR5/CFIS.${tile_id}.r.cog.cat $ROOT_DIR/cats"
vcp "vos:/cfis/tiles_DR5/CFIS.${tile_id}.r.cog.cat" "$ROOT_DIR/cats"

echo "Filtering catalog with Gaia mask:"
echo "  r0=${gaia_radius_r0_arcsec}\"  mag0=${gaia_radius_mag0}  clip=[${gaia_radius_min_arcsec}\",${gaia_radius_max_arcsec}\"]"
echo "Applying quality cuts:"
echo "  FLAGS<=${quality_flags_max}  point(min=${point_min_criteria}, dmag<${point_delta_mag_max}, fwhm_ratio<${point_fwhm_ratio_max}, b/a>${point_axis_ratio_min}, kron<=${point_kron_max})"
echo "  faint(mag>(predmaglim-${faint_margin_mag}), magerr_cog<=${faint_magerr_cog_max}, magerr_auto<=${faint_magerr_auto_max})"
"$python_bin" "$SCRIPT_DIR/filter_cfis_catalog_gaia.py" \
    --catalog "$catalog_path" \
    --output "$filtered_catalog_path" \
    --output-clean "$clean_catalog_path" \
    --stars-region "$stars_region_path" \
    --gaia-radius-r0-arcsec "$gaia_radius_r0_arcsec" \
    --gaia-radius-mag0 "$gaia_radius_mag0" \
    --gaia-radius-min-arcsec "$gaia_radius_min_arcsec" \
    --gaia-radius-max-arcsec "$gaia_radius_max_arcsec" \
    --flags-max "$quality_flags_max" \
    --point-min-criteria "$point_min_criteria" \
    --point-delta-mag-max "$point_delta_mag_max" \
    --point-fwhm-ratio-max "$point_fwhm_ratio_max" \
    --point-axis-ratio-min "$point_axis_ratio_min" \
    --point-kron-max "$point_kron_max" \
    --faint-margin-mag "$faint_margin_mag" \
    --faint-magerr-cog-max "$faint_magerr_cog_max" \
    --faint-magerr-auto-max "$faint_magerr_auto_max"

echo "Full catalog: $catalog_path"
echo "No-Gaia catalog: $filtered_catalog_path"
echo "Clean catalog (no Gaia + no point-like + no faint): $clean_catalog_path"
echo "Matched Gaia stars region: $stars_region_path"

echo "Generating DS9 region from full catalog: $full_region_path"
"$python_bin" "$SCRIPT_DIR/cog_catalog_to_region.py" \
    --catalog "$catalog_path" \
    --output "$full_region_path"

echo "Generating DS9 region from filtered catalog: $filtered_region_path"
"$python_bin" "$SCRIPT_DIR/cog_catalog_to_region.py" \
    --catalog "$filtered_catalog_path" \
    --output "$filtered_region_path"

echo "Generating DS9 region from clean catalog: $clean_region_path"
"$python_bin" "$SCRIPT_DIR/cog_catalog_to_region.py" \
    --catalog "$clean_catalog_path" \
    --output "$clean_region_path"

echo "vcp vos:/cfis/tiles_DR5/CFIS.${tile_id}.r.fits $ROOT_DIR/tiles"
vcp "vos:/cfis/tiles_DR5/CFIS.${tile_id}.r.fits" "$ROOT_DIR/tiles"

echo "Generating cutouts from clean catalog (size = ${cutout_kron_multiplier} * KRON_RADIUS)"
cutout_cmd=(
    "$python_bin" "$SCRIPT_DIR/generate_cfis_cutouts.py"
    --catalog "$clean_catalog_path"
    --image "$ROOT_DIR/tiles/CFIS.${tile_id}.r.fits"
    --tile-id "$tile_id"
    --output-dir "$cutout_output_dir"
    --kron-multiplier "$cutout_kron_multiplier"
    --min-half-size "$cutout_min_half_size"
    --max-half-size "$cutout_max_half_size"
    --default-kron-radius "$cutout_default_kron_radius"
)
if [[ -n "$cutout_min_snr_sb_cog" ]]; then
    echo "Applying cutout SNR threshold: SNR_SB_COG >= $cutout_min_snr_sb_cog"
    cutout_cmd+=(--min-snr-sb-cog "$cutout_min_snr_sb_cog")
fi
echo "Applying cutout KRON threshold: KRON_RADIUS_PIX > $cutout_min_kron_radius px"
cutout_cmd+=(--min-kron-radius "$cutout_min_kron_radius")
cutout_cmd+=(--overwrite)
"${cutout_cmd[@]}"

if [[ "$run_mfmtk" == "1" ]]; then
    echo "Running Morfometryka on cutouts and generating merged morphology catalog"
    mfmtk_cmd=(
        "$python_bin" "$SCRIPT_DIR/run_mfmtk_on_cutouts.py"
        --catalog "$clean_catalog_path"
        --cutouts-dir "$cutout_output_dir"
        --tile-id "$tile_id"
        --output "$mfmtk_output_path"
        --psf-fwhm-pix "$mfmtk_psf_fwhm_pix"
        --psf-size "$mfmtk_psf_size"
        --timeout "$mfmtk_timeout"
    )
    if [[ -n "$mfmtk_psf_path" ]]; then
        mfmtk_cmd+=(--psf "$mfmtk_psf_path")
    fi
    if [[ -n "$mfmtk_max_sources" ]]; then
        mfmtk_cmd+=(--max-sources "$mfmtk_max_sources")
    fi
    if [[ -n "$mfmtk_catalog_min_snr_sb_cog" ]]; then
        mfmtk_cmd+=(--catalog-min-snr-sb-cog "$mfmtk_catalog_min_snr_sb_cog")
    fi
    if [[ -n "$mfmtk_catalog_min_snr_sb_2arc" ]]; then
        mfmtk_cmd+=(--catalog-min-snr-sb-2arc "$mfmtk_catalog_min_snr_sb_2arc")
    fi
    if [[ -n "$mfmtk_catalog_min_kron_radius_pix" ]]; then
        mfmtk_cmd+=(--catalog-min-kron-radius-pix "$mfmtk_catalog_min_kron_radius_pix")
    fi
    "${mfmtk_cmd[@]}"
    echo "Morphology catalog: $mfmtk_output_path"
fi

echo "vcp vos:/cfis/tiles_DR5/CFIS.${tile_id}.u.fits $ROOT_DIR/tiles"
vcp "vos:/cfis/tiles_DR5/CFIS.${tile_id}.u.fits" "$ROOT_DIR/tiles"

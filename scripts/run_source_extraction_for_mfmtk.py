#!/usr/bin/env python3
"""
Source extraction + MFMTK feature pipeline.

This script detects sources in a JWST mosaic, builds per-source cutouts,
runs MFMTK on each cutout, and writes a source-level CSV catalog.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import re
import shutil
import signal
import subprocess
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from tqdm import tqdm

from gaia_mask_utils import build_gaia_exclusion_mask as _build_gaia_exclusion_mask
from mfmtk import Photometry, Stamp, config as mfmtk_config

try:
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources
except Exception:  # pragma: no cover - optional when using SExtractor mode
    Background2D = None
    MedianBackground = None
    SourceCatalog = None
    deblend_sources = None
    detect_sources = None

warnings.filterwarnings("ignore")
mfmtk_config.verbose = 0

ERROR_LOG_PATH = Path("err.log")
DEBUG_LOG_PATH = Path("debug.log")
DEBUG_ENABLED = False
DEBUG_HEARTBEAT_SECONDS = 60.0
FILTER_TOKEN_RE = re.compile(r"^F\d{3,4}[A-Z0-9]*$", re.IGNORECASE)
FIELD_VERSION_SUFFIX_RE = re.compile(r"v\d+$", re.IGNORECASE)


class MfmtkTimeoutError(Exception):
    """Raised when a per-source MFMTK timeout is reached."""


def _mfmtk_alarm_handler(signum, frame):  # noqa: ARG001
    raise MfmtkTimeoutError


@contextmanager
def mfmtk_timeout(seconds: float):
    if seconds <= 0:
        yield
        return
    old_handler = signal.signal(signal.SIGALRM, _mfmtk_alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def set_error_log_path(path: Path) -> None:
    global ERROR_LOG_PATH
    ERROR_LOG_PATH = path


def set_debug_log_path(path: Path, enabled: bool, heartbeat_seconds: float) -> None:
    global DEBUG_LOG_PATH, DEBUG_ENABLED, DEBUG_HEARTBEAT_SECONDS
    DEBUG_LOG_PATH = path
    DEBUG_ENABLED = bool(enabled)
    DEBUG_HEARTBEAT_SECONDS = max(1.0, float(heartbeat_seconds))


def get_error_logger() -> logging.Logger:
    logger = logging.getLogger("source_extraction_errors")
    logger.setLevel(logging.ERROR)
    log_path = str(ERROR_LOG_PATH)
    if getattr(logger, "_log_path", None) != log_path:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        handler = logging.FileHandler(log_path, mode="a")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(process)d %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger._log_path = log_path
    return logger


def get_debug_logger() -> logging.Logger:
    logger = logging.getLogger("source_extraction_debug")
    logger.setLevel(logging.INFO)
    log_path = str(DEBUG_LOG_PATH)
    if getattr(logger, "_log_path", None) != log_path:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        handler = logging.FileHandler(log_path, mode="a")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(process)d %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger._log_path = log_path
    return logger


def debug_log(msg: str, *args) -> None:
    if DEBUG_ENABLED:
        get_debug_logger().info(msg, *args)


def debug_exception(msg: str, *args) -> None:
    if DEBUG_ENABLED:
        get_debug_logger().exception(msg, *args)


def _init_worker_logging(
    err_log_path: str,
    debug_log_path: str,
    debug_enabled: bool,
    debug_interval: float,
) -> None:
    set_error_log_path(Path(err_log_path))
    get_error_logger()
    set_debug_log_path(Path(debug_log_path), debug_enabled, debug_interval)
    debug_log("worker_initialized pid=%d", os.getpid())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source extraction and MFMTK feature extraction."
    )
    parser.add_argument("--input", required=True, help="JWST mosaic FITS path")
    parser.add_argument(
        "--output",
        default="catalog_source_extraction_mfmtk.csv",
        help="Output CSV path",
    )
    parser.add_argument("--psf", default="psf/jwst_psf_f444w.fits", help="MFMTK PSF FITS")
    parser.add_argument(
        "--detection-engine",
        choices=("sextractor", "photutils"),
        default="sextractor",
        help="Detection/catalog engine (default: sextractor).",
    )
    parser.add_argument(
        "--sextractor-bin",
        default="",
        help="Path to SExtractor executable (source-extractor/sex/sextractor).",
    )
    parser.add_argument(
        "--detect-stack-inputs",
        nargs="+",
        default=None,
        help=(
            "Optional FITS mosaics to stack for detection (dual-image mode). "
            "If omitted, siblings in the same directory are auto-discovered by field."
        ),
    )
    parser.add_argument(
        "--detect-stack-filters",
        dest="detect_stack_filters",
        action="store_true",
        default=True,
        help="Auto-stack same-field filter mosaics for detection (default: enabled).",
    )
    parser.add_argument(
        "--no-detect-stack-filters",
        dest="detect_stack_filters",
        action="store_false",
        help="Disable multi-filter stacked detection; detect on --input only.",
    )

    parser.add_argument("--nsigma", type=float, default=2.5, help="Detection sigma threshold")
    parser.add_argument("--npixels", type=int, default=50, help="Minimum connected pixels")
    parser.add_argument("--deblend", action="store_true", help="Enable deblending")
    parser.add_argument("--deblend-nlevels", type=int, default=32, help="Deblend levels")
    parser.add_argument("--deblend-contrast", type=float, default=0.001, help="Deblend contrast")
    parser.add_argument("--box-size", type=int, default=64, help="Background box size")
    parser.add_argument("--filter-size", type=int, default=3, help="Background filter size")
    parser.add_argument(
        "--kernel-fwhm",
        type=float,
        default=2.0,
        help="Gaussian smoothing kernel FWHM (pixels)",
    )

    parser.add_argument(
        "--cutout-mode",
        choices=("kron", "fixed"),
        default="kron",
        help="Cutout sizing mode",
    )
    parser.add_argument(
        "--size-mult",
        type=float,
        default=2.0,
        help="When --cutout-mode kron: half-size = size_mult * kron_radius",
    )
    parser.add_argument("--fixed-half", type=float, default=96.0, help="Fixed cutout half-size")
    parser.add_argument("--min-half", type=float, default=64.0, help="Minimum cutout half-size")
    parser.add_argument("--max-half", type=float, default=512.0, help="Maximum cutout half-size")

    parser.add_argument(
        "--mfmtk-workers",
        type=int,
        default=14,
        help="Parallel workers for MFMTK step (0 means serial)",
    )
    parser.add_argument(
        "--mfmtk-timeout",
        type=float,
        default=0.0,
        help="Per-source timeout in seconds for MFMTK (0 disables timeout)",
    )

    parser.add_argument(
        "--mosaic-cutout",
        type=int,
        default=0,
        help="If >0, process only a square cutout of this size (pixels)",
    )
    parser.add_argument(
        "--mosaic-center",
        default="",
        help="Cutout center as 'x,y' in pixels (default: mosaic center)",
    )
    parser.add_argument(
        "--tile-max",
        type=int,
        default=0,
        help="Maximum number of tiles (0 disables tiling)",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=200,
        help="Tile overlap in pixels",
    )
    parser.add_argument(
        "--tile-workers",
        type=int,
        default=0,
        help="Parallel workers for tiles (0 means auto)",
    )

    parser.add_argument("--sci-ext", default="SCI", help="SCI extension name")
    parser.add_argument("--err-ext", default="ERR", help="ERR extension name")
    parser.add_argument("--wht-ext", default="WHT", help="WHT extension name (fallback)")
    parser.add_argument(
        "--skip-err",
        action="store_true",
        help="Skip ERR/WHT loading (faster, no error map)",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not apply FITS BSCALE/BZERO when opening data",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug log file")
    parser.add_argument(
        "--debug-log",
        default="debug.log",
        help="Debug log filename in output directory",
    )
    parser.add_argument(
        "--debug-interval",
        type=float,
        default=60.0,
        help="Seconds between heartbeat logs while waiting on futures",
    )

    parser.add_argument(
        "--exclude-gaia-stars",
        dest="exclude_gaia_stars",
        action="store_true",
        default=True,
        help="Exclude Gaia-star areas from source detection (default: enabled).",
    )
    parser.add_argument(
        "--no-exclude-gaia-stars",
        dest="exclude_gaia_stars",
        action="store_false",
        help="Disable Gaia-star exclusion masking.",
    )
    parser.add_argument(
        "--gaia-stars-csv",
        default="",
        help="Optional Gaia CSV (e.g., from identify-gaia-stars.py). If empty, query Gaia and cache it.",
    )
    parser.add_argument(
        "--gaia-cache-csv",
        default="",
        help="Output CSV for Gaia auto-query (default: <output_dir>/gaia_stars_in_field.csv).",
    )
    parser.add_argument(
        "--gaia-table",
        default="gaiadr3.gaia_source",
        help="Gaia table for ADQL query when auto-query is enabled.",
    )
    parser.add_argument(
        "--gaia-row-limit",
        type=int,
        default=200000,
        help="Gaia query row limit.",
    )
    parser.add_argument(
        "--gaia-mag-limit",
        type=float,
        default=None,
        help="Optional magnitude cut (phot_g_mean_mag <= value).",
    )
    parser.add_argument("--gaia-ra-col", default="ra", help="RA column name in Gaia CSV.")
    parser.add_argument("--gaia-dec-col", default="dec", help="Dec column name in Gaia CSV.")
    parser.add_argument(
        "--gaia-mag-col",
        default="phot_g_mean_mag",
        help="Magnitude column name in Gaia CSV.",
    )
    parser.add_argument(
        "--gaia-r0",
        type=float,
        default=8.0,
        help="Gaia mask radius (arcsec) at reference magnitude mag0.",
    )
    parser.add_argument(
        "--gaia-mag0",
        type=float,
        default=15.0,
        help="Reference magnitude for Gaia mask radius scaling.",
    )
    parser.add_argument(
        "--gaia-minr",
        type=float,
        default=1.0,
        help="Minimum Gaia mask radius in arcsec.",
    )
    parser.add_argument(
        "--gaia-maxr",
        type=float,
        default=40.0,
        help="Maximum Gaia mask radius in arcsec.",
    )
    parser.add_argument(
        "--gaia-mask-shape",
        choices=("star12", "star8", "star4", "jwst", "circle"),
        default="star12",
        help=(
            "Gaia exclusion mask shape (default: star12 core+arms every 30 deg; "
            "'jwst' keeps the older hex+6-spike template)."
        ),
    )
    parser.add_argument(
        "--gaia-mask-rotation-deg",
        type=float,
        default=0.0,
        help="Rotation angle (deg) for JWST Gaia mask template.",
    )
    parser.add_argument(
        "--gaia-mask-xshift-pix",
        type=float,
        default=0.0,
        help="Global Gaia mask X shift in pixels (positive shifts mask right).",
    )
    parser.add_argument(
        "--gaia-mask-yshift-pix",
        type=float,
        default=0.0,
        help="Global Gaia mask Y shift in pixels (positive shifts mask up).",
    )
    return parser.parse_args()


def parse_xy(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Center must be 'x,y' in pixels.")
    return float(parts[0]), float(parts[1])


def to_float_array(val) -> np.ndarray:
    if hasattr(val, "to_value"):
        return np.asarray(val.to_value(), dtype=np.float64)
    return np.asarray(val, dtype=np.float64)


def _normalize_metric_name(name: str) -> str:
    if name.startswith(("P_", "S_")):
        return name[2:]
    return name


def _supports_kwarg(fn, kwarg_name: str) -> bool:
    try:
        return kwarg_name in inspect.signature(fn).parameters
    except Exception:
        return False


def _extract_filter_and_field_token(path: Path) -> tuple[str | None, str | None]:
    stem = path.name
    if stem.endswith(".fits.gz"):
        stem = stem[:-8]
    elif stem.endswith(".fits"):
        stem = stem[:-5]

    prefix = stem.split("_", 1)[0]
    parts = [p for p in prefix.split("-") if p]
    for i, part in enumerate(parts):
        if FILTER_TOKEN_RE.fullmatch(part):
            field_token = parts[i + 1] if (i + 1) < len(parts) else None
            return part.upper(), field_token
    return None, None


def _normalize_field_token(field_token: str | None) -> str:
    if not field_token:
        return ""
    return FIELD_VERSION_SUFFIX_RE.sub("", str(field_token).strip()).upper()


def _looks_like_fits(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".fits") or name.endswith(".fits.gz")


def _resolve_detection_stack_paths(
    input_path: Path,
    args: argparse.Namespace,
) -> list[Path]:
    input_path = input_path.expanduser().resolve()

    explicit = getattr(args, "detect_stack_inputs", None)
    if explicit:
        ordered: list[Path] = [input_path]
        seen = {input_path}
        for item in explicit:
            p = Path(str(item)).expanduser().resolve()
            if p == input_path or p in seen:
                continue
            if not p.exists():
                raise FileNotFoundError(f"Detection stack input not found: {p}")
            ordered.append(p)
            seen.add(p)
        debug_log("detect_stack_explicit n=%d", int(len(ordered)))
        return ordered

    if not bool(getattr(args, "detect_stack_filters", True)):
        return [input_path]

    in_filter, in_field = _extract_filter_and_field_token(input_path)
    norm_field = _normalize_field_token(in_field)
    if not in_filter or not norm_field:
        debug_log("detect_stack_auto_skip reason=filename_parse_failed input=%s", input_path.name)
        return [input_path]

    parent = input_path.parent
    candidates: list[Path] = []
    for p in sorted(parent.iterdir()):
        if not p.is_file() or not _looks_like_fits(p):
            continue
        filt, field = _extract_filter_and_field_token(p)
        if not filt:
            continue
        if _normalize_field_token(field) != norm_field:
            continue
        candidates.append(p.resolve())

    # Keep deterministic order and ensure the measurement image is first.
    unique: list[Path] = []
    seen: set[Path] = set()
    for p in [input_path, *candidates]:
        if p in seen:
            continue
        unique.append(p)
        seen.add(p)

    debug_log(
        "detect_stack_auto input=%s field=%s n_candidates=%d",
        input_path.name,
        norm_field,
        int(len(unique)),
    )
    return unique or [input_path]


def _footprint_polygon_icrs(wcs: WCS, nx: int, ny: int) -> np.ndarray:
    try:
        fp = wcs.calc_footprint(axes=(nx, ny))
        return np.asarray(fp, dtype=float)
    except Exception:
        corners = np.array(
            [
                [0, 0],
                [nx - 1, 0],
                [nx - 1, ny - 1],
                [0, ny - 1],
            ],
            dtype=float,
        )
        ra, dec = wcs.pixel_to_world_values(corners[:, 0], corners[:, 1])
        return np.column_stack((np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)))


def _build_gaia_adql_query(
    fp_deg: np.ndarray,
    table: str,
    mag_limit: float | None,
) -> str:
    coords = ", ".join([f"{ra:.12f}, {dec:.12f}" for ra, dec in fp_deg])
    poly = f"POLYGON('ICRS', {coords})"
    where = f"CONTAINS(POINT('ICRS', ra, dec), {poly}) = 1"
    if mag_limit is not None:
        where += f" AND phot_g_mean_mag <= {float(mag_limit)}"
    return (
        "SELECT source_id, ra, dec, phot_g_mean_mag "
        f"FROM {table} "
        f"WHERE {where}"
    )


def _query_gaia_catalog(
    wcs: WCS,
    nx: int,
    ny: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    try:
        from astroquery.gaia import Gaia
    except ImportError as exc:
        raise RuntimeError(
            "astroquery is required for Gaia auto-query. Install astroquery or provide --gaia-stars-csv."
        ) from exc

    fp = _footprint_polygon_icrs(wcs, nx, ny)
    query = _build_gaia_adql_query(fp, str(args.gaia_table), args.gaia_mag_limit)
    Gaia.ROW_LIMIT = int(args.gaia_row_limit)
    debug_log("gaia_query_start row_limit=%d", int(args.gaia_row_limit))
    job = Gaia.launch_job_async(query)
    tab = job.get_results()
    if len(tab) == 0:
        return pd.DataFrame(
            columns=["source_id", "ra", "dec", "phot_g_mean_mag"],
            dtype=float,
        )
    df = tab.to_pandas()
    keep_cols = [c for c in ("source_id", "ra", "dec", "phot_g_mean_mag") if c in df.columns]
    return df[keep_cols].copy()


@lru_cache(maxsize=8)
def _read_gaia_catalog_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _sanitize_gaia_catalog(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    ra_col = str(args.gaia_ra_col)
    dec_col = str(args.gaia_dec_col)
    mag_col = str(args.gaia_mag_col)
    required = [ra_col, dec_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Gaia CSV is missing required column(s): {missing}")

    out = pd.DataFrame(
        {
            "ra": pd.to_numeric(df[ra_col], errors="coerce"),
            "dec": pd.to_numeric(df[dec_col], errors="coerce"),
        }
    )
    has_mag_col = mag_col in df.columns
    if has_mag_col:
        out["phot_g_mean_mag"] = pd.to_numeric(df[mag_col], errors="coerce")
    else:
        out["phot_g_mean_mag"] = np.nan

    keep = np.isfinite(out["ra"].to_numpy()) & np.isfinite(out["dec"].to_numpy())
    out = out.loc[keep].reset_index(drop=True)
    if args.gaia_mag_limit is not None and has_mag_col:
        mag = out["phot_g_mean_mag"].to_numpy()
        keep_mag = np.isfinite(mag) & (mag <= float(args.gaia_mag_limit))
        out = out.loc[keep_mag].reset_index(drop=True)
    return out


def _load_gaia_catalog_from_args(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(str(args.gaia_stars_csv)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Gaia CSV not found: {path}")
    raw = _read_gaia_catalog_csv(str(path))
    return _sanitize_gaia_catalog(raw, args)


def _prepare_gaia_catalog_if_needed(
    args: argparse.Namespace,
    wcs: WCS,
    nx: int,
    ny: int,
    output_dir: Path,
) -> None:
    if not args.exclude_gaia_stars:
        return

    if str(args.gaia_stars_csv).strip():
        args.gaia_stars_csv = str(Path(str(args.gaia_stars_csv)).expanduser().resolve())
        gaia_df = _load_gaia_catalog_from_args(args)
        debug_log("gaia_catalog_loaded path=%s rows=%d", args.gaia_stars_csv, int(len(gaia_df)))
        print(f"Using Gaia catalog: {args.gaia_stars_csv} ({len(gaia_df)} rows)")
        return

    gaia_df = _query_gaia_catalog(wcs, nx, ny, args)
    cache_path = (
        Path(str(args.gaia_cache_csv)).expanduser()
        if str(args.gaia_cache_csv).strip()
        else (output_dir / "gaia_stars_in_field.csv")
    )
    cache_path = cache_path.resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gaia_df.to_csv(cache_path, index=False)
    args.gaia_stars_csv = str(cache_path)
    debug_log("gaia_catalog_queried path=%s rows=%d", cache_path, int(len(gaia_df)))
    print(f"Wrote Gaia catalog: {cache_path} ({len(gaia_df)} rows)")


def extract_scalar_metrics(obj, skip: set[str] | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    skip = skip or set()
    for name, val in vars(obj).items():
        if name.startswith("_") or callable(val):
            continue
        if isinstance(val, (dict, list, tuple, set, np.ndarray)):
            continue
        if hasattr(val, "value") and np.isscalar(val.value):
            val = val.value
        if isinstance(val, (np.number, int, float, bool, np.bool_)):
            key = _normalize_metric_name(name)
            if key not in skip:
                out[key] = float(val)
    return out


def run_mfmtk_on_cutout(
    cut_path: str,
    psf_path: str,
    timeout_seconds: float = 0.0,
) -> dict[str, float]:
    logger = get_error_logger()
    metrics: dict[str, float] = {
        "mfmtk_stamp_ok": 0.0,
        "mfmtk_phot_ok": 0.0,
        "mfmtk_timeout": 0.0,
    }
    try:
        with mfmtk_timeout(timeout_seconds):
            stamp = Stamp(cut_path, psf_path)
            metrics["mfmtk_stamp_ok"] = 1.0
            metrics.update(extract_scalar_metrics(stamp, skip=set(metrics)))
            try:
                phot = Photometry(stamp)
                metrics["mfmtk_phot_ok"] = 1.0
                metrics.update(extract_scalar_metrics(phot, skip=set(metrics)))
                metrics["P_Rp"] = float(getattr(phot, "Rp", np.nan))
                metrics["P_q"] = float(getattr(phot, "P_q", np.nan))
                if hasattr(phot, "LT") and phot.LT is not None:
                    lt_val = np.nanmax(phot.LT) if np.size(phot.LT) > 1 else float(phot.LT)
                    metrics["P_LT"] = float(lt_val)
                elif hasattr(phot, "LR") and phot.LR is not None and np.size(phot.LR) > 0:
                    metrics["P_LT"] = float(np.nanmax(phot.LR))
                else:
                    metrics["P_LT"] = np.nan
            except MfmtkTimeoutError:
                raise
            except Exception as exc:
                metrics["mfmtk_phot_ok"] = 0.0
                logger.exception("Photometry failed for %s: %s", cut_path, exc)
    except MfmtkTimeoutError:
        metrics["mfmtk_stamp_ok"] = 0.0
        metrics["mfmtk_phot_ok"] = 0.0
        metrics["mfmtk_timeout"] = 1.0
        logger.error("MFMTK timed out for %s after %.2fs", cut_path, float(timeout_seconds))
    except Exception as exc:
        metrics["mfmtk_stamp_ok"] = 0.0
        metrics["mfmtk_phot_ok"] = 0.0
        logger.exception("Stamp failed for %s: %s", cut_path, exc)
    return metrics


def pixscale_from_wcs(wcs: WCS) -> float:
    scales = proj_plane_pixel_scales(wcs)
    if scales.size == 0:
        return np.nan
    return float(np.mean(scales) * 3600.0)


def _split_range(total: int, parts: int) -> list[tuple[int, int]]:
    if parts <= 0:
        raise ValueError("parts must be > 0")
    base = total // parts
    rem = total % parts
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < rem else 0)
        end = start + size
        out.append((start, end))
        start = end
    return out


def _choose_tile_grid(width: int, height: int, max_tiles: int) -> tuple[int, int]:
    max_tiles = max(1, int(max_tiles))
    best = None
    for nrows in range(1, max_tiles + 1):
        for ncols in range(1, max_tiles + 1):
            ntiles = nrows * ncols
            if ntiles > max_tiles:
                continue
            tile_w = width / ncols
            tile_h = height / nrows
            if tile_w <= 0 or tile_h <= 0:
                continue
            score = abs(np.log(tile_w / tile_h))
            if best is None or ntiles > best[0] or (ntiles == best[0] and score < best[1]):
                best = (ntiles, score, nrows, ncols)
    if best is None:
        return 1, 1
    return int(best[2]), int(best[3])


def _build_tiles(
    width: int,
    height: int,
    nrows: int,
    ncols: int,
    overlap: int,
) -> list[dict[str, int]]:
    x_splits = _split_range(width, ncols)
    y_splits = _split_range(height, nrows)
    tiles: list[dict[str, int]] = []
    for r, (y0_base, y1_base) in enumerate(y_splits):
        for c, (x0_base, x1_base) in enumerate(x_splits):
            x0 = max(0, x0_base - overlap)
            x1 = min(width, x1_base + overlap)
            y0 = max(0, y0_base - overlap)
            y1 = min(height, y1_base + overlap)
            tiles.append(
                {
                    "row": r,
                    "col": c,
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "core_x0": x0_base,
                    "core_x1": x1_base,
                    "core_y0": y0_base,
                    "core_y1": y1_base,
                }
            )
    return tiles


def _load_error_map(
    hdul: fits.HDUList,
    args: argparse.Namespace,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray | None:
    if args.skip_err:
        return None

    err_map = None
    wht_map = None
    if args.err_ext in hdul and hdul[args.err_ext].data is not None:
        err_map = hdul[args.err_ext].section[y_slice, x_slice]
    elif args.wht_ext in hdul and hdul[args.wht_ext].data is not None:
        wht_map = hdul[args.wht_ext].section[y_slice, x_slice]

    if err_map is None and wht_map is not None:
        err_map = np.full_like(wht_map, np.nan, dtype=np.float32)
        good = np.isfinite(wht_map) & (wht_map > 0)
        err_map[good] = 1.0 / np.sqrt(wht_map[good])
    return err_map


def _load_sci_data_slice(
    fits_path: Path,
    args: argparse.Namespace,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    with fits.open(
        fits_path,
        memmap=True,
        mode="readonly",
        lazy_load_hdus=True,
        do_not_scale_image_data=args.no_scale,
    ) as hdul:
        if args.sci_ext not in hdul:
            raise KeyError(f"SCI extension '{args.sci_ext}' not found in {fits_path}")
        sci_hdu = hdul[args.sci_ext]
        if sci_hdu.data is None:
            raise RuntimeError(f"SCI extension '{args.sci_ext}' has no data in {fits_path}")
        arr = sci_hdu.section[y_slice, x_slice]
    return np.array(arr, dtype=np.float32, copy=True)


def _build_detection_stack_for_region(
    stack_paths: list[Path],
    args: argparse.Namespace,
    y_slice: slice,
    x_slice: slice,
    context: str,
) -> tuple[np.ndarray | None, list[Path]]:
    if len(stack_paths) <= 1:
        return None, []

    sum_img: np.ndarray | None = None
    count_img: np.ndarray | None = None
    used_paths: list[Path] = []

    for p in stack_paths:
        try:
            img = _load_sci_data_slice(p, args, y_slice, x_slice)
        except Exception as exc:
            debug_exception("detect_stack_read_failed context=%s path=%s err=%s", context, p, exc)
            continue

        if sum_img is None:
            sum_img = np.zeros_like(img, dtype=np.float32)
            count_img = np.zeros_like(img, dtype=np.uint16)
        elif img.shape != sum_img.shape:
            debug_log(
                "detect_stack_skip_shape context=%s path=%s shape=%s expected=%s",
                context,
                p,
                tuple(img.shape),
                tuple(sum_img.shape),
            )
            continue

        finite = np.isfinite(img)
        if not np.any(finite):
            debug_log("detect_stack_skip_allnan context=%s path=%s", context, p)
            continue

        sum_img[finite] += img[finite]
        count_img[finite] += 1
        used_paths.append(p)

    if sum_img is None or count_img is None or len(used_paths) <= 1:
        return None, used_paths

    det_img = np.zeros_like(sum_img, dtype=np.float32)
    good = count_img > 0
    det_img[good] = sum_img[good] / count_img[good]

    debug_log(
        "detect_stack_built context=%s n_inputs=%d shape=%dx%d",
        context,
        int(len(used_paths)),
        int(det_img.shape[0]),
        int(det_img.shape[1]),
    )
    return det_img, used_paths


def _resolve_sextractor_binary(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "sextractor_bin", "")).strip()
    if explicit:
        return explicit
    for name in ("source-extractor", "sex", "sextractor"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "SExtractor executable not found. Install 'source-extractor' (or 'sex') "
        "or pass --sextractor-bin /path/to/binary."
    )


def _write_fits_image(path: Path, data: np.ndarray, wcs: WCS | None = None) -> None:
    header = wcs.to_header() if wcs is not None else None
    fits.PrimaryHDU(np.asarray(data), header=header).writeto(path, overwrite=True)


def _gaia_mask_export_path(tmp_dir: Path, context: str) -> Path:
    # For mosaic runs, save beside the final output directory (parent of tmp_mfmtk).
    # For tile runs, save within the tile temp directory to avoid clutter/collisions.
    if context == "mosaic":
        return tmp_dir.parent / "gaia_mask_mosaic.fits"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(context))
    return tmp_dir / f"gaia_mask_{safe}.fits"


def _export_gaia_mask_fits(
    mask: np.ndarray | None,
    wcs: WCS,
    tmp_dir: Path,
    context: str,
) -> Path | None:
    if mask is None:
        return None
    path = _gaia_mask_export_path(tmp_dir, context)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_fits_image(path, mask.astype(np.uint8), wcs=wcs)
    debug_log("gaia_mask_fits_written context=%s path=%s", context, path)
    return path


def _sextractor_gaussian_kernel(size: int, sigma_pix: float) -> np.ndarray:
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    sigma = max(float(sigma_pix), 1.0e-3)
    c = size // 2
    yy, xx = np.mgrid[-c : c + 1, -c : c + 1]
    kern = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    if np.sum(kern) <= 0:
        kern[c, c] = 1.0
    return kern / np.sum(kern)


def _write_sextractor_conv_file(path: Path, args: argparse.Namespace) -> None:
    sigma_pix = max(float(args.kernel_fwhm) / 2.355, 0.5)
    size = int(np.ceil(6.0 * sigma_pix))
    if size % 2 == 0:
        size += 1
    size = max(3, size)
    kern = _sextractor_gaussian_kernel(size, sigma_pix)
    lines = ["CONV NORM", f"# {size}x{size} Gaussian kernel (generated)"]
    for row in kern:
        lines.append(" ".join([f"{float(v):.8f}" for v in row]))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_sextractor_params_file(path: Path) -> None:
    # Minimal columns to mirror the photutils-based downstream expectations.
    params = [
        "NUMBER",
        "X_IMAGE",
        "Y_IMAGE",
        "A_IMAGE",
        "B_IMAGE",
        "KRON_RADIUS",
        "FLUX_ISO",
        "ISOAREA_IMAGE",
        "ALPHA_J2000",
        "DELTA_J2000",
        "FLAGS",
        "IMAFLAGS_ISO",
    ]
    path.write_text("\n".join(params) + "\n", encoding="ascii")


def _read_sextractor_fits_catalog(path: Path) -> fits.FITS_rec:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None:
                return hdu.data.copy()
    raise RuntimeError(f"SExtractor catalog has no binary table HDU: {path}")


def _sextractor_dual_image_catalog(
    data: np.ndarray,
    detect_data: np.ndarray,
    wcs: WCS,
    gaia_mask: np.ndarray | None,
    args: argparse.Namespace,
    tmp_dir: Path,
    context: str,
) -> dict[str, np.ndarray]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sex_bin = _resolve_sextractor_binary(args)

    # Export binary mask FITS for QA/reuse and also use it as FLAG_IMAGE for SExtractor.
    gaia_mask_path = _export_gaia_mask_fits(gaia_mask, wcs, tmp_dir, context)

    det_arr = np.array(detect_data, dtype=np.float32, copy=True)
    meas_arr = np.array(data, dtype=np.float32, copy=True)
    if gaia_mask is not None and gaia_mask.shape == det_arr.shape:
        for arr in (det_arr, meas_arr):
            finite = np.isfinite(arr)
            fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
            arr[gaia_mask] = fill

    det_path = tmp_dir / "sex_detect.fits"
    meas_path = tmp_dir / "sex_measure.fits"
    cat_path = tmp_dir / "sex_catalog.fits"
    conv_path = tmp_dir / "sex.conv"
    params_path = tmp_dir / "sex.params"

    _write_fits_image(det_path, det_arr, wcs=wcs)
    _write_fits_image(meas_path, meas_arr, wcs=wcs)
    _write_sextractor_conv_file(conv_path, args)
    _write_sextractor_params_file(params_path)

    deblend_nthresh = int(args.deblend_nlevels) if bool(args.deblend) else 1
    deblend_mincont = float(args.deblend_contrast) if bool(args.deblend) else 1.0

    cmd = [
        sex_bin,
        f"{det_path},{meas_path}",
        "-CATALOG_NAME",
        str(cat_path),
        "-CATALOG_TYPE",
        "FITS_1.0",
        "-PARAMETERS_NAME",
        str(params_path),
        "-DETECT_MINAREA",
        str(int(args.npixels)),
        "-DETECT_THRESH",
        str(float(args.nsigma)),
        "-ANALYSIS_THRESH",
        str(float(args.nsigma)),
        "-THRESH_TYPE",
        "RELATIVE",
        "-FILTER",
        "Y",
        "-FILTER_NAME",
        str(conv_path),
        "-BACK_SIZE",
        str(int(args.box_size)),
        "-BACK_FILTERSIZE",
        str(int(args.filter_size)),
        "-DEBLEND_NTHRESH",
        str(max(1, deblend_nthresh)),
        "-DEBLEND_MINCONT",
        str(float(deblend_mincont)),
        "-VERBOSE_TYPE",
        "QUIET",
        "-CHECKIMAGE_TYPE",
        "NONE",
    ]
    if gaia_mask_path is not None:
        cmd.extend(["-FLAG_IMAGE", str(gaia_mask_path), "-FLAG_TYPE", "OR"])

    debug_log("sextractor_start context=%s cmd=%s", context, " ".join(cmd))
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "SExtractor failed "
            f"(exit={proc.returncode}) for {context}: "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )

    rec = _read_sextractor_fits_catalog(cat_path)
    if rec is None or len(rec) == 0:
        raise RuntimeError("No sources detected.")

    names = {str(n).upper(): str(n) for n in rec.dtype.names or ()}

    def col(name: str, default=None):
        key = names.get(name.upper())
        if key is None:
            return default
        return np.asarray(rec[key])

    x_image = np.asarray(col("X_IMAGE", np.array([], dtype=float)), dtype=float)
    y_image = np.asarray(col("Y_IMAGE", np.array([], dtype=float)), dtype=float)
    if x_image.size == 0:
        raise RuntimeError("SExtractor catalog missing X_IMAGE/Y_IMAGE.")

    # SExtractor image coordinates are 1-based; convert to numpy 0-based pixel coords.
    x_arr = x_image - 1.0
    y_arr = y_image - 1.0
    sma_arr = np.asarray(col("A_IMAGE", np.full_like(x_arr, np.nan)), dtype=float)
    smi_arr = np.asarray(col("B_IMAGE", np.full_like(x_arr, np.nan)), dtype=float)
    kron_arr = np.asarray(col("KRON_RADIUS", np.full_like(x_arr, np.nan)), dtype=float)
    seg_flux_arr = np.asarray(col("FLUX_ISO", np.full_like(x_arr, np.nan)), dtype=float)
    seg_area_arr = np.asarray(col("ISOAREA_IMAGE", np.full_like(x_arr, np.nan)), dtype=float)
    sex_flags_arr = np.asarray(col("FLAGS", np.zeros_like(x_arr)), dtype=float)
    imaflags_arr = np.asarray(col("IMAFLAGS_ISO", np.zeros_like(x_arr)), dtype=float)

    ra_arr = col("ALPHA_J2000")
    dec_arr = col("DELTA_J2000")
    if ra_arr is None or dec_arr is None:
        ra_arr, dec_arr = wcs.pixel_to_world_values(x_arr, y_arr)
    ra_arr = np.asarray(ra_arr, dtype=float)
    dec_arr = np.asarray(dec_arr, dtype=float)

    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    if imaflags_arr is not None:
        keep &= np.asarray(imaflags_arr, dtype=float) == 0.0

    if not np.any(keep):
        raise RuntimeError("No sources detected after Gaia-mask filtering.")

    out = {
        "x": x_arr[keep],
        "y": y_arr[keep],
        "sma": sma_arr[keep],
        "smi": smi_arr[keep],
        "kron": kron_arr[keep],
        "segment_flux": seg_flux_arr[keep],
        "segment_area": seg_area_arr[keep],
        "ra": ra_arr[keep],
        "dec": dec_arr[keep],
        "sex_flags": sex_flags_arr[keep],
        "sex_imaflags_iso": imaflags_arr[keep],
    }
    debug_log(
        "sextractor_done context=%s n_raw=%d n_kept=%d",
        context,
        int(len(x_arr)),
        int(len(out["x"])),
    )
    return out


def _compute_cutout_size(
    i: int,
    args: argparse.Namespace,
    kron_arr: np.ndarray,
    sma_arr: np.ndarray,
) -> tuple[float, int]:
    if args.cutout_mode == "kron":
        if np.isfinite(kron_arr[i]) and np.isfinite(sma_arr[i]):
            half = args.size_mult * float(kron_arr[i]) * float(sma_arr[i])
        else:
            half = args.fixed_half
    else:
        half = args.fixed_half

    half = max(float(args.min_half), min(float(args.max_half), float(half)))
    size = max(2, int(np.ceil(2.0 * half)))
    return half, size


def _run_mfmtk_batch(
    cut_paths: list[Path],
    psf_path: Path,
    args: argparse.Namespace,
    context: str,
) -> list[dict[str, float]]:
    workers = int(args.mfmtk_workers)
    timeout_seconds = float(args.mfmtk_timeout)
    if workers <= 0:
        workers = 1
    workers = min(workers, os.cpu_count() or workers)
    metrics_out: list[dict[str, float] | None] = [None] * len(cut_paths)

    debug_log(
        "stage_mfmtk_start context=%s workers=%d timeout=%.2fs",
        context,
        int(workers),
        float(timeout_seconds),
    )
    t0 = time.monotonic()

    if workers == 1:
        for i, path in enumerate(tqdm(cut_paths, desc="mfmtk metrics")):
            metrics_out[i] = run_mfmtk_on_cutout(str(path), str(psf_path), timeout_seconds)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker_logging,
            initargs=(
                str(ERROR_LOG_PATH),
                str(DEBUG_LOG_PATH),
                bool(DEBUG_ENABLED),
                float(DEBUG_HEARTBEAT_SECONDS),
            ),
        ) as ex:
            futures = {
                ex.submit(run_mfmtk_on_cutout, str(path), str(psf_path), timeout_seconds): i
                for i, path in enumerate(cut_paths)
            }
            if DEBUG_ENABLED:
                pending = set(futures.keys())
                completed = 0
                with tqdm(total=len(futures), desc="mfmtk metrics") as pbar:
                    while pending:
                        done, pending = wait(
                            pending,
                            timeout=float(DEBUG_HEARTBEAT_SECONDS),
                            return_when=FIRST_COMPLETED,
                        )
                        if not done:
                            debug_log(
                                "stage_mfmtk_wait context=%s completed=%d/%d pending=%d elapsed=%.1fs",
                                context,
                                int(completed),
                                int(len(futures)),
                                int(len(pending)),
                                float(time.monotonic() - t0),
                            )
                            continue
                        for fut in done:
                            idx = futures[fut]
                            metrics_out[idx] = fut.result()
                            completed += 1
                            pbar.update(1)
            else:
                for fut in tqdm(as_completed(futures), total=len(futures), desc="mfmtk metrics"):
                    idx = futures[fut]
                    metrics_out[idx] = fut.result()

    debug_log(
        "stage_mfmtk_done context=%s n_cutouts=%d elapsed=%.1fs",
        context,
        int(len(cut_paths)),
        float(time.monotonic() - t0),
    )

    all_keys: set[str] = set()
    for m in metrics_out:
        if m:
            all_keys.update(m.keys())
    normalized: list[dict[str, float]] = []
    for m in metrics_out:
        item = dict(m or {})
        for key in all_keys:
            if key not in item:
                item[key] = np.nan
        normalized.append(item)
    return normalized


def _run_pipeline_on_data(
    data: np.ndarray,
    wcs: WCS,
    err_map: np.ndarray | None,
    detect_data: np.ndarray | None,
    args: argparse.Namespace,
    tmp_dir: Path,
    context: str = "mosaic",
) -> pd.DataFrame:
    t0 = time.monotonic()
    pixscale_arcsec = pixscale_from_wcs(wcs)
    if not np.isfinite(pixscale_arcsec):
        raise RuntimeError("Could not determine pixel scale from WCS.")

    debug_log(
        "stage_start context=%s shape=%dx%d tmp_dir=%s",
        context,
        int(data.shape[0]),
        int(data.shape[1]),
        tmp_dir,
    )

    if detect_data is None:
        detect_data = data
    elif detect_data.shape != data.shape:
        raise RuntimeError(
            f"Detection image shape {detect_data.shape} does not match measurement image shape {data.shape}"
        )

    gaia_mask = None
    if args.exclude_gaia_stars:
        gaia_catalog = _load_gaia_catalog_from_args(args)
        gaia_mask = _build_gaia_exclusion_mask(
            data.shape,
            wcs,
            pixscale_arcsec,
            gaia_catalog,
            args,
        )
        if gaia_mask is not None:
            masked_px = int(np.count_nonzero(gaia_mask))
            debug_log(
                "gaia_mask context=%s stars=%d masked_pixels=%d frac=%.4f",
                context,
                int(len(gaia_catalog)),
                masked_px,
                float(masked_px / gaia_mask.size),
            )

    sex_flags_arr: np.ndarray | None = None
    sex_imaflags_arr: np.ndarray | None = None
    if str(getattr(args, "detection_engine", "sextractor")).lower() == "sextractor":
        sex_cat = _sextractor_dual_image_catalog(
            data=data,
            detect_data=detect_data,
            wcs=wcs,
            gaia_mask=gaia_mask,
            args=args,
            tmp_dir=tmp_dir,
            context=context,
        )
        x_centroid = np.asarray(sex_cat["x"], dtype=float)
        y_centroid = np.asarray(sex_cat["y"], dtype=float)
        sma_arr = np.asarray(sex_cat["sma"], dtype=float)
        smi_arr = np.asarray(sex_cat["smi"], dtype=float)
        kron_arr = np.asarray(sex_cat["kron"], dtype=float)
        seg_flux_arr = np.asarray(sex_cat["segment_flux"], dtype=float)
        seg_area_arr = np.asarray(sex_cat["segment_area"], dtype=float)
        ra_arr = np.asarray(sex_cat["ra"], dtype=float)
        dec_arr = np.asarray(sex_cat["dec"], dtype=float)
        sex_flags_arr = np.asarray(sex_cat.get("sex_flags"), dtype=float)
        sex_imaflags_arr = np.asarray(sex_cat.get("sex_imaflags_iso"), dtype=float)
    else:
        if any(x is None for x in (Background2D, MedianBackground, SourceCatalog, detect_sources)):
            raise RuntimeError(
                "photutils detection engine requested but photutils is not installed."
            )
        sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
        bkg = Background2D(
            data,
            box_size=(args.box_size, args.box_size),
            filter_size=(args.filter_size, args.filter_size),
            sigma_clip=sigma_clip,
            bkg_estimator=MedianBackground(),
        )
        data_sub = data - bkg.background
        detect_bkg = Background2D(
            detect_data,
            box_size=(args.box_size, args.box_size),
            filter_size=(args.filter_size, args.filter_size),
            sigma_clip=sigma_clip,
            bkg_estimator=MedianBackground(),
        )
        detect_data_sub = detect_data - detect_bkg.background
        threshold = args.nsigma * detect_bkg.background_rms

        kernel = Gaussian2DKernel(x_stddev=args.kernel_fwhm / 2.355)
        smooth = convolve(detect_data_sub, kernel, normalize_kernel=True)

        detect_kwargs = {}
        if gaia_mask is not None and _supports_kwarg(detect_sources, "mask"):
            detect_kwargs["mask"] = gaia_mask
            segm = detect_sources(smooth, threshold, npixels=args.npixels, **detect_kwargs)
        elif gaia_mask is not None:
            smooth_for_detection = np.array(smooth, copy=True)
            threshold_for_detection = np.array(threshold, copy=True)
            smooth_for_detection[gaia_mask] = 0.0
            threshold_for_detection[gaia_mask] = np.inf
            segm = detect_sources(smooth_for_detection, threshold_for_detection, npixels=args.npixels)
        else:
            segm = detect_sources(smooth, threshold, npixels=args.npixels)

        if segm is None:
            raise RuntimeError("No sources detected.")
        if args.deblend:
            deblend_kwargs = {}
            if gaia_mask is not None and _supports_kwarg(deblend_sources, "mask"):
                deblend_kwargs["mask"] = gaia_mask
            segm = deblend_sources(
                smooth,
                segm,
                npixels=args.npixels,
                nlevels=args.deblend_nlevels,
                contrast=args.deblend_contrast,
                **deblend_kwargs,
            )

        cat = SourceCatalog(data_sub, segm, wcs=wcs, error=err_map)
        x_centroid = to_float_array(cat.xcentroid)
        y_centroid = to_float_array(cat.ycentroid)
        sma_arr = to_float_array(cat.semimajor_sigma)
        smi_arr = to_float_array(cat.semiminor_sigma)
        kron_arr = to_float_array(cat.kron_radius)
        seg_flux_arr = to_float_array(cat.segment_flux)
        seg_area_arr = to_float_array(cat.segment_area)
        sky = cat.sky_centroid
        ra_arr = to_float_array(sky.ra.deg)
        dec_arr = to_float_array(sky.dec.deg)

    debug_log(
        "stage_detect_done context=%s n_sources=%d pixscale_arcsec=%.6f detect_image=%s engine=%s",
        context,
        int(len(x_centroid)),
        float(pixscale_arcsec),
        "stacked" if detect_data is not data else "input",
        str(getattr(args, "detection_engine", "sextractor")),
    )

    psf_path = Path(args.psf)
    if not psf_path.exists():
        raise RuntimeError(f"PSF file not found: {psf_path}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cut_paths: list[Path] = []
    cutout_half_sizes: list[float] = []
    cutout_sizes: list[int] = []

    # Build one cutout per detected source and keep source pixel position in header.
    for i, (x, y) in enumerate(tqdm(np.column_stack((x_centroid, y_centroid)), desc="mfmtk cutouts")):
        half, size = _compute_cutout_size(i, args, kron_arr, sma_arr)
        cut = Cutout2D(
            data,
            (x, y),
            (size, size),
            wcs=wcs,
            mode="partial",
            fill_value=0.0,
        )
        cut_header = cut.wcs.to_header()
        cut_header["OBJXPIX"] = float(cut.position_cutout[0])
        cut_header["OBJYPIX"] = float(cut.position_cutout[1])
        cut_path = tmp_dir / f"source_{i + 1:06d}.fits"
        fits.PrimaryHDU(cut.data.astype(np.float32), header=cut_header).writeto(
            cut_path, overwrite=True
        )
        cut_paths.append(cut_path)
        cutout_half_sizes.append(float(half))
        cutout_sizes.append(int(size))

    debug_log("stage_cutouts_done context=%s n_cutouts=%d", context, int(len(cut_paths)))

    mfmtk_metrics = _run_mfmtk_batch(cut_paths, psf_path, args, context=context)

    rows: list[dict[str, float]] = []
    for i in range(len(x_centroid)):
        sma = float(sma_arr[i]) if np.isfinite(sma_arr[i]) else np.nan
        smi = float(smi_arr[i]) if np.isfinite(smi_arr[i]) else np.nan
        kron = float(kron_arr[i]) if np.isfinite(kron_arr[i]) else np.nan
        kron_radius = kron * sma if np.isfinite(kron) and np.isfinite(sma) else np.nan
        row: dict[str, float | int] = {
            "id": i + 1,
            "x": float(x_centroid[i]),
            "y": float(y_centroid[i]),
            "ra": float(ra_arr[i]),
            "dec": float(dec_arr[i]),
            "sma": sma,
            "smi": smi,
            "kron": kron,
            "kron_radius": kron_radius,
            "segment_flux": float(seg_flux_arr[i]) if np.isfinite(seg_flux_arr[i]) else np.nan,
            "segment_area": float(seg_area_arr[i]) if np.isfinite(seg_area_arr[i]) else np.nan,
            "pixscale_arcsec": float(pixscale_arcsec),
            "cutout_half_pix": float(cutout_half_sizes[i]),
            "cutout_size_pix": int(cutout_sizes[i]),
        }
        if sex_flags_arr is not None and i < len(sex_flags_arr):
            row["sex_flags"] = (
                float(sex_flags_arr[i]) if np.isfinite(sex_flags_arr[i]) else np.nan
            )
        if sex_imaflags_arr is not None and i < len(sex_imaflags_arr):
            row["sex_imaflags_iso"] = (
                float(sex_imaflags_arr[i]) if np.isfinite(sex_imaflags_arr[i]) else np.nan
            )
        if i < len(mfmtk_metrics):
            row.update(mfmtk_metrics[i])
        rows.append(row)

    df = pd.DataFrame(rows)
    debug_log(
        "stage_done context=%s rows=%d elapsed=%.1fs",
        context,
        int(len(df)),
        float(time.monotonic() - t0),
    )
    return df


def _process_tile(
    input_path: Path,
    args: argparse.Namespace,
    tile: dict[str, int],
    output_dir: Path,
) -> pd.DataFrame:
    set_error_log_path(output_dir / "err.log")
    get_error_logger()
    set_debug_log_path(output_dir / args.debug_log, bool(args.debug), float(args.debug_interval))

    tile_label = f"r{int(tile['row']):02d}c{int(tile['col']):02d}"
    tile_t0 = time.monotonic()
    debug_log(
        "tile_start tile=%s x=[%d,%d) y=[%d,%d)",
        tile_label,
        int(tile["x0"]),
        int(tile["x1"]),
        int(tile["y0"]),
        int(tile["y1"]),
    )

    with fits.open(
        input_path,
        memmap=True,
        mode="readonly",
        lazy_load_hdus=True,
        do_not_scale_image_data=args.no_scale,
    ) as hdul:
        sci_hdu = hdul[args.sci_ext]
        x0, x1 = int(tile["x0"]), int(tile["x1"])
        y0, y1 = int(tile["y0"]), int(tile["y1"])
        data = sci_hdu.section[y0:y1, x0:x1]
        wcs_full = WCS(sci_hdu.header)
        wcs = wcs_full.slice((slice(y0, y1), slice(x0, x1)))
        err_map = _load_error_map(hdul, args, slice(y0, y1), slice(x0, x1))

    detect_data = None
    stack_paths = [Path(p) for p in getattr(args, "_detect_stack_paths", [])]
    if stack_paths:
        detect_data, used_stack_paths = _build_detection_stack_for_region(
            stack_paths,
            args,
            slice(y0, y1),
            slice(x0, x1),
            context=tile_label,
        )
        if detect_data is None:
            debug_log(
                "detect_stack_fallback context=%s used_inputs=%d",
                tile_label,
                int(len(used_stack_paths)),
            )

    tmp_dir = output_dir / "tmp_mfmtk" / f"tile_{tile['row']:02d}_{tile['col']:02d}"
    df = _run_pipeline_on_data(
        data,
        wcs,
        err_map,
        detect_data,
        args,
        tmp_dir,
        context=tile_label,
    )

    if not df.empty:
        df["x"] = pd.to_numeric(df["x"], errors="coerce") + float(tile["x0"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce") + float(tile["y0"])

        core_x0 = tile["core_x0"] - tile["x0"]
        core_x1 = tile["core_x1"] - tile["x0"]
        core_y0 = tile["core_y0"] - tile["y0"]
        core_y1 = tile["core_y1"] - tile["y0"]

        x_rel = pd.to_numeric(df["x"], errors="coerce") - float(tile["x0"])
        y_rel = pd.to_numeric(df["y"], errors="coerce") - float(tile["y0"])
        keep = (x_rel >= core_x0) & (x_rel < core_x1) & (y_rel >= core_y0) & (y_rel < core_y1)
        df = df.loc[keep].reset_index(drop=True)

    debug_log(
        "tile_done tile=%s rows=%d elapsed=%.1fs",
        tile_label,
        int(len(df)),
        float(time.monotonic() - tile_t0),
    )
    return df


def _write_catalog(df: pd.DataFrame, output_path: Path) -> None:
    if not df.empty:
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        df = df.reset_index(drop=True)
        df.insert(0, "id", np.arange(1, len(df) + 1, dtype=int))
    df.to_csv(output_path, index=False)
    debug_log("pipeline_done output=%s rows=%d", output_path, int(len(df)))
    print(f"Wrote {output_path} ({len(df)} sources)")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    input_path = input_path.expanduser().resolve()

    set_error_log_path(output_path.parent / "err.log")
    get_error_logger()
    set_debug_log_path(output_path.parent / args.debug_log, bool(args.debug), float(args.debug_interval))
    if args.debug:
        print(f"Debug logging enabled: {output_path.parent / args.debug_log}")

    debug_log(
        "pipeline_start pid=%d input=%s output=%s tile_max=%d tile_workers=%d mfmtk_workers=%d mfmtk_timeout=%.2fs",
        os.getpid(),
        input_path,
        output_path,
        int(args.tile_max),
        int(args.tile_workers),
        int(args.mfmtk_workers),
        float(args.mfmtk_timeout),
    )

    detect_stack_paths = _resolve_detection_stack_paths(input_path, args)
    args._detect_stack_paths = [str(p) for p in detect_stack_paths]
    if len(detect_stack_paths) > 1:
        print(f"Using stacked detection image from {len(detect_stack_paths)} mosaics")
        debug_log(
            "detect_stack_paths %s",
            ", ".join([p.name for p in detect_stack_paths]),
        )
    else:
        debug_log("detect_stack_disabled_or_single input_only=%s", input_path.name)

    if args.tile_max and args.tile_max > 1 and args.mosaic_cutout and args.mosaic_cutout > 0:
        raise RuntimeError("--tile-max and --mosaic-cutout are mutually exclusive.")

    if args.tile_max and args.tile_max > 1:
        with fits.open(
            input_path,
            memmap=True,
            mode="readonly",
            lazy_load_hdus=True,
            do_not_scale_image_data=args.no_scale,
        ) as hdul:
            sci_hdu = hdul[args.sci_ext]
            height, width = sci_hdu.shape
            wcs_full = WCS(sci_hdu.header)

        if args.exclude_gaia_stars:
            _prepare_gaia_catalog_if_needed(args, wcs_full, width, height, output_path.parent)

        nrows, ncols = _choose_tile_grid(width, height, args.tile_max)
        tiles = _build_tiles(width, height, nrows, ncols, max(0, int(args.tile_overlap)))

        tile_workers = int(args.tile_workers)
        if tile_workers <= 0:
            tile_workers = min(len(tiles), os.cpu_count() or len(tiles))
        tile_workers = min(tile_workers, len(tiles))
        debug_log(
            "tile_plan nrows=%d ncols=%d ntiles=%d tile_workers=%d overlap=%d",
            int(nrows),
            int(ncols),
            int(len(tiles)),
            int(tile_workers),
            int(args.tile_overlap),
        )

        dfs: list[pd.DataFrame] = []
        if tile_workers <= 1:
            for tile in tiles:
                dfs.append(_process_tile(input_path, args, tile, output_path.parent))
        else:
            with ProcessPoolExecutor(
                max_workers=tile_workers,
                initializer=_init_worker_logging,
                initargs=(
                    str(output_path.parent / "err.log"),
                    str(output_path.parent / args.debug_log),
                    bool(args.debug),
                    float(args.debug_interval),
                ),
            ) as ex:
                futures = {
                    ex.submit(_process_tile, input_path, args, tile, output_path.parent): tile
                    for tile in tiles
                }
                if args.debug:
                    pending = set(futures.keys())
                    completed = 0
                    tile_t0 = time.monotonic()
                    with tqdm(total=len(futures), desc="tiles") as pbar:
                        while pending:
                            done, pending = wait(
                                pending,
                                timeout=float(args.debug_interval),
                                return_when=FIRST_COMPLETED,
                            )
                            if not done:
                                debug_log(
                                    "tile_wait completed=%d/%d pending=%d elapsed=%.1fs",
                                    int(completed),
                                    int(len(futures)),
                                    int(len(pending)),
                                    float(time.monotonic() - tile_t0),
                                )
                                continue
                            for fut in done:
                                tile_df = fut.result()
                                dfs.append(tile_df)
                                completed += 1
                                pbar.update(1)
                else:
                    for fut in tqdm(as_completed(futures), total=len(futures), desc="tiles"):
                        dfs.append(fut.result())

        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        _write_catalog(df, output_path)
        return

    with fits.open(
        input_path,
        memmap=True,
        mode="readonly",
        lazy_load_hdus=True,
        do_not_scale_image_data=args.no_scale,
    ) as hdul:
        sci_hdu = hdul[args.sci_ext]
        header = sci_hdu.header
        wcs_full = WCS(header)
        height, width = sci_hdu.shape

        if args.mosaic_cutout and args.mosaic_cutout > 0:
            if args.mosaic_center:
                center = parse_xy(args.mosaic_center)
            else:
                center = (width / 2.0, height / 2.0)
            size = int(args.mosaic_cutout)
            half = int(np.ceil(size / 2))
            cx = int(np.round(center[0]))
            cy = int(np.round(center[1]))
            x1 = max(0, cx - half)
            x2 = min(width, cx + half)
            y1 = max(0, cy - half)
            y2 = min(height, cy + half)
            data = sci_hdu.section[y1:y2, x1:x2]
            wcs = wcs_full.slice((slice(y1, y2), slice(x1, x2)))
            err_map = _load_error_map(hdul, args, slice(y1, y2), slice(x1, x2))
            print(f"Using mosaic section size={size} center=({center[0]:.1f},{center[1]:.1f})")
        else:
            data = sci_hdu.data
            wcs = wcs_full
            err_map = _load_error_map(hdul, args, slice(0, height), slice(0, width))

    if args.exclude_gaia_stars:
        _prepare_gaia_catalog_if_needed(
            args,
            wcs,
            int(data.shape[1]),
            int(data.shape[0]),
            output_path.parent,
        )

    tmp_dir = output_path.parent / "tmp_mfmtk"
    if args.mosaic_cutout and args.mosaic_cutout > 0:
        detect_y_slice = slice(y1, y2)
        detect_x_slice = slice(x1, x2)
    else:
        detect_y_slice = slice(0, int(data.shape[0]))
        detect_x_slice = slice(0, int(data.shape[1]))

    detect_data = None
    if getattr(args, "_detect_stack_paths", None):
        detect_data, used_stack_paths = _build_detection_stack_for_region(
            [Path(p) for p in args._detect_stack_paths],
            args,
            detect_y_slice,
            detect_x_slice,
            context="mosaic",
        )
        if detect_data is None and len(args._detect_stack_paths) > 1:
            print(
                "Falling back to single-image detection (stacked detection image could not be built)."
            )
            debug_log(
                "detect_stack_fallback context=mosaic used_inputs=%d",
                int(len(used_stack_paths)),
            )

    df = _run_pipeline_on_data(data, wcs, err_map, detect_data, args, tmp_dir, context="mosaic")
    _write_catalog(df, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        get_error_logger().exception("Unhandled exception")
        raise

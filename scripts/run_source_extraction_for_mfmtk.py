#!/usr/bin/env python3
"""
Source extraction + MFMTK feature pipeline.

This script detects sources in a JWST mosaic, builds per-source cutouts,
runs MFMTK on each cutout, and writes a source-level CSV catalog.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources
from tqdm import tqdm

from mfmtk import Photometry, Stamp, config as mfmtk_config

warnings.filterwarnings("ignore")
mfmtk_config.verbose = 0

ERROR_LOG_PATH = Path("err.log")
DEBUG_LOG_PATH = Path("debug.log")
DEBUG_ENABLED = False
DEBUG_HEARTBEAT_SECONDS = 60.0


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

    sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
    bkg = Background2D(
        data,
        box_size=(args.box_size, args.box_size),
        filter_size=(args.filter_size, args.filter_size),
        sigma_clip=sigma_clip,
        bkg_estimator=MedianBackground(),
    )
    data_sub = data - bkg.background
    threshold = args.nsigma * bkg.background_rms

    kernel = Gaussian2DKernel(x_stddev=args.kernel_fwhm / 2.355)
    smooth = convolve(data_sub, kernel, normalize_kernel=True)
    segm = detect_sources(smooth, threshold, npixels=args.npixels)
    if segm is None:
        raise RuntimeError("No sources detected.")
    if args.deblend:
        segm = deblend_sources(
            smooth,
            segm,
            npixels=args.npixels,
            nlevels=args.deblend_nlevels,
            contrast=args.deblend_contrast,
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
        "stage_detect_done context=%s n_sources=%d pixscale_arcsec=%.6f",
        context,
        int(len(x_centroid)),
        float(pixscale_arcsec),
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

    tmp_dir = output_dir / "tmp_mfmtk" / f"tile_{tile['row']:02d}_{tile['col']:02d}"
    df = _run_pipeline_on_data(data, wcs, err_map, args, tmp_dir, context=tile_label)

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
            height, width = hdul[args.sci_ext].shape

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

    tmp_dir = output_path.parent / "tmp_mfmtk"
    df = _run_pipeline_on_data(data, wcs, err_map, args, tmp_dir, context="mosaic")
    _write_catalog(df, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        get_error_logger().exception("Unhandled exception")
        raise

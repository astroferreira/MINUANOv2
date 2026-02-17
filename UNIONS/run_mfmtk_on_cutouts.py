#!/usr/bin/env python3
"""
Run Morfometryka (Stamp + Photometry) on pre-generated CFIS cutouts and
join the output metrics with the input CFIS catalog.

This reuses the same metric extraction style from:
MINUANOv2/scripts/run_source_extraction_for_mfmtk.py
"""

from __future__ import annotations

import argparse
import math
import os
import re
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# Keep matplotlib/fontconfig caches in writable temp paths.
_tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_tmp_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp_root / ".cache"))

from mfmtk import Photometry, Stamp, config as mfmtk_config

mfmtk_config.verbose = 0
# Use MFMTK internal segmentation (no photutils segmentation pass on cutouts).
mfmtk_config.Segmentation = "mfmtk"
if not hasattr(mfmtk_config, "segK"):
    # segK is required by mfmtk segmentation mode.
    mfmtk_config.segK = float(getattr(mfmtk_config, "SegThreshold", 5.0))

HEADER_RE = re.compile(r"^#\s*(\d+)\s+([A-Za-z0-9_]+)\b")
CUTOUT_RE = re.compile(r"^tile_(?P<tile>.+?)_source_(?P<source>.+)\.fits$")


class MfmtkTimeoutError(Exception):
    """Raised when per-source MFMTK timeout is reached."""


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


def parse_source_id(raw: str) -> str:
    raw = raw.strip()
    try:
        value = float(raw)
        if math.isfinite(value):
            nearest = int(round(value))
            if abs(value - nearest) < 1e-6:
                return str(nearest)
    except Exception:
        pass
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    return safe or "unknown"


def parse_numeric_or_string(token: str):
    value = token.strip()
    if not value:
        return np.nan
    lower = value.lower()
    if lower in {"nan", "+nan", "-nan"}:
        return np.nan
    try:
        parsed = float(value)
    except Exception:
        return value
    return parsed if math.isfinite(parsed) else np.nan


def load_catalog_dataframe(path: Path) -> pd.DataFrame:
    column_map: dict[str, int] = {}
    rows: list[dict] = []

    with path.open("r", encoding="ascii", errors="ignore") as fin:
        for line in fin:
            if line.startswith("#"):
                match = HEADER_RE.match(line)
                if match:
                    idx_1based = int(match.group(1))
                    name = match.group(2)
                    column_map[name] = idx_1based - 1
                continue

            if not line.strip():
                continue

            if not column_map:
                raise RuntimeError(f"No catalog header with indexed columns in {path}")

            parts = line.split()
            row: dict[str, float | str] = {}
            for name, idx in sorted(column_map.items(), key=lambda item: item[1]):
                token = parts[idx] if idx < len(parts) else "nan"
                row[name] = parse_numeric_or_string(token)

            if "NUMBER" not in row:
                raise RuntimeError("Catalog must include NUMBER column")

            row["source_id"] = parse_source_id(str(row["NUMBER"]))
            rows.append(row)

    return pd.DataFrame(rows)


def estimate_pixel_scale_arcsec_from_cutout(path: Path) -> float | None:
    try:
        with fits.open(path, memmap=True) as hdul:
            if len(hdul) == 0:
                return None
            wcs = WCS(hdul[0].header)
            scales_deg = proj_plane_pixel_scales(wcs)
            if scales_deg.size == 0:
                return None
            scale = float(np.mean(np.abs(scales_deg)) * 3600.0)
            if math.isfinite(scale) and scale > 0:
                return scale
    except Exception:
        pass
    return None


def parse_cutout_entry(path: Path) -> tuple[str, str] | None:
    match = CUTOUT_RE.match(path.name)
    if match:
        tile_id = match.group("tile")
        source_id = parse_source_id(match.group("source"))
        return tile_id, source_id

    # Fallback to FITS header if the filename does not follow the pattern.
    try:
        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header
            tile_id = str(header.get("TILEID", "")).strip()
            source_id = parse_source_id(str(header.get("SRCID", "")).strip())
            if tile_id and source_id:
                return tile_id, source_id
    except Exception:
        return None
    return None


def collect_cutouts(cutouts_dir: Path, tile_id: str | None) -> pd.DataFrame:
    entries: list[dict[str, str]] = []
    for path in sorted(cutouts_dir.glob("*.fits")):
        parsed = parse_cutout_entry(path)
        if parsed is None:
            continue
        tile, source_id = parsed
        if tile_id and tile != tile_id:
            continue
        entries.append(
            {
                "tile_id": tile,
                "source_id": source_id,
                "cutout_path": str(path.resolve()),
            }
        )
    return pd.DataFrame(entries)


def maybe_add_kron_radius_pix(
    catalog_df: pd.DataFrame,
    pixel_scale_arcsec: float | None,
) -> pd.DataFrame:
    if catalog_df.empty:
        return catalog_df
    if "KRON_RADIUS_PIX" in catalog_df.columns:
        return catalog_df
    if "KRON_RADIUS" not in catalog_df.columns or "A_WORLD" not in catalog_df.columns:
        return catalog_df
    if pixel_scale_arcsec is None or not math.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0.0:
        return catalog_df

    out = catalog_df.copy()
    kron = pd.to_numeric(out["KRON_RADIUS"], errors="coerce")
    a_world = pd.to_numeric(out["A_WORLD"], errors="coerce")
    out["KRON_RADIUS_PIX"] = kron * ((a_world * 3600.0) / float(pixel_scale_arcsec))
    return out


def apply_catalog_cuts(
    catalog_df: pd.DataFrame,
    min_snr_sb_cog: float | None,
    min_snr_sb_2arc: float | None,
    min_kron_radius_pix: float | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "catalog_rows_input": int(len(catalog_df)),
        "catalog_rows_after_snr_sb_cog": int(len(catalog_df)),
        "catalog_rows_after_snr_sb_2arc": int(len(catalog_df)),
        "catalog_rows_after_kron_pix": int(len(catalog_df)),
    }
    out = catalog_df

    if min_snr_sb_cog is not None:
        if "SNR_SB_COG" not in out.columns:
            raise RuntimeError("Catalog does not include SNR_SB_COG required by cut.")
        keep = pd.to_numeric(out["SNR_SB_COG"], errors="coerce") >= float(min_snr_sb_cog)
        out = out.loc[keep].copy()
        stats["catalog_rows_after_snr_sb_cog"] = int(len(out))

    if min_snr_sb_2arc is not None:
        if "SNR_SB_2ARC" not in out.columns:
            raise RuntimeError("Catalog does not include SNR_SB_2ARC required by cut.")
        keep = pd.to_numeric(out["SNR_SB_2ARC"], errors="coerce") >= float(min_snr_sb_2arc)
        out = out.loc[keep].copy()
        stats["catalog_rows_after_snr_sb_2arc"] = int(len(out))

    if min_kron_radius_pix is not None:
        if "KRON_RADIUS_PIX" not in out.columns:
            raise RuntimeError(
                "KRON_RADIUS_PIX is unavailable; cannot apply --catalog-min-kron-radius-pix."
            )
        keep = pd.to_numeric(out["KRON_RADIUS_PIX"], errors="coerce") >= float(min_kron_radius_pix)
        out = out.loc[keep].copy()
        stats["catalog_rows_after_kron_pix"] = int(len(out))

    return out, stats


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
    cutout_path: str,
    psf_path: str,
    timeout_seconds: float,
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "mfmtk_stamp_ok": 0.0,
        "mfmtk_phot_ok": 0.0,
        "mfmtk_timeout": 0.0,
        "mfmtk_error": "",
    }
    try:
        with mfmtk_timeout(timeout_seconds):
            stamp = Stamp(cutout_path, psf_path)
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
                metrics["mfmtk_error"] = f"photometry_failed: {exc}"
                metrics["mfmtk_phot_ok"] = 0.0
    except MfmtkTimeoutError:
        metrics["mfmtk_stamp_ok"] = 0.0
        metrics["mfmtk_phot_ok"] = 0.0
        metrics["mfmtk_timeout"] = 1.0
        metrics["mfmtk_error"] = f"timeout_after_{timeout_seconds:.2f}s"
    except Exception as exc:
        metrics["mfmtk_stamp_ok"] = 0.0
        metrics["mfmtk_phot_ok"] = 0.0
        metrics["mfmtk_error"] = f"stamp_failed: {exc}"
    return metrics


def run_mfmtk_batch(
    cutout_paths: list[str],
    psf_path: Path,
    timeout_seconds: float,
) -> list[dict[str, float | str]]:
    if not cutout_paths:
        return []

    out: list[dict[str, float | str]] = []
    start = time.monotonic()
    for i, path in enumerate(cutout_paths, start=1):
        out.append(run_mfmtk_on_cutout(path, str(psf_path), timeout_seconds))
        if i % 25 == 0 or i == len(cutout_paths):
            elapsed = time.monotonic() - start
            print(f"mfmtk_progress={i}/{len(cutout_paths)} elapsed_s={elapsed:.1f}")
    return out


def build_gaussian_psf(size: int, fwhm_pix: float) -> np.ndarray:
    if size < 7:
        raise ValueError("PSF size must be >= 7")
    if size % 2 == 0:
        size += 1
    if fwhm_pix <= 0:
        raise ValueError("PSF FWHM must be > 0")

    sigma = float(fwhm_pix) / 2.3548200450309493
    yy, xx = np.indices((size, size), dtype=np.float64)
    c = 0.5 * (size - 1)
    rr2 = (xx - c) ** 2 + (yy - c) ** 2
    psf = np.exp(-0.5 * rr2 / (sigma * sigma))
    norm = float(psf.sum())
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Synthetic PSF normalization failed.")
    psf /= norm
    return psf.astype(np.float32)


def resolve_psf_path(
    psf_path: Path | None,
    output_path: Path,
    psf_size: int,
    psf_fwhm_pix: float,
) -> Path:
    if psf_path is not None:
        if not psf_path.exists():
            raise FileNotFoundError(f"PSF file not found: {psf_path}")
        return psf_path

    auto_path = output_path.parent / f"{output_path.stem}.synthetic_psf.fits"
    if not auto_path.exists():
        auto_path.parent.mkdir(parents=True, exist_ok=True)
        psf = build_gaussian_psf(psf_size, psf_fwhm_pix)
        fits.PrimaryHDU(psf).writeto(auto_path, overwrite=True)
    return auto_path


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "tile_id",
        "source_id",
        "cutout_path",
        "mfmtk_stamp_ok",
        "mfmtk_phot_ok",
        "mfmtk_timeout",
        "mfmtk_error",
        "P_Rp",
        "P_q",
        "P_LT",
        "SNR_SB_COG",
        "SNR_SB_2ARC",
        "KRON_RADIUS_PIX",
        "MAG_COG",
        "MAG_2ARC",
        "MAG_AUTO",
        "PREDMAGLIM",
        "FLAGS",
    ]
    head = [c for c in preferred if c in df.columns]
    tail = [c for c in df.columns if c not in head]
    return df.loc[:, head + tail]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Morfometryka on CFIS cutouts and merge with catalog columns."
    )
    parser.add_argument("--catalog", required=True, help="Input CFIS catalog used to create cutouts")
    parser.add_argument("--cutouts-dir", required=True, help="Directory with cutout FITS files")
    parser.add_argument("--output", required=True, help="Output morphology CSV path")
    parser.add_argument("--tile-id", default=None, help="Optional tile id filter, e.g. 042.345")
    parser.add_argument("--psf", default=None, help="PSF FITS path (optional)")
    parser.add_argument(
        "--psf-fwhm-pix",
        type=float,
        default=3.0,
        help="FWHM (pixels) for synthetic Gaussian PSF if --psf is not provided",
    )
    parser.add_argument(
        "--psf-size",
        type=int,
        default=33,
        help="Size (pixels) for synthetic PSF if --psf is not provided",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Per-source timeout in seconds for MFMTK (0 disables timeout)",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Optional maximum number of cutouts to process",
    )
    parser.add_argument(
        "--catalog-min-snr-sb-cog",
        type=float,
        default=None,
        help="Optional catalog cut: keep only sources with SNR_SB_COG >= threshold",
    )
    parser.add_argument(
        "--catalog-min-snr-sb-2arc",
        type=float,
        default=None,
        help="Optional catalog cut: keep only sources with SNR_SB_2ARC >= threshold",
    )
    parser.add_argument(
        "--catalog-min-kron-radius-pix",
        type=float,
        default=None,
        help="Optional catalog cut: keep only sources with KRON_RADIUS_PIX >= threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    catalog_path = Path(args.catalog)
    cutouts_dir = Path(args.cutouts_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not cutouts_dir.exists():
        raise FileNotFoundError(f"Cutout directory not found: {cutouts_dir}")

    cutout_df = collect_cutouts(cutouts_dir, args.tile_id)
    if cutout_df.empty:
        raise RuntimeError(f"No cutouts found in {cutouts_dir} for tile filter={args.tile_id!r}")

    scale_arcsec = estimate_pixel_scale_arcsec_from_cutout(Path(cutout_df.iloc[0]["cutout_path"]))
    catalog_df = load_catalog_dataframe(catalog_path)
    catalog_df = maybe_add_kron_radius_pix(catalog_df, scale_arcsec)
    catalog_df, cut_stats = apply_catalog_cuts(
        catalog_df,
        min_snr_sb_cog=args.catalog_min_snr_sb_cog,
        min_snr_sb_2arc=args.catalog_min_snr_sb_2arc,
        min_kron_radius_pix=args.catalog_min_kron_radius_pix,
    )

    if catalog_df.empty:
        print("No catalog rows after cuts; writing empty output.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    allowed_sources = set(catalog_df["source_id"].astype(str))
    cutout_df["source_id"] = cutout_df["source_id"].astype(str)
    cutout_df = cutout_df.loc[cutout_df["source_id"].isin(allowed_sources)].reset_index(drop=True)

    if args.max_sources is not None:
        if args.max_sources <= 0:
            raise ValueError("--max-sources must be > 0")
        cutout_df = cutout_df.head(args.max_sources).copy()

    if cutout_df.empty:
        print("No cutouts left after catalog-source matching and cuts; writing empty output.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    psf_path = resolve_psf_path(
        Path(args.psf) if args.psf else None,
        output_path=output_path,
        psf_size=int(args.psf_size),
        psf_fwhm_pix=float(args.psf_fwhm_pix),
    )

    print(f"catalog={catalog_path}")
    print(f"cutouts_dir={cutouts_dir}")
    print(f"tile_filter={args.tile_id}")
    print(f"pixel_scale_arcsec={scale_arcsec}")
    print(f"psf={psf_path}")
    print(f"catalog_rows={len(catalog_df)}")
    print(f"cutouts_rows={len(cutout_df)}")
    for key, value in cut_stats.items():
        print(f"{key}={value}")

    metrics = run_mfmtk_batch(
        cutout_paths=cutout_df["cutout_path"].tolist(),
        psf_path=psf_path,
        timeout_seconds=float(args.timeout),
    )
    metrics_df = pd.DataFrame(metrics)
    merged = pd.concat([cutout_df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)

    catalog_df = catalog_df.copy()
    catalog_df["source_id"] = catalog_df["source_id"].astype(str)
    out_df = merged.merge(catalog_df, on="source_id", how="left", suffixes=("", "_CAT"))
    out_df = order_columns(out_df)

    out_df.to_csv(output_path, index=False)
    print(f"output={output_path}")
    print(f"rows_written={len(out_df)}")


if __name__ == "__main__":
    main()

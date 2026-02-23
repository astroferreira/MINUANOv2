#!/usr/bin/env python3
"""
Apply a Gaia-based star mask to a FITS image and write a masked copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from gaia_mask_utils import build_gaia_exclusion_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask Gaia stars in a FITS image and write a new FITS file.",
    )
    parser.add_argument("fits_path", help="Input FITS file")
    parser.add_argument("gaia_csv", help="Gaia CSV with RA/Dec/(optional) magnitude columns")
    parser.add_argument(
        "--sci-ext",
        default="SCI",
        help="Image extension to mask (name or index). Default: SCI",
    )
    parser.add_argument("--gaia-ra-col", default="ra", help="RA column in Gaia CSV")
    parser.add_argument("--gaia-dec-col", default="dec", help="Dec column in Gaia CSV")
    parser.add_argument(
        "--gaia-mag-col",
        default="phot_g_mean_mag",
        help="Gaia G-mag column in Gaia CSV (optional; missing values fall back to default size)",
    )

    parser.add_argument("--gaia-r0", type=float, default=8.0)
    parser.add_argument("--gaia-mag0", type=float, default=15.0)
    parser.add_argument("--gaia-minr", type=float, default=1.0)
    parser.add_argument("--gaia-maxr", type=float, default=40.0)
    parser.add_argument(
        "--gaia-mask-shape",
        choices=("star12", "star8", "star4", "jwst", "circle"),
        default="star12",
    )
    parser.add_argument("--gaia-mask-rotation-deg", type=float, default=0.0)
    parser.add_argument("--gaia-mask-xshift-pix", type=float, default=0.0)
    parser.add_argument("--gaia-mask-yshift-pix", type=float, default=0.0)
    parser.add_argument(
        "--fill-value",
        default="nan",
        help="Value written into masked pixels (default: nan)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path (default: input filename + '.gaia_masked')",
    )
    return parser.parse_args()


def _parse_hdu_selector(text: str):
    try:
        return int(text)
    except ValueError:
        return text


def _default_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".fits.gz"):
        base = name[:-8]
        return input_path.with_name(f"{base}.gaia_masked.fits.gz")
    if name.endswith(".fits"):
        base = name[:-5]
        return input_path.with_name(f"{base}.gaia_masked.fits")
    return input_path.with_name(f"{name}.gaia_masked.fits")


def _pixscale_arcsec_from_wcs(wcs: WCS) -> float:
    scales = proj_plane_pixel_scales(wcs)
    if scales.size == 0:
        return np.nan
    return float(np.mean(scales) * 3600.0)


def _load_gaia_catalog(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.gaia_csv)
    ra_col = str(args.gaia_ra_col)
    dec_col = str(args.gaia_dec_col)
    mag_col = str(args.gaia_mag_col)

    missing = [c for c in (ra_col, dec_col) if c not in df.columns]
    if missing:
        raise RuntimeError(f"Gaia CSV is missing required column(s): {missing}")

    out = pd.DataFrame(
        {
            "ra": pd.to_numeric(df[ra_col], errors="coerce"),
            "dec": pd.to_numeric(df[dec_col], errors="coerce"),
        }
    )
    if mag_col in df.columns:
        out["phot_g_mean_mag"] = pd.to_numeric(df[mag_col], errors="coerce")
    else:
        out["phot_g_mean_mag"] = np.nan

    keep = np.isfinite(out["ra"].to_numpy()) & np.isfinite(out["dec"].to_numpy())
    return out.loc[keep].reset_index(drop=True)


def _parse_fill_value(text: str) -> float:
    if str(text).strip().lower() == "nan":
        return float("nan")
    return float(text)


def main() -> None:
    args = _parse_args()
    in_path = Path(args.fits_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input FITS not found: {in_path}")

    out_path = (
        Path(args.output).expanduser().resolve()
        if str(args.output).strip()
        else _default_output_path(in_path)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gaia_catalog = _load_gaia_catalog(args)
    fill_value = _parse_fill_value(args.fill_value)

    hdu_sel = _parse_hdu_selector(str(args.sci_ext))
    with fits.open(in_path) as hdul:
        try:
            hdu = hdul[hdu_sel]
        except Exception as exc:
            raise KeyError(f"HDU '{args.sci_ext}' not found in {in_path}") from exc
        if hdu.data is None:
            raise RuntimeError(f"HDU '{args.sci_ext}' has no image data")

        data = np.asarray(hdu.data)
        if data.ndim != 2:
            raise RuntimeError(f"Expected a 2D image in HDU '{args.sci_ext}', got shape {data.shape}")

        wcs = WCS(hdu.header)
        pixscale_arcsec = _pixscale_arcsec_from_wcs(wcs)
        if not np.isfinite(pixscale_arcsec) or pixscale_arcsec <= 0:
            raise RuntimeError("Could not determine a valid pixel scale from the FITS WCS header")

        mask = build_gaia_exclusion_mask(
            shape=data.shape,
            wcs=wcs,
            pixscale_arcsec=pixscale_arcsec,
            gaia_catalog=gaia_catalog,
            args=args,
        )

        if mask is None:
            mask = np.zeros(data.shape, dtype=bool)

        if np.isnan(fill_value):
            out_data = np.array(data, dtype=np.float32, copy=True)
        else:
            out_data = np.array(data, copy=True)
            if not np.can_cast(type(fill_value), out_data.dtype, casting="safe"):
                out_data = out_data.astype(np.float32, copy=False)

        out_data[mask] = fill_value
        hdu.data = out_data
        hdul.writeto(out_path, overwrite=True)

    masked_px = int(np.count_nonzero(mask))
    print(f"Input FITS:   {in_path}")
    print(f"Gaia CSV:     {Path(args.gaia_csv).expanduser().resolve()}")
    print(f"Output FITS:  {out_path}")
    print(f"Image HDU:    {args.sci_ext}")
    print(f"Gaia rows:    {len(gaia_catalog)}")
    print(f"Masked pixels:{masked_px}")


if __name__ == "__main__":
    main()

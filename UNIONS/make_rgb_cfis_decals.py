#!/usr/bin/env python3
"""
Create an RGB image by projecting CFIS bands onto a DECaLS cutout WCS.

Default channel mapping:
  R <- CFIS r
  G <- DECaLS g (reference grid)
  B <- CFIS u
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import map_coordinates

# Keep matplotlib/fontconfig caches in writable temp paths.
_tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_tmp_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp_root / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_fits_image(path: Path) -> tuple[np.ndarray, WCS]:
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) == 0 or hdul[0].data is None:
            raise RuntimeError(f"No image data in FITS: {path}")
        data = np.asarray(hdul[0].data, dtype=np.float32)
        while data.ndim > 2:
            data = data[0]
        wcs = WCS(hdul[0].header)
    if data.ndim != 2:
        raise RuntimeError(f"Expected 2D image in {path}, got shape={data.shape}")
    return data, wcs


def mean_pixel_scale_arcsec(wcs: WCS) -> float:
    scales = proj_plane_pixel_scales(wcs) * 3600.0
    scales = np.asarray(scales, dtype=float)
    good = np.isfinite(scales) & (scales > 0)
    if not np.any(good):
        raise RuntimeError("Could not determine positive pixel scale from WCS")
    return float(np.mean(scales[good]))


def world_center_from_wcs(wcs: WCS, shape: tuple[int, int]) -> tuple[float, float]:
    h, w = shape
    x0 = 0.5 * (w - 1)
    y0 = 0.5 * (h - 1)
    ra, dec = wcs.all_pix2world(x0, y0, 0)
    return float(ra), float(dec)


def reproject_to_reference(
    src_data: np.ndarray,
    src_wcs: WCS,
    ref_wcs: WCS,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    h, w = ref_shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    ra, dec = ref_wcs.all_pix2world(xx, yy, 0)
    src_x, src_y = src_wcs.all_world2pix(ra, dec, 0)

    coords = np.array([src_y, src_x], dtype=np.float64)
    projected = map_coordinates(
        src_data,
        coords,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    return projected.astype(np.float32)


def robust_asinh_stretch(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.7,
    asinh_a: float = 8.0,
) -> np.ndarray:
    valid = np.isfinite(img)
    if not np.any(valid):
        return np.zeros_like(img, dtype=np.float32)

    values = img[valid]
    lo = float(np.percentile(values, p_low))
    hi = float(np.percentile(values, p_high))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    norm = (img - lo) / (hi - lo)
    norm = np.clip(np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    stretched = np.arcsinh(asinh_a * norm) / np.arcsinh(asinh_a)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def maybe_cutout(
    data: np.ndarray,
    wcs: WCS,
    ra: float | None,
    dec: float | None,
    size: int | None,
) -> tuple[np.ndarray, WCS]:
    if ra is None or dec is None or size is None:
        return data, wcs
    if size <= 0:
        raise ValueError("--size-pix must be > 0")
    center = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
    cut = Cutout2D(
        data,
        position=center,
        size=(int(size), int(size)),
        wcs=wcs,
        mode="trim",
        copy=True,
    )
    return np.asarray(cut.data, dtype=np.float32), cut.wcs


def build_cfis_reference_from_decals_footprint(
    cfis_data: np.ndarray,
    cfis_wcs: WCS,
    decals_shape: tuple[int, int],
    decals_wcs: WCS,
    ra: float | None,
    dec: float | None,
    size_pix: int | None,
) -> tuple[np.ndarray, WCS, tuple[float, float], tuple[int, int]]:
    if ra is not None and dec is not None and size_pix is not None:
        ref_data, ref_wcs = maybe_cutout(cfis_data, cfis_wcs, ra, dec, size_pix)
        return ref_data, ref_wcs, (float(ra), float(dec)), ref_data.shape

    center_ra, center_dec = world_center_from_wcs(decals_wcs, decals_shape)
    decals_scale = mean_pixel_scale_arcsec(decals_wcs)
    cfis_scale = mean_pixel_scale_arcsec(cfis_wcs)

    size_y_cfis = int(np.ceil(decals_shape[0] * decals_scale / cfis_scale))
    size_x_cfis = int(np.ceil(decals_shape[1] * decals_scale / cfis_scale))
    size_y_cfis = max(2, size_y_cfis)
    size_x_cfis = max(2, size_x_cfis)

    center = SkyCoord(center_ra * u.deg, center_dec * u.deg, frame="icrs")
    cut = Cutout2D(
        cfis_data,
        position=center,
        size=(size_y_cfis, size_x_cfis),
        wcs=cfis_wcs,
        mode="trim",
        copy=True,
    )
    return (
        np.asarray(cut.data, dtype=np.float32),
        cut.wcs,
        (center_ra, center_dec),
        (int(size_y_cfis), int(size_x_cfis)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RGB PNG from CFIS u/r + DECaLS g around a target."
    )
    parser.add_argument("--r-fits", required=True, help="CFIS r FITS image path")
    parser.add_argument("--g-fits", required=True, help="DECaLS g FITS image path (reference WCS)")
    parser.add_argument("--u-fits", required=True, help="CFIS u FITS image path")
    parser.add_argument("--output", required=True, help="Output RGB PNG path")
    parser.add_argument(
        "--reference-grid",
        choices=("decals", "cfis"),
        default="decals",
        help="Output pixel grid: decals (default) or cfis",
    )
    parser.add_argument(
        "--ra",
        type=float,
        default=None,
        help="Optional center RA (deg) to cut reference image before RGB assembly",
    )
    parser.add_argument(
        "--dec",
        type=float,
        default=None,
        help="Optional center Dec (deg) to cut reference image before RGB assembly",
    )
    parser.add_argument(
        "--size-pix",
        type=int,
        default=None,
        help="Optional reference cutout size in pixels (requires --ra and --dec)",
    )
    parser.add_argument(
        "--p-low",
        type=float,
        default=1.0,
        help="Lower percentile for per-channel stretch (default: 1.0)",
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=99.7,
        help="Upper percentile for per-channel stretch (default: 99.7)",
    )
    parser.add_argument(
        "--asinh-a",
        type=float,
        default=8.0,
        help="Asinh stretch parameter (default: 8.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Output gamma correction (>1 darkens background and bright regions; default: 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    r_path = Path(args.r_fits)
    g_path = Path(args.g_fits)
    u_path = Path(args.u_fits)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    r_data, r_wcs = load_fits_image(r_path)
    g_data, g_wcs = load_fits_image(g_path)
    u_data, u_wcs = load_fits_image(u_path)

    if args.reference_grid == "decals":
        g_data, g_wcs = maybe_cutout(g_data, g_wcs, args.ra, args.dec, args.size_pix)
        ref_data = g_data
        ref_wcs = g_wcs
        r_on_ref = reproject_to_reference(r_data, r_wcs, ref_wcs, ref_shape=ref_data.shape)
        g_on_ref = ref_data
        b_on_ref = reproject_to_reference(u_data, u_wcs, ref_wcs, ref_shape=ref_data.shape)
        ref_desc = "decals"
    else:
        ref_data, ref_wcs, center_ra_dec, est_size = build_cfis_reference_from_decals_footprint(
            cfis_data=r_data,
            cfis_wcs=r_wcs,
            decals_shape=g_data.shape,
            decals_wcs=g_wcs,
            ra=args.ra,
            dec=args.dec,
            size_pix=args.size_pix,
        )
        r_on_ref = ref_data
        g_on_ref = reproject_to_reference(g_data, g_wcs, ref_wcs, ref_shape=ref_data.shape)
        b_on_ref = reproject_to_reference(u_data, u_wcs, ref_wcs, ref_shape=ref_data.shape)
        ref_desc = (
            f"cfis center_ra_dec={center_ra_dec[0]:.8f},{center_ra_dec[1]:.8f} "
            f"size_pix={est_size[1]}x{est_size[0]}"
        )

    r_st = robust_asinh_stretch(r_on_ref, p_low=args.p_low, p_high=args.p_high, asinh_a=args.asinh_a)
    g_st = robust_asinh_stretch(g_on_ref, p_low=args.p_low, p_high=args.p_high, asinh_a=args.asinh_a)
    b_st = robust_asinh_stretch(b_on_ref, p_low=args.p_low, p_high=args.p_high, asinh_a=args.asinh_a)

    rgb = np.dstack([r_st, g_st, b_st])
    if args.gamma <= 0:
        raise ValueError("--gamma must be > 0")
    if abs(args.gamma - 1.0) > 1.0e-6:
        rgb = np.clip(np.power(rgb, float(args.gamma)), 0.0, 1.0)
    plt.imsave(out_path, rgb, origin="lower")

    print(f"r_fits={r_path}")
    print(f"g_fits={g_path}")
    print(f"u_fits={u_path}")
    print(f"reference_grid={args.reference_grid}")
    print(f"reference_info={ref_desc}")
    if args.ra is not None and args.dec is not None and args.size_pix is not None:
        print(f"center_ra_dec={args.ra:.8f},{args.dec:.8f}")
        print(f"size_pix={args.size_pix}")
    print(f"p_low={args.p_low}")
    print(f"p_high={args.p_high}")
    print(f"asinh_a={args.asinh_a}")
    print(f"gamma={args.gamma}")
    print(f"output={out_path}")
    print(f"shape={rgb.shape[1]}x{rgb.shape[0]}")


if __name__ == "__main__":
    main()

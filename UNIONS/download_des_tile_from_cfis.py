#!/usr/bin/env python3
"""
Download Legacy Survey DR10 FITS cutouts in g-band covering one CFIS tile area.

This uses the documented Legacy Survey cutout endpoint:
https://www.legacysurvey.org/viewer/fits-cutout

Examples:
  # Native-scale mosaic (many cutouts, default)
  python download_des_tile_from_cfis.py --tile 042.345 --outdir des_tiles

  # One coarse cutout covering the full tile (lower resolution)
  python download_des_tile_from_cfis.py --tile 042.345 --single --outdir des_tiles
"""

from __future__ import annotations

import argparse
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "https://www.legacysurvey.org/viewer/fits-cutout"
DEFAULT_LAYER = "ls-dr10-south"


def cfis_tile_center(tile_id: str) -> tuple[float, float]:
    """
    Convert CFIS tile index xxx.yyy into tile center (RA, Dec) in deg.
    """
    try:
        xxx_str, yyy_str = tile_id.split(".")
        xxx = int(xxx_str)
        yyy = int(yyy_str)
    except Exception as exc:
        raise ValueError(f"Invalid tile id '{tile_id}'. Use format xxx.yyy") from exc

    dec = yyy / 2.0 - 90.0
    cosf = math.cos(math.radians(dec))
    if abs(cosf) < 1e-10:
        raise ValueError(f"Tile '{tile_id}' is too close to the pole for stable RA conversion")

    ra = (xxx / (2.0 * cosf)) % 360.0
    return ra, dec


def wrap_ra(ra_deg: float) -> float:
    return ra_deg % 360.0


def build_cutout_url(
    base_url: str,
    ra: float,
    dec: float,
    width: int,
    height: int,
    pixscale: float,
    layer: str,
    band: str,
    subimage: str | None = None,
) -> str:
    params = {
        "ra": f"{ra:.8f}",
        "dec": f"{dec:.8f}",
        "width": str(width),
        "height": str(height),
        "pixscale": f"{pixscale:.6f}",
        "layer": layer,
        "bands": band,
    }
    if subimage:
        params["subimage"] = subimage
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)


def tile_grid_for_area(
    center_ra: float,
    center_dec: float,
    tile_size_deg: float,
    pixscale: float,
    max_size: int,
) -> tuple[int, int, list[tuple[int, int, float, float]]]:
    """
    Return (n_grid, cutout_size_px, list of (ix, iy, ra, dec)).
    """
    side_arcsec = tile_size_deg * 3600.0
    total_px = int(math.ceil(side_arcsec / pixscale))
    n_grid = max(1, int(math.ceil(total_px / max_size)))
    cutout_size_px = int(math.ceil(total_px / n_grid))

    if cutout_size_px > max_size:
        raise RuntimeError("Internal error: computed cutout size exceeds max_size")

    dec_step_deg = (cutout_size_px * pixscale) / 3600.0
    cells = []
    mid = (n_grid - 1) / 2.0

    for iy in range(n_grid):
        dec_i = center_dec + (iy - mid) * dec_step_deg
        cosd = math.cos(math.radians(dec_i))
        if abs(cosd) < 1e-6:
            cosd = 1e-6
        ra_step_deg = dec_step_deg / abs(cosd)
        for ix in range(n_grid):
            ra_i = wrap_ra(center_ra + (ix - mid) * ra_step_deg)
            cells.append((ix, iy, ra_i, dec_i))

    return n_grid, cutout_size_px, cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Legacy Survey DR10 g-band cutouts for one CFIS tile area."
    )
    parser.add_argument(
        "--tile",
        required=True,
        help="CFIS tile id in xxx.yyy format (example: 042.345)",
    )
    parser.add_argument(
        "--band",
        default="g",
        help="Band selection passed to Legacy cutout API (default: g)",
    )
    parser.add_argument(
        "--layer",
        default=DEFAULT_LAYER,
        help=f"Legacy Survey layer (default: {DEFAULT_LAYER})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Legacy Survey FITS cutout endpoint (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--tile-size-deg",
        type=float,
        default=0.516,
        help="Target CFIS-like tile size in Dec (deg), default 0.516",
    )
    parser.add_argument(
        "--pixscale",
        type=float,
        default=0.262,
        help="Cutout pixel scale in arcsec/pixel (default 0.262)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Maximum width/height in pixels per cutout (default 512)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Force one cutout by coarsening pixscale to fit max-size",
    )
    parser.add_argument(
        "--outdir",
        default="des_tiles",
        help="Output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned URLs/files without downloading",
    )
    args = parser.parse_args()

    if args.pixscale <= 0:
        raise ValueError("--pixscale must be > 0")
    if args.max_size <= 0:
        raise ValueError("--max-size must be > 0")
    if args.tile_size_deg <= 0:
        raise ValueError("--tile-size-deg must be > 0")

    center_ra, center_dec = cfis_tile_center(args.tile)
    side_arcsec = args.tile_size_deg * 3600.0

    pixscale = args.pixscale
    if args.single:
        required = side_arcsec / args.max_size
        if required > pixscale:
            pixscale = required

    n_grid, cutout_size_px, cells = tile_grid_for_area(
        center_ra=center_ra,
        center_dec=center_dec,
        tile_size_deg=args.tile_size_deg,
        pixscale=pixscale,
        max_size=args.max_size,
    )

    outdir = Path(args.outdir)
    planned = []
    for ix, iy, ra_i, dec_i in cells:
        if n_grid == 1:
            filename = f"DES_{args.band}_{args.tile}.fits"
            subimage = f"{args.tile}_{args.band}"
        else:
            filename = f"DES_{args.band}_{args.tile}_x{ix:02d}_y{iy:02d}.fits"
            subimage = f"{args.tile}_{args.band}_x{ix:02d}_y{iy:02d}"

        url = build_cutout_url(
            base_url=args.base_url,
            ra=ra_i,
            dec=dec_i,
            width=cutout_size_px,
            height=cutout_size_px,
            pixscale=pixscale,
            layer=args.layer,
            band=args.band,
            subimage=subimage,
        )
        planned.append((url, outdir / filename))

    print(f"tile={args.tile}")
    print(f"center_ra_dec={center_ra:.8f},{center_dec:.8f}")
    print(f"band={args.band}")
    print(f"layer={args.layer}")
    print(f"tile_size_deg={args.tile_size_deg}")
    print(f"pixscale_arcsec_per_pix={pixscale:.6f}")
    print(f"cutout_size_px={cutout_size_px}")
    print(f"grid={n_grid}x{n_grid}")
    print(f"n_cutouts={len(planned)}")
    print(f"outdir={outdir}")

    ok = 0
    failed = 0
    for i, (url, path) in enumerate(planned, start=1):
        print(f"cutout_{i:03d}_url={url}")
        print(f"cutout_{i:03d}_file={path}")
        if not args.dry_run:
            try:
                download_file(url, path)
                ok += 1
            except urllib.error.HTTPError as exc:
                failed += 1
                print(
                    f"cutout_{i:03d}_error=HTTP {exc.code} {exc.reason}; url={url}"
                )
            except Exception as exc:
                failed += 1
                print(f"cutout_{i:03d}_error={exc}; url={url}")

    if not args.dry_run:
        print(f"download_success={ok}")
        print(f"download_failed={failed}")


if __name__ == "__main__":
    main()

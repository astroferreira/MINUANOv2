#!/usr/bin/env python3
"""
Generate FITS cutouts for CFIS catalog sources from a tile image.

Output file naming:
  tile_<tile_id>_source_<source_id>.fits
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


HEADER_RE = re.compile(r"^#\s*(\d+)\s+([A-Za-z0-9_]+)\b")


@dataclass
class SourceRow:
    source_id: str
    x_image: float
    y_image: float
    kron_radius: float
    snr_sb_cog: float | None = None
    a_world: float | None = None


def parse_float(parts: list[str], index: int | None) -> float | None:
    if index is None or index < 0 or index >= len(parts):
        return None
    try:
        value = float(parts[index])
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


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


def parse_catalog_sources(
    catalog_path: Path,
    default_kron_radius: float,
    min_snr_sb_cog: float | None = None,
) -> tuple[list[SourceRow], int, int]:
    column_map: dict[str, int] = {}
    rows: list[SourceRow] = []
    n_data_rows = 0
    n_snr_rejected = 0

    with catalog_path.open("r", encoding="ascii", errors="ignore") as fin:
        for line in fin:
            if line.startswith("#"):
                match = HEADER_RE.match(line)
                if match:
                    index_1based = int(match.group(1))
                    name = match.group(2)
                    column_map[name] = index_1based - 1
                continue

            if not line.strip():
                continue

            n_data_rows += 1
            number_idx = column_map.get("NUMBER")
            x_idx = column_map.get("X_IMAGE")
            y_idx = column_map.get("Y_IMAGE")
            kron_idx = column_map.get("KRON_RADIUS")
            a_world_idx = column_map.get("A_WORLD")
            snr_sb_cog_idx = column_map.get("SNR_SB_COG")
            if number_idx is None or x_idx is None or y_idx is None:
                raise RuntimeError("Catalog must include NUMBER, X_IMAGE, and Y_IMAGE columns")
            if min_snr_sb_cog is not None and snr_sb_cog_idx is None:
                raise RuntimeError(
                    "Catalog does not include SNR_SB_COG; cannot apply --min-snr-sb-cog."
                )

            parts = line.split()
            if number_idx >= len(parts):
                continue

            x_image = parse_float(parts, x_idx)
            y_image = parse_float(parts, y_idx)
            if x_image is None or y_image is None:
                continue

            kron = parse_float(parts, kron_idx)
            if kron is None or kron <= 0:
                kron = default_kron_radius
            a_world = parse_float(parts, a_world_idx)

            snr_sb_cog = parse_float(parts, snr_sb_cog_idx)
            if min_snr_sb_cog is not None:
                if snr_sb_cog is None or snr_sb_cog < min_snr_sb_cog:
                    n_snr_rejected += 1
                    continue

            source_id = parse_source_id(parts[number_idx])
            rows.append(
                SourceRow(
                    source_id=source_id,
                    x_image=x_image,
                    y_image=y_image,
                    kron_radius=kron,
                    snr_sb_cog=snr_sb_cog,
                    a_world=a_world,
                )
            )

    return rows, n_data_rows, n_snr_rejected


def estimate_pixel_scale_arcsec(image_header) -> float | None:
    try:
        w = WCS(image_header)
        scales_deg = proj_plane_pixel_scales(w)
        scale_arcsec = float(abs(scales_deg[0]) * 3600.0)
        if math.isfinite(scale_arcsec) and scale_arcsec > 0:
            return scale_arcsec
    except Exception:
        pass
    return None


def kron_radius_pixels(src: SourceRow, pixel_scale_arcsec: float | None) -> float | None:
    """
    Convert KRON_RADIUS (in units of A) into pixels using A_WORLD and WCS pixel scale.
    """
    if (
        pixel_scale_arcsec is None
        or src.a_world is None
        or src.a_world <= 0.0
        or src.kron_radius <= 0.0
    ):
        return None
    a_pixels = (src.a_world * 3600.0) / pixel_scale_arcsec
    if not math.isfinite(a_pixels) or a_pixels <= 0.0:
        return None
    value = src.kron_radius * a_pixels
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def create_cutouts(
    image_path: Path,
    tile_id: str,
    sources: list[SourceRow],
    output_dir: Path,
    kron_multiplier: float,
    min_half_size: int,
    max_half_size: int,
    max_sources: int | None,
    min_kron_radius: float | None,
    overwrite: bool,
) -> tuple[int, int, int, int]:
    with fits.open(image_path, memmap=True) as hdul:
        if len(hdul) == 0 or hdul[0].data is None:
            raise RuntimeError(f"No image data in {image_path}")

        data = hdul[0].data
        while data.ndim > 2:
            data = data[0]

        pixel_scale_arcsec = estimate_pixel_scale_arcsec(hdul[0].header)
        try:
            wcs = WCS(hdul[0].header)
        except Exception:
            wcs = None

        output_dir.mkdir(parents=True, exist_ok=True)

        n_requested = 0
        n_written = 0
        n_skipped = 0
        n_kron_rejected = 0

        for src in sources:
            kron_pix = kron_radius_pixels(src, pixel_scale_arcsec)
            if min_kron_radius is not None:
                if kron_pix is None or kron_pix <= min_kron_radius:
                    n_kron_rejected += 1
                    continue

            if max_sources is not None and n_requested >= max_sources:
                break
            n_requested += 1

            # Keep cutout sizing based on the catalog KRON_RADIUS value
            # (legacy behavior expected by current pipeline tuning).
            half_size = int(math.ceil(kron_multiplier * src.kron_radius))
            half_size = max(min_half_size, half_size)
            if max_half_size > 0:
                half_size = min(max_half_size, half_size)
            size = (2 * half_size + 1, 2 * half_size + 1)

            # SExtractor X/Y_IMAGE are 1-based pixel coordinates.
            position = (src.x_image - 1.0, src.y_image - 1.0)

            try:
                cutout = Cutout2D(
                    data,
                    position=position,
                    size=size,
                    wcs=wcs,
                    mode="trim",
                    copy=True,
                )
            except Exception:
                n_skipped += 1
                continue

            if cutout.data.size == 0:
                n_skipped += 1
                continue

            out_path = output_dir / f"tile_{tile_id}_source_{src.source_id}.fits"

            if cutout.wcs is not None:
                out_header = cutout.wcs.to_header()
            else:
                out_header = fits.Header()
            out_header["TILEID"] = (tile_id, "CFIS tile id")
            out_header["SRCID"] = (src.source_id, "Source NUMBER from catalog")
            out_header["XIMAGE"] = (float(src.x_image), "Catalog X_IMAGE (pixel, 1-based)")
            out_header["YIMAGE"] = (float(src.y_image), "Catalog Y_IMAGE (pixel, 1-based)")
            out_header["KRONRAD"] = (float(src.kron_radius), "Catalog KRON_RADIUS")
            if src.a_world is not None:
                out_header["AWORLD"] = (float(src.a_world), "Catalog A_WORLD (deg)")
            if kron_pix is not None:
                out_header["KRONPX"] = (float(kron_pix), "KRON radius converted to pixels")
            out_header["CKMULT"] = (float(kron_multiplier), "Cutout half-size multiplier over KRON_RADIUS_PIX")
            out_header["CHALF"] = (int(half_size), "Cutout half-size in pixels")

            fits.PrimaryHDU(data=cutout.data, header=out_header).writeto(out_path, overwrite=overwrite)
            n_written += 1

    return n_requested, n_written, n_skipped, n_kron_rejected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FITS cutouts from a CFIS tile image for catalog sources."
    )
    parser.add_argument("--catalog", required=True, help="Input CFIS COG catalog path")
    parser.add_argument("--image", required=True, help="Input CFIS tile FITS image")
    parser.add_argument("--tile-id", required=True, help="Tile id for output naming, e.g. 042.345")
    parser.add_argument("--output-dir", default="cutouts", help="Directory for cutout FITS files")
    parser.add_argument(
        "--kron-multiplier",
        type=float,
        default=12.0,
        help="Cutout half-size = ceil(kron_multiplier * KRON_RADIUS) pixels",
    )
    parser.add_argument(
        "--default-kron-radius",
        type=float,
        default=3.5,
        help="Fallback KRON_RADIUS if missing/non-positive",
    )
    parser.add_argument(
        "--min-half-size",
        type=int,
        default=12,
        help="Minimum cutout half-size in pixels",
    )
    parser.add_argument(
        "--max-half-size",
        type=int,
        default=1024,
        help="Maximum cutout half-size in pixels (set <=0 for no cap)",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Optional max number of sources to process",
    )
    parser.add_argument(
        "--min-snr-sb-cog",
        type=float,
        default=None,
        help="Only create cutouts for rows with SNR_SB_COG >= this value",
    )
    parser.add_argument(
        "--min-kron-radius",
        type=float,
        default=None,
        help="Only create cutouts for rows with KRON_RADIUS > this value (pixels)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cutout files",
    )
    args = parser.parse_args()

    if args.kron_multiplier <= 0:
        raise ValueError("--kron-multiplier must be > 0")
    if args.default_kron_radius <= 0:
        raise ValueError("--default-kron-radius must be > 0")
    if args.min_half_size <= 0:
        raise ValueError("--min-half-size must be > 0")
    if args.max_sources is not None and args.max_sources <= 0:
        raise ValueError("--max-sources must be > 0 when provided")
    if args.min_snr_sb_cog is not None and not math.isfinite(args.min_snr_sb_cog):
        raise ValueError("--min-snr-sb-cog must be finite when provided")
    if args.min_kron_radius is not None and not math.isfinite(args.min_kron_radius):
        raise ValueError("--min-kron-radius must be finite when provided")
    if args.min_kron_radius is not None and args.min_kron_radius < 0:
        raise ValueError("--min-kron-radius must be >= 0 when provided")

    catalog_path = Path(args.catalog)
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    sources, n_data_rows, n_snr_rejected = parse_catalog_sources(
        catalog_path,
        default_kron_radius=args.default_kron_radius,
        min_snr_sb_cog=args.min_snr_sb_cog,
    )
    n_requested, n_written, n_skipped, n_kron_rejected = create_cutouts(
        image_path=image_path,
        tile_id=args.tile_id,
        sources=sources,
        output_dir=output_dir,
        kron_multiplier=args.kron_multiplier,
        min_half_size=args.min_half_size,
        max_half_size=args.max_half_size,
        max_sources=args.max_sources,
        min_kron_radius=args.min_kron_radius,
        overwrite=args.overwrite,
    )

    print(f"catalog={catalog_path}")
    print(f"image={image_path}")
    print(f"output_dir={output_dir}")
    print(f"tile_id={args.tile_id}")
    print(f"rows_catalog_total={n_data_rows}")
    print(f"sources_catalog={len(sources)}")
    print(f"rows_rejected_snr_sb_cog={n_snr_rejected}")
    print(f"rows_rejected_kron_radius={n_kron_rejected}")
    if args.min_snr_sb_cog is not None:
        print(f"min_snr_sb_cog={args.min_snr_sb_cog}")
    if args.min_kron_radius is not None:
        print(f"min_kron_radius={args.min_kron_radius}")
    print(f"sources_requested={n_requested}")
    print(f"cutouts_written={n_written}")
    print(f"cutouts_skipped={n_skipped}")
    print(f"kron_multiplier={args.kron_multiplier}")
    print(f"min_half_size={args.min_half_size}")
    print(f"max_half_size={args.max_half_size}")


if __name__ == "__main__":
    main()

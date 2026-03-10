#!/usr/bin/env python3
"""
Download Euclid MER FITS cutouts for a CSV catalogue with columns:
    id, ra, dec, size

The script:
1. Reads the input CSV.
2. For each row, queries the Euclid Q1 mosaic_product table to find the VIS MER
   mosaic covering the target position and requested cutout radius.
3. Downloads a FITS cutout using astroquery.esa.euclid.Euclid.get_cutout().
4. Writes a summary CSV with status and output filenames.

Notes:
- This uses public Euclid Q1 MER mosaics.
- Euclid cutouts in astroquery currently work for MER images.
- "size" is interpreted in arcsec by default; you can change that with
  --size-unit.

Example:
    python euclid_stamp_downloader.py catalogue.csv outdir \
        --instrument VIS --filter-name VIS --size-unit arcsec
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.esa.euclid import Euclid


LOG = logging.getLogger("euclid_stamp_downloader")


@dataclass
class Target:
    obj_id: str
    ra: float
    dec: float
    size: float


def sanitize_filename(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Euclid MER FITS cutouts from a CSV catalogue."
    )
    parser.add_argument("input_csv", type=Path, help="CSV with id, ra, dec, size columns")
    parser.add_argument("output_dir", type=Path, help="Directory where FITS cutouts will be saved")
    parser.add_argument(
        "--size-unit",
        choices=["arcsec", "arcmin", "deg"],
        default="arcsec",
        help="Unit of the size column (default: arcsec)",
    )
    parser.add_argument(
        "--size-mode",
        choices=["radius", "diameter"],
        default="diameter",
        help="Interpret 'size' as a radius or diameter (default: diameter)",
    )
    parser.add_argument(
        "--instrument",
        default="VIS",
        help="Instrument to select in q1.mosaic_product (default: VIS)",
    )
    parser.add_argument(
        "--filter-name",
        default=None,
        help="Optional filter_name restriction in q1.mosaic_product",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.2,
        help="Pause in seconds between downloads (default: 0.2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite FITS files that already exist",
    )
    parser.add_argument(
        "--summary-name",
        default="download_summary.csv",
        help="Output summary CSV filename (default: download_summary.csv)",
    )
    return parser.parse_args()


def size_to_radius(size: float, unit_name: str, mode: str) -> u.Quantity:
    unit_map = {
        "arcsec": u.arcsec,
        "arcmin": u.arcmin,
        "deg": u.deg,
    }
    q = size * unit_map[unit_name]
    if mode == "diameter":
        q = q / 2.0
    return q


def load_catalogue(path: Path) -> list[Target]:
    df = pd.read_csv(path)
    required = {"id", "ra", "dec", "size"}
    missing = required - set(df.columns.str.lower())

    # Allow case-insensitive column names by normalizing.
    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    targets: list[Target] = []
    for _, row in df.iterrows():
        targets.append(
            Target(
                obj_id=str(row["id"]),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                size=float(row["size"]),
            )
        )
    return targets


def adql_float(x: float) -> str:
    return f"{x:.10f}"


def find_covering_mosaic(
    ra: float,
    dec: float,
    radius_deg: float,
    instrument: str,
    filter_name: Optional[str] = None,
):
    where = [
        f"instrument_name='{instrument}'",
        (
            "INTERSECTS("
            f"CIRCLE({adql_float(ra)}, {adql_float(dec)}, {radius_deg:.10f}), fov"
            ")=1"
        ),
    ]
    if filter_name:
        where.append(f"filter_name='{filter_name}'")

    query = f"""
        SELECT file_name, file_path, instrument_name, filter_name,
               creation_date, tile_index, patch_id_list
        FROM q1.mosaic_product
        WHERE {' AND '.join(where)}
        ORDER BY creation_date DESC
    """
    LOG.debug("ADQL query:\n%s", query)
    job = Euclid.launch_job_async(query)
    return job.get_results()


def download_cutout(
    *,
    file_path: str,
    ra: float,
    dec: float,
    radius: u.Quantity,
    output_file: Path,
):
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    result = Euclid.get_cutout(
        file_path=file_path,
        coordinate=coord,
        radius=radius,
        output_file=str(output_file),
        instrument="None",
        id="None",
    )
    if not result:
        raise RuntimeError("Euclid.get_cutout returned no output path")
    return Path(result[0])


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        Euclid.get_status_messages()
    except Exception as exc:
        LOG.warning("Could not read Euclid service status: %s", exc)

    try:
        targets = load_catalogue(args.input_csv)
    except Exception as exc:
        LOG.error("Failed to read catalogue: %s", exc)
        return 1

    results_rows = []

    for i, target in enumerate(targets, start=1):
        safe_id = sanitize_filename(target.obj_id)
        radius = size_to_radius(target.size, args.size_unit, args.size_mode)
        radius_deg = radius.to(u.deg).value
        outname = f"{safe_id}_ra{target.ra:.6f}_dec{target.dec:.6f}_{args.instrument}_cutout.fits"
        outfile = args.output_dir / outname

        LOG.info("[%d/%d] id=%s ra=%.6f dec=%.6f radius=%s", i, len(targets), target.obj_id, target.ra, target.dec, radius)

        if outfile.exists() and not args.overwrite:
            LOG.info("File exists, skipping: %s", outfile)
            results_rows.append(
                {
                    "id": target.obj_id,
                    "ra": target.ra,
                    "dec": target.dec,
                    "size": target.size,
                    "radius_arcsec": radius.to(u.arcsec).value,
                    "status": "skipped_exists",
                    "output_file": str(outfile),
                    "selected_file_name": "",
                    "selected_file_path": "",
                    "message": "file already exists",
                }
            )
            continue

        try:
            res = find_covering_mosaic(
                ra=target.ra,
                dec=target.dec,
                radius_deg=radius_deg,
                instrument=args.instrument,
                filter_name=args.filter_name,
            )

            if len(res) == 0:
                raise RuntimeError("No covering mosaic found in q1.mosaic_product")

            selected = res[0]
            full_file_path = f"{selected['file_path']}/{selected['file_name']}"

            saved = download_cutout(
                file_path=full_file_path,
                ra=target.ra,
                dec=target.dec,
                radius=radius,
                output_file=outfile,
            )

            results_rows.append(
                {
                    "id": target.obj_id,
                    "ra": target.ra,
                    "dec": target.dec,
                    "size": target.size,
                    "radius_arcsec": radius.to(u.arcsec).value,
                    "status": "ok",
                    "output_file": str(saved),
                    "selected_file_name": str(selected["file_name"]),
                    "selected_file_path": str(selected["file_path"]),
                    "message": "",
                }
            )
            LOG.info("Saved: %s", saved)

        except Exception as exc:
            LOG.exception("Failed for id=%s", target.obj_id)
            results_rows.append(
                {
                    "id": target.obj_id,
                    "ra": target.ra,
                    "dec": target.dec,
                    "size": target.size,
                    "radius_arcsec": radius.to(u.arcsec).value,
                    "status": "error",
                    "output_file": str(outfile),
                    "selected_file_name": "",
                    "selected_file_path": "",
                    "message": str(exc),
                }
            )

        if args.pause > 0:
            time.sleep(args.pause)

    summary_path = args.output_dir / args.summary_name
    pd.DataFrame(results_rows).to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    LOG.info("Wrote summary: %s", summary_path)

    n_ok = sum(r["status"] == "ok" for r in results_rows)
    n_err = sum(r["status"] == "error" for r in results_rows)
    n_skip = sum(r["status"] == "skipped_exists" for r in results_rows)
    LOG.info("Done. ok=%d error=%d skipped=%d", n_ok, n_err, n_skip)
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

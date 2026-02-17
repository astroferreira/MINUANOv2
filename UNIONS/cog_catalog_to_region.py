#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path


HEADER_RE = re.compile(r"^#\s*(\d+)\s+([A-Za-z0-9_]+)\b")


def parse_column_map(catalog_path):
    column_map = {}
    with catalog_path.open("r", encoding="ascii", errors="ignore") as fin:
        for line in fin:
            if not line.startswith("#"):
                break
            match = HEADER_RE.match(line)
            if not match:
                continue
            index_1based = int(match.group(1))
            name = match.group(2)
            column_map[name] = index_1based - 1
    return column_map


def parse_float(parts, index):
    if index is None or index >= len(parts):
        return None
    try:
        return float(parts[index])
    except Exception:
        return None


def theta_j2000_to_ds9(theta_j2000_deg):
    # SExtractor THETA_J2000 is measured east of north.
    # DS9 ellipse angles are measured from the local +X axis, so convert.
    theta_ds9 = 90.0 - float(theta_j2000_deg)
    return ((theta_ds9 + 180.0) % 360.0) - 180.0


def create_ds9_regions(catalog_path, output_path, min_radius_arcsec=0.10):
    column_map = parse_column_map(catalog_path)

    ra_idx = column_map.get("ALPHA_J2000")
    dec_idx = column_map.get("DELTA_J2000")
    a_world_idx = column_map.get("A_WORLD")
    b_world_idx = column_map.get("B_WORLD")
    theta_idx = column_map.get("THETA_J2000")
    kron_idx = column_map.get("KRON_RADIUS")
    fwhm_world_idx = column_map.get("FWHM_WORLD")

    missing = [name for name, idx in [("ALPHA_J2000", ra_idx), ("DELTA_J2000", dec_idx)] if idx is None]
    if missing:
        raise RuntimeError(f"Missing required columns: {', '.join(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    written_rows = 0

    with (
        catalog_path.open("r", encoding="ascii", errors="ignore") as fin,
        output_path.open("w", encoding="ascii") as fout,
    ):
        fout.write("# Region file format: DS9 version 4.1\n")
        fout.write(
            'global color=cyan dashlist=8 3 width=1 font="helvetica 10 normal" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        fout.write("fk5\n")

        for line in fin:
            if not line.strip() or line.startswith("#"):
                continue

            total_rows += 1
            parts = line.split()

            ra = parse_float(parts, ra_idx)
            dec = parse_float(parts, dec_idx)
            if not (isinstance(ra, float) and isinstance(dec, float)):
                continue
            if not (math.isfinite(ra) and math.isfinite(dec)):
                continue

            theta = parse_float(parts, theta_idx)
            if not (isinstance(theta, float) and math.isfinite(theta)):
                theta = 0.0
            else:
                theta = theta_j2000_to_ds9(theta)

            kron = parse_float(parts, kron_idx)
            kron_scale = kron if (isinstance(kron, float) and math.isfinite(kron) and kron > 0.0) else 1.0

            a_world = parse_float(parts, a_world_idx)
            b_world = parse_float(parts, b_world_idx)

            if (
                isinstance(a_world, float)
                and isinstance(b_world, float)
                and math.isfinite(a_world)
                and math.isfinite(b_world)
                and a_world > 0.0
                and b_world > 0.0
            ):
                a_arcsec = max(a_world * 3600.0 * kron_scale, min_radius_arcsec)
                b_arcsec = max(b_world * 3600.0 * kron_scale, min_radius_arcsec)
                fout.write(
                    f'ellipse({ra:.8f},{dec:.8f},{a_arcsec:.3f}",{b_arcsec:.3f}",{theta:.2f})\n'
                )
                written_rows += 1
                continue

            fwhm_world = parse_float(parts, fwhm_world_idx)
            if isinstance(fwhm_world, float) and math.isfinite(fwhm_world) and fwhm_world > 0.0:
                radius = max((fwhm_world * 3600.0) / 2.0, min_radius_arcsec)
                fout.write(
                    f'ellipse({ra:.8f},{dec:.8f},{radius:.3f}",{radius:.3f}",{theta:.2f})\n'
                )
                written_rows += 1

    return total_rows, written_rows, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert CFIS COG ASCII catalog to DS9 region ellipses."
    )
    parser.add_argument(
        "--catalog",
        default="catalogs_perseus",
        help="Input COG catalog (ASCII table with # column header lines)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output DS9 region path (default: <catalog>.reg)",
    )
    parser.add_argument(
        "--min-radius-arcsec",
        type=float,
        default=0.10,
        help="Minimum ellipse semi-axis in arcsec",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    output_path = Path(args.output) if args.output else Path(f"{catalog_path}.reg")

    total_rows, written_rows, output = create_ds9_regions(
        catalog_path, output_path, min_radius_arcsec=args.min_radius_arcsec
    )
    print(f"catalog={catalog_path}")
    print(f"rows_total={total_rows}")
    print(f"rows_written={written_rows}")
    print(f"region={output}")


if __name__ == "__main__":
    main()

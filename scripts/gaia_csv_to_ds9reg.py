#!/usr/bin/env python3
"""Convert a source catalog CSV into a SAOImage DS9 region file.

Expected columns by default: xcentroid_pix, ycentroid_pix,
sc_equivalent_radius, sc_elongation, sc_orientation.
Here sc_equivalent_radius is used as semi-major axis ``a`` and
sc_elongation is interpreted as ``b/a`` (so ``b = a * (b/a)``).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a catalog CSV to a DS9 ellipse region file."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="catalog_source_extraction_mfmtk.csv",
        help="Input catalog CSV path (default: catalog_source_extraction_mfmtk.csv)",
    )
    parser.add_argument(
        "output_reg",
        nargs="?",
        default=None,
        help="Output DS9 region file path (default: input filename with .reg extension)",
    )
    parser.add_argument(
        "--x-col", default="xcentroid_pix", help="Column name for x center"
    )
    parser.add_argument(
        "--y-col", default="ycentroid_pix", help="Column name for y center"
    )
    parser.add_argument(
        "--a-col",
        default="sc_equivalent_radius",
        help="Column name for semi-major axis a",
    )
    parser.add_argument(
        "--ba-col",
        default="sc_elongation",
        help="Column name for axis ratio b/a (semi-minor over semi-major)",
    )
    parser.add_argument(
        "--pa-col",
        default="sc_orientation",
        help="Column name for ellipse position angle in degrees",
    )
    return parser.parse_args()


def _to_float(value: str) -> float:
    if value is None:
        raise ValueError("missing value")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite value")
    return parsed


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_reg) if args.output_reg else input_path.with_suffix(".reg")

    if not input_path.exists():
        print(f"Error: input CSV not found: {input_path}", file=sys.stderr)
        return 1

    required = [args.x_col, args.y_col, args.a_col, args.ba_col, args.pa_col]

    count_written = 0
    count_skipped = 0

    with input_path.open("r", newline="") as fin:
        reader = csv.DictReader(fin)
        missing_cols = [c for c in required if c not in (reader.fieldnames or [])]
        if missing_cols:
            print(
                f"Error: missing required column(s): {', '.join(missing_cols)}",
                file=sys.stderr,
            )
            return 2

        with output_path.open("w", newline="") as fout:
            fout.write("# Region file format: DS9 version 4.1\n")
            fout.write("global color=green dashlist=8 3 width=1 font='helvetica 10 normal' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
            fout.write("image\n")

            for rownum, row in enumerate(reader, start=2):
                try:
                    x = _to_float(row.get(args.x_col, ""))
                    y = _to_float(row.get(args.y_col, ""))
                    a = _to_float(row.get(args.a_col, ""))
                    b_over_a = _to_float(row.get(args.ba_col, ""))
                    pa = _to_float(row.get(args.pa_col, ""))
                    if a <= 0 or b_over_a <= 0:
                        raise ValueError("semi-axis and axis ratio must be > 0")
                    b = a * b_over_a
                    # Catalog orientation is 90 deg offset from DS9's ellipse angle convention.
                    pa = (pa + 90.0) % 180.0
                except ValueError:
                    count_skipped += 1
                    continue

                fout.write(f"ellipse({x:.6f},{y:.6f},{a:.6f},{b:.6f},{pa:.6f})\n")
                count_written += 1

    print(
        f"Wrote {count_written} regions to {output_path} "
        f"(skipped {count_skipped} invalid rows)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

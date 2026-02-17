#!/usr/bin/env python3
"""
Build a CFIS sky-coverage footprint map from a tile list text file.

Accepted line formats include:
  - CFIS_LSB.042.345.r.fits
  - CFIS_LSB.042.345.r.cat
  - 042.345
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from make_catalog_footprint_map import make_svg_footprint


PATTERNS = [
    re.compile(r"CFIS_LSB\.(\d{3})\.(\d{3})\.r\.", re.IGNORECASE),
    re.compile(r"\b(\d{3})\.(\d{3})\b"),
]


def parse_tile_tuple(line: str) -> tuple[int, int] | None:
    text = line.strip()
    if not text:
        return None
    for pattern in PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def load_tiles(tile_list_path: Path) -> set[tuple[int, int]]:
    tiles: set[tuple[int, int]] = set()
    with tile_list_path.open("r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            tile = parse_tile_tuple(line)
            if tile is not None:
                tiles.add(tile)
    return tiles


def cfis_tile_from_radec(ra: float, dec: float) -> tuple[int, int]:
    yyy = int(round((dec + 90.0) * 2.0))
    cosf = math.cos((yyy / 2.0 - 90.0) * math.pi / 180.0)
    xxx = int(round(ra * 2.0 * cosf))
    return xxx, yyy


def parse_efigi_rows(efigi_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    header_names: list[str] = []
    rows: list[dict[str, str]] = []

    with efigi_path.open("r", encoding="utf-8", errors="ignore") as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("# PGC_name"):
                    header_names = line[1:].strip().split()
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            row: dict[str, str] = {}
            if header_names and len(parts) >= len(header_names):
                for idx, name in enumerate(header_names):
                    row[name] = parts[idx]
            else:
                # Fallback to a minimal schema.
                row["PGC_name"] = parts[0]
                row["RA"] = parts[1]
                row["DEC"] = parts[2]
                if len(parts) > 9:
                    row["z_all"] = parts[9]

            rows.append(row)

    if not rows:
        raise RuntimeError(f"No EFIGI galaxy rows parsed from: {efigi_path}")
    return rows, header_names


def filter_efigi_inside_tiles(
    rows: list[dict[str, str]],
    tiles: set[tuple[int, int]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    inside: list[dict[str, str]] = []
    outside: list[dict[str, str]] = []

    for row in rows:
        try:
            ra = float(row["RA"])
            dec = float(row["DEC"])
        except Exception:
            outside.append(dict(row))
            continue

        xxx, yyy = cfis_tile_from_radec(ra, dec)
        tile_id = f"{xxx:03d}.{yyy:03d}"

        out_row = dict(row)
        out_row["CFIS_TILE"] = tile_id
        out_row["IN_CFIS"] = "1" if (xxx, yyy) in tiles else "0"

        if (xxx, yyy) in tiles:
            inside.append(out_row)
        else:
            outside.append(out_row)

    return inside, outside


def write_rows_csv(path: Path, rows: list[dict[str, str]], preferred_order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    discovered = {key for row in rows for key in row.keys()}
    ordered = [name for name in preferred_order if name in discovered]
    ordered.extend(sorted(name for name in discovered if name not in ordered))

    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def overlay_galaxies_on_svg(svg_path: Path, galaxies_inside: list[dict[str, str]]) -> None:
    if not galaxies_inside:
        return
    text = svg_path.read_text(encoding="ascii", errors="ignore")

    width = 1800
    height = 980
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_from_ra(ra_deg: float) -> float:
        return margin_left + (360.0 - (ra_deg % 360.0)) / 360.0 * plot_w

    def y_from_dec(dec_deg: float) -> float:
        return margin_top + (90.0 - dec_deg) / 180.0 * plot_h

    additions: list[str] = []
    additions.append('<g id="efigi_inside_cfis">')
    for row in galaxies_inside:
        try:
            ra = float(row["RA"])
            dec = float(row["DEC"])
        except Exception:
            continue
        x = x_from_ra(ra)
        y = y_from_dec(dec)
        if x < margin_left or x > margin_left + plot_w:
            continue
        if y < margin_top or y > margin_top + plot_h:
            continue
        additions.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.8" fill="#dc2626" fill-opacity="0.85" stroke="none"/>'
        )
    additions.append("</g>")
    additions.append(
        '<rect x="1540" y="86" width="12" height="12" fill="#dc2626" fill-opacity="0.85" stroke="none"/>'
    )
    additions.append(
        '<text x="1558" y="96" text-anchor="start" font-family="Helvetica, Arial, sans-serif" '
        'font-size="12" fill="#334155">EFIGI inside CFIS</text>'
    )
    additions.append(
        f'<text x="1558" y="112" text-anchor="start" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="12" fill="#334155">N={len(galaxies_inside)}</text>'
    )

    marker_block = "\n".join(additions) + "\n</svg>"
    text = text.replace("</svg>", marker_block)
    svg_path.write_text(text, encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CFIS sky coverage SVG from a tile list text file."
    )
    parser.add_argument(
        "--tiles-list",
        default="all_tiles.txt",
        help="Text file listing CFIS tiles or filenames (default: all_tiles.txt)",
    )
    parser.add_argument(
        "--output",
        default="cats/CFIS_all_tiles_coverage.svg",
        help="Output SVG path (default: cats/CFIS_all_tiles_coverage.svg)",
    )
    parser.add_argument(
        "--title",
        default="CFIS Available Tiles Coverage",
        help="Map title",
    )
    parser.add_argument(
        "--efigi-catalog",
        default="",
        help="Optional EFIGI text catalog path to filter against CFIS footprint",
    )
    parser.add_argument(
        "--efigi-output-inside",
        default="cats/EFIGI_inside_CFIS.csv",
        help="CSV output for EFIGI galaxies inside CFIS footprint",
    )
    parser.add_argument(
        "--efigi-output-outside",
        default="cats/EFIGI_outside_CFIS.csv",
        help="CSV output for EFIGI galaxies outside CFIS footprint",
    )
    args = parser.parse_args()

    tile_list_path = Path(args.tiles_list)
    output_path = Path(args.output)
    if not tile_list_path.exists():
        raise FileNotFoundError(f"Tile list file not found: {tile_list_path}")

    tiles = load_tiles(tile_list_path)
    if not tiles:
        raise RuntimeError(f"No CFIS tile IDs found in: {tile_list_path}")

    make_svg_footprint(tiles, output_path, args.title)

    print(f"tiles_list={tile_list_path}")
    print(f"unique_tiles={len(tiles)}")
    print(f"output={output_path}")

    if args.efigi_catalog:
        efigi_path = Path(args.efigi_catalog)
        if not efigi_path.exists():
            raise FileNotFoundError(f"EFIGI catalog not found: {efigi_path}")

        efigi_rows, header_names = parse_efigi_rows(efigi_path)
        inside_rows, outside_rows = filter_efigi_inside_tiles(efigi_rows, tiles)

        preferred = list(header_names) if header_names else ["PGC_name", "RA", "DEC", "z_all"]
        preferred += ["CFIS_TILE", "IN_CFIS"]
        inside_path = Path(args.efigi_output_inside)
        outside_path = Path(args.efigi_output_outside)
        write_rows_csv(inside_path, inside_rows, preferred_order=preferred)
        write_rows_csv(outside_path, outside_rows, preferred_order=preferred)

        overlay_galaxies_on_svg(output_path, inside_rows)

        print(f"efigi_catalog={efigi_path}")
        print(f"efigi_total={len(efigi_rows)}")
        print(f"efigi_inside_cfis={len(inside_rows)}")
        print(f"efigi_outside_cfis={len(outside_rows)}")
        print(f"efigi_inside_output={inside_path}")
        print(f"efigi_outside_output={outside_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import math
import re
import struct
from pathlib import Path


BLOCK = 2880


def read_header(fobj):
    cards = []
    while True:
        block = fobj.read(BLOCK)
        if not block:
            return None
        if len(block) != BLOCK:
            raise RuntimeError("Truncated FITS header block")
        cards.extend(block[i : i + 80].decode("ascii") for i in range(0, BLOCK, 80))
        if any(card.startswith("END") for card in cards[-36:]):
            break

    header = {}
    for card in cards:
        key = card[:8].strip()
        if not key or key in ("END", "COMMENT", "HISTORY"):
            continue
        if card[8:10] != "= ":
            continue
        raw = card[10:].split("/", 1)[0].strip()
        if raw.startswith("'") and raw.endswith("'"):
            val = raw[1:-1].strip()
        elif raw in ("T", "F"):
            val = raw == "T"
        else:
            try:
                if any(ch in raw for ch in ".EeDd"):
                    val = float(raw.replace("D", "E"))
                else:
                    val = int(raw)
            except Exception:
                val = raw
        header[key] = val
    return header


def round_up_2880(size):
    return ((size + BLOCK - 1) // BLOCK) * BLOCK


def tform_size(tform):
    match = re.fullmatch(r"(\d*)([LXBIJKAEDCMPQ])", tform)
    if not match:
        raise ValueError(f"Unsupported TFORM: {tform}")

    n = int(match.group(1) or "1")
    code = match.group(2)
    if code in ("L", "X", "B", "A"):
        return n
    if code == "I":
        return 2 * n
    if code == "J":
        return 4 * n
    if code == "K":
        return 8 * n
    if code == "E":
        return 4 * n
    if code == "D":
        return 8 * n
    if code == "C":
        return 8 * n
    if code == "M":
        return 16 * n
    if code == "P":
        return 8
    if code == "Q":
        return 16
    raise ValueError(f"Unsupported TFORM: {tform}")


def cfis_tile_from_radec(ra, dec):
    yyy = int(round((dec + 90.0) * 2.0))
    cosf = math.cos((yyy / 2.0 - 90.0) * math.pi / 180.0)
    xxx = int(round(ra * 2.0 * cosf))
    return xxx, yyy


def extract_unique_tiles(catalog_path):
    tiles = set()

    with catalog_path.open("rb") as fin:
        primary = read_header(fin)
        if primary is None:
            raise RuntimeError("Could not read PRIMARY HDU")

        # Skip primary data payload (usually none for this catalog).
        naxis = int(primary.get("NAXIS", 0))
        bitpix = abs(int(primary.get("BITPIX", 8)))
        pcount = int(primary.get("PCOUNT", 0))
        gcount = int(primary.get("GCOUNT", 1))
        size = 0
        if naxis > 0:
            size = 1
            for i in range(1, naxis + 1):
                size *= int(primary.get(f"NAXIS{i}", 0))
            size = size * bitpix // 8
        size = (size + pcount) * gcount
        if size:
            fin.seek(round_up_2880(size), 1)

        table = read_header(fin)
        if table is None or table.get("XTENSION") != "BINTABLE":
            raise RuntimeError("Expected BINTABLE in extension 1")

        row_len = int(table["NAXIS1"])
        nrows = int(table["NAXIS2"])
        tfields = int(table["TFIELDS"])

        offset = 0
        ra_offset = None
        dec_offset = None
        for idx in range(1, tfields + 1):
            ttype = table.get(f"TTYPE{idx}")
            tform = table.get(f"TFORM{idx}")
            if ttype == "RA":
                ra_offset = offset
            if ttype == "DEC":
                dec_offset = offset
            offset += tform_size(tform)

        if ra_offset is None or dec_offset is None:
            raise RuntimeError("RA/DEC columns not found")
        if offset != row_len:
            raise RuntimeError(f"Row length mismatch: computed {offset}, header {row_len}")

        rows_per_chunk = 200000
        for start in range(0, nrows, rows_per_chunk):
            take = min(rows_per_chunk, nrows - start)
            raw = fin.read(take * row_len)
            if len(raw) != take * row_len:
                raise RuntimeError(f"Truncated table data near row {start}")

            for i in range(take):
                base = i * row_len
                ra = struct.unpack_from(">d", raw, base + ra_offset)[0]
                dec = struct.unpack_from(">d", raw, base + dec_offset)[0]
                if not (math.isfinite(ra) and math.isfinite(dec)):
                    continue
                tiles.add(cfis_tile_from_radec(ra, dec))

    return tiles


def tile_center_radec(tile):
    xxx, yyy = tile
    dec = yyy / 2.0 - 90.0
    cosf = math.cos(math.radians(dec))
    if abs(cosf) < 1.0e-9:
        return None
    ra = (xxx / (2.0 * cosf)) % 360.0
    return ra, dec


def split_wrap_segments(ra_min, ra_max):
    if ra_min < 0.0:
        return [(ra_min + 360.0, 360.0), (0.0, ra_max)]
    if ra_max >= 360.0:
        return [(ra_min, 360.0), (0.0, ra_max - 360.0)]
    return [(ra_min, ra_max)]


def make_svg_footprint(tiles, output_path, title):
    width = 1800
    height = 980
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_from_ra(ra_deg):
        # RA increases to the left in this display.
        return margin_left + (360.0 - ra_deg) / 360.0 * plot_w

    def y_from_dec(dec_deg):
        return margin_top + (90.0 - dec_deg) / 180.0 * plot_h

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<text x="{width/2:.1f}" y="36" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#0f172a">'
        f"{title}</text>"
    )

    # Grid
    for dec in range(-75, 91, 15):
        y = y_from_dec(float(dec))
        color = "#cbd5e1" if dec != 0 else "#94a3b8"
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#475569">'
            f"{dec} deg</text>"
        )

    for ra in range(0, 361, 30):
        x = x_from_ra(float(ra))
        color = "#cbd5e1"
        lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_h}" '
            f'stroke="{color}" stroke-width="1"/>'
        )
        label = f"{ra} deg"
        lines.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_h + 22}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#475569">'
            f"{label}</text>"
        )

    lines.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#64748b" stroke-width="1.5"/>'
    )

    # Footprint: draw approximate 0.5 deg x 0.5 deg tile cells.
    fill = "#0ea5e9"
    for tile in sorted(tiles, key=lambda v: (v[1], v[0])):
        center = tile_center_radec(tile)
        if center is None:
            continue
        ra_c, dec_c = center
        cosf = math.cos(math.radians(dec_c))
        if abs(cosf) < 1.0e-6:
            continue

        dec_half = 0.25
        ra_half = 0.25 / abs(cosf)
        dec_min = max(-90.0, dec_c - dec_half)
        dec_max = min(90.0, dec_c + dec_half)
        ra_min = ra_c - ra_half
        ra_max = ra_c + ra_half

        for seg_min, seg_max in split_wrap_segments(ra_min, ra_max):
            x1 = x_from_ra(seg_min)
            x2 = x_from_ra(seg_max)
            y1 = y_from_dec(dec_min)
            y2 = y_from_dec(dec_max)
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w <= 0.0 or h <= 0.0:
                continue
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'fill="{fill}" fill-opacity="0.65" stroke="none"/>'
            )

    lines.append(
        f'<text x="{margin_left}" y="{margin_top - 20}" text-anchor="start" '
        'font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#334155">'
        f"Unique CFIS tiles: {len(tiles)}</text>"
    )
    lines.append(
        f'<text x="{margin_left + plot_w}" y="{margin_top - 20}" text-anchor="end" '
        'font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#334155">'
        "Projection: Equatorial (RA/Dec), equirectangular</text>"
    )
    lines.append(
        f'<text x="{margin_left + plot_w/2:.2f}" y="{height - 16}" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#475569">'
        "Right Ascension (degrees, increasing to the left)</text>"
    )
    lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a sky footprint SVG for CFIS catalog coverage."
    )
    parser.add_argument(
        "--catalog",
        default="catalogs/CFIS_DR5_DESI_DR1_share_r.fits",
        help="Input FITS catalog",
    )
    parser.add_argument(
        "--output",
        default="catalogs/CFIS_DR5_DESI_DR1_share_r_footprint.svg",
        help="Output SVG path",
    )
    parser.add_argument(
        "--title",
        default="CFIS_DR5_DESI_DR1_share_r Catalog Footprint",
        help="Map title",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    output_path = Path(args.output)
    tiles = extract_unique_tiles(catalog_path)
    make_svg_footprint(tiles, output_path, args.title)

    print(f"catalog={catalog_path}")
    print(f"unique_tiles={len(tiles)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import math
import re
import struct
import tempfile
from pathlib import Path


BLOCK = 2880


def read_header_cards_and_bytes(fobj):
    cards = []
    raw = bytearray()
    while True:
        block = fobj.read(BLOCK)
        if not block:
            return None, None
        if len(block) != BLOCK:
            raise RuntimeError("Truncated FITS header block")
        raw.extend(block)
        cards.extend(block[i : i + 80].decode("ascii") for i in range(0, BLOCK, 80))
        if any(card.startswith("END") for card in cards[-36:]):
            return cards, bytes(raw)


def parse_card_value(card):
    if card[8:10] != "= ":
        return None

    raw = card[10:]
    value_part = raw.split("/", 1)[0].strip()
    if value_part.startswith("'") and value_part.endswith("'"):
        return value_part[1:-1].strip()
    if value_part == "T":
        return True
    if value_part == "F":
        return False

    try:
        if any(ch in value_part for ch in ".EeDd"):
            return float(value_part.replace("D", "E"))
        return int(value_part)
    except Exception:
        return value_part


def cards_to_dict(cards):
    out = {}
    for card in cards:
        key = card[:8].strip()
        if not key or key in ("END", "COMMENT", "HISTORY"):
            continue
        val = parse_card_value(card)
        if val is not None:
            out[key] = val
    return out


def split_comment(card):
    if card[8:10] != "= ":
        return None
    rest = card[10:]
    if "/" in rest:
        return rest.split("/", 1)[1].strip()
    return None


def make_int_card(key, value, comment=None):
    text = f"{key:<8}= {value:>20d}"
    if comment:
        text += f" / {comment}"
    return text[:80].ljust(80)


def build_updated_extension_header(cards, rows_selected):
    out_cards = []
    replaced_naxis2 = False

    for card in cards:
        key = card[:8].strip()
        if key == "END":
            continue
        if key in ("CHECKSUM", "DATASUM"):
            continue
        if key == "NAXIS2":
            out_cards.append(make_int_card("NAXIS2", rows_selected, split_comment(card)))
            replaced_naxis2 = True
            continue
        out_cards.append(card)

    if not replaced_naxis2:
        out_cards.append(make_int_card("NAXIS2", rows_selected))

    out_cards.append("END".ljust(80))
    payload = "".join(out_cards).encode("ascii")
    pad = (BLOCK - (len(payload) % BLOCK)) % BLOCK
    return payload + (b" " * pad)


def round_up_2880(size):
    return ((size + BLOCK - 1) // BLOCK) * BLOCK


def tform_size(tform):
    match = re.fullmatch(r"(\d*)([LXBIJKAEDCMPQ])", tform)
    if not match:
        raise ValueError(f"Unsupported TFORM value: {tform}")

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

    raise ValueError(f"Unsupported TFORM value: {tform}")


def unpack_scalar_value(tform, raw):
    match = re.fullmatch(r"(\d*)([LXBIJKAEDCMPQ])", tform)
    if not match:
        return None

    n = int(match.group(1) or "1")
    code = match.group(2)

    if code == "D" and n == 1:
        return struct.unpack(">d", raw)[0]
    if code == "E" and n == 1:
        return struct.unpack(">f", raw)[0]
    if code == "J" and n == 1:
        return struct.unpack(">i", raw)[0]
    if code == "K" and n == 1:
        return struct.unpack(">q", raw)[0]
    return None


def parse_tile_id(tile_text):
    match = re.search(r"(\d{3})\.(\d{3})", tile_text)
    if not match:
        raise ValueError(
            "Tile must contain xxx.yyy, e.g. 068.167 or CFIS_LSB.068.167.r.fits"
        )
    return int(match.group(1)), int(match.group(2))


def source_tile(ra, dec):
    yyy = int(round((dec + 90.0) * 2.0))
    cosf = math.cos((yyy / 2.0 - 90.0) * math.pi / 180.0)
    xxx = int(round(ra * 2.0 * cosf))
    return xxx, yyy


def theta_j2000_to_ds9(theta_j2000_deg):
    # SExtractor THETA_J2000 is measured east of north.
    # DS9 ellipse angles are measured from the local +X axis, so convert.
    theta_ds9 = 90.0 - float(theta_j2000_deg)
    return ((theta_ds9 + 180.0) % 360.0) - 180.0


def write_ds9_region(region_path, region_rows_file):
    region_path.parent.mkdir(parents=True, exist_ok=True)
    with region_path.open("w", encoding="ascii") as reg:
        reg.write("# Region file format: DS9 version 4.1\n")
        reg.write(
            'global color=green dashlist=8 3 width=1 font="helvetica 10 normal" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        reg.write("fk5\n")
        region_rows_file.seek(0)
        while True:
            chunk = region_rows_file.read(1024 * 1024)
            if not chunk:
                break
            reg.write(chunk)


def copy_tile_subset(catalog_path, tile_xxx, tile_yyy, output_path, region_path):
    with catalog_path.open("rb") as fin:
        primary_cards, primary_header_bytes = read_header_cards_and_bytes(fin)
        if primary_cards is None:
            raise RuntimeError("Could not read PRIMARY header")

        primary = cards_to_dict(primary_cards)
        if int(primary.get("NAXIS", 0)) != 0:
            raise RuntimeError("Expected NAXIS=0 in PRIMARY HDU for this catalog")

        ext_cards, _ext_header_bytes = read_header_cards_and_bytes(fin)
        if ext_cards is None:
            raise RuntimeError("Could not read table extension header")

        ext = cards_to_dict(ext_cards)
        if ext.get("XTENSION") != "BINTABLE":
            raise RuntimeError("Expected first extension to be BINTABLE")

        row_len = int(ext["NAXIS1"])
        nrows = int(ext["NAXIS2"])
        tfields = int(ext["TFIELDS"])

        offset = 0
        ra_offset = None
        dec_offset = None
        a_world_info = None
        b_world_info = None
        theta_info = None
        kron_info = None
        for idx in range(1, tfields + 1):
            ttype = ext.get(f"TTYPE{idx}")
            tform = ext.get(f"TFORM{idx}")
            if ttype == "RA":
                ra_offset = offset
            if ttype == "DEC":
                dec_offset = offset
            if ttype == "A_WORLD":
                a_world_info = (tform, offset)
            if ttype == "B_WORLD":
                b_world_info = (tform, offset)
            if ttype == "THETA_J2000":
                theta_info = (tform, offset)
            if ttype == "KRON_RADIUS":
                kron_info = (tform, offset)
            offset += tform_size(tform)

        if offset != row_len:
            raise RuntimeError(f"Row size mismatch: computed {offset}, header {row_len}")
        if ra_offset is None or dec_offset is None:
            raise RuntimeError("RA/DEC columns not found in table")

        rows_selected = 0
        rows_per_chunk = 200000
        table_data_bytes = nrows * row_len
        table_data_pad = round_up_2880(table_data_bytes) - table_data_bytes

        with (
            tempfile.TemporaryFile() as temp_rows,
            tempfile.TemporaryFile(mode="w+t", encoding="ascii") as temp_regions,
        ):
            for start in range(0, nrows, rows_per_chunk):
                take = min(rows_per_chunk, nrows - start)
                raw = fin.read(take * row_len)
                if len(raw) != take * row_len:
                    raise RuntimeError(f"Truncated table data at row {start}")

                for i in range(take):
                    base = i * row_len
                    ra = struct.unpack_from(">d", raw, base + ra_offset)[0]
                    dec = struct.unpack_from(">d", raw, base + dec_offset)[0]
                    if not (math.isfinite(ra) and math.isfinite(dec)):
                        continue

                    xxx, yyy = source_tile(ra, dec)
                    if xxx == tile_xxx and yyy == tile_yyy:
                        temp_rows.write(raw[base : base + row_len])

                        # Build a sky ellipse in arcsec. Use Kron-scaled semiaxes
                        # when available, otherwise fall back to A/B_WORLD directly.
                        a_arcsec = None
                        b_arcsec = None
                        theta = 0.0
                        kron = 1.0

                        if kron_info is not None:
                            kron_raw = raw[base + kron_info[1] : base + kron_info[1] + tform_size(kron_info[0])]
                            kron_value = unpack_scalar_value(kron_info[0], kron_raw)
                            if isinstance(kron_value, (int, float)) and math.isfinite(kron_value) and kron_value > 0:
                                kron = float(kron_value)

                        if a_world_info is not None:
                            a_raw = raw[base + a_world_info[1] : base + a_world_info[1] + tform_size(a_world_info[0])]
                            a_world = unpack_scalar_value(a_world_info[0], a_raw)
                            if isinstance(a_world, (int, float)) and math.isfinite(a_world) and a_world > 0:
                                a_arcsec = float(a_world) * 3600.0 * kron

                        if b_world_info is not None:
                            b_raw = raw[base + b_world_info[1] : base + b_world_info[1] + tform_size(b_world_info[0])]
                            b_world = unpack_scalar_value(b_world_info[0], b_raw)
                            if isinstance(b_world, (int, float)) and math.isfinite(b_world) and b_world > 0:
                                b_arcsec = float(b_world) * 3600.0 * kron

                        if theta_info is not None:
                            t_raw = raw[base + theta_info[1] : base + theta_info[1] + tform_size(theta_info[0])]
                            t_val = unpack_scalar_value(theta_info[0], t_raw)
                            if isinstance(t_val, (int, float)) and math.isfinite(t_val):
                                theta = theta_j2000_to_ds9(float(t_val))

                        if a_arcsec is None or b_arcsec is None:
                            temp_regions.write(f"point({ra:.8f},{dec:.8f}) # point=circle\n")
                        else:
                            # Keep a minimum display radius so tiny sources remain visible.
                            a_arcsec = max(a_arcsec, 0.10)
                            b_arcsec = max(b_arcsec, 0.10)
                            temp_regions.write(
                                f'ellipse({ra:.8f},{dec:.8f},{a_arcsec:.3f}",{b_arcsec:.3f}",{theta:.2f})\n'
                            )
                        rows_selected += 1

            if table_data_pad:
                fin.seek(table_data_pad, 1)

            ext_header_bytes = build_updated_extension_header(ext_cards, rows_selected)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as fout:
                fout.write(primary_header_bytes)
                fout.write(ext_header_bytes)
                temp_rows.seek(0)
                while True:
                    chunk = temp_rows.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)

                data_bytes = rows_selected * row_len
                pad = (BLOCK - (data_bytes % BLOCK)) % BLOCK
                if pad:
                    fout.write(b"\x00" * pad)

            write_ds9_region(region_path, temp_regions)

    return rows_selected, nrows


def main():
    parser = argparse.ArgumentParser(
        description="Extract rows for one CFIS tile into a new FITS table."
    )
    parser.add_argument(
        "--catalog",
        default="catalogs/CFIS_DR5_DESI_DR1_share_r.fits",
        help="Input FITS catalog path",
    )
    parser.add_argument(
        "--tile",
        required=True,
        help="Tile id (xxx.yyy) or full tile filename containing xxx.yyy",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output FITS path (default: catalogs/<catalog_stem>_tile_<xxx.yyy>.fits)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Output DS9 region path (default: same as --output with .reg extension)",
    )
    args = parser.parse_args()

    tile_xxx, tile_yyy = parse_tile_id(args.tile)
    tile_id = f"{tile_xxx:03d}.{tile_yyy:03d}"

    catalog_path = Path(args.catalog)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = catalog_path.with_name(f"{catalog_path.stem}_tile_{tile_id}.fits")

    if args.region:
        region_path = Path(args.region)
    else:
        region_path = output_path.with_suffix(".reg")

    selected, total = copy_tile_subset(
        catalog_path, tile_xxx, tile_yyy, output_path, region_path
    )
    print(f"tile={tile_id}")
    print(f"rows_selected={selected}")
    print(f"rows_total={total}")
    print(f"output={output_path}")
    print(f"region={region_path}")


if __name__ == "__main__":
    main()

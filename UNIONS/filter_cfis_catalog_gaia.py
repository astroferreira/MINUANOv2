#!/usr/bin/env python3
"""
Filter a CFIS COG ASCII catalog by removing entries matched to Gaia sources.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord


HEADER_RE = re.compile(r"^#\s*(\d+)\s+([A-Za-z0-9_]+)\b")


@dataclass
class CatalogRow:
    line: str
    ra: float | None = None
    dec: float | None = None
    mag_auto: float | None = None
    mag_apauto: float | None = None
    fwhm_world: float | None = None
    prediq: float | None = None
    a_world: float | None = None
    b_world: float | None = None
    kron_radius: float | None = None
    flags: float | None = None
    mag_cog: float | None = None
    mag_2arc: float | None = None
    predmaglim: float | None = None
    magerr_cog: float | None = None
    magerr_2arc: float | None = None
    magerr_auto: float | None = None


@dataclass
class GaiaSource:
    ra: float
    dec: float
    mag: float


@dataclass
class QualityFilterConfig:
    flags_max: float
    point_min_criteria: int
    point_delta_mag_max: float
    point_fwhm_ratio_max: float
    point_axis_ratio_min: float
    point_kron_max: float
    faint_margin_mag: float
    faint_magerr_cog_max: float
    faint_magerr_auto_max: float
    disable_point_filter: bool
    disable_faint_filter: bool


SB_ERR_TO_SNR = 1.0857362047581296


def mag_to_radius_arcsec(mag, r0=8.0, mag0=15.0, min_r=1.0, max_r=40.0):
    """
    Raio em arcsec ~ r0 * 10^(-0.2*(mag-mag0)).
    Brilho maior (mag menor) => raio maior.
    Clampa em [min_r, max_r].
    """
    m = np.array(mag, dtype=float)
    r = r0 * (10.0 ** (-0.2 * (m - mag0)))
    return np.clip(r, min_r, max_r)


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


def parse_catalog(catalog_path: Path) -> tuple[list[str], list[CatalogRow]]:
    column_map: dict[str, int] = {}
    header_lines: list[str] = []
    rows: list[CatalogRow] = []

    with catalog_path.open("r", encoding="ascii", errors="ignore") as fin:
        idx: dict[str, int | None] | None = None
        for line in fin:
            if line.startswith("#"):
                header_lines.append(line)
                match = HEADER_RE.match(line)
                if match:
                    index_1based = int(match.group(1))
                    name = match.group(2)
                    column_map[name] = index_1based - 1
                continue
            if idx is None:
                idx = {
                    "ra": column_map.get("ALPHA_J2000", column_map.get("RA")),
                    "dec": column_map.get("DELTA_J2000", column_map.get("DEC")),
                    "mag_auto": column_map.get("MAG_AUTO"),
                    "mag_apauto": column_map.get("MAG_APAUTO"),
                    "fwhm_world": column_map.get("FWHM_WORLD"),
                    "prediq": column_map.get("PREDIQ"),
                    "a_world": column_map.get("A_WORLD"),
                    "b_world": column_map.get("B_WORLD"),
                    "kron_radius": column_map.get("KRON_RADIUS"),
                    "flags": column_map.get("FLAGS"),
                    "mag_cog": column_map.get("MAG_COG"),
                    "mag_2arc": column_map.get("MAG_2ARC"),
                    "predmaglim": column_map.get("PREDMAGLIM"),
                    "magerr_cog": column_map.get("MAGERR_COG"),
                    "magerr_2arc": column_map.get("MAGERR_2ARC"),
                    "magerr_auto": column_map.get("MAGERR_AUTO"),
                }
                if idx["ra"] is None or idx["dec"] is None:
                    raise RuntimeError(
                        "Could not find ALPHA_J2000/DELTA_J2000 (or RA/DEC) columns"
                    )

            row = CatalogRow(line=line)
            if line.strip():
                parts = line.split()
                row.ra = parse_float(parts, idx["ra"])
                row.dec = parse_float(parts, idx["dec"])
                row.mag_auto = parse_float(parts, idx["mag_auto"])
                row.mag_apauto = parse_float(parts, idx["mag_apauto"])
                row.fwhm_world = parse_float(parts, idx["fwhm_world"])
                row.prediq = parse_float(parts, idx["prediq"])
                row.a_world = parse_float(parts, idx["a_world"])
                row.b_world = parse_float(parts, idx["b_world"])
                row.kron_radius = parse_float(parts, idx["kron_radius"])
                row.flags = parse_float(parts, idx["flags"])
                row.mag_cog = parse_float(parts, idx["mag_cog"])
                row.mag_2arc = parse_float(parts, idx["mag_2arc"])
                row.predmaglim = parse_float(parts, idx["predmaglim"])
                row.magerr_cog = parse_float(parts, idx["magerr_cog"])
                row.magerr_2arc = parse_float(parts, idx["magerr_2arc"])
                row.magerr_auto = parse_float(parts, idx["magerr_auto"])
            rows.append(row)

    return header_lines, rows


def header_max_index(header_lines: list[str]) -> int:
    max_idx = 0
    for line in header_lines:
        match = HEADER_RE.match(line)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx


def build_clean_header_lines(header_lines: list[str]) -> list[str]:
    out = list(header_lines)
    next_idx = header_max_index(header_lines) + 1
    extra_columns = [
        ("AREA_KRON_ARCSEC2", "Kron ellipse area [arcsec**2]"),
        ("MU_2ARC", "SB from MAG_2ARC in 1 arcsec radius aperture [mag/arcsec**2]"),
        ("SNR_SB_2ARC", "SB SNR proxy from MAGERR_2ARC"),
        ("MU_COG_KRON", "SB from MAG_COG and Kron area [mag/arcsec**2]"),
        ("SNR_SB_COG", "SB SNR proxy from MAGERR_COG"),
    ]
    for name, description in extra_columns:
        out.append(f"# {next_idx:3d} {name:<16} {description}\n")
        next_idx += 1
    return out


def compute_sb_metrics(
    row: CatalogRow,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    area_kron_arcsec2 = None
    if (
        row.a_world is not None
        and row.b_world is not None
        and row.kron_radius is not None
        and row.a_world > 0.0
        and row.b_world > 0.0
        and row.kron_radius > 0.0
    ):
        a_arcsec = row.a_world * 3600.0 * row.kron_radius
        b_arcsec = row.b_world * 3600.0 * row.kron_radius
        if a_arcsec > 0.0 and b_arcsec > 0.0:
            area_kron_arcsec2 = math.pi * a_arcsec * b_arcsec

    mu_2arc = None
    if row.mag_2arc is not None:
        # MAG_2ARC is through a 2" diameter aperture => radius = 1".
        mu_2arc = row.mag_2arc + 2.5 * math.log10(math.pi)

    snr_sb_2arc = None
    if row.magerr_2arc is not None and row.magerr_2arc > 0.0:
        snr_sb_2arc = SB_ERR_TO_SNR / row.magerr_2arc

    mu_cog_kron = None
    if row.mag_cog is not None and area_kron_arcsec2 is not None and area_kron_arcsec2 > 0.0:
        mu_cog_kron = row.mag_cog + 2.5 * math.log10(area_kron_arcsec2)

    snr_sb_cog = None
    if row.magerr_cog is not None and row.magerr_cog > 0.0:
        snr_sb_cog = SB_ERR_TO_SNR / row.magerr_cog

    return area_kron_arcsec2, mu_2arc, snr_sb_2arc, mu_cog_kron, snr_sb_cog


def format_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def footprint_from_rows(rows: list[CatalogRow], padding_deg: float) -> tuple[float, float, float]:
    valid = [(r.ra, r.dec) for r in rows if r.ra is not None and r.dec is not None]
    if not valid:
        raise RuntimeError("Catalog has no valid sky coordinates to query Gaia")

    ra_values = [ra for ra, _ in valid]
    dec_values = [dec for _, dec in valid]

    coords = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg, frame="icrs")
    cart = coords.cartesian
    center_cart = SkyCoord(
        x=cart.x.mean(),
        y=cart.y.mean(),
        z=cart.z.mean(),
        representation_type="cartesian",
        frame="icrs",
    )
    center = SkyCoord(
        ra=center_cart.spherical.lon,
        dec=center_cart.spherical.lat,
        frame="icrs",
    )

    max_sep = center.separation(coords).max().deg
    query_radius = max_sep + padding_deg
    return float(center.ra.deg), float(center.dec.deg), float(query_radius)


def get_case_insensitive(row: dict[str, str], key: str) -> str | None:
    key_l = key.lower()
    for current_key, value in row.items():
        if current_key is not None and current_key.lower() == key_l:
            return value
    return None


def parse_gaia_csv_text(csv_text: str) -> list[GaiaSource]:
    rows: list[GaiaSource] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        ra_raw = get_case_insensitive(row, "ra")
        dec_raw = get_case_insensitive(row, "dec")
        if ra_raw is None or dec_raw is None:
            continue
        try:
            ra = float(ra_raw)
            dec = float(dec_raw)
        except Exception:
            continue
        if math.isfinite(ra) and math.isfinite(dec):
            mag_raw = (
                get_case_insensitive(row, "phot_g_mean_mag")
                or get_case_insensitive(row, "gmag")
                or get_case_insensitive(row, "mag")
            )
            try:
                mag = float(mag_raw) if mag_raw is not None else float("nan")
            except Exception:
                mag = float("nan")
            rows.append(GaiaSource(ra=ra, dec=dec, mag=mag))
    return rows


def fetch_gaia_csv(
    tap_url: str,
    gaia_table: str,
    center_ra: float,
    center_dec: float,
    query_radius_deg: float,
    maxrec: int,
    timeout_s: int,
) -> str:
    adql = (
        f"SELECT TOP {maxrec} source_id, ra, dec, phot_g_mean_mag "
        f"FROM {gaia_table} "
        f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {center_ra:.10f}, {center_dec:.10f}, {query_radius_deg:.10f}))"
    )
    payload = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "MAXREC": str(maxrec),
        "QUERY": adql,
    }
    response = requests.post(tap_url, data=payload, timeout=timeout_s)
    response.raise_for_status()
    return response.text


def load_gaia_sources(
    gaia_csv_path: Path | None,
    tap_url: str,
    gaia_table: str,
    center_ra: float,
    center_dec: float,
    query_radius_deg: float,
    maxrec: int,
    timeout_s: int,
) -> list[GaiaSource]:
    if gaia_csv_path is not None:
        csv_text = gaia_csv_path.read_text(encoding="utf-8")
        return parse_gaia_csv_text(csv_text)

    csv_text = fetch_gaia_csv(
        tap_url=tap_url,
        gaia_table=gaia_table,
        center_ra=center_ra,
        center_dec=center_dec,
        query_radius_deg=query_radius_deg,
        maxrec=maxrec,
        timeout_s=timeout_s,
    )
    return parse_gaia_csv_text(csv_text)


def write_gaia_region(
    region_path: Path,
    gaia_rows: list[GaiaSource],
    gaia_radius_arcsec: np.ndarray,
    indices: set[int],
) -> None:
    region_path.parent.mkdir(parents=True, exist_ok=True)
    with region_path.open("w", encoding="ascii") as reg:
        reg.write("# Region file format: DS9 version 4.1\n")
        reg.write(
            'global color=yellow dashlist=8 3 width=1 font="helvetica 10 normal" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        reg.write("fk5\n")
        for idx in sorted(indices):
            src = gaia_rows[idx]
            radius = float(gaia_radius_arcsec[idx])
            reg.write(f'circle({src.ra:.8f},{src.dec:.8f},{radius:.3f}")\n')


def filter_rows(
    rows: list[CatalogRow],
    gaia_rows: list[GaiaSource],
    gaia_radius_r0_arcsec: float,
    gaia_radius_mag0: float,
    gaia_radius_min_arcsec: float,
    gaia_radius_max_arcsec: float,
) -> tuple[list[CatalogRow], int, int, float, float, np.ndarray, set[int]]:
    valid_indices = [idx for idx, row in enumerate(rows) if row.ra is not None and row.dec is not None]
    valid_ra = [rows[idx].ra for idx in valid_indices]
    valid_dec = [rows[idx].dec for idx in valid_indices]

    if not valid_indices or not gaia_rows:
        return rows, len(valid_indices), 0, 0.0, 0.0, np.array([], dtype=np.float64), set()

    gaia_ra_rad = np.radians(np.array([row.ra for row in gaia_rows], dtype=np.float64))
    gaia_dec_rad = np.radians(np.array([row.dec for row in gaia_rows], dtype=np.float64))
    gaia_mag = np.array([row.mag for row in gaia_rows], dtype=np.float64)
    gaia_mag = np.where(np.isfinite(gaia_mag), gaia_mag, gaia_radius_mag0)
    gaia_radius_arcsec = mag_to_radius_arcsec(
        gaia_mag,
        r0=gaia_radius_r0_arcsec,
        mag0=gaia_radius_mag0,
        min_r=gaia_radius_min_arcsec,
        max_r=gaia_radius_max_arcsec,
    ).astype(np.float64)
    gaia_radius_rad = np.radians(gaia_radius_arcsec / 3600.0)

    # Sort Gaia by Dec so each catalog row only evaluates nearby candidates.
    order = np.argsort(gaia_dec_rad)
    gaia_dec_sorted = gaia_dec_rad[order]
    gaia_ra_sorted = gaia_ra_rad[order]
    gaia_radius_sorted = gaia_radius_rad[order]
    gaia_index_sorted = order
    max_radius_rad = float(np.max(gaia_radius_sorted))

    matched_catalog_indices: set[int] = set()
    matched_gaia_indices: set[int] = set()
    for local_idx, cat_idx in enumerate(valid_indices):
        ra0 = math.radians(valid_ra[local_idx])
        dec0 = math.radians(valid_dec[local_idx])

        lo = int(np.searchsorted(gaia_dec_sorted, dec0 - max_radius_rad, side="left"))
        hi = int(np.searchsorted(gaia_dec_sorted, dec0 + max_radius_rad, side="right"))
        if lo >= hi:
            continue

        cand_dec = gaia_dec_sorted[lo:hi]
        cand_ra = gaia_ra_sorted[lo:hi]
        cand_radius = gaia_radius_sorted[lo:hi]
        cand_idx = gaia_index_sorted[lo:hi]

        delta_ra = np.abs(cand_ra - ra0)
        delta_ra = np.minimum(delta_ra, 2.0 * np.pi - delta_ra)

        # Quick RA prefilter using the maximum Gaia exclusion radius.
        cos_dec0 = abs(math.cos(dec0))
        if cos_dec0 > 1e-8:
            ra_limit = min(math.pi, max_radius_rad / cos_dec0)
            keep = delta_ra <= ra_limit
            if not np.any(keep):
                continue
            cand_dec = cand_dec[keep]
            delta_ra = delta_ra[keep]
            cand_radius = cand_radius[keep]
            cand_idx = cand_idx[keep]

        sin_ddec = np.sin((cand_dec - dec0) * 0.5)
        sin_dra = np.sin(delta_ra * 0.5)
        a = sin_ddec * sin_ddec + math.cos(dec0) * np.cos(cand_dec) * sin_dra * sin_dra
        sep = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(a, 0.0))))
        matched_here = sep <= cand_radius
        if np.any(matched_here):
            matched_catalog_indices.add(cat_idx)
            matched_gaia_indices.update(int(i) for i in cand_idx[matched_here])

    filtered = [row for idx, row in enumerate(rows) if idx not in matched_catalog_indices]
    return (
        filtered,
        len(valid_indices),
        len(matched_catalog_indices),
        float(np.min(gaia_radius_arcsec)),
        float(np.max(gaia_radius_arcsec)),
        gaia_radius_arcsec,
        matched_gaia_indices,
    )


def apply_post_gaia_filters(
    rows: list[CatalogRow],
    cfg: QualityFilterConfig,
) -> tuple[list[CatalogRow], dict[str, int]]:
    clean_rows: list[CatalogRow] = []

    n_flags_bad = 0
    n_point_like = 0
    n_faint_bad = 0
    n_removed = 0

    for row in rows:
        flags_bad = False
        if cfg.flags_max >= 0 and row.flags is not None:
            flags_bad = row.flags > cfg.flags_max

        point_like = False
        if not cfg.disable_point_filter:
            point_votes = 0
            point_available = 0

            if row.mag_apauto is not None and row.mag_auto is not None:
                point_available += 1
                if (row.mag_apauto - row.mag_auto) < cfg.point_delta_mag_max:
                    point_votes += 1

            if (
                row.fwhm_world is not None
                and row.prediq is not None
                and row.prediq > 0.0
                and row.fwhm_world > 0.0
            ):
                point_available += 1
                fwhm_world_arcsec = row.fwhm_world * 3600.0
                if (fwhm_world_arcsec / row.prediq) < cfg.point_fwhm_ratio_max:
                    point_votes += 1

            if row.a_world is not None and row.b_world is not None and row.a_world > 0.0:
                point_available += 1
                if (row.b_world / row.a_world) > cfg.point_axis_ratio_min:
                    point_votes += 1

            if row.kron_radius is not None and row.kron_radius > 0.0:
                point_available += 1
                if row.kron_radius <= cfg.point_kron_max:
                    point_votes += 1

            if point_available >= cfg.point_min_criteria and point_votes >= cfg.point_min_criteria:
                point_like = True

        faint_bad = False
        if not cfg.disable_faint_filter:
            faint_criteria = []
            if row.mag_cog is not None and row.predmaglim is not None:
                faint_criteria.append(row.mag_cog > (row.predmaglim - cfg.faint_margin_mag))
            if row.magerr_cog is not None:
                faint_criteria.append(row.magerr_cog > cfg.faint_magerr_cog_max)
            if row.magerr_auto is not None:
                faint_criteria.append(row.magerr_auto > cfg.faint_magerr_auto_max)
            faint_bad = any(faint_criteria)

        removed = flags_bad or point_like or faint_bad
        if removed:
            n_removed += 1
            if flags_bad:
                n_flags_bad += 1
            if point_like:
                n_point_like += 1
            if faint_bad:
                n_faint_bad += 1
            continue

        clean_rows.append(row)

    stats = {
        "rows_input": len(rows),
        "rows_removed_total": n_removed,
        "rows_removed_flags": n_flags_bad,
        "rows_removed_point_like": n_point_like,
        "rows_removed_faint": n_faint_bad,
        "rows_written": len(clean_rows),
    }
    return clean_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-match a CFIS COG catalog with Gaia and remove matched stars."
    )
    parser.add_argument("--catalog", required=True, help="Input CFIS COG catalog path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output filtered catalog path (default: <catalog>.nogaia)",
    )
    parser.add_argument(
        "--stars-region",
        default=None,
        help="Optional DS9 region output containing matched Gaia stars only",
    )
    parser.add_argument(
        "--output-clean",
        default=None,
        help="Output catalog after Gaia + quality cuts (default: <output>.clean)",
    )
    parser.add_argument(
        "--gaia-csv",
        default=None,
        help="Optional Gaia CSV file with ra/dec columns (skip TAP query)",
    )
    parser.add_argument(
        "--tap-url",
        default="https://gea.esac.esa.int/tap-server/tap/sync",
        help="Gaia TAP sync endpoint",
    )
    parser.add_argument(
        "--gaia-table",
        default="gaiadr3.gaia_source",
        help="Gaia table to query",
    )
    parser.add_argument(
        "--gaia-radius-r0-arcsec",
        type=float,
        default=8.0,
        help="Reference exclusion radius r0 (arcsec) for Gaia star masking",
    )
    parser.add_argument(
        "--gaia-radius-mag0",
        type=float,
        default=15.0,
        help="Reference magnitude mag0 for Gaia star masking",
    )
    parser.add_argument(
        "--gaia-radius-min-arcsec",
        type=float,
        default=1.0,
        help="Minimum exclusion radius (arcsec) for Gaia star masking",
    )
    parser.add_argument(
        "--gaia-radius-max-arcsec",
        type=float,
        default=40.0,
        help="Maximum exclusion radius (arcsec) for Gaia star masking",
    )
    parser.add_argument(
        "--query-padding-deg",
        type=float,
        default=0.05,
        help="Added angular padding around catalog footprint for Gaia query",
    )
    parser.add_argument(
        "--max-query-radius-deg",
        type=float,
        default=2.0,
        help="Safety limit for computed Gaia query radius",
    )
    parser.add_argument(
        "--maxrec",
        type=int,
        default=3000000,
        help="Maximum Gaia rows returned by TAP query",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds for Gaia TAP request",
    )
    parser.add_argument(
        "--flags-max",
        type=float,
        default=3.0,
        help="Reject rows with FLAGS > this value (set <0 to disable)",
    )
    parser.add_argument(
        "--point-min-criteria",
        type=int,
        default=3,
        help="Minimum number of point-like criteria required to reject as point source",
    )
    parser.add_argument(
        "--point-delta-mag-max",
        type=float,
        default=0.3,
        help="Reject point-like if (MAG_APAUTO - MAG_AUTO) < this value",
    )
    parser.add_argument(
        "--point-fwhm-ratio-max",
        type=float,
        default=1.35,
        help="Reject point-like if (FWHM_WORLD*3600 / PREDIQ) < this value",
    )
    parser.add_argument(
        "--point-axis-ratio-min",
        type=float,
        default=0.75,
        help="Reject point-like if (B_WORLD / A_WORLD) > this value",
    )
    parser.add_argument(
        "--point-kron-max",
        type=float,
        default=3.8,
        help="Reject point-like if KRON_RADIUS <= this value",
    )
    parser.add_argument(
        "--faint-margin-mag",
        type=float,
        default=0.5,
        help="Reject faint if MAG_COG > (PREDMAGLIM - this margin)",
    )
    parser.add_argument(
        "--faint-magerr-cog-max",
        type=float,
        default=0.2,
        help="Reject faint if MAGERR_COG > this value",
    )
    parser.add_argument(
        "--faint-magerr-auto-max",
        type=float,
        default=0.2,
        help="Reject faint if MAGERR_AUTO > this value",
    )
    parser.add_argument(
        "--disable-point-filter",
        action="store_true",
        help="Disable point-source rejection stage",
    )
    parser.add_argument(
        "--disable-faint-filter",
        action="store_true",
        help="Disable faint-source rejection stage",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    output_path = Path(args.output) if args.output else Path(f"{catalog_path}.nogaia")
    gaia_csv_path = Path(args.gaia_csv) if args.gaia_csv else None
    stars_region_path = Path(args.stars_region) if args.stars_region else None
    output_clean_path = Path(args.output_clean) if args.output_clean else Path(f"{output_path}.clean")

    if args.gaia_radius_r0_arcsec <= 0:
        raise ValueError("--gaia-radius-r0-arcsec must be > 0")
    if args.gaia_radius_min_arcsec <= 0:
        raise ValueError("--gaia-radius-min-arcsec must be > 0")
    if args.gaia_radius_max_arcsec <= 0:
        raise ValueError("--gaia-radius-max-arcsec must be > 0")
    if args.gaia_radius_min_arcsec > args.gaia_radius_max_arcsec:
        raise ValueError("--gaia-radius-min-arcsec must be <= --gaia-radius-max-arcsec")
    if args.query_padding_deg < 0:
        raise ValueError("--query-padding-deg must be >= 0")
    if args.max_query_radius_deg <= 0:
        raise ValueError("--max-query-radius-deg must be > 0")
    if args.maxrec <= 0:
        raise ValueError("--maxrec must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.point_min_criteria <= 0:
        raise ValueError("--point-min-criteria must be > 0")
    if args.point_fwhm_ratio_max <= 0:
        raise ValueError("--point-fwhm-ratio-max must be > 0")
    if args.point_axis_ratio_min <= 0:
        raise ValueError("--point-axis-ratio-min must be > 0")
    if args.point_kron_max <= 0:
        raise ValueError("--point-kron-max must be > 0")

    header_lines, rows = parse_catalog(catalog_path)
    center_ra, center_dec, query_radius_deg = footprint_from_rows(rows, args.query_padding_deg)
    if query_radius_deg > args.max_query_radius_deg:
        raise RuntimeError(
            f"Computed Gaia query radius {query_radius_deg:.4f} deg exceeds safety limit "
            f"{args.max_query_radius_deg:.4f} deg"
        )

    gaia_sources = load_gaia_sources(
        gaia_csv_path=gaia_csv_path,
        tap_url=args.tap_url,
        gaia_table=args.gaia_table,
        center_ra=center_ra,
        center_dec=center_dec,
        query_radius_deg=query_radius_deg,
        maxrec=args.maxrec,
        timeout_s=args.timeout,
    )

    (
        filtered_rows,
        n_valid_rows,
        n_removed,
        gaia_radius_min_used,
        gaia_radius_max_used,
        gaia_radius_arcsec,
        matched_gaia_indices,
    ) = filter_rows(
        rows=rows,
        gaia_rows=gaia_sources,
        gaia_radius_r0_arcsec=args.gaia_radius_r0_arcsec,
        gaia_radius_mag0=args.gaia_radius_mag0,
        gaia_radius_min_arcsec=args.gaia_radius_min_arcsec,
        gaia_radius_max_arcsec=args.gaia_radius_max_arcsec,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as fout:
        for line in header_lines:
            fout.write(line)
        for row in filtered_rows:
            fout.write(row.line)

    quality_cfg = QualityFilterConfig(
        flags_max=args.flags_max,
        point_min_criteria=args.point_min_criteria,
        point_delta_mag_max=args.point_delta_mag_max,
        point_fwhm_ratio_max=args.point_fwhm_ratio_max,
        point_axis_ratio_min=args.point_axis_ratio_min,
        point_kron_max=args.point_kron_max,
        faint_margin_mag=args.faint_margin_mag,
        faint_magerr_cog_max=args.faint_magerr_cog_max,
        faint_magerr_auto_max=args.faint_magerr_auto_max,
        disable_point_filter=args.disable_point_filter,
        disable_faint_filter=args.disable_faint_filter,
    )
    clean_rows, clean_stats = apply_post_gaia_filters(filtered_rows, quality_cfg)

    clean_header_lines = build_clean_header_lines(header_lines)
    output_clean_path.parent.mkdir(parents=True, exist_ok=True)
    with output_clean_path.open("w", encoding="ascii") as fout:
        for line in clean_header_lines:
            fout.write(line)
        for row in clean_rows:
            (
                area_kron_arcsec2,
                mu_2arc,
                snr_sb_2arc,
                mu_cog_kron,
                snr_sb_cog,
            ) = compute_sb_metrics(row)
            fout.write(
                row.line.rstrip("\n")
                + " "
                + " ".join(
                    [
                        format_metric(area_kron_arcsec2),
                        format_metric(mu_2arc),
                        format_metric(snr_sb_2arc),
                        format_metric(mu_cog_kron),
                        format_metric(snr_sb_cog),
                    ]
                )
                + "\n"
            )

    if stars_region_path is not None:
        write_gaia_region(
            region_path=stars_region_path,
            gaia_rows=gaia_sources,
            gaia_radius_arcsec=gaia_radius_arcsec,
            indices=matched_gaia_indices,
        )

    print(f"catalog={catalog_path}")
    print(f"output={output_path}")
    print(f"output_clean={output_clean_path}")
    print(f"rows_total={len(rows)}")
    print(f"rows_valid_coords={n_valid_rows}")
    print(f"gaia_sources={len(gaia_sources)}")
    print(f"gaia_stars_matched={len(matched_gaia_indices)}")
    print(f"rows_removed={n_removed}")
    print(f"rows_written={len(filtered_rows)}")
    print(f"rows_removed_gaia={n_removed}")
    print(f"rows_written_gaia={len(filtered_rows)}")
    print(f"rows_removed_quality={clean_stats['rows_removed_total']}")
    print(f"rows_removed_quality_flags={clean_stats['rows_removed_flags']}")
    print(f"rows_removed_quality_point_like={clean_stats['rows_removed_point_like']}")
    print(f"rows_removed_quality_faint={clean_stats['rows_removed_faint']}")
    print(f"rows_written_clean={clean_stats['rows_written']}")
    if stars_region_path is not None:
        print(f"stars_region={stars_region_path}")
    print(
        "gaia_mask_radius_arcsec="
        f"[{gaia_radius_min_used:.3f},{gaia_radius_max_used:.3f}] "
        f"(r0={args.gaia_radius_r0_arcsec:.3f}, mag0={args.gaia_radius_mag0:.3f}, "
        f"clip=[{args.gaia_radius_min_arcsec:.3f},{args.gaia_radius_max_arcsec:.3f}])"
    )
    print(f"query_center_ra_deg={center_ra:.8f}")
    print(f"query_center_dec_deg={center_dec:.8f}")
    print(f"query_radius_deg={query_radius_deg:.8f}")


if __name__ == "__main__":
    main()

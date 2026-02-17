#!/usr/bin/env python3
"""
Query MAST for observations around a sky position and list JWST matches.

Examples:
  python MINUANOv2/UNIONS/query_jwst_mast.py --ra 150.116321 --dec 2.205830 --radius-arcsec 120
  python MINUANOv2/UNIONS/query_jwst_mast.py --ra 150.116321 --dec 2.205830 --radius-arcsec 60 --output jwst_matches.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_MAST_URL = "https://mast.stsci.edu/api/v0/invoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check MAST for JWST observations at a given RA/Dec cone search."
    )
    parser.add_argument("--ra", type=float, required=True, help="Right ascension in degrees")
    parser.add_argument("--dec", type=float, required=True, help="Declination in degrees")

    radius_group = parser.add_mutually_exclusive_group()
    radius_group.add_argument(
        "--radius-arcsec",
        type=float,
        default=120.0,
        help="Search radius in arcsec (default: 120)",
    )
    radius_group.add_argument(
        "--radius-deg",
        type=float,
        default=None,
        help="Search radius in degrees (overrides --radius-arcsec)",
    )

    parser.add_argument(
        "--mast-url",
        default=DEFAULT_MAST_URL,
        help=f"MAST API endpoint (default: {DEFAULT_MAST_URL})",
    )
    parser.add_argument(
        "--service",
        default="Mast.Caom.Cone",
        help="MAST service to query (default: Mast.Caom.Cone)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=2000,
        help="Rows per API page (default: 2000)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max pages to fetch (0 = all pages)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="HTTP timeout in seconds (default: 45)",
    )
    parser.add_argument(
        "--include-non-jwst",
        action="store_true",
        help="Do not filter to JWST; return all collections",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=20,
        help="Max rows to print to terminal (default: 20)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output CSV path",
    )
    parser.add_argument(
        "--fail-if-empty",
        action="store_true",
        help="Exit with code 1 if no rows are found after filtering",
    )
    return parser.parse_args()


def validate_inputs(ra: float, dec: float, radius_deg: float, page_size: int, max_pages: int) -> None:
    if not (0.0 <= ra < 360.0):
        raise ValueError("--ra must be in [0, 360)")
    if not (-90.0 <= dec <= 90.0):
        raise ValueError("--dec must be in [-90, 90]")
    if not math.isfinite(radius_deg) or radius_deg <= 0.0:
        raise ValueError("Search radius must be finite and > 0")
    if page_size <= 0:
        raise ValueError("--page-size must be > 0")
    if max_pages < 0:
        raise ValueError("--max-pages must be >= 0")


def mast_invoke(
    mast_url: str,
    service: str,
    params: dict[str, float],
    page: int,
    page_size: int,
    timeout_s: float,
) -> list[dict]:
    request_payload = {
        "service": service,
        "params": params,
        "format": "json",
        "pagesize": page_size,
        "page": page,
    }

    body = urllib.parse.urlencode({"request": json.dumps(request_payload)}).encode("utf-8")
    req = urllib.request.Request(mast_url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "application/json, text/plain, */*")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400].replace("\n", " ")
        raise RuntimeError(f"MAST HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"MAST connection error: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        snippet = raw[:400].decode("utf-8", errors="replace").replace("\n", " ")
        raise RuntimeError(f"Invalid JSON from MAST: {snippet}") from exc

    if isinstance(payload, dict):
        data = payload.get("data")
        if data is None:
            data = payload.get("Data")
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]

    raise RuntimeError("Unexpected MAST response structure")


def query_all_pages(
    mast_url: str,
    service: str,
    ra: float,
    dec: float,
    radius_deg: float,
    page_size: int,
    max_pages: int,
    timeout_s: float,
) -> list[dict]:
    params = {"ra": float(ra), "dec": float(dec), "radius": float(radius_deg)}
    rows: list[dict] = []

    page = 1
    while True:
        batch = mast_invoke(
            mast_url=mast_url,
            service=service,
            params=params,
            page=page,
            page_size=page_size,
            timeout_s=timeout_s,
        )
        rows.extend(batch)

        if not batch or len(batch) < page_size:
            break
        if max_pages > 0 and page >= max_pages:
            break
        page += 1

    return rows


def get_value(row: dict, *keys: str):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return ""


def is_jwst_row(row: dict) -> bool:
    value = str(get_value(row, "obs_collection", "obs_collection_name")).strip().upper()
    return value == "JWST"


def maybe_float(value) -> float:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return float("inf")


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            maybe_float(get_value(row, "distance", "dist")),
            maybe_float(get_value(row, "t_min")),
        ),
    )


def summarize(rows: list[dict]) -> tuple[int, int]:
    unique_obsids = set()
    unique_proposals = set()
    for row in rows:
        obsid = str(get_value(row, "obsid", "obs_id", "observation_id")).strip()
        proposal = str(get_value(row, "proposal_id", "proposal", "proposal_pi")).strip()
        if obsid:
            unique_obsids.add(obsid)
        if proposal:
            unique_proposals.add(proposal)
    return len(unique_obsids), len(unique_proposals)


def print_preview(rows: list[dict], limit: int) -> None:
    if limit <= 0 or not rows:
        return
    n = min(limit, len(rows))
    print(f"preview_rows={n}")
    for i, row in enumerate(rows[:n], start=1):
        print(
            f"{i:03d} "
            f"obs_id={get_value(row, 'obs_id', 'obsid', 'observation_id')} "
            f"proposal={get_value(row, 'proposal_id', 'proposal')} "
            f"instrument={get_value(row, 'instrument_name', 'instrument')} "
            f"filters={get_value(row, 'filters', 'filter')} "
            f"target={get_value(row, 'target_name')} "
            f"t_min={get_value(row, 't_min')} "
            f"distance={get_value(row, 'distance', 'dist')}"
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as fout:
            fout.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    radius_deg = args.radius_deg if args.radius_deg is not None else args.radius_arcsec / 3600.0
    validate_inputs(args.ra, args.dec, radius_deg, args.page_size, args.max_pages)

    try:
        all_rows = query_all_pages(
            mast_url=args.mast_url,
            service=args.service,
            ra=args.ra,
            dec=args.dec,
            radius_deg=radius_deg,
            page_size=args.page_size,
            max_pages=args.max_pages,
            timeout_s=args.timeout,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}")

    if args.include_non_jwst:
        selected_rows = all_rows
        selected_collection = "ALL"
    else:
        selected_rows = [row for row in all_rows if is_jwst_row(row)]
        selected_collection = "JWST"

    selected_rows = sort_rows(selected_rows)
    n_unique_obs, n_unique_proposal = summarize(selected_rows)

    print(f"query_ra_deg={args.ra:.8f}")
    print(f"query_dec_deg={args.dec:.8f}")
    print(f"query_radius_deg={radius_deg:.8f}")
    print(f"query_radius_arcsec={radius_deg * 3600.0:.3f}")
    print(f"rows_total_all_collections={len(all_rows)}")
    print(f"rows_selected_collection={selected_collection}")
    print(f"rows_selected={len(selected_rows)}")
    print(f"unique_observations={n_unique_obs}")
    print(f"unique_proposals={n_unique_proposal}")

    print_preview(selected_rows, args.print_limit)

    if args.output:
        out_path = Path(args.output)
        try:
            write_csv(out_path, selected_rows)
        except Exception as exc:
            raise SystemExit(f"ERROR: could not write CSV '{out_path}': {exc}")
        print(f"output_csv={out_path}")

    if args.fail_if_empty and not selected_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

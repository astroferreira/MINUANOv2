#!/usr/bin/env python3
"""
Download an SDSS g-band FITS cutout for a CFIS tile footprint via SkyServer SIAP.

This uses the SDSS SkyServer SIAP endpoint documented in SkyServer API pages.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import html
import math
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_ENDPOINTS = [
    "https://skyserver.sdss.org/dr18/SkyServerWS/SIAP/getSIAP",
    "https://skyserver.sdss.org/dr16/SkyServerWS/SIAP/getSIAP",
    "https://skyserver.sdss.org/dr12/SkyServerWS/SIAP/getSIAP",
]


def cfis_tile_center(tile_id: str) -> tuple[float, float]:
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


def build_siap_url(endpoint: str, ra: float, dec: float, size_deg: float, band: str) -> str:
    params = {
        "POS": f"{ra:.8f},{dec:.8f}",
        "SIZE": f"{size_deg:.6f}",
        "FORMAT": "image/fits",
        "bandpass": band,
    }
    return f"{endpoint}?{urllib.parse.urlencode(params)}"


def http_get(url: str, timeout: float = 60.0) -> tuple[bytes, str]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = resp.read()
        ctype = resp.headers.get("Content-Type", "")
    return payload, ctype


def looks_like_fits_bytes(payload: bytes) -> bool:
    if len(payload) < 80:
        return False
    head = payload[:2880]
    return b"SIMPLE  =" in head or b"XTENSION=" in head


def decode_fits_payload(payload: bytes, content_type: str) -> bytes | None:
    ctype = content_type.lower()
    if looks_like_fits_bytes(payload):
        return payload

    if payload.startswith(b"BZh"):
        try:
            unpacked = bz2.decompress(payload)
            if looks_like_fits_bytes(unpacked):
                return unpacked
        except Exception:
            pass

    if payload[:2] == b"\x1f\x8b":
        try:
            unpacked = gzip.decompress(payload)
            if looks_like_fits_bytes(unpacked):
                return unpacked
        except Exception:
            pass

    # Some services mislabel FITS as octet-stream.
    if ("fits" in ctype or "octet-stream" in ctype) and looks_like_fits_bytes(payload):
        return payload

    return None


def parse_votable_access_rows(payload: bytes, endpoint: str) -> list[tuple[str, str, str]]:
    """
    Return list of (access_url, image_format, bandpass) rows extracted from SIAP VOTable.
    """
    text = payload.decode("utf-8", errors="ignore")
    try:
        root = ET.fromstring(text)
    except Exception:
        return []

    rows_out: list[tuple[str, str, str]] = []
    for table in root.findall(".//{*}TABLE"):
        fields = table.findall("{*}FIELD")
        if not fields:
            continue

        access_candidates = []
        format_candidates = []
        band_candidates = []
        for idx, field in enumerate(fields):
            name = (field.get("name") or "").strip().lower()
            ucd = (field.get("ucd") or "").strip().lower()
            utype = (field.get("utype") or "").strip().lower()
            ident = " ".join([name, ucd, utype])

            if "image_accessreference" in ident or "accessreference" in ident or "access_url" in ident:
                access_candidates.append(idx)
            if "image_format" in ident or name == "format" or "mime" in ident:
                format_candidates.append(idx)
            if (
                "bandpass" in ident
                or name == "band"
                or name == "filter"
                or "em.wl" in ident
            ):
                band_candidates.append(idx)

        if not access_candidates:
            continue

        tabledata = table.find(".//{*}TABLEDATA")
        if tabledata is None:
            continue

        for tr in tabledata.findall("{*}TR"):
            vals = [((td.text or "").strip()) for td in tr.findall("{*}TD")]
            if not vals:
                continue

            access = ""
            for idx in access_candidates:
                if idx < len(vals) and vals[idx]:
                    access = vals[idx]
                    break
            if not access:
                continue

            image_format = ""
            for idx in format_candidates:
                if idx < len(vals) and vals[idx]:
                    image_format = vals[idx]
                    break

            bandpass = ""
            for idx in band_candidates:
                if idx < len(vals) and vals[idx]:
                    bandpass = vals[idx]
                    break

            access = urllib.parse.urljoin(endpoint, access)
            rows_out.append((access, image_format, bandpass))

    return rows_out


def extract_urls_from_xml_like(payload: bytes) -> list[str]:
    text = payload.decode("utf-8", errors="ignore")
    text = html.unescape(text)
    urls = re.findall(r"https?://[^\\s\"'<>]+", text)
    seen = set()
    ordered = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        ordered.append(u)
    return ordered


def select_fits_url(urls: list[str], endpoint: str) -> str | None:
    endpoint_base = endpoint.lower().split("?", 1)[0]
    for u in urls:
        ul = u.lower()
        if ul.split("?", 1)[0] == endpoint_base:
            continue
        if "getsiap?" in ul:
            continue
        if ul.endswith(".fits") or "format=image/fits" in ul or "fits" in ul:
            return u
    return urls[0] if urls else None


def infer_band_from_url(url: str) -> str:
    ul = url.lower()
    # SDSS frame naming usually includes frame-g-, frame-r-, etc.
    m = re.search(r"frame-([ugriz])-", ul)
    if m:
        return m.group(1)
    m = re.search(r"[?&](?:band|bands|filter)=([ugriz])(?:[&$])?", ul)
    if m:
        return m.group(1)
    return ""


def save_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SDSS SIAP FITS cutout covering one CFIS tile area."
    )
    parser.add_argument("--tile", required=True, help="CFIS tile id, e.g. 042.345")
    parser.add_argument("--band", default="g", help="SDSS bandpass (u,g,r,i,z), default g")
    parser.add_argument(
        "--size-deg",
        type=float,
        default=0.516,
        help="Requested SIAP angular size in degrees (default 0.516)",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINTS[0],
        help=f"Primary SIAP endpoint (default {DEFAULT_ENDPOINTS[0]})",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback endpoints (dr16/dr12)",
    )
    parser.add_argument("--outdir", default="sdss_tiles", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Do not download file")
    args = parser.parse_args()

    if args.size_deg <= 0:
        raise ValueError("--size-deg must be > 0")

    ra, dec = cfis_tile_center(args.tile)
    outpath = Path(args.outdir) / f"SDSS_{args.band}_{args.tile}.fits"

    endpoints = [args.endpoint]
    if not args.no_fallback:
        for ep in DEFAULT_ENDPOINTS:
            if ep not in endpoints:
                endpoints.append(ep)

    print(f"tile={args.tile}")
    print(f"center_ra_dec={ra:.8f},{dec:.8f}")
    print(f"band={args.band}")
    print(f"size_deg={args.size_deg}")
    print(f"output={outpath}")

    if args.dry_run:
        for endpoint in endpoints:
            siap_url = build_siap_url(endpoint, ra, dec, args.size_deg, args.band)
            print(f"query_url={siap_url}")
        print("result=dry_run_no_network")
        return

    errors = []
    for endpoint in endpoints:
        siap_url = build_siap_url(endpoint, ra, dec, args.size_deg, args.band)
        print(f"query_url={siap_url}")
        try:
            payload, ctype = http_get(siap_url)
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
            print(f"endpoint_error={endpoint} {exc}")
            continue

        decoded = decode_fits_payload(payload, ctype)
        if decoded is not None:
            save_bytes(outpath, decoded)
            print(f"selected_endpoint={endpoint}")
            print("result=direct_fits_response")
            print(f"saved={outpath}")
            return

        access_rows = parse_votable_access_rows(payload, endpoint)
        target_band = args.band.lower().strip()
        fits_url = None
        preferred = []
        fallback = []
        for url, image_format, bandpass in access_rows:
            bandpass_l = bandpass.lower().strip()
            inferred = infer_band_from_url(url)
            if target_band and (target_band in bandpass_l or inferred == target_band):
                preferred.append((url, image_format))
            else:
                fallback.append((url, image_format))

        for url, image_format in preferred + fallback:
            if "fits" in image_format.lower() or url.lower().endswith(".fits.bz2"):
                fits_url = url
                break
        if fits_url is None and (preferred or fallback):
            fits_url = (preferred + fallback)[0][0]

        if fits_url is None:
            urls = extract_urls_from_xml_like(payload)
            # Prefer target band URLs when possible.
            band_urls = [u for u in urls if infer_band_from_url(u) == target_band]
            if band_urls:
                urls = band_urls
            fits_url = select_fits_url(urls, endpoint)

        if not fits_url:
            errors.append(f"{endpoint}: no FITS URL in SIAP response")
            continue

        print(f"access_url={fits_url}")
        try:
            fits_payload, fits_ctype = http_get(fits_url)
            decoded_access = decode_fits_payload(fits_payload, fits_ctype)
            if decoded_access is None:
                errors.append(f"{endpoint}: access URL did not return FITS")
                continue
            save_bytes(outpath, decoded_access)
            print(f"selected_endpoint={endpoint}")
            print("result=siap_access_url")
            print(f"saved={outpath}")
            return
        except Exception as exc:
            errors.append(f"{endpoint}: failed downloading access URL ({exc})")
            continue

    raise RuntimeError(
        "Could not download SDSS FITS cutout from any SIAP endpoint.\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


if __name__ == "__main__":
    main()

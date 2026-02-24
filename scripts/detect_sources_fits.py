#!/usr/bin/env python3
"""Detect sources in a FITS image using a photutils-style segmentation workflow."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

# Default SourceCatalog-derived parameters written to the "extended" catalog.
# Update this list to change which photutils SourceCatalog quantities are exported.
# "centroid" expands to "xcentroid" and "ycentroid".
# "r50" is computed as fluxfrac_radius(0.5).
DEFAULT_EXTENDED_SOURCECAT_PARAMS = [
    "area",
    "centroid",
    "elongation",
    "equivalent_radius",
    "kron_radius",
    "orientation",
    "r50",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a FITS image and detect sources via thresholded segmentation."
    )
    parser.add_argument("fits_path", type=Path, help="Input FITS file path")
    parser.add_argument(
        "--hdu",
        default=0,
        help="HDU index or name to read (default: 0)",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=3.0,
        help="Detection threshold in sigma above background (default: 3.0)",
    )
    parser.add_argument(
        "--npixels",
        type=int,
        default=5,
        help="Minimum connected pixels per source (default: 5)",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Pixel connectivity for segmentation (4 or 8; default: 8)",
    )
    parser.add_argument(
        "--smooth-fwhm",
        type=float,
        default=2.0,
        help="Gaussian smoothing FWHM in pixels before detection (default: 2.0)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable Gaussian smoothing before detection",
    )
    parser.add_argument(
        "--segm-out",
        type=Path,
        default=None,
        help="Optional output FITS path for segmentation labels",
    )
    parser.add_argument(
        "--segm-dtype",
        choices=("auto", "int16", "int32"),
        default="auto",
        help="Segmentation FITS integer dtype (default: auto => int16 if possible else int32)",
    )
    parser.add_argument(
        "--basic-catalog-out",
        "--catalog-out",
        dest="basic_catalog_out",
        type=Path,
        default=None,
        help="Optional output CSV path for basic source summary",
    )
    parser.add_argument(
        "--extended-catalog-out",
        "--source-catalog-out",
        dest="extended_catalog_out",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path for selected SourceCatalog-derived "
            "parameters (extended catalog)"
        ),
    )
    parser.add_argument(
        "--merged-catalog-out",
        type=Path,
        default=None,
        help=(
            "Optional output CSV merging the basic catalog with extended "
            "SourceCatalog columns (prefixed with sc_) using label"
        ),
    )
    parser.add_argument(
        "--ds9-region-out",
        type=Path,
        default=None,
        help=(
            "Optional SAOImage DS9 region file with source ellipses "
            "(centroid/orientation/elongation/equivalent_radius)"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Worker threads for per-source row serialization/merge (default: 1). "
            "Source detection itself is handled by photutils and remains serial."
        ),
    )
    return parser.parse_args()


def load_2d_fits_data(path: Path, hdu: str | int) -> np.ndarray:
    from astropy.io import fits

    data = fits.getdata(path, hdu=hdu)
    arr = np.asarray(data, dtype=float)

    if arr.ndim == 2:
        return arr
    if arr.ndim > 2:
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            return np.asarray(squeezed, dtype=float)
    raise ValueError(f"Expected a 2D image, got shape {arr.shape}")


def compute_catalog(segm_data: np.ndarray) -> list[dict[str, float | int]]:
    """Fast simple catalog using grouped reductions over labeled pixels."""
    yy, xx = np.nonzero(segm_data > 0)
    if xx.size == 0:
        return []

    labels = segm_data[yy, xx].astype(np.int64, copy=False)
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    xx = xx[order]
    yy = yy[order]

    starts = np.concatenate(([0], np.flatnonzero(np.diff(labels)) + 1))
    uniq = labels[starts]
    ends = np.concatenate((starts[1:], [labels.size]))
    counts = ends - starts

    xsum = np.add.reduceat(xx.astype(np.float64, copy=False), starts)
    ysum = np.add.reduceat(yy.astype(np.float64, copy=False), starts)
    xmin = np.minimum.reduceat(xx, starts)
    xmax = np.maximum.reduceat(xx, starts)
    ymin = np.minimum.reduceat(yy, starts)
    ymax = np.maximum.reduceat(yy, starts)

    rows: list[dict[str, float | int]] = []
    for i, label in enumerate(uniq):
        npix = int(counts[i])
        rows.append(
            {
                "label": int(label),
                "npix": npix,
                "xcentroid_pix": float(xsum[i] / npix),
                "ycentroid_pix": float(ysum[i] / npix),
                "xmin_pix": int(xmin[i]),
                "xmax_pix": int(xmax[i]),
                "ymin_pix": int(ymin[i]),
                "ymax_pix": int(ymax[i]),
            }
        )
    return rows


def cast_segm_for_output(segm_data: np.ndarray, dtype_name: str) -> np.ndarray:
    max_label = int(segm_data.max(initial=0))
    if dtype_name == "auto":
        dtype_name = "int16" if max_label <= np.iinfo(np.int16).max else "int32"

    out_dtype = np.int16 if dtype_name == "int16" else np.int32
    if max_label > np.iinfo(out_dtype).max:
        raise ValueError(
            f"Segmentation label {max_label} exceeds {dtype_name} range; use --segm-dtype int32"
        )
    return segm_data.astype(out_dtype, copy=False)


def write_catalog_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            fieldnames = [
                "label",
                "npix",
                "xcentroid_pix",
                "ycentroid_pix",
                "xmin_pix",
                "xmax_pix",
                "ymin_pix",
                "ymax_pix",
            ]
        else:
            fieldnames = []
            seen: set[str] = set()
            for row in rows:
                for key in row.keys():
                    if key not in seen:
                        seen.add(key)
                        fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stringify_value(value: object) -> object:
    if value is None:
        return ""

    try:
        if np.ma.is_masked(value):
            return ""
    except Exception:
        pass

    if hasattr(value, "unit") and hasattr(value, "value"):
        base = getattr(value, "value")
        unit = str(getattr(value, "unit"))
        return f"{_stringify_value(base)} {unit}".strip()

    if isinstance(value, (np.generic,)):
        return value.item()

    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(_stringify_value(v)) for v in value) + "]"

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _stringify_value(value.item())
        return np.array2string(value, separator=",", max_line_width=10_000)

    return value


def _column_dict_rows_chunk(
    columns: dict[str, list[object]],
    colnames: list[str],
    start: int,
    stop: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(start, stop):
        rows.append({name: _stringify_value(columns[name][i]) for name in colnames})
    return rows


def _table_rows_chunk(
    table,
    colnames: list[str],
    start: int,
    stop: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(start, stop):
        row = table[i]
        rows.append({name: _stringify_value(row[name]) for name in colnames})
    return rows


def table_to_serializable_rows(table, jobs: int = 1) -> list[dict[str, object]]:
    colnames = list(table.colnames)
    nrows = len(table)
    if nrows == 0:
        return []

    jobs = max(1, int(jobs))
    if jobs == 1 or nrows < 64:
        return _table_rows_chunk(table, colnames, 0, nrows)

    jobs = min(jobs, os.cpu_count() or 1, nrows)
    chunk = (nrows + jobs - 1) // jobs
    ranges = [(i, min(i + chunk, nrows)) for i in range(0, nrows, chunk)]

    out: list[dict[str, object]] = []
    # Threading avoids copying the astropy table to child processes.
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = [ex.submit(_table_rows_chunk, table, colnames, a, b) for a, b in ranges]
        for fut in futures:
            out.extend(fut.result())
    return out


def column_dict_to_rows(columns: dict[str, list[object]], jobs: int = 1) -> list[dict[str, object]]:
    if not columns:
        return []
    colnames = list(columns.keys())
    nrows = len(columns[colnames[0]])
    if nrows == 0:
        return []

    jobs = max(1, int(jobs))
    if jobs == 1 or nrows < 64:
        return _column_dict_rows_chunk(columns, colnames, 0, nrows)

    jobs = min(jobs, os.cpu_count() or 1, nrows)
    chunk = (nrows + jobs - 1) // jobs
    ranges = [(i, min(i + chunk, nrows)) for i in range(0, nrows, chunk)]

    out: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = [
            ex.submit(_column_dict_rows_chunk, columns, colnames, a, b)
            for a, b in ranges
        ]
        for fut in futures:
            out.extend(fut.result())
    return out


def _normalize_param_names(param_names: list[str]) -> list[str]:
    normalized: list[str] = []
    for name in param_names:
        if name == "centroid":
            normalized.extend(["xcentroid", "ycentroid"])
        else:
            normalized.append(name)
    return normalized


def extract_extended_source_columns(src_catalog, param_names: list[str]) -> dict[str, list[object]]:
    col_names = _normalize_param_names(param_names)
    nsrc = len(src_catalog)
    columns: dict[str, list[object]] = {
        "label": [int(x) for x in np.asarray(src_catalog.labels).tolist()]
    }

    for name in col_names:
        if name == "r50":
            values = src_catalog.fluxfrac_radius(0.5)
            out_name = "r50"
        else:
            values = getattr(src_catalog, name)
            out_name = name

        columns[out_name] = [values[i] for i in range(nsrc)]
    return columns


def write_ds9_regions(path: Path, extended_rows: list[dict[str, object]]) -> None:
    """Write DS9 image-coordinate ellipses using centroid/orientation/elongation/Req."""
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(
            "global color=green width=3 select=1 highlite=1 dash=0 "
            'font="helvetica 4 normal"\n'
        )
        f.write("image\n")

        for row in extended_rows:
            try:
                x = float(str(row["xcentroid"]).split()[0]) + 1.0
                y = float(str(row["ycentroid"]).split()[0]) + 1.0
                elong = float(str(row["elongation"]).split()[0])
                req = float(str(row["equivalent_radius"]).split()[0])
                ang = float(str(row["orientation"]).split()[0])
            except Exception:
                continue

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(elong) and math.isfinite(req)):
                continue
            if elong <= 0 or req <= 0:
                continue

            # Preserve equivalent-area radius and elongation in the DS9 ellipse axes.
            axis_ratio = max(elong, 1.0e-8)
            a = req * math.sqrt(axis_ratio)
            b = req / math.sqrt(axis_ratio)
            if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
                continue

            label = row.get("label", "")
            f.write(
                f"ellipse({x:.6f},{y:.6f},{a:.6f},{b:.6f},{ang:.6f}) "
                f"# text={{{label}}}\n"
            )


def merge_catalog_rows(
    simple_rows: list[dict[str, object]],
    source_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    source_by_label: dict[int, dict[str, object]] = {}
    for row in source_rows:
        if "label" not in row:
            continue
        try:
            label = int(row["label"])
        except Exception:
            continue
        source_by_label[label] = row

    merged: list[dict[str, object]] = []
    for srow in simple_rows:
        row = dict(srow)
        try:
            label = int(srow["label"])
        except Exception:
            merged.append(row)
            continue

        sc_row = source_by_label.get(label)
        if sc_row:
            for key, value in sc_row.items():
                if key == "label":
                    continue
                row[f"sc_{key}"] = value
        merged.append(row)
    return merged


def main() -> int:
    args = parse_args()

    try:
        from astropy.convolution import Gaussian2DKernel, convolve
        from astropy.io import fits
        from astropy.stats import sigma_clipped_stats
        from photutils.segmentation import SourceCatalog, detect_sources
    except ImportError as exc:  # pragma: no cover
        print(
            "Missing dependency. Install: pip install astropy photutils",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    hdu = int(args.hdu) if str(args.hdu).isdigit() else args.hdu
    data = load_2d_fits_data(args.fits_path, hdu)

    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        raise ValueError("Input image has no finite pixels")

    mean, median, std = sigma_clipped_stats(data, mask=~finite_mask, sigma=3.0)
    threshold = median + args.threshold_sigma * std

    work = np.array(data, copy=True)
    work[~finite_mask] = median

    if args.no_smooth:
        detect_image = work
    else:
        sigma = max(args.smooth_fwhm, 0.0) / 2.355
        if sigma > 0:
            kernel = Gaussian2DKernel(sigma)
            detect_image = convolve(work, kernel, normalize_kernel=True)
        else:
            detect_image = work

    segm = detect_sources(
        detect_image,
        threshold=threshold,
        npixels=args.npixels,
        connectivity=args.connectivity,
        mask=~finite_mask,
    )

    need_extended = bool(
        args.extended_catalog_out or args.merged_catalog_out or args.ds9_region_out
    )

    if segm is None:
        print("No sources detected.")
        if args.basic_catalog_out:
            write_catalog_csv(args.basic_catalog_out, [])
        if args.extended_catalog_out:
            write_catalog_csv(args.extended_catalog_out, [], fieldnames=["label"])
        if args.merged_catalog_out:
            write_catalog_csv(args.merged_catalog_out, [])
        if args.extended_catalog_out or args.ds9_region_out:
            print("Skipping SourceCatalog-derived outputs because no sources were detected.")
        return 0

    segm_data = np.asarray(segm.data, dtype=np.int32)
    basic_rows = compute_catalog(segm_data)
    extended_rows: list[dict[str, object]] | None = None

    print(f"Detected {len(basic_rows)} sources")
    print(f"Background median={median:.6g}, std={std:.6g}, threshold={threshold:.6g}")

    if args.segm_out:
        segm_out_data = cast_segm_for_output(segm_data, args.segm_dtype)
        fits.writeto(args.segm_out, segm_out_data, overwrite=True)
        print(f"Wrote segmentation FITS: {args.segm_out}")

    if args.basic_catalog_out:
        write_catalog_csv(args.basic_catalog_out, basic_rows)
        print(f"Wrote basic catalog CSV: {args.basic_catalog_out}")

    if need_extended:
        src_catalog = SourceCatalog(
            data=work,
            segment_img=segm,
            mask=~finite_mask,
            background=np.full(work.shape, median, dtype=float),
        )
        ext_columns = extract_extended_source_columns(
            src_catalog,
            DEFAULT_EXTENDED_SOURCECAT_PARAMS,
        )
        extended_rows = column_dict_to_rows(ext_columns, jobs=args.jobs)

        if args.extended_catalog_out:
            write_catalog_csv(args.extended_catalog_out, extended_rows)
            print(f"Wrote extended catalog CSV: {args.extended_catalog_out}")

        if args.ds9_region_out:
            write_ds9_regions(args.ds9_region_out, extended_rows)
            print(f"Wrote DS9 region file: {args.ds9_region_out}")

        if args.merged_catalog_out:
            merged_rows = merge_catalog_rows(basic_rows, extended_rows)
            write_catalog_csv(args.merged_catalog_out, merged_rows)
            print(f"Wrote merged catalogs CSV: {args.merged_catalog_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

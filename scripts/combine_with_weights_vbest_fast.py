#!/usr/bin/env python3
"""
Weighted coadd of aligned FITS images with SCI/WHT (and optional ERR) extensions.

Features (best-of merge of previous variants, tuned for speed):
- Accept a folder of FITS files OR an explicit file list.
- Weighted mean coadd using WHT as weights.
- Optional sigma clipping across the image stack before weighted averaging.
- Robust masking of non-finite pixels and configurable handling of non-positive WHT.
- Explicit shape checks and clearer error messages.
- Configurable ERR output from WHT (assuming WHT is inverse variance).
- Lower-memory streaming path when sigma clipping is disabled.
- Faster defaults: float32 accumulation and memmap enabled.

Output FITS extensions:
- SCI: weighted coadd
- WHT: sum of weights
- ERR: sqrt(1 / WHT) if requested (assuming inverse-variance WHT)

Examples:
  python combine_with_weights_vbest.py /path/to/folder -o coadd.fits
  python combine_with_weights_vbest.py img1.fits img2.fits img3.fits -o coadd.fits
  python combine_with_weights_vbest.py /path/to/folder --sigma-clip 3.0 --pattern "*.fits"
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits

try:
    from astropy.stats import sigma_clip
except Exception:  # pragma: no cover
    sigma_clip = None


EXT_NAMES = ("SCI", "WHT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine aligned FITS images using WHT-weighted mean coaddition."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Either one folder path OR a list of FITS file paths.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="coadd.fits",
        help="Output FITS filename (default: coadd.fits)",
    )
    parser.add_argument(
        "--pattern",
        default="*.fits",
        help="Glob pattern when input is a folder (default: *.fits)",
    )
    parser.add_argument(
        "--allow-nonpositive-wht",
        action="store_true",
        help="Do not automatically zero WHT<=0 (default behavior zeros non-positive WHT).",
    )
    parser.add_argument(
        "--wht-floor",
        type=float,
        default=0.0,
        help="Set WHT values below this threshold to zero (default: 0.0).",
    )
    parser.add_argument(
        "--mask-nonfinite",
        action="store_true",
        default=True,
        help="Mask non-finite SCI/WHT by zeroing weights (default: enabled).",
    )
    parser.add_argument(
        "--no-mask-nonfinite",
        dest="mask_nonfinite",
        action="store_false",
        help="Disable masking of NaN/Inf values (not recommended).",
    )
    parser.add_argument(
        "--sigma-clip",
        type=float,
        default=None,
        help=(
            "Apply sigma clipping to SCI stack before weighting "
            "(requires loading stack into memory). Example: --sigma-clip 3.0"
        ),
    )
    parser.add_argument(
        "--sigma-maxiters",
        type=int,
        default=5,
        help="Maximum sigma-clipping iterations (default: 5).",
    )
    parser.add_argument(
        "--err-mode",
        choices=("from-wht", "nan", "inf", "zero"),
        default="from-wht",
        help=(
            "ERR extension behavior. 'from-wht' computes sqrt(1/WHT) assuming "
            "WHT is inverse variance (default)."
        ),
    )
    parser.add_argument(
        "--empty-sci",
        choices=("nan", "zero"),
        default="nan",
        help="Fill value for SCI pixels with zero total weight (default: nan).",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Data type for SCI/ERR/WHT output arrays (default: float32).",
    )
    parser.add_argument(
        "--accum-dtype",
        choices=("float32", "float64"),
        default="float32",
        help=(
            "Internal accumulation dtype. float32 is faster/lighter; "
            "float64 is more precise (default: float32)."
        ),
    )
    parser.add_argument(
        "--memmap",
        action="store_true",
        default=True,
        help="Open FITS files with memmap=True (default: enabled).",
    )
    parser.add_argument(
        "--no-memmap",
        dest="memmap",
        action="store_false",
        help="Disable FITS memory mapping.",
    )
    parser.add_argument(
        "--expect-n",
        type=int,
        default=None,
        help="Optional expected number of input files; errors if mismatch.",
    )
    parser.add_argument(
        "--no-preserve-err-header",
        action="store_true",
        help="Do not copy ERR header from the first file (if present).",
    )
    return parser


def resolve_inputs(inputs: list[str], pattern: str) -> list[Path]:
    """
    Resolve either:
    - a single folder path -> glob within folder
    - explicit file paths
    """
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        folder = Path(inputs[0])
        files = sorted(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matched {pattern!r} in {folder}")
        return [p for p in files if p.is_file()]

    files = [Path(p) for p in inputs]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing file(s): {', '.join(missing)}")
    return files


def read_ext_data(
    hdul: fits.HDUList,
    name: str,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, fits.Header]:
    """Read a named image extension using the requested dtype."""
    try:
        hdu = hdul[name]
    except Exception as exc:
        raise KeyError(f"Missing extension {name!r}") from exc
    if hdu.data is None:
        raise ValueError(f"Extension {name!r} has no data")
    return np.asarray(hdu.data, dtype=dtype), hdu.header.copy()


def sanitize_sci_wht(
    sci: np.ndarray,
    wht: np.ndarray,
    *,
    mask_nonfinite: bool,
    allow_nonpositive_wht: bool,
    wht_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply masking policy by zeroing weights where pixels should not contribute.
    SCI is left unchanged where weight becomes zero because it no longer matters.
    """
    if sci.shape != wht.shape:
        raise ValueError(f"SCI/WHT shape mismatch: {sci.shape} vs {wht.shape}")

    out_wht = wht
    bad: np.ndarray | None = None

    if wht_floor > 0:
        bad = out_wht < wht_floor if bad is None else (bad | (out_wht < wht_floor))

    if not allow_nonpositive_wht:
        bad = out_wht <= 0 if bad is None else (bad | (out_wht <= 0))

    if mask_nonfinite:
        nonfinite = ~np.isfinite(sci) | ~np.isfinite(out_wht)
        bad = nonfinite if bad is None else (bad | nonfinite)

    if bad is not None and np.any(bad):
        out_wht = out_wht.copy()
        out_wht[bad] = 0.0

    return sci, out_wht


def validate_shape(current: np.ndarray, expected_shape: tuple[int, ...], file_path: Path, ext_name: str) -> None:
    if current.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch in {file_path} extension {ext_name}: "
            f"{current.shape} != expected {expected_shape}"
        )


def collect_headers_and_shape(
    first_file: Path,
    *,
    memmap: bool,
    preserve_err_header: bool,
    accum_dtype: np.dtype,
) -> tuple[tuple[int, ...], fits.Header, fits.Header, fits.Header, fits.Header]:
    with fits.open(first_file, memmap=memmap) as hd0:
        sci0, sci_hdr = read_ext_data(hd0, "SCI", dtype=accum_dtype)
        wht0, wht_hdr = read_ext_data(hd0, "WHT", dtype=accum_dtype)
        if sci0.shape != wht0.shape:
            raise ValueError(f"SCI/WHT shape mismatch in first file {first_file}: {sci0.shape} vs {wht0.shape}")

        prim_hdr = hd0[0].header.copy()
        if preserve_err_header and "ERR" in hd0 and hd0["ERR"].data is not None:
            err_hdr = hd0["ERR"].header.copy()
        else:
            err_hdr = fits.Header()

        return sci0.shape, prim_hdr, sci_hdr, err_hdr, wht_hdr


def streaming_weighted_combine(
    files: list[Path],
    *,
    memmap: bool,
    mask_nonfinite: bool,
    allow_nonpositive_wht: bool,
    wht_floor: float,
    empty_sci: str,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient path: iterate file-by-file and accumulate only sums.
    """
    with fits.open(files[0], memmap=memmap) as hd0:
        sci0, _ = read_ext_data(hd0, "SCI", dtype=accum_dtype)
        wht0, _ = read_ext_data(hd0, "WHT", dtype=accum_dtype)
        shape = sci0.shape

    sum_w = np.zeros(shape, dtype=accum_dtype)
    sum_wx = np.zeros(shape, dtype=accum_dtype)
    scratch = np.empty(shape, dtype=accum_dtype)

    for file_path in files:
        with fits.open(file_path, memmap=memmap) as hdul:
            sci, _ = read_ext_data(hdul, "SCI", dtype=accum_dtype)
            wht, _ = read_ext_data(hdul, "WHT", dtype=accum_dtype)

        validate_shape(sci, shape, file_path, "SCI")
        validate_shape(wht, shape, file_path, "WHT")
        sci, wht = sanitize_sci_wht(
            sci,
            wht,
            mask_nonfinite=mask_nonfinite,
            allow_nonpositive_wht=allow_nonpositive_wht,
            wht_floor=wht_floor,
        )

        np.add(sum_w, wht, out=sum_w)
        np.multiply(wht, sci, out=scratch)
        np.add(sum_wx, scratch, out=sum_wx)

    sci_out = np.full(shape, np.nan if empty_sci == "nan" else 0.0, dtype=accum_dtype)
    good = sum_w > 0
    np.divide(sum_wx, sum_w, out=sci_out, where=good)
    return sci_out, sum_w


def stack_weighted_combine_with_sigma_clip(
    files: list[Path],
    *,
    memmap: bool,
    mask_nonfinite: bool,
    allow_nonpositive_wht: bool,
    wht_floor: float,
    sigma: float,
    sigma_maxiters: int,
    empty_sci: str,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stacked path for optional sigma clipping.
    Loads all SCI/WHT arrays into memory, clips SCI outliers, and zeroes weights
    for clipped pixels before computing the weighted mean.
    """
    if sigma_clip is None:
        raise RuntimeError("astropy.stats.sigma_clip is unavailable; install astropy with stats support.")

    sci_list: list[np.ndarray] = []
    wht_list: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None

    for file_path in files:
        with fits.open(file_path, memmap=memmap) as hdul:
            sci, _ = read_ext_data(hdul, "SCI", dtype=accum_dtype)
            wht, _ = read_ext_data(hdul, "WHT", dtype=accum_dtype)

        if expected_shape is None:
            expected_shape = sci.shape
        validate_shape(sci, expected_shape, file_path, "SCI")
        validate_shape(wht, expected_shape, file_path, "WHT")

        sci, wht = sanitize_sci_wht(
            sci,
            wht,
            mask_nonfinite=mask_nonfinite,
            allow_nonpositive_wht=allow_nonpositive_wht,
            wht_floor=wht_floor,
        )
        sci_list.append(sci)
        wht_list.append(wht)

    sci_stack = np.stack(sci_list, axis=0).astype(accum_dtype, copy=False)
    wht_stack = np.stack(wht_list, axis=0).astype(accum_dtype, copy=False)

    # Sigma clip SCI across the stack for each pixel location.
    clipped = sigma_clip(sci_stack, sigma=sigma, axis=0, maxiters=sigma_maxiters, masked=True)
    clipped_mask = np.asarray(clipped.mask, dtype=bool)

    # Zero weights for clipped pixels and for any non-finite SCI (if masking disabled,
    # sigma_clip may still leave problematic values in sci_stack).
    w = wht_stack.copy()
    w[clipped_mask] = 0.0

    if not mask_nonfinite:
        invalid_num = ~np.isfinite(sci_stack) | ~np.isfinite(w)
        if np.any(invalid_num):
            w[invalid_num] = 0.0

    wsum = np.sum(w, axis=0, dtype=accum_dtype)
    np.multiply(sci_stack, w, out=sci_stack)
    num = np.sum(sci_stack, axis=0, dtype=accum_dtype)

    sci_out = np.full(wsum.shape, np.nan if empty_sci == "nan" else 0.0, dtype=accum_dtype)
    good = wsum > 0
    np.divide(num, wsum, out=sci_out, where=good)
    return sci_out, wsum


def build_err_from_wht(wht_out: np.ndarray, mode: str) -> np.ndarray:
    """
    Build ERR extension from WHT or fill according to requested policy.
    Assumes WHT is inverse variance when mode == 'from-wht'.
    """
    if mode == "nan":
        return np.full_like(wht_out, np.nan, dtype=np.float64)
    if mode == "inf":
        return np.full_like(wht_out, np.inf, dtype=np.float64)
    if mode == "zero":
        return np.zeros_like(wht_out, dtype=np.float64)

    err_out = np.full_like(wht_out, np.nan, dtype=np.float64)
    good = wht_out > 0
    err_out[good] = np.sqrt(1.0 / wht_out[good])
    return err_out


def clean_image_header_for_output(header: fits.Header) -> fits.Header:
    """
    Remove structural keywords that astropy/fits will set automatically.
    Keeps science metadata while avoiding output verification conflicts.
    """
    h = header.copy()
    for key in list(h.keys()):
        if key.startswith("NAXIS") or key in {"BITPIX"}:
            h.pop(key, None)
    return h


def cast_output_array(arr: np.ndarray, dtype_name: str) -> np.ndarray:
    return arr.astype(np.float32 if dtype_name == "float32" else np.float64, copy=False)


def add_provenance_history(
    hdu_sci: fits.ImageHDU,
    hdu_err: fits.ImageHDU,
    hdu_wht: fits.ImageHDU,
    *,
    files: Iterable[Path],
    sigma_clip_value: float | None,
    err_mode: str,
    wht_floor: float,
    mask_nonfinite: bool,
    allow_nonpositive_wht: bool,
    accum_dtype: str,
    output_dtype: str,
) -> None:
    nfiles = sum(1 for _ in files)
    # Compact method description for downstream reproducibility / paper notes.
    hdu_sci.header["COADDMTH"] = ("WHTMEAN", "Weighted mean coadd using WHT weights")
    hdu_sci.header["COADDWHT"] = ("INVVAR?", "Interpret WHT as inverse variance if true")
    hdu_sci.header["COADDSIG"] = (
        "NONE" if sigma_clip_value is None else f"{sigma_clip_value:g}",
        "SCI sigma-clipping threshold; NONE means disabled",
    )
    hdu_sci.header["COADDMNF"] = (bool(mask_nonfinite), "Non-finite SCI/WHT excluded by zeroing weight")
    hdu_sci.header["COADDNPW"] = (
        not bool(allow_nonpositive_wht),
        "If true, WHT<=0 excluded from coadd",
    )
    hdu_sci.header["COADDFLR"] = (float(wht_floor), "Weights below this floor are excluded")
    hdu_sci.header["COADDACT"] = (accum_dtype, "Internal accumulation dtype")
    hdu_sci.header["COADDOUT"] = (output_dtype, "Output array dtype")
    hdu_sci.header["COADDERR"] = (err_mode, "ERR extension construction mode")

    hdu_sci.header["HISTORY"] = f"Weighted coadd from {nfiles} input files"
    hdu_sci.header["HISTORY"] = "SCI = sum(SCI_i * WHT_i) / sum(WHT_i)"
    hdu_sci.header["HISTORY"] = "Assumes aligned images on common pixel grid."
    hdu_sci.header["HISTORY"] = "If WHT is inverse variance, SCI is minimum-variance estimator."
    if sigma_clip_value is not None:
        hdu_sci.header["HISTORY"] = f"Sigma clipping applied to SCI stack: sigma={sigma_clip_value}"
        hdu_sci.header["HISTORY"] = "Clipped SCI outliers are excluded by setting corresponding WHT=0."
    hdu_sci.header["HISTORY"] = f"mask_nonfinite={mask_nonfinite}"
    hdu_sci.header["HISTORY"] = f"allow_nonpositive_wht={allow_nonpositive_wht}"
    hdu_sci.header["HISTORY"] = f"wht_floor={wht_floor}"

    if err_mode == "from-wht":
        hdu_err.header["HISTORY"] = "ERR = sqrt(1 / WHT), assuming WHT is inverse variance"
    else:
        hdu_err.header["HISTORY"] = f"ERR filled with mode={err_mode}"

    hdu_wht.header["HISTORY"] = "WHT = sum(WHT_i)"


def write_output_fits(
    output_path: Path,
    *,
    sci_out: np.ndarray,
    err_out: np.ndarray,
    wht_out: np.ndarray,
    prim_hdr: fits.Header,
    sci_hdr: fits.Header,
    err_hdr: fits.Header,
    wht_hdr: fits.Header,
    output_dtype: str,
    files: list[Path],
    sigma_clip_value: float | None,
    err_mode: str,
    wht_floor: float,
    mask_nonfinite: bool,
    allow_nonpositive_wht: bool,
    accum_dtype: str,
) -> None:
    phdu = fits.PrimaryHDU(header=prim_hdr)

    sci_hdu = fits.ImageHDU(
        data=cast_output_array(sci_out, output_dtype),
        header=clean_image_header_for_output(sci_hdr),
        name="SCI",
    )
    err_hdu = fits.ImageHDU(
        data=cast_output_array(err_out, output_dtype),
        header=clean_image_header_for_output(err_hdr),
        name="ERR",
    )
    wht_hdu = fits.ImageHDU(
        data=cast_output_array(wht_out, output_dtype),
        header=clean_image_header_for_output(wht_hdr),
        name="WHT",
    )

    add_provenance_history(
        sci_hdu,
        err_hdu,
        wht_hdu,
        files=files,
        sigma_clip_value=sigma_clip_value,
        err_mode=err_mode,
        wht_floor=wht_floor,
        mask_nonfinite=mask_nonfinite,
        allow_nonpositive_wht=allow_nonpositive_wht,
        accum_dtype=accum_dtype,
        output_dtype=output_dtype,
    )

    fits.HDUList([phdu, sci_hdu, err_hdu, wht_hdu]).writeto(
        output_path,
        overwrite=True,
        output_verify="fix",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    files = resolve_inputs(args.inputs, args.pattern)
    if args.expect_n is not None and len(files) != args.expect_n:
        raise SystemExit(f"Expected {args.expect_n} inputs, found {len(files)}")

    accum_dtype = np.float32 if args.accum_dtype == "float32" else np.float64

    print(f"Found {len(files)} input file(s)")
    if args.sigma_clip is not None:
        print("Mode: stacked weighted combine with sigma clipping")
    else:
        print("Mode: streaming weighted combine")
    print(f"Accumulation dtype: {args.accum_dtype}; memmap={args.memmap}")

    shape, prim_hdr, sci_hdr, err_hdr, wht_hdr = collect_headers_and_shape(
        files[0],
        memmap=args.memmap,
        preserve_err_header=not args.no_preserve_err_header,
        accum_dtype=accum_dtype,
    )
    print(f"Reference shape: {shape}")

    if args.sigma_clip is None:
        sci_out, wht_out = streaming_weighted_combine(
            files,
            memmap=args.memmap,
            mask_nonfinite=args.mask_nonfinite,
            allow_nonpositive_wht=args.allow_nonpositive_wht,
            wht_floor=args.wht_floor,
            empty_sci=args.empty_sci,
            accum_dtype=accum_dtype,
        )
    else:
        sci_out, wht_out = stack_weighted_combine_with_sigma_clip(
            files,
            memmap=args.memmap,
            mask_nonfinite=args.mask_nonfinite,
            allow_nonpositive_wht=args.allow_nonpositive_wht,
            wht_floor=args.wht_floor,
            sigma=args.sigma_clip,
            sigma_maxiters=args.sigma_maxiters,
            empty_sci=args.empty_sci,
            accum_dtype=accum_dtype,
        )

    err_out = build_err_from_wht(wht_out, args.err_mode)
    output_path = Path(args.output)
    write_output_fits(
        output_path,
        sci_out=sci_out,
        err_out=err_out,
        wht_out=wht_out,
        prim_hdr=prim_hdr,
        sci_hdr=sci_hdr,
        err_hdr=err_hdr,
        wht_hdr=wht_hdr,
        output_dtype=args.output_dtype,
        files=files,
        sigma_clip_value=args.sigma_clip,
        err_mode=args.err_mode,
        wht_floor=args.wht_floor,
        mask_nonfinite=args.mask_nonfinite,
        allow_nonpositive_wht=args.allow_nonpositive_wht,
        accum_dtype=args.accum_dtype,
    )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

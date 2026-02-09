#!/usr/bin/env python3
"""Accelerated PSF/template matching in FITS images.

Quick guide:
- Basic run:
  python match_psf_fourier_accel.py <target.fits> <pattern.fits>
- Backend:
  auto mode is always used (GPU preferred with automatic CPU fallback)
- Common options:
  --crop XMIN XMAX YMIN YMAX     process a sub-image
  --fft-workers N                CPU FFT threads (scipy.fft)
  --refine-workers N             parallel workers for refinement
  --out-map/--out-csv/--out-reg  output products

Features:
- CPU path with multi-threaded FFT via scipy.fft workers.
- Optional GPU path via CuPy FFT (if installed).
- Automatic GPU->CPU fallback on runtime GPU errors.
- Parallel refinement stage on multiple CPU workers.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from scipy import fft as spfft
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Accelerated Fourier matched-filter detection of template-like structures."
    )
    parser.add_argument(
        "target_image",
        nargs="?",
        default="CWEB-F444W-7B_i2dnobg_small.fits",
        help="Target image FITS where structures are searched",
    )
    parser.add_argument(
        "pattern_image",
        nargs="?",
        default="jwst_psf_f444w.fits",
        help="Pattern/template image FITS used for matching (e.g. PSF)",
    )
    parser.add_argument(
        "--image-hdu",
        type=int,
        default=-1,
        help="HDU index for image data. Use -1 to auto-select first 2D HDU.",
    )
    parser.add_argument(
        "--pattern-hdu",
        type=int,
        default=-1,
        help="HDU index for pattern data. Use -1 to auto-select first 2D HDU.",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Optional crop in 0-based pixel coordinates before filtering.",
    )
    parser.add_argument(
        "--fft-workers",
        type=int,
        default=0,
        help="CPU FFT worker threads for scipy.fft (0 = all available cores).",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=5.0,
        help="Detection threshold in SNR units",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=5.0,
        help="Minimum pixel separation between detections",
    )
    parser.add_argument(
        "--fit-window-scale",
        type=float,
        default=1.0,
        help="Half-window size for local fit, in pattern half-size units.",
    )
    parser.add_argument(
        "--min-fit-corr",
        type=float,
        default=0.2,
        help="Minimum local pattern-data correlation to keep a detection.",
    )
    parser.add_argument(
        "--max-fit-rms-ratio",
        type=float,
        default=1.0,
        help="Maximum residual RMS / data RMS in local fit window.",
    )
    parser.add_argument(
        "--refine-workers",
        type=int,
        default=0,
        help="Number of CPU workers for refinement stage (0 = all available cores).",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.85,
        help="Minimum template scale factor for multi-scale matching.",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.15,
        help="Maximum template scale factor for multi-scale matching.",
    )
    parser.add_argument(
        "--scale-steps",
        type=int,
        default=7,
        help="Number of scale samples between --scale-min and --scale-max (>=1).",
    )
    parser.add_argument(
        "--out-map",
        default=None,
        help="Output SNR map FITS",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output candidates CSV (default: <target>_matchedPSF.csv)",
    )
    parser.add_argument(
        "--out-reg",
        default=None,
        help="Output DS9 region file (default: <target>_matchedPSF.reg)",
    )
    return parser.parse_args()


def load_fits_2d(path: Path, hdu: int) -> tuple[np.ndarray, int]:
    with fits.open(path) as hdul:
        if hdu >= 0:
            data = hdul[hdu].data
            shape = () if data is None else np.shape(data)
            if data is None or np.ndim(data) != 2:
                raise ValueError(f"{path} HDU {hdu} is not 2D (shape={shape})")
            return np.asarray(data, dtype=float), hdu

        for idx, h in enumerate(hdul):
            data = h.data
            if data is not None and np.ndim(data) == 2:
                return np.asarray(data, dtype=float), idx

    raise ValueError(f"{path}: no 2D image HDU found")


def finite_fill(a: np.ndarray, fill_value: float | None = None) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    mask = np.isfinite(out)
    if not np.any(mask):
        raise ValueError("array has no finite values")
    if fill_value is None:
        fill_value = float(np.nanmedian(out[mask]))
    out[~mask] = fill_value
    return out


def fft_convolve_same_cpu(image: np.ndarray, kernel: np.ndarray, workers: int) -> np.ndarray:
    kh, kw = kernel.shape
    full_shape = (image.shape[0] + kh - 1, image.shape[1] + kw - 1)
    fft_kwargs = {"s": full_shape, "axes": (0, 1)}
    if workers > 0:
        fft_kwargs["workers"] = workers

    f_img = spfft.rfftn(image, **fft_kwargs)
    f_ker = spfft.rfftn(kernel[::-1, ::-1], **fft_kwargs)
    conv_full = spfft.irfftn(f_img * f_ker, **fft_kwargs)

    y0 = (kh - 1) // 2
    x0 = (kw - 1) // 2
    y1 = y0 + image.shape[0]
    x1 = x0 + image.shape[1]
    return conv_full[y0:y1, x0:x1]


def resolve_workers(requested: int) -> int:
    if requested > 0:
        return requested
    return max(1, os.cpu_count() or 1)


def fft_convolve_same_gpu(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError(
            "GPU backend requested but CuPy is unavailable. Install cupy-cudaXX first."
        ) from exc

    kh, kw = kernel.shape
    full_shape = (image.shape[0] + kh - 1, image.shape[1] + kw - 1)
    g_img = cp.asarray(image)
    g_ker = cp.asarray(kernel[::-1, ::-1])

    f_img = cp.fft.rfftn(g_img, s=full_shape, axes=(0, 1))
    f_ker = cp.fft.rfftn(g_ker, s=full_shape, axes=(0, 1))
    conv_full = cp.fft.irfftn(f_img * f_ker, s=full_shape, axes=(0, 1))

    y0 = (kh - 1) // 2
    x0 = (kw - 1) // 2
    y1 = y0 + image.shape[0]
    x1 = x0 + image.shape[1]
    out = conv_full[y0:y1, x0:x1]
    return cp.asnumpy(out)


def robust_sigma(a: np.ndarray) -> float:
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        sigma = float(np.std(a))
    if sigma <= 0:
        raise ValueError("cannot estimate non-zero noise sigma")
    return sigma


def local_maxima_8conn(arr: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    peaks = arr >= threshold
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
            peaks &= arr >= shifted

    peaks[[0, -1], :] = False
    peaks[:, [0, -1]] = False
    return np.where(peaks)


def suppress_close_points(
    y: np.ndarray, x: np.ndarray, score: np.ndarray, min_separation: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(score)[::-1]
    ys = y[order]
    xs = x[order]
    ss = score[order]
    keep_y: list[int] = []
    keep_x: list[int] = []
    keep_s: list[float] = []
    min_sep2 = min_separation * min_separation

    for yi, xi, si in zip(ys, xs, ss):
        too_close = False
        for yj, xj in zip(keep_y, keep_x):
            if (yi - yj) * (yi - yj) + (xi - xj) * (xi - xj) < min_sep2:
                too_close = True
                break
        if not too_close:
            keep_y.append(int(yi))
            keep_x.append(int(xi))
            keep_s.append(float(si))

    return np.asarray(keep_y), np.asarray(keep_x), np.asarray(keep_s)


def _refine_chunk(
    image: np.ndarray,
    patterns: list[np.ndarray],
    best_scale_idx: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    score: np.ndarray,
    fit_window_scale: float,
    min_fit_corr: float,
    max_fit_rms_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep_y: list[int] = []
    keep_x: list[int] = []
    keep_snr: list[float] = []
    keep_amp: list[float] = []
    keep_corr: list[float] = []
    keep_rmsr: list[float] = []

    for yi, xi, si in zip(y, x, score):
        p = patterns[int(best_scale_idx[yi, xi])]
        ph, pw = p.shape
        half_h = max(2, min(ph // 2, int(round((ph // 2) * fit_window_scale))))
        half_w = max(2, min(pw // 2, int(round((pw // 2) * fit_window_scale))))

        py0 = (ph // 2) - half_h
        py1 = (ph // 2) + half_h + 1
        px0 = (pw // 2) - half_w
        px1 = (pw // 2) + half_w + 1
        if py0 < 0 or px0 < 0 or py1 > ph or px1 > pw:
            continue

        t = p[py0:py1, px0:px1]
        t = t - np.mean(t)
        denom = np.sum(t * t)
        if denom <= 0:
            continue

        y0 = yi - half_h
        y1 = yi + half_h + 1
        x0 = xi - half_w
        x1 = xi + half_w + 1
        if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
            continue
        d = image[y0:y1, x0:x1]
        if d.shape != t.shape:
            continue
        d = d - np.mean(d)

        amp = float(np.sum(d * t) / denom)
        model = amp * t
        resid = d - model

        dnorm = np.sqrt(np.sum(d * d))
        corr = float(np.sum(d * t) / (dnorm * np.sqrt(denom))) if dnorm > 0 else 0.0
        data_rms = float(np.sqrt(np.mean(d * d)))
        resid_rms = float(np.sqrt(np.mean(resid * resid)))
        rms_ratio = resid_rms / data_rms if data_rms > 0 else np.inf

        if corr < min_fit_corr:
            continue
        if rms_ratio > max_fit_rms_ratio:
            continue

        keep_y.append(int(yi))
        keep_x.append(int(xi))
        keep_snr.append(float(si))
        keep_amp.append(amp)
        keep_corr.append(corr)
        keep_rmsr.append(rms_ratio)

    return (
        np.asarray(keep_y),
        np.asarray(keep_x),
        np.asarray(keep_snr),
        np.asarray(keep_amp),
        np.asarray(keep_corr),
        np.asarray(keep_rmsr),
    )


def refine_candidates_parallel(
    image: np.ndarray,
    patterns: list[np.ndarray],
    best_scale_idx: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    score: np.ndarray,
    fit_window_scale: float,
    min_fit_corr: float,
    max_fit_rms_ratio: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(score) == 0:
        empty = np.asarray([])
        return empty, empty, empty, empty, empty, empty

    workers = max(1, workers)
    if workers == 1:
        return _refine_chunk(
            image,
            patterns,
            best_scale_idx,
            y,
            x,
            score,
            fit_window_scale,
            min_fit_corr,
            max_fit_rms_ratio,
        )

    idx_chunks = [chunk for chunk in np.array_split(np.arange(len(score)), workers) if len(chunk) > 0]
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx in idx_chunks:
            futures.append(
                ex.submit(
                    _refine_chunk,
                    image,
                    patterns,
                    best_scale_idx,
                    y[idx],
                    x[idx],
                    score[idx],
                    fit_window_scale,
                    min_fit_corr,
                    max_fit_rms_ratio,
                )
            )
        for f in futures:
            results.append(f.result())

    y_all = np.concatenate([r[0] for r in results]) if results else np.asarray([])
    x_all = np.concatenate([r[1] for r in results]) if results else np.asarray([])
    s_all = np.concatenate([r[2] for r in results]) if results else np.asarray([])
    a_all = np.concatenate([r[3] for r in results]) if results else np.asarray([])
    c_all = np.concatenate([r[4] for r in results]) if results else np.asarray([])
    r_all = np.concatenate([r[5] for r in results]) if results else np.asarray([])

    if len(s_all) > 0:
        order = np.argsort(s_all)[::-1]
        y_all = y_all[order]
        x_all = x_all[order]
        s_all = s_all[order]
        a_all = a_all[order]
        c_all = c_all[order]
        r_all = r_all[order]

    return y_all, x_all, s_all, a_all, c_all, r_all


def write_candidates_csv(
    path: Path,
    y: np.ndarray,
    x: np.ndarray,
    score: np.ndarray,
    amp: np.ndarray,
    corr: np.ndarray,
    rms_ratio: np.ndarray,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "snr", "fit_amp", "fit_corr", "fit_rms_ratio"])
        for i, (yi, xi, si, ai, ci, ri) in enumerate(
            zip(y, x, score, amp, corr, rms_ratio), start=1
        ):
            writer.writerow([i, xi + 1.0, yi + 1.0, si, ai, ci, ri])


def write_ds9_regions(
    path: Path, y: np.ndarray, x: np.ndarray, score: np.ndarray, corr: np.ndarray
) -> None:
    with path.open("w", newline="") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("global color=red width=1 select=1 highlite=1 move=1 delete=1 source=1\n")
        f.write("image\n")
        for yi, xi in zip(y, x):
            f.write(f"point({xi + 1.0:.3f},{yi + 1.0:.3f}) # point=x\n")


def pick_backend(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        return "gpu"

    # auto mode
    try:
        import cupy  # noqa: F401

        return "gpu"
    except Exception:
        return "cpu"


def normalized_pattern(pattern: np.ndarray) -> np.ndarray:
    p = pattern - np.median(pattern)
    pnorm = np.sqrt(np.sum(p * p))
    if pnorm <= 0:
        raise ValueError("pattern normalization failed (all-zero template)")
    return p / pnorm


def resize_pattern(pattern: np.ndarray, scale: float) -> np.ndarray:
    if np.isclose(scale, 1.0):
        return np.array(pattern, copy=True)
    resized = ndimage.zoom(pattern, zoom=scale, order=3, mode="nearest")
    if resized.shape[0] < 3 or resized.shape[1] < 3:
        raise ValueError(f"scaled template too small at scale={scale:.3f}, shape={resized.shape}")
    return resized


def fits_stem(path: str) -> str:
    name = Path(path).name
    if name.endswith(".fits.gz"):
        return name[:-8]
    return Path(name).stem


def main() -> int:
    args = parse_args()

    image_raw, used_image_hdu = load_fits_2d(Path(args.target_image), args.image_hdu)
    pattern_raw, used_pattern_hdu = load_fits_2d(Path(args.pattern_image), args.pattern_hdu)

    if args.crop:
        xmin, xmax, ymin, ymax = args.crop
        image_raw = image_raw[ymin:ymax, xmin:xmax]

    image = finite_fill(image_raw)
    pattern_base = finite_fill(pattern_raw, fill_value=0.0)

    image = image - np.median(image)
    fft_workers = resolve_workers(args.fft_workers)
    refine_workers = resolve_workers(args.refine_workers)

    if args.scale_steps < 1:
        raise ValueError("--scale-steps must be >= 1")
    if args.scale_min <= 0 or args.scale_max <= 0:
        raise ValueError("--scale-min and --scale-max must be > 0")
    if args.scale_min > args.scale_max:
        raise ValueError("--scale-min must be <= --scale-max")

    scales = np.linspace(args.scale_min, args.scale_max, args.scale_steps, dtype=float)
    patterns: list[np.ndarray] = [normalized_pattern(resize_pattern(pattern_base, s)) for s in scales]

    backend = pick_backend("auto")
    fallback_used = False
    fallback_reason = ""

    responses: list[np.ndarray] = []
    for pattern in patterns:
        if backend == "gpu":
            try:
                responses.append(fft_convolve_same_gpu(image, pattern))
                continue
            except Exception as exc:
                fallback_used = True
                fallback_reason = str(exc)
                backend = "cpu"
        try:
            responses.append(fft_convolve_same_cpu(image, pattern, fft_workers))
        except Exception as cpu_exc:
            if fallback_used:
                print(
                    "Error: GPU failed and CPU fallback also failed: "
                    f"gpu_error={fallback_reason} cpu_error={cpu_exc}",
                    file=sys.stderr,
                )
            else:
                print(f"Error: {cpu_exc}", file=sys.stderr)
            return 2

    response_stack = np.stack(responses, axis=0)
    best_scale_idx = np.argmax(response_stack, axis=0)
    response = np.take_along_axis(
        response_stack,
        np.expand_dims(best_scale_idx, axis=0),
        axis=0,
    )[0]

    sigma = robust_sigma(response)
    snr = response / sigma

    y, x = local_maxima_8conn(snr, args.threshold_sigma)
    scores = snr[y, x]
    y, x, scores = suppress_close_points(y, x, scores, args.min_separation)
    raw_n = len(scores)

    y, x, scores, amps, corrs, rms_ratios = refine_candidates_parallel(
        image=image,
        patterns=patterns,
        best_scale_idx=best_scale_idx,
        y=y,
        x=x,
        score=scores,
        fit_window_scale=args.fit_window_scale,
        min_fit_corr=args.min_fit_corr,
        max_fit_rms_ratio=args.max_fit_rms_ratio,
        workers=refine_workers,
    )

    target_base = f"{fits_stem(args.target_image)}_matchedPSF"
    out_map = args.out_map or f"{target_base}_snr.fits"
    out_csv = args.out_csv or f"{target_base}.csv"
    out_reg = args.out_reg or f"{target_base}.reg"

    fits.PrimaryHDU(data=snr.astype(np.float32)).writeto(out_map, overwrite=True)
    write_candidates_csv(Path(out_csv), y, x, scores, amps, corrs, rms_ratios)
    write_ds9_regions(Path(out_reg), y, x, scores, corrs)

    if fallback_used:
        print(
            f"Warning: requested GPU path failed and fell back to CPU: {fallback_reason}",
            file=sys.stderr,
        )

    print(
        f"backend={backend} fft_workers={fft_workers} refine_workers={refine_workers} | "
        f"scales=[{args.scale_min:.3f},{args.scale_max:.3f}] steps={args.scale_steps} | "
        f"detections raw={raw_n} kept={len(scores)} | threshold={args.threshold_sigma:.2f} sigma | "
        f"fit_corr>={args.min_fit_corr:.2f} rms_ratio<={args.max_fit_rms_ratio:.2f} | "
        f"image_hdu={used_image_hdu} pattern_hdu={used_pattern_hdu} | "
        f"map={out_map} csv={out_csv} reg={out_reg}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

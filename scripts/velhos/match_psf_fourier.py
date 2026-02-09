#!/usr/bin/env python3
"""Find PSF-like structures in an image using FFT-based matched filtering.

This script computes a matched-filter response by convolving the image with
the PSF template in Fourier space, builds an SNR map, then extracts local
maxima above a sigma threshold.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect PSF-like structures with Fourier matched filtering."
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
        help="HDU index for image data. Use -1 to auto-select first 2D HDU (default).",
    )
    parser.add_argument(
        "--psf-hdu",
        type=int,
        default=-1,
        help="HDU index for PSF data. Use -1 to auto-select first 2D HDU (default).",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Optional crop in 0-based pixel coordinates before filtering.",
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
        help="Half-window size for local PSF fit, in PSF half-size units.",
    )
    parser.add_argument(
        "--min-fit-corr",
        type=float,
        default=0.2,
        help="Minimum local template-data correlation to keep a detection.",
    )
    parser.add_argument(
        "--max-fit-rms-ratio",
        type=float,
        default=1.0,
        help="Maximum residual RMS / data RMS in local fit window.",
    )
    parser.add_argument(
        "--out-map",
        default="matched_filter_snr.fits",
        help="Output SNR map FITS",
    )
    parser.add_argument(
        "--out-csv",
        default="matched_filter_candidates.csv",
        help="Output candidates CSV",
    )
    parser.add_argument(
        "--out-reg",
        default="matched_filter_candidates.reg",
        help="Output DS9 region file",
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


def fft_convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    full_shape = (image.shape[0] + kh - 1, image.shape[1] + kw - 1)
    f_img = np.fft.rfftn(image, s=full_shape, axes=(0, 1))
    f_ker = np.fft.rfftn(kernel[::-1, ::-1], s=full_shape, axes=(0, 1))
    conv_full = np.fft.irfftn(f_img * f_ker, s=full_shape, axes=(0, 1))

    y0 = (kh - 1) // 2
    x0 = (kw - 1) // 2
    y1 = y0 + image.shape[0]
    x1 = x0 + image.shape[1]
    return conv_full[y0:y1, x0:x1]


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


def refine_candidates(
    image: np.ndarray,
    psf: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    score: np.ndarray,
    fit_window_scale: float,
    min_fit_corr: float,
    max_fit_rms_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ph, pw = psf.shape
    half_h = max(2, min(ph // 2, int(round((ph // 2) * fit_window_scale))))
    half_w = max(2, min(pw // 2, int(round((pw // 2) * fit_window_scale))))

    keep_y: list[int] = []
    keep_x: list[int] = []
    keep_snr: list[float] = []
    keep_amp: list[float] = []
    keep_corr: list[float] = []
    keep_rmsr: list[float] = []

    for yi, xi, si in zip(y, x, score):
        y0 = yi - half_h
        y1 = yi + half_h + 1
        x0 = xi - half_w
        x1 = xi + half_w + 1
        if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
            continue
        patch = image[y0:y1, x0:x1]
        if patch.size < 25:
            continue

        # Use a centered PSF window with the same shape as the local image patch.
        py0 = (ph // 2) - half_h
        py1 = (ph // 2) + half_h + 1
        px0 = (pw // 2) - half_w
        px1 = (pw // 2) + half_w + 1
        if py0 < 0 or px0 < 0 or py1 > ph or px1 > pw:
            continue
        t = psf[py0:py1, px0:px1]
        if t.shape != patch.shape:
            continue

        d = patch - np.mean(patch)
        t = t - np.mean(t)
        denom = np.sum(t * t)
        if denom <= 0:
            continue

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
            # DS9 image coordinates are 1-based.
            writer.writerow([i, xi + 1.0, yi + 1.0, si, ai, ci, ri])


def write_ds9_regions(
    path: Path, y: np.ndarray, x: np.ndarray, score: np.ndarray, corr: np.ndarray
) -> None:
    with path.open("w", newline="") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("global color=red width=1 select=1 highlite=1 move=1 delete=1 source=1\n")
        f.write("image\n")
        for yi, xi, si, ci in zip(y, x, score, corr):
            f.write(
                f"point({xi + 1.0:.3f},{yi + 1.0:.3f}) "
                f"# point=x text={{SNR={si:.2f},corr={ci:.2f}}}\n"
            )


def main() -> int:
    args = parse_args()
    image_raw, used_image_hdu = load_fits_2d(Path(args.target_image), args.image_hdu)
    psf_raw, used_psf_hdu = load_fits_2d(Path(args.pattern_image), args.psf_hdu)

    if args.crop:
        xmin, xmax, ymin, ymax = args.crop
        image_raw = image_raw[ymin:ymax, xmin:xmax]

    image = finite_fill(image_raw)
    psf = finite_fill(psf_raw, fill_value=0.0)

    image = image - np.median(image)
    psf = psf - np.median(psf)
    psf_norm = np.sqrt(np.sum(psf * psf))
    if psf_norm <= 0:
        raise ValueError("PSF normalization failed (all-zero template)")
    psf = psf / psf_norm

    response = fft_convolve_same(image, psf)
    sigma = robust_sigma(response)
    snr = response / sigma

    y, x = local_maxima_8conn(snr, args.threshold_sigma)
    scores = snr[y, x]
    y, x, scores = suppress_close_points(y, x, scores, args.min_separation)
    raw_n = len(scores)

    y, x, scores, amps, corrs, rms_ratios = refine_candidates(
        image=image,
        psf=psf,
        y=y,
        x=x,
        score=scores,
        fit_window_scale=args.fit_window_scale,
        min_fit_corr=args.min_fit_corr,
        max_fit_rms_ratio=args.max_fit_rms_ratio,
    )

    fits.PrimaryHDU(data=snr.astype(np.float32)).writeto(args.out_map, overwrite=True)
    write_candidates_csv(Path(args.out_csv), y, x, scores, amps, corrs, rms_ratios)
    write_ds9_regions(Path(args.out_reg), y, x, scores, corrs)

    print(
        f"Detections raw={raw_n} kept={len(scores)} | "
        f"threshold={args.threshold_sigma:.2f} sigma | "
        f"fit_corr>={args.min_fit_corr:.2f} rms_ratio<={args.max_fit_rms_ratio:.2f} | "
        f"image_hdu={used_image_hdu} psf_hdu={used_psf_hdu} | "
        f"map={args.out_map} csv={args.out_csv} reg={args.out_reg}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

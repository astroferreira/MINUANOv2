#!/usr/bin/env python3
"""
Gaia star exclusion mask generation utilities.

Extracted from `run_source_extraction_for_mfmtk.py` so the mask logic can be
reused independently of the full source-extraction pipeline.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from astropy.wcs import WCS

# Slightly expand Gaia exclusion radii to better suppress stellar contamination.
GAIA_MASK_RADIUS_SCALE = 1.10

GAIA_JWST_MASK_CORE_FRAC = 0.72
GAIA_JWST_MASK_SPIKE_LENGTH_FRAC = 1.20
GAIA_JWST_MASK_SPIKE_HALF_WIDTH_FRAC = 0.08
GAIA_JWST_MASK_SPIKE_HALF_WIDTH_MIN_PIX = 1.5

# Empirical axis/diagonal spike mask (recommended for COSMOS-Web style mosaics).
GAIA_STAR_MASK_CORE_FRAC = 0.36
GAIA_STAR_MASK_ARM_HALF_WIDTH_FRAC = 0.06
GAIA_STAR_MASK_ARM_HALF_WIDTH_MIN_PIX = 1.0
GAIA_STAR_MASK_MAIN_ARM_LENGTH_FRAC = 1.10
GAIA_STAR_MASK_DIAG_ARM_LENGTH_FRAC = 0.80
GAIA_STAR_MASK_DIAG_ARM_WIDTH_SCALE = 0.75

# Asymmetric horizontal arm scaling for the custom 30-degree template.
GAIA_STAR_MASK_ARM_LENGTH_SCALE_0_DEG = 0.40
GAIA_STAR_MASK_ARM_LENGTH_SCALE_180_DEG = 0.28

# Custom 30-degree arm set requested for the mosaics.
GAIA_STAR30_CUSTOM_ANGLES_DEG = [0.0, 30.0, 90.0, 150.0, 180.0, 210.0, 270.0, 330.0]

# Non-linear size response: faint/small stars get larger cores but shorter arms,
# while bright/large stars get longer arms.
GAIA_STAR_MASK_NL_PIVOT_PIX = 18.0
GAIA_STAR_MASK_CORE_FAINT_SCALE_MAX = 1.75
GAIA_STAR_MASK_CORE_FAINT_GAMMA = 1.35
GAIA_STAR_MASK_ARM_SCALE_MIN = 0.70
GAIA_STAR_MASK_ARM_SCALE_MAX = 1.25
GAIA_STAR_MASK_ARM_SCALE_GAMMA = 1.20


def _mag_to_radius_arcsec(
    mag: np.ndarray,
    r0: float,
    mag0: float,
    min_r: float,
    max_r: float,
) -> np.ndarray:
    m = np.asarray(mag, dtype=float)
    r = float(r0) * (10.0 ** (-0.2 * (m - float(mag0))))
    r = np.clip(r, float(min_r), float(max_r))
    if not np.all(np.isfinite(r)):
        r = np.where(np.isfinite(r), r, float(r0))
    return r


def _regular_ngon_mask(
    dx: np.ndarray,
    dy: np.ndarray,
    radius: float,
    nsides: int = 6,
    rotation_rad: float = 0.0,
) -> np.ndarray:
    if not (np.isfinite(radius) and radius > 0):
        return np.zeros_like(dx, dtype=bool)
    rr = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx) - float(rotation_rad)
    period = 2.0 * np.pi / float(nsides)
    theta_mod = (theta + np.pi) % period - (period / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_boundary = float(radius) * np.cos(np.pi / float(nsides)) / np.cos(theta_mod)
    return rr <= r_boundary


def _jwst_diffraction_template_mask(
    dx: np.ndarray,
    dy: np.ndarray,
    radius_pix: float,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    rr = float(radius_pix)
    if not (np.isfinite(rr) and rr > 0):
        return np.zeros_like(dx, dtype=bool)

    rot = np.deg2rad(float(rotation_deg))
    core_radius = GAIA_JWST_MASK_CORE_FRAC * rr
    spike_len = GAIA_JWST_MASK_SPIKE_LENGTH_FRAC * rr
    spike_half_width = max(
        float(GAIA_JWST_MASK_SPIKE_HALF_WIDTH_MIN_PIX),
        float(GAIA_JWST_MASK_SPIKE_HALF_WIDTH_FRAC) * rr,
    )

    mask = _regular_ngon_mask(dx, dy, core_radius, nsides=6, rotation_rad=rot)
    for k in range(6):
        ang = rot + (k * np.pi / 3.0)
        ca = np.cos(ang)
        sa = np.sin(ang)
        along = dx * ca + dy * sa
        perp = -dx * sa + dy * ca
        body = (along >= 0.0) & (along <= spike_len) & (np.abs(perp) <= spike_half_width)
        tip = ((along - spike_len) ** 2 + perp**2) <= (spike_half_width**2)
        mask |= body | tip
    return mask


def _gaia_star_nonlinear_scales(radius_pix: float) -> tuple[float, float]:
    rr = float(radius_pix)
    if not (np.isfinite(rr) and rr > 0):
        return 1.0, 1.0

    pivot = max(1.0e-6, float(GAIA_STAR_MASK_NL_PIVOT_PIX))
    bright_t = rr / (rr + pivot)
    bright_t = float(np.clip(bright_t, 0.0, 1.0))
    faint_t = 1.0 - bright_t

    core_faint_gamma = max(1.0e-6, float(GAIA_STAR_MASK_CORE_FAINT_GAMMA))
    arm_gamma = max(1.0e-6, float(GAIA_STAR_MASK_ARM_SCALE_GAMMA))

    core_scale = 1.0 + (faint_t**core_faint_gamma) * (
        float(GAIA_STAR_MASK_CORE_FAINT_SCALE_MAX) - 1.0
    )
    arm_scale = float(GAIA_STAR_MASK_ARM_SCALE_MIN) + (bright_t**arm_gamma) * (
        float(GAIA_STAR_MASK_ARM_SCALE_MAX) - float(GAIA_STAR_MASK_ARM_SCALE_MIN)
    )
    return float(core_scale), float(arm_scale)


def _spike_arm_star_template_mask(
    dx: np.ndarray,
    dy: np.ndarray,
    radius_pix: float,
    mode: str = "star12",
    rotation_deg: float = 0.0,
) -> np.ndarray:
    rr = float(radius_pix)
    if not (np.isfinite(rr) and rr > 0):
        return np.zeros_like(dx, dtype=bool)

    core_scale, arm_scale = _gaia_star_nonlinear_scales(rr)
    core_radius = max(1.0, float(GAIA_STAR_MASK_CORE_FRAC) * rr * core_scale)
    base_half_width = max(
        float(GAIA_STAR_MASK_ARM_HALF_WIDTH_MIN_PIX),
        float(GAIA_STAR_MASK_ARM_HALF_WIDTH_FRAC) * rr,
    )
    rot = np.deg2rad(float(rotation_deg))

    mask = (dx**2 + dy**2) <= core_radius**2

    mode_lc = str(mode).lower()
    axis_angles = [0.0, 90.0, 180.0, 270.0]
    diag_angles = [45.0, 135.0, 225.0, 315.0]
    if mode_lc == "star4":
        angles = axis_angles
    elif mode_lc == "star8":
        angles = axis_angles + diag_angles
    else:
        angles = list(GAIA_STAR30_CUSTOM_ANGLES_DEG)

    for ang_deg in angles:
        ang = np.deg2rad(ang_deg) + rot
        ca = np.cos(ang)
        sa = np.sin(ang)
        along = dx * ca + dy * sa
        perp = -dx * sa + dy * ca

        is_diag = (mode_lc == "star8") and ((ang_deg % 90.0) != 0.0)
        arm_len = (
            (GAIA_STAR_MASK_DIAG_ARM_LENGTH_FRAC if is_diag else GAIA_STAR_MASK_MAIN_ARM_LENGTH_FRAC)
            * rr
        )
        if ang_deg == 0.0:
            arm_len *= float(GAIA_STAR_MASK_ARM_LENGTH_SCALE_0_DEG)
        elif ang_deg == 180.0:
            arm_len *= float(GAIA_STAR_MASK_ARM_LENGTH_SCALE_180_DEG)
        arm_len *= arm_scale
        half_width = base_half_width * (GAIA_STAR_MASK_DIAG_ARM_WIDTH_SCALE if is_diag else 1.0)

        arm_inner = core_radius
        body = (along >= arm_inner) & (along <= arm_inner + arm_len) & (np.abs(perp) <= half_width)
        tip = ((along - (arm_inner + arm_len)) ** 2 + perp**2) <= (half_width**2)
        mask |= body | tip

    return mask


def _gaia_star_mask_patch(
    dx: np.ndarray,
    dy: np.ndarray,
    radius_pix: float,
    args: argparse.Namespace,
) -> np.ndarray:
    shape = str(getattr(args, "gaia_mask_shape", "jwst")).strip().lower()
    if shape == "circle":
        return dx**2 + dy**2 <= float(radius_pix) ** 2
    if shape in {"star12", "star8", "star4"}:
        return _spike_arm_star_template_mask(
            dx,
            dy,
            radius_pix,
            mode=shape,
            rotation_deg=float(getattr(args, "gaia_mask_rotation_deg", 0.0)),
        )
    return _jwst_diffraction_template_mask(
        dx,
        dy,
        radius_pix,
        rotation_deg=float(getattr(args, "gaia_mask_rotation_deg", 0.0)),
    )


def _gaia_star_mask_extent_pix(radius_pix: float, args: argparse.Namespace) -> float:
    rr = float(radius_pix)
    shape = str(getattr(args, "gaia_mask_shape", "jwst")).strip().lower()
    if shape == "circle":
        return rr
    if shape in {"star12", "star8", "star4"}:
        core_scale, arm_scale = _gaia_star_nonlinear_scales(rr)
        base_half_width = max(
            float(GAIA_STAR_MASK_ARM_HALF_WIDTH_MIN_PIX),
            float(GAIA_STAR_MASK_ARM_HALF_WIDTH_FRAC) * rr,
        )
        max_half_width = base_half_width
        if shape == "star8":
            max_half_width = max(
                max_half_width,
                base_half_width * float(GAIA_STAR_MASK_DIAG_ARM_WIDTH_SCALE),
            )
        max_len = max(
            float(GAIA_STAR_MASK_MAIN_ARM_LENGTH_FRAC) * rr,
            float(GAIA_STAR_MASK_DIAG_ARM_LENGTH_FRAC) * rr if shape == "star8" else 0.0,
        )
        if shape == "star12":
            max_len = max(
                max_len,
                float(GAIA_STAR_MASK_MAIN_ARM_LENGTH_FRAC) * rr
                * float(GAIA_STAR_MASK_ARM_LENGTH_SCALE_0_DEG),
                float(GAIA_STAR_MASK_MAIN_ARM_LENGTH_FRAC) * rr
                * float(GAIA_STAR_MASK_ARM_LENGTH_SCALE_180_DEG),
            )
        max_len *= arm_scale
        core_radius = max(1.0, float(GAIA_STAR_MASK_CORE_FRAC) * rr * core_scale)
        return max(rr, core_radius + max_len + max_half_width)

    spike_half_width = max(
        float(GAIA_JWST_MASK_SPIKE_HALF_WIDTH_MIN_PIX),
        float(GAIA_JWST_MASK_SPIKE_HALF_WIDTH_FRAC) * rr,
    )
    return max(rr, float(GAIA_JWST_MASK_SPIKE_LENGTH_FRAC) * rr + spike_half_width)


def build_gaia_exclusion_mask(
    shape: tuple[int, int],
    wcs: WCS,
    pixscale_arcsec: float,
    gaia_catalog: pd.DataFrame,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if gaia_catalog.empty:
        return None

    ny, nx = int(shape[0]), int(shape[1])
    ra = gaia_catalog["ra"].to_numpy(dtype=float)
    dec = gaia_catalog["dec"].to_numpy(dtype=float)
    mag = gaia_catalog["phot_g_mean_mag"].to_numpy(dtype=float)

    x, y = wcs.world_to_pixel_values(ra, dec)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x += float(getattr(args, "gaia_mask_xshift_pix", 0.0))
    y += float(getattr(args, "gaia_mask_yshift_pix", 0.0))

    radii_arcsec = _mag_to_radius_arcsec(
        mag,
        r0=float(args.gaia_r0),
        mag0=float(args.gaia_mag0),
        min_r=float(args.gaia_minr),
        max_r=float(args.gaia_maxr),
    )
    radii_arcsec = np.asarray(radii_arcsec, dtype=float) * float(GAIA_MASK_RADIUS_SCALE)
    pixscale = max(float(pixscale_arcsec), 1.0e-9)
    radii_pix = radii_arcsec / pixscale

    mask = np.zeros((ny, nx), dtype=bool)
    for cx, cy, rr in zip(x, y, radii_pix):
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(rr) and rr > 0):
            continue

        mask_extent = _gaia_star_mask_extent_pix(float(rr), args)
        x0 = max(0, int(np.floor(cx - mask_extent)))
        x1 = min(nx, int(np.ceil(cx + mask_extent)) + 1)
        y0 = max(0, int(np.floor(cy - mask_extent)))
        y1 = min(ny, int(np.ceil(cy + mask_extent)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx = np.asarray(xx, dtype=float) - float(cx)
        dy = np.asarray(yy, dtype=float) - float(cy)
        patch = _gaia_star_mask_patch(dx, dy, float(rr), args)
        mask[y0:y1, x0:x1] |= patch

    if not np.any(mask):
        return None
    return mask


__all__ = ["build_gaia_exclusion_mask"]

#!/usr/bin/env python3
"""
Interactive local web UI to tune RGB rendering from:
  R <- CFIS r
  G <- DECaLS g
  B <- CFIS u

By default, rendering uses CFIS as reference grid (so DECaLS is upsampled).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import astropy.units as u
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import map_coordinates

# Keep matplotlib/fontconfig caches in writable temp paths.
_tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_tmp_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp_root / ".cache"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RGB Tuner</title>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #131e34;
      --text: #d6dfef;
      --muted: #9fb2d6;
      --accent: #4cc9f0;
    }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at 20% 20%, #1a2744 0%, var(--bg) 45%);
      color: var(--text);
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }
    .controls {
      background: linear-gradient(180deg, #162441 0%, #101a2d 100%);
      padding: 16px 18px;
      border-right: 1px solid #243252;
      overflow: auto;
    }
    .viewer {
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 1.15rem;
      color: #f3f7ff;
    }
    p {
      margin: 0 0 12px 0;
      color: var(--muted);
      font-size: 0.88rem;
    }
    .row { margin-bottom: 10px; }
    label {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      color: #c5d3eb;
      margin-bottom: 3px;
    }
    input[type=number], select {
      width: 100%;
      box-sizing: border-box;
      background: #0d1526;
      color: var(--text);
      border: 1px solid #2a3a60;
      border-radius: 8px;
      padding: 6px 8px;
    }
    input[type=range] { width: 100%; }
    .btn {
      width: 100%;
      padding: 9px 10px;
      border: 0;
      border-radius: 10px;
      background: linear-gradient(90deg, #2d7ef7, var(--accent));
      color: #04101c;
      font-weight: 700;
      cursor: pointer;
      margin-top: 8px;
    }
    .btn:disabled { opacity: 0.6; cursor: default; }
    .status {
      font-size: 0.82rem;
      color: var(--muted);
      min-height: 1.2rem;
      margin-top: 6px;
    }
    .canvas-wrap {
      background: #05090f;
      border: 1px solid #28395f;
      border-radius: 10px;
      padding: 8px;
      flex: 1;
      min-height: 260px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    img {
      max-width: 100%;
      max-height: calc(100vh - 120px);
      border-radius: 6px;
    }
    .meta {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.75rem;
      color: #97add4;
      white-space: pre-wrap;
      background: #0c1425;
      border: 1px solid #24365c;
      border-radius: 8px;
      padding: 8px;
    }
  </style>
</head>
<body>
  <div class="controls">
    <h1>RGB Tuner</h1>
    <p>Tune background/core/color interactively. Default mode keeps CFIS resolution and upsamples DECaLS g.</p>

    <div class="row">
      <label>RA (deg)</label>
      <input id="ra" type="number" step="0.000001" value="141.8479167">
    </div>
    <div class="row">
      <label>Dec (deg)</label>
      <input id="dec" type="number" step="0.000001" value="30.4408611">
    </div>
    <div class="row">
      <label>Size (pixels)</label>
      <input id="size_pix" type="number" step="1" min="256" max="5000" value="1300">
    </div>
    <div class="row">
      <label>Reference Grid</label>
      <select id="reference_grid">
        <option value="cfis" selected>cfis (DECaLS upsampled)</option>
        <option value="decals">decals (CFIS downsampled)</option>
      </select>
    </div>

    <div class="row">
      <label>p_low <span id="p_low_val"></span></label>
      <input id="p_low" type="range" min="0.1" max="10" step="0.1" value="4.2">
    </div>
    <div class="row">
      <label>p_high <span id="p_high_val"></span></label>
      <input id="p_high" type="range" min="95" max="99.999" step="0.001" value="99.99">
      <input id="p_high_num" type="number" min="95" max="99.999" step="0.001" value="99.99" style="margin-top:4px;">
    </div>
    <div class="row">
      <label>asinh_a <span id="asinh_a_val"></span></label>
      <input id="asinh_a" type="range" min="1" max="80" step="0.1" value="20">
    </div>
    <div class="row">
      <label>gamma <span id="gamma_val"></span></label>
      <input id="gamma" type="range" min="0.1" max="2.2" step="0.005" value="0.6">
    </div>

    <div class="row">
      <label>R gain <span id="r_gain_val"></span></label>
      <input id="r_gain" type="range" min="0.3" max="2.0" step="0.01" value="0.36">
    </div>
    <div class="row">
      <label>G gain <span id="g_gain_val"></span></label>
      <input id="g_gain" type="range" min="0.3" max="2.0" step="0.01" value="0.46">
    </div>
    <div class="row">
      <label>B gain <span id="b_gain_val"></span></label>
      <input id="b_gain" type="range" min="0.3" max="2.0" step="0.01" value="0.36">
    </div>

    <button class="btn" id="render_btn">Render</button>
    <div class="status" id="status"></div>
  </div>

  <div class="viewer">
    <div class="canvas-wrap">
      <img id="preview" alt="RGB preview"/>
    </div>
    <div class="meta" id="meta">No render yet.</div>
  </div>

<script>
  const ids = ["p_low","p_high","asinh_a","gamma","r_gain","g_gain","b_gain"];
  const pHighRange = document.getElementById("p_high");
  const pHighNum = document.getElementById("p_high_num");
  function syncLabels() {
    ids.forEach(k => {
      document.getElementById(k + "_val").textContent = document.getElementById(k).value;
    });
    pHighNum.value = pHighRange.value;
  }
  ids.forEach(k => document.getElementById(k).addEventListener("input", syncLabels));
  pHighNum.addEventListener("input", () => {
    const v = parseFloat(pHighNum.value);
    if (Number.isFinite(v)) {
      const clamped = Math.min(99.999, Math.max(95.0, v));
      pHighRange.value = clamped.toFixed(3);
      syncLabels();
    }
  });
  syncLabels();

  async function render() {
    const btn = document.getElementById("render_btn");
    const status = document.getElementById("status");
    btn.disabled = true;
    status.textContent = "Rendering...";

    const payload = {
      ra: parseFloat(document.getElementById("ra").value),
      dec: parseFloat(document.getElementById("dec").value),
      size_pix: parseInt(document.getElementById("size_pix").value),
      reference_grid: document.getElementById("reference_grid").value,
      p_low: parseFloat(document.getElementById("p_low").value),
      p_high: parseFloat(document.getElementById("p_high").value),
      asinh_a: parseFloat(document.getElementById("asinh_a").value),
      gamma: parseFloat(document.getElementById("gamma").value),
      r_gain: parseFloat(document.getElementById("r_gain").value),
      g_gain: parseFloat(document.getElementById("g_gain").value),
      b_gain: parseFloat(document.getElementById("b_gain").value)
    };

    try {
      const resp = await fetch("/render", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) throw new Error(data.error || "render failed");

      document.getElementById("preview").src = data.image_url + "?t=" + Date.now();
      document.getElementById("meta").textContent = JSON.stringify(data.meta, null, 2);
      status.textContent = "Done";
    } catch (err) {
      status.textContent = "Error: " + err.message;
    } finally {
      btn.disabled = false;
    }
  }

  document.getElementById("render_btn").addEventListener("click", render);
  render();
</script>
</body>
</html>
"""


def robust_asinh_stretch(
    img: np.ndarray,
    p_low: float,
    p_high: float,
    asinh_a: float,
) -> np.ndarray:
    valid = np.isfinite(img)
    if not np.any(valid):
        return np.zeros_like(img, dtype=np.float32)
    values = img[valid]
    lo = float(np.percentile(values, p_low))
    hi = float(np.percentile(values, p_high))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    norm = (img - lo) / (hi - lo)
    norm = np.clip(np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    stretched = np.arcsinh(asinh_a * norm) / np.arcsinh(asinh_a)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def reproject_to_reference(
    src_data: np.ndarray,
    src_wcs: WCS,
    ref_wcs: WCS,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    h, w = ref_shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    ra, dec = ref_wcs.all_pix2world(xx, yy, 0)
    src_x, src_y = src_wcs.all_world2pix(ra, dec, 0)
    coords = np.array([src_y, src_x], dtype=np.float64)
    projected = map_coordinates(
        src_data,
        coords,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    return projected.astype(np.float32)


def world_center_from_wcs(wcs: WCS, shape: tuple[int, int]) -> tuple[float, float]:
    h, w = shape
    x0 = 0.5 * (w - 1)
    y0 = 0.5 * (h - 1)
    ra, dec = wcs.all_pix2world(x0, y0, 0)
    return float(ra), float(dec)


def mean_pixel_scale_arcsec(wcs: WCS) -> float:
    scales = proj_plane_pixel_scales(wcs) * 3600.0
    scales = np.asarray(scales, dtype=float)
    good = np.isfinite(scales) & (scales > 0)
    if not np.any(good):
        raise RuntimeError("Could not determine positive pixel scale from WCS")
    return float(np.mean(scales[good]))


def cutout_world(
    data: np.ndarray,
    wcs: WCS,
    ra: float,
    dec: float,
    size_y: int,
    size_x: int,
) -> tuple[np.ndarray, WCS]:
    center = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
    cut = Cutout2D(
        data,
        position=center,
        size=(int(size_y), int(size_x)),
        wcs=wcs,
        mode="trim",
        copy=True,
    )
    return np.asarray(cut.data, dtype=np.float32), cut.wcs


@dataclass
class FITSImage:
    path: Path
    hdul: fits.HDUList
    data: np.ndarray
    wcs: WCS


class RGBEngine:
    def __init__(self, r_path: Path, g_path: Path, u_path: Path):
        self.r = self._open(r_path)
        self.g = self._open(g_path)
        self.u = self._open(u_path)

    def _open(self, path: Path) -> FITSImage:
        hdul = fits.open(path, memmap=True)
        if len(hdul) == 0 or hdul[0].data is None:
            hdul.close()
            raise RuntimeError(f"No image data in FITS: {path}")
        data = hdul[0].data
        while data.ndim > 2:
            data = data[0]
        wcs = WCS(hdul[0].header)
        return FITSImage(path=path, hdul=hdul, data=data, wcs=wcs)

    def close(self):
        for item in (self.r, self.g, self.u):
            try:
                item.hdul.close()
            except Exception:
                pass

    def render(self, params: dict, out_path: Path) -> dict:
        ra = float(params["ra"])
        dec = float(params["dec"])
        size_pix = int(params.get("size_pix", 1300))
        reference_grid = str(params.get("reference_grid", "cfis")).strip().lower()
        p_low = float(params.get("p_low", 4.2))
        p_high = float(params.get("p_high", 99.99))
        asinh_a = float(params.get("asinh_a", 20.0))
        gamma = float(params.get("gamma", 0.6))
        r_gain = float(params.get("r_gain", 0.36))
        g_gain = float(params.get("g_gain", 0.46))
        b_gain = float(params.get("b_gain", 0.36))

        if not (0.0 <= ra < 360.0):
            raise ValueError("RA must be in [0, 360)")
        if not (-90.0 <= dec <= 90.0):
            raise ValueError("Dec must be in [-90, 90]")
        if size_pix <= 32:
            raise ValueError("size_pix must be > 32")
        if size_pix > 8000:
            raise ValueError("size_pix must be <= 8000")
        if reference_grid not in ("cfis", "decals"):
            raise ValueError("reference_grid must be 'cfis' or 'decals'")
        if p_high <= p_low:
            raise ValueError("p_high must be > p_low")
        if asinh_a <= 0 or gamma <= 0:
            raise ValueError("asinh_a and gamma must be > 0")
        if min(r_gain, g_gain, b_gain) <= 0:
            raise ValueError("channel gains must be > 0")

        if reference_grid == "decals":
            ref_data, ref_wcs = cutout_world(self.g.data, self.g.wcs, ra, dec, size_pix, size_pix)
            r_on_ref = reproject_to_reference(self.r.data, self.r.wcs, ref_wcs, ref_data.shape)
            g_on_ref = ref_data
            b_on_ref = reproject_to_reference(self.u.data, self.u.wcs, ref_wcs, ref_data.shape)
            ref_scale = mean_pixel_scale_arcsec(ref_wcs)
        else:
            decals_scale = mean_pixel_scale_arcsec(self.g.wcs)
            cfis_scale = mean_pixel_scale_arcsec(self.r.wcs)
            # Preserve sky footprint requested in DECaLS-sized pixels, converted to CFIS pixel grid.
            cfis_size = int(np.ceil(size_pix * decals_scale / cfis_scale))
            cfis_size = max(64, min(8000, cfis_size))
            ref_data, ref_wcs = cutout_world(self.r.data, self.r.wcs, ra, dec, cfis_size, cfis_size)
            r_on_ref = ref_data
            g_on_ref = reproject_to_reference(self.g.data, self.g.wcs, ref_wcs, ref_data.shape)
            b_on_ref = reproject_to_reference(self.u.data, self.u.wcs, ref_wcs, ref_data.shape)
            ref_scale = mean_pixel_scale_arcsec(ref_wcs)

        r_st = robust_asinh_stretch(r_on_ref, p_low=p_low, p_high=p_high, asinh_a=asinh_a) * r_gain
        g_st = robust_asinh_stretch(g_on_ref, p_low=p_low, p_high=p_high, asinh_a=asinh_a) * g_gain
        b_st = robust_asinh_stretch(b_on_ref, p_low=p_low, p_high=p_high, asinh_a=asinh_a) * b_gain
        rgb = np.dstack([r_st, g_st, b_st])
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = np.power(rgb, gamma)
        rgb = np.clip(rgb, 0.0, 1.0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(out_path, rgb, origin="lower")

        gray = rgb.mean(axis=2)
        sat_frac = float(((rgb >= 0.995).all(axis=2)).mean())
        near_black_frac = float((gray < 0.03).mean())
        center_ra, center_dec = world_center_from_wcs(ref_wcs, ref_data.shape)

        return {
            "shape": [int(ref_data.shape[1]), int(ref_data.shape[0])],
            "reference_grid": reference_grid,
            "center_ra_dec": [center_ra, center_dec],
            "pixel_scale_arcsec": ref_scale,
            "sat_frac": sat_frac,
            "near_black_frac": near_black_frac,
            "output": str(out_path),
        }


class RGBHandler(BaseHTTPRequestHandler):
    server_version = "RGBTuner/1.0"

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text: str):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(404, "Not found")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._html(HTML_PAGE)
            return
        if parsed.path.startswith("/images/"):
            filename = parsed.path.split("/", 2)[-1]
            safe = Path(filename).name
            self._file(self.server.output_dir / safe)  # type: ignore[attr-defined]
            return
        if parsed.path == "/health":
            self._json(200, {"ok": True})
            return
        self.send_error(404, "Not found")

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/render":
            self.send_error(404, "Not found")
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len)
            params = json.loads(raw.decode("utf-8"))
            stamp = int(time.time() * 1000)
            out_name = f"rgb_render_{stamp}.png"
            out_path = self.server.output_dir / out_name  # type: ignore[attr-defined]
            meta = self.server.engine.render(params, out_path)  # type: ignore[attr-defined]
        except Exception as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return

        self._json(
            200,
            {
                "ok": True,
                "image_url": f"/images/{out_name}",
                "meta": meta,
            },
        )

    def log_message(self, fmt: str, *args):
        # Keep console clean; only explicit startup message is printed.
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RGB tuning local web server.")
    parser.add_argument(
        "--r-fits",
        default="CFIS_LSB.244.241.r.fits",
        help="CFIS r FITS (default: CFIS_LSB.244.241.r.fits)",
    )
    parser.add_argument(
        "--g-fits",
        default="des_tiles/DECaLS_g_ra141.847917_dec30.440861_w2048.fits",
        help=(
            "DECaLS g FITS "
            "(default: des_tiles/DECaLS_g_ra141.847917_dec30.440861_w2048.fits)"
        ),
    )
    parser.add_argument(
        "--u-fits",
        default="CFIS.244.241.u.fits",
        help="CFIS u FITS (default: CFIS.244.241.u.fits)",
    )
    parser.add_argument(
        "--output-dir",
        default="des_tiles/rgb_tuner",
        help="Directory to store rendered PNGs",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7870, help="Bind port (default: 7870)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = RGBEngine(
        r_path=Path(args.r_fits),
        g_path=Path(args.g_fits),
        u_path=Path(args.u_fits),
    )

    class Server(ThreadingHTTPServer):
        pass

    server = Server((args.host, int(args.port)), RGBHandler)
    server.output_dir = output_dir  # type: ignore[attr-defined]
    server.engine = engine  # type: ignore[attr-defined]

    try:
        print(f"RGB tuner running on http://{args.host}:{args.port}")
        print(f"R={args.r_fits}")
        print(f"G={args.g_fits}")
        print(f"B={args.u_fits}")
        print(f"output_dir={output_dir}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()

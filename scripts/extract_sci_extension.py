#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from astropy.io import fits


def output_path_for(path: Path) -> Path:
    return path.with_name(f"{path.stem}_SCI.fits")


def extract_sci(input_path: Path, overwrite: bool = True) -> Path:
    out_path = output_path_for(input_path)

    with fits.open(input_path) as hdul:
        sci_hdu = None
        for hdu in hdul:
            if hdu.header.get("EXTNAME", "").strip().upper() == "SCI":
                sci_hdu = hdu.copy()
                break

        if sci_hdu is None:
            raise RuntimeError(f"Nenhuma extensão SCI encontrada em {input_path}")

        primary = fits.PrimaryHDU()
        out_hdul = fits.HDUList([primary, sci_hdu])
        out_hdul.writeto(out_path, overwrite=overwrite)

    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extrai a extensão SCI de arquivos FITS e salva como <nome>_SCI.fits"
    )
    ap.add_argument("fits", nargs="+", help="Arquivo(s) FITS para processar")
    ap.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Não sobrescrever arquivos de saída existentes",
    )
    args = ap.parse_args()

    files = [Path(p) for p in args.fits]

    overwrite = not args.no_overwrite
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        if path.name.endswith("_SCI.fits"):
            print(f"Pulando arquivo já extraído: {path.name}")
            continue
        out = extract_sci(path, overwrite=overwrite)
        print(f"{path.name} -> {out.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

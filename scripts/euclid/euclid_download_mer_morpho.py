#!/usr/bin/env python3
"""
download_euclid_morpho_catalog_robust.py

Baixa uma amostra do Euclid Q1 MER catalogue via IRSA TAP com query simples
e faz os filtros de morfometria no Python.

Saída:
    id,ra,dec,size

Uso:
    python download_euclid_morpho_catalog_robust.py \
        --output euclid_morpho.csv \
        --top-fetch 5000 \
        --top-save 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from astroquery.ipac.irsa import Irsa

TABLE_NAME = "euclid_q1_mer_catalogue"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("euclid_morpho_catalog.csv"))
    p.add_argument("--top-fetch", type=int, default=5000,
                   help="Quantas linhas buscar do TAP antes de filtrar no Python.")
    p.add_argument("--top-save", type=int, default=100,
                   help="Quantas linhas salvar no CSV final.")
    p.add_argument("--snr-min", type=float, default=30.0)
    p.add_argument("--extended-prob-min", type=float, default=0.9)
    p.add_argument("--point-like-prob-max", type=float, default=0.1)
    p.add_argument("--fwhm-min", type=float, default=0.25)
    p.add_argument("--size-factor", type=float, default=6.0)
    p.add_argument("--size-min", type=float, default=3.0)
    p.add_argument("--size-max", type=float, default=15.0)
    p.add_argument("--ra-min", type=float, default=None)
    p.add_argument("--ra-max", type=float, default=None)
    p.add_argument("--dec-min", type=float, default=None)
    p.add_argument("--dec-max", type=float, default=None)
    return p.parse_args()


def build_simple_query(args: argparse.Namespace) -> str:
    # Só filtros simples, sem CASE, sem divisão, sem OR complicado.
    where = [
        "object_id IS NOT NULL",
        "ra IS NOT NULL",
        "dec IS NOT NULL",
        "fwhm IS NOT NULL",
        "flux_vis_psf IS NOT NULL",
        "fluxerr_vis_psf IS NOT NULL",
        "extended_prob IS NOT NULL",
        "flux_vis_psf > 0",
        "fluxerr_vis_psf > 0",
        "fwhm > 0",
    ]

    if args.ra_min is not None:
        where.append(f"ra >= {args.ra_min}")
    if args.ra_max is not None:
        where.append(f"ra <= {args.ra_max}")
    if args.dec_min is not None:
        where.append(f"dec >= {args.dec_min}")
    if args.dec_max is not None:
        where.append(f"dec <= {args.dec_max}")

    where_clause = " AND ".join(where)

    q = f"""
SELECT TOP {args.top_fetch}
    object_id,
    ra,
    dec,
    fwhm,
    flux_vis_psf,
    fluxerr_vis_psf,
    extended_prob,
    point_like_prob
FROM {TABLE_NAME}
WHERE {where_clause}
ORDER BY flux_vis_psf DESC
"""
    return q.strip()


def main() -> int:
    args = parse_args()
    query = build_simple_query(args)

    print("Executando query TAP simples...")
    print(query)
    print()

    try:
        result = Irsa.query_tap(query)
        table = result.to_table()
    except Exception as exc:
        print(f"Erro na query TAP: {exc}", file=sys.stderr)
        return 2

    if len(table) == 0:
        print("Nenhuma linha retornada.")
        return 0

    df = table.to_pandas()

    # Garantir tipos numéricos
    for col in [
        "ra", "dec", "fwhm", "flux_vis_psf", "fluxerr_vis_psf",
        "extended_prob", "point_like_prob"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["object_id"] = df["object_id"].astype(str)

    # Filtros morfológicos no Python
    df = df.dropna(subset=["object_id", "ra", "dec", "fwhm",
                           "flux_vis_psf", "fluxerr_vis_psf", "extended_prob"]).copy()

    df["snr_vis"] = df["flux_vis_psf"] / df["fluxerr_vis_psf"]

    mask = (
        (df["snr_vis"] >= args.snr_min) &
        (df["extended_prob"] >= args.extended_prob_min) &
        (df["fwhm"] >= args.fwhm_min)
    )

    # point_like_prob pode ser NaN; nesse caso, aceita
    if "point_like_prob" in df.columns:
        mask &= (
            df["point_like_prob"].isna() |
            (df["point_like_prob"] <= args.point_like_prob_max)
        )

    df = df.loc[mask].copy()

    if len(df) == 0:
        print("Nenhuma fonte passou nos filtros no Python.")
        return 0

    # size do stamp em arcsec
    df["size"] = args.size_factor * df["fwhm"]
    df["size"] = df["size"].clip(lower=args.size_min, upper=args.size_max)

    # Ordena por S/N no Python
    df = df.sort_values("snr_vis", ascending=False)

    # Limita saída final
    df = df.head(args.top_save)

    out = pd.DataFrame({
        "id": df["object_id"],
        "ra": df["ra"],
        "dec": df["dec"],
        "size": df["size"],
    })

    out.to_csv(args.output, index=False)

    print(f"Fontes buscadas do TAP: {len(table)}")
    print(f"Fontes após filtro: {len(df)}")
    print(f"Arquivo salvo: {args.output}")
    print()
    print(out.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

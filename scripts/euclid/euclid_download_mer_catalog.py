#!/usr/bin/env python3
"""
download_euclid_mer_catalog.py

Baixa o catálogo Euclid Q1 MER da IRSA TAP.

Tabela:
    euclid_q1_mer_catalogue

Modos:
1. Baixar tudo (todas as colunas) em chunks
2. Baixar um subconjunto de colunas
3. Contar quantas linhas existem
4. Salvar em CSV ou FITS

Exemplos:
    python download_euclid_mer_catalog.py --count

    python download_euclid_mer_catalog.py \
        --output euclid_mer_subset.csv \
        --columns object_id,ra,dec,fwhm,flux_vis_psf,fluxerr_vis_psf \
        --chunk-size 200000

    python download_euclid_mer_catalog.py \
        --output euclid_mer_full.csv \
        --all-columns \
        --chunk-size 100000

Dependências:
    pip install astroquery astropy pandas
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from astroquery.ipac.irsa import Irsa
from astropy.table import vstack, Table


TABLE_NAME = "euclid_q1_mer_catalogue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa o catálogo Euclid Q1 MER da IRSA TAP em chunks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("euclid_q1_mer_catalogue.csv"),
        help="Arquivo de saída (.csv ou .fits). Default: euclid_q1_mer_catalogue.csv",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="object_id,ra,dec",
        help=(
            "Lista de colunas separadas por vírgula. "
            "Ignorado se --all-columns for usado."
        ),
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Baixa todas as colunas da tabela.",
    )
    parser.add_argument(
        "--where",
        type=str,
        default="",
        help=(
            "Cláusula WHERE sem a palavra WHERE. "
            "Ex.: \"flux_vis_psf > 0 AND fluxerr_vis_psf > 0\""
        ),
    )
    parser.add_argument(
        "--order-by",
        type=str,
        default="object_id",
        help="Coluna(s) para ORDER BY. Default: object_id",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Número de linhas por chunk. Default: 100000",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limita o número total de linhas baixadas.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Só conta o número de linhas e sai.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Pausa em segundos entre chunks. Default: 0",
    )
    return parser.parse_args()


def run_tap(query: str) -> Table:
    return Irsa.query_tap(query).to_table()


def count_rows(where_clause: str) -> int:
    q = f"SELECT COUNT(*) AS n FROM {TABLE_NAME}"
    if where_clause.strip():
        q += f" WHERE {where_clause}"
    tab = run_tap(q)
    return int(tab["n"][0])


def build_select_query(
    columns: str,
    where_clause: str,
    order_by: str,
    chunk_size: int,
    offset: int,
) -> str:
    q = (
        f"SELECT {columns} "
        f"FROM {TABLE_NAME} "
    )
    if where_clause.strip():
        q += f"WHERE {where_clause} "
    q += f"ORDER BY {order_by} "
    q += f"OFFSET {offset} ROWS FETCH NEXT {chunk_size} ROWS ONLY"
    return q


def fetch_chunk(
    columns: str,
    where_clause: str,
    order_by: str,
    chunk_size: int,
    offset: int,
) -> Table:
    query = build_select_query(
        columns=columns,
        where_clause=where_clause,
        order_by=order_by,
        chunk_size=chunk_size,
        offset=offset,
    )
    return run_tap(query)


def table_to_pandas(tab: Table) -> pd.DataFrame:
    return tab.to_pandas()


def append_csv(df: pd.DataFrame, path: Path, write_header: bool) -> None:
    mode = "w" if write_header else "a"
    df.to_csv(path, mode=mode, header=write_header, index=False)


def main() -> int:
    args = parse_args()

    if args.all_columns:
        columns = "*"
    else:
        columns = ",".join([c.strip() for c in args.columns.split(",") if c.strip()])
        if not columns:
            print("Erro: nenhuma coluna válida em --columns", file=sys.stderr)
            return 1

    # Conta linhas, se pedido
    if args.count:
        try:
            n = count_rows(args.where)
            print(n)
            return 0
        except Exception as exc:
            print(f"Erro ao contar linhas: {exc}", file=sys.stderr)
            return 2

    output = args.output
    suffix = output.suffix.lower()
    if suffix not in {".csv", ".fits"}:
        print("Erro: use saída .csv ou .fits", file=sys.stderr)
        return 1

    # Descobre total de linhas
    try:
        total_rows = count_rows(args.where)
    except Exception as exc:
        print(f"Erro ao contar linhas: {exc}", file=sys.stderr)
        return 2

    if args.max_rows is not None:
        total_rows = min(total_rows, args.max_rows)

    if total_rows == 0:
        print("Nenhuma linha encontrada.")
        return 0

    print(f"Total de linhas a baixar: {total_rows}")

    fetched = 0
    offset = 0
    chunk_idx = 0
    all_tables = []

    while fetched < total_rows:
        this_chunk = min(args.chunk_size, total_rows - fetched)
        print(
            f"Baixando chunk {chunk_idx + 1} | "
            f"offset={offset} | rows={this_chunk}"
        )

        try:
            tab = fetch_chunk(
                columns=columns,
                where_clause=args.where,
                order_by=args.order_by,
                chunk_size=this_chunk,
                offset=offset,
            )
        except Exception as exc:
            print(f"Erro ao baixar chunk offset={offset}: {exc}", file=sys.stderr)
            return 3

        if len(tab) == 0:
            print("Chunk vazio recebido; encerrando.")
            break

        if suffix == ".csv":
            df = table_to_pandas(tab)
            append_csv(df, output, write_header=(chunk_idx == 0))
        else:
            all_tables.append(tab)

        got = len(tab)
        fetched += got
        offset += got
        chunk_idx += 1

        print(f"  -> recebido {got} | acumulado {fetched}/{total_rows}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    if suffix == ".fits":
        if not all_tables:
            print("Nada para gravar.")
            return 0
        merged = vstack(all_tables, metadata_conflicts="silent")
        merged.write(output, overwrite=True)

    print(f"Arquivo salvo em: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
import getpass

#Force the archive to the central mirror 
#Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" 


# Use this if you need higher quotas or private tables
#user = "fferrari"
#Gaia.login(user=user, password=getpass.getpass(f"Gaia password for {user}: "))



def get_2d_wcs_and_shape(fits_path: str, ext: int | None):
    hdul = fits.open(fits_path)
    try:
        if ext is None:
            # primeiro HDU com dados >= 2D
            for i, hdu in enumerate(hdul):
                if getattr(hdu, "data", None) is None:
                    continue
                if hdu.data is not None and hdu.data.ndim >= 2:
                    ext = i
                    break
            if ext is None:
                raise RuntimeError("Não achei nenhum HDU com dados 2D (ou maiores).")

        hdu = hdul[ext]
        data = hdu.data
        if data is None or data.ndim < 2:
            raise RuntimeError(f"HDU {ext} não tem imagem 2D.")

        # Se for cube, pega o último plano (ajuste se quiser outro)
        if data.ndim > 2:
            data2d = data[-1, ...]
        else:
            data2d = data

        w = WCS(hdu.header)
        ny, nx = data2d.shape
        return w, (nx, ny), ext
    finally:
        hdul.close()


def footprint_polygon_icrs(w: WCS, nx: int, ny: int) -> np.ndarray:
    """
    Retorna os cantos do frame (ra,dec) em graus, array (4,2), sistema ICRS.
    """
    try:
        fp = w.calc_footprint(axes=(nx, ny))  # (4,2) em deg (ra,dec)
        return np.array(fp, dtype=float)
    except Exception:
        corners = np.array([
            [0,    0],
            [nx-1, 0],
            [nx-1, ny-1],
            [0,    ny-1],
        ], dtype=float)
        sky = w.pixel_to_world(corners[:, 0], corners[:, 1])
        return np.vstack([sky.ra.deg, sky.dec.deg]).T


GAIA_SOURCE_COLUMNS = {
    "solution_id",
    "designation",
    "source_id",
    "random_index",
    "ref_epoch",
    "ra",
    "ra_error",
    "dec",
    "dec_error",
    "parallax",
    "parallax_error",
    "pm",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "radial_velocity",
    "radial_velocity_error",
    "ruwe",
    "non_single_star",
    "in_qso_candidates",
    "in_galaxy_candidates",
}

GAIA_ASTROPHYS_COLUMNS = {
    "classprob_dsc_combmod_star",
    "classprob_dsc_combmod_galaxy",
    "classprob_dsc_combmod_quasar",
}


def qualify_adql_columns(cols: str) -> str:
    qualified: list[str] = []
    for raw_col in cols.split(","):
        col = raw_col.strip()
        if not col:
            continue
        if "." in col or "(" in col or " " in col:
            qualified.append(col)
            continue
        if col in GAIA_ASTROPHYS_COLUMNS:
            qualified.append(f"ap.{col}")
        elif col in GAIA_SOURCE_COLUMNS:
            qualified.append(f"gs.{col}")
        else:
            qualified.append(col)
    return ", ".join(qualified)


def build_adql_polygon(
    fp_deg: np.ndarray,
    table: str,
    cols: str,
    mag_limit: float | None,
    only_stars: bool,
    star_prob_min: float,
):
    # ADQL POLYGON: POLYGON('ICRS', ra1, dec1, ra2, dec2, ...)
    coords = ", ".join([f"{r:.12f}, {d:.12f}" for r, d in fp_deg])
    poly = f"POLYGON('ICRS', {coords})"
    select_cols = qualify_adql_columns(cols)

    where = f"CONTAINS(POINT('ICRS', gs.ra, gs.dec), {poly}) = 1"
    if mag_limit is not None:
        where += f" AND gs.phot_g_mean_mag <= {mag_limit}"
    if only_stars:
        where += f"""
        AND ap.classprob_dsc_combmod_star IS NOT NULL
        AND ap.classprob_dsc_combmod_galaxy IS NOT NULL
        AND ap.classprob_dsc_combmod_quasar IS NOT NULL
        AND ap.classprob_dsc_combmod_star >= {float(star_prob_min)}
        AND ap.classprob_dsc_combmod_star > ap.classprob_dsc_combmod_galaxy
        AND ap.classprob_dsc_combmod_star > ap.classprob_dsc_combmod_quasar
        """

    query = f"""
    SELECT {select_cols}
    FROM {table} AS gs
    JOIN gaiadr3.astrophysical_parameters AS ap USING (source_id)
    WHERE {where}
    """
    return query


def mag_to_radius_arcsec(mag, r0=8.0, mag0=15.0, min_r=1.0, max_r=40.0):
    """
    Raio em arcsec ~ r0 * 10^(-0.2*(mag-mag0)).
    Brilho maior (mag menor) => raio maior.
    Clampa em [min_r, max_r].
    """
    m = np.array(mag, dtype=float)
    r = r0 * (10.0 ** (-0.2 * (m - mag0)))
    return np.clip(r, min_r, max_r)


def write_ds9_regions_fk5(tab, reg_path, mag_col="phot_g_mean_mag",
                          r0=8.0, mag0=15.0, min_r=1.0, max_r=40.0,
                          color="green", width=1, show_text=False):
    """
    DS9 regions (fk5) com círculos em RA/Dec e raio proporcional à magnitude.
    """
    if mag_col not in tab.colnames:
        raise RuntimeError(f"Coluna '{mag_col}' não existe na tabela. Colunas: {tab.colnames}")

    radii = mag_to_radius_arcsec(tab[mag_col], r0=r0, mag0=mag0, min_r=min_r, max_r=max_r)

    with open(reg_path, "w", encoding="utf-8") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(
            f'global color={color} width={width} font="helvetica 10 normal" '
            f"select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0\n"
        )
        f.write("fk5\n")
        for row, r_arcsec in zip(tab, radii):
            ra = float(row["ra"])
            dec = float(row["dec"])
            if show_text:
                sid = row["source_id"] if "source_id" in row.colnames else ""
                magv = float(row[mag_col])
                f.write(f'circle({ra:.10f},{dec:.10f},{r_arcsec:.3f}") # text={{id={sid} G={magv:.2f}}}\n')
            else:
                f.write(f'circle({ra:.10f},{dec:.10f},{r_arcsec:.3f}")\n')


def main():
    ap = argparse.ArgumentParser(
        description="Lê WCS de um FITS, consulta Gaia no footprint, salva CSV e DS9 regions."
    )
    ap.add_argument("fits", help="caminho do FITS")
    ap.add_argument("--ext", type=int, default=None, help="HDU/extensão com a imagem (default: auto)")

    ap.add_argument("--table", default="gaiadr3.gaia_source", help="tabela Gaia (default: gaiadr3.gaia_source)")
    ap.add_argument(
        "--cols",
        default=(
            "source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, "
            "classprob_dsc_combmod_star, classprob_dsc_combmod_galaxy, classprob_dsc_combmod_quasar"
        ),
        help="colunas ADQL"
    )
    ap.add_argument("--mag", type=float, default=None, help="corte em G (phot_g_mean_mag <= mag)")
    ap.add_argument("--row-limit", type=int, default=200000, help="limite de linhas (segurança)")
    ap.add_argument(
        "--allow-nonstellar",
        action="store_true",
        help="desliga o filtro de classe e retorna qualquer fonte Gaia no footprint",
    )
    ap.add_argument(
        "--star-prob-min",
        type=float,
        default=0.8,
        help="probabilidade minima em classprob_dsc_combmod_star para aceitar a fonte como estrela (default: 0.8)",
    )

    ap.add_argument("--out", default=None, help="arquivo CSV de saída (default: <fits>_gaia.csv)")
    ap.add_argument("--add-xy", action="store_true", help="adiciona x,y (pixels) via WCS ao CSV")

    # DS9
    ap.add_argument("--ds9-magcol", default="phot_g_mean_mag", help="coluna de magnitude p/ escala do DS9")
    ap.add_argument("--ds9-r0", type=float, default=8.0, help="raio (arcsec) em mag0")
    ap.add_argument("--ds9-mag0", type=float, default=15.0, help="magnitude de referência p/ r0")
    ap.add_argument("--ds9-minr", type=float, default=1.0, help="raio mínimo (arcsec)")
    ap.add_argument("--ds9-maxr", type=float, default=40.0, help="raio máximo (arcsec)")
    ap.add_argument("--ds9-color", default="green", help="cor no DS9")
    ap.add_argument("--ds9-width", type=int, default=1, help="largura da linha no DS9")
    ap.add_argument("--ds9-text", action="store_true", help="coloca texto com source_id e magnitude")

    args = ap.parse_args()

    w, (nx, ny), used_ext = get_2d_wcs_and_shape(args.fits, args.ext)
    fp = footprint_polygon_icrs(w, nx, ny)

    Gaia.ROW_LIMIT = args.row_limit
    query = build_adql_polygon(
        fp,
        args.table,
        args.cols,
        args.mag,
        only_stars=not args.allow_nonstellar,
        star_prob_min=args.star_prob_min,
    )
    #job = Gaia.launch_job_async(query)
    # synchronous call
    job = Gaia.launch_job(query)
    tab = job.get_results()

    if args.add_xy and len(tab) > 0:
        sc = SkyCoord(tab["ra"], tab["dec"], unit="deg", frame="icrs")
        x, y = w.world_to_pixel(sc)
        tab["x"] = x
        tab["y"] = y

    csv_path = args.out or str(Path(args.fits).with_suffix("")) + "_gaia.csv"
    tab.write(csv_path, format="csv", overwrite=True)
    print(f"Wrote {len(tab)} rows to {csv_path}  (HDU ext={used_ext}, nx={nx}, ny={ny})")

    ds9_path = str(Path(args.fits).with_suffix("")) + "_gaia.reg"
    write_ds9_regions_fk5(
        tab, ds9_path,
        mag_col=args.ds9_magcol,
        r0=args.ds9_r0, mag0=args.ds9_mag0,
        min_r=args.ds9_minr, max_r=args.ds9_maxr,
        color=args.ds9_color, width=args.ds9_width,
        show_text=args.ds9_text
    )
    print(f"Wrote DS9 regions to {ds9_path}")


if __name__ == "__main__":
    sys.exit(main())

from astroquery.ipac.irsa import Irsa

q = """
SELECT TOP 5 object_id, ra, dec, fwhm
FROM euclid_q1_mer_catalogue
WHERE flux_vis_psf > 0
"""

q2 = """
SELECT TOP 1000 object_id, ra, dec, fwhm, flux_vis_psf, fluxerr_vis_psf,
       extended_prob, point_like_prob
FROM euclid_q1_mer_catalogue
WHERE flux_vis_psf > 0
ORDER BY flux_vis_psf DESC
"""


tab = Irsa.query_tap(q2).to_table()
print(tab)

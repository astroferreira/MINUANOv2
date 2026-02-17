import pandas as pd
import numpy as np

from skimage.transform import resize
import numpy as np
import pandas as pd

from astropy.io import fits


def normalize(im_dat, all_size=128):
    im_dat[np.isnan(im_dat)] = 0
    im_dat[im_dat < -10] = 0
    im_dat = resize(im_dat,(all_size, all_size), anti_aliasing=True)
    im_dat = im_dat-np.amin(im_dat)
    im_dat = im_dat/np.amax(im_dat)
   
    return im_dat
def CFIS_tile_radec(ra, dec):
    # return tile name (see Stacking in docs)
    # https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/community/cfis/datadoc.html
    # CFIS tiles centers are on a cartesian grid spaced by exactly 0.5 degrees in RA
    # but tiles cover a sphere, so translation in Dec is more involved
    yyy = int(np.rint((dec+90)*2))
    cosf = np.cos((yyy/2-90)*np.pi/180.)
    xxx = int(np.rint(ra*2*cosf))
    tile = f'CFIS_LSB.{xxx:03d}.{yyy:03d}.r.fits'
    return tile

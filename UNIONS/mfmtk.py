#!/usr/bin/env python
r"""
                          __   __  
    Morfometryka         (  \,/  ) 
                          \_ | _/  
  Fabricio Ferrari        (_/ \_)  
fabricio@ferrari.pro.br  2012-2025

arxiv.org/abs/1509.05430
arxiv.org/abs/1707.02863
arxiv.org/abs/1907.10188
"""

__version__ = 9.66
__author__  = 'Fabricio Ferrari, Leonardo Ferreira, Geferson Lucatelli'
__email__   = 'fabricio@ferrari.pro.br'
__date__    = '20250328'


r'''
 _                            _   
(_)_ __ ___  _ __   ___  _ __| |_ 
| | '_ ` _ \| '_ \ / _ \| '__| __|
| | | | | | | |_) | (_) | |  | |_ 
|_|_| |_| |_| .__/ \___/|_|   \__|
            |_|                   
'''

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)


import os
import sys

# Limit numerical backends to a single thread per process to avoid oversubscription.
for _var in (
   "OMP_NUM_THREADS",
   "OPENBLAS_NUM_THREADS",
   "MKL_NUM_THREADS",
   "VECLIB_MAXIMUM_THREADS",
   "NUMEXPR_NUM_THREADS",
):
   os.environ[_var] = "1"

# must be called before anything that deals with Matplotlib
if 'noshow' in sys.argv:
   from matplotlib import use as mpluse
   mpluse('Agg')
   

import morfometrykalibstable96 as mfmlib 
#from morfometrykalib96  import galclean

import numpy as np
import numpy.ma as ma
np.round_ = np.round

from scipy.stats   import spearmanr
from scipy.optimize  import   leastsq, fmin, curve_fit
import scipy.interpolate
import scipy.signal as sig
import scipy.ndimage as nd


import matplotlib.pyplot as pl
import astropy.io.fits as pyfits # just to be backward compatible
import photutils as pu 
from astropy.stats import sigma_clipped_stats




###TOBE### poderia colcoar dentro do Config ?
# General setup
np.random.seed(10)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def log_message(level, msg):
    if config.verbose >= level:
        print(msg)



r'''               __ _       
  ___ ___  _ __  / _(_) __ _ 
 / __/ _ \| '_ \| |_| |/ _` |
| (_| (_) | | | |  _| | (_| |
 \___\___/|_| |_|_| |_|\__, |
                       |___/ 
'''
class config():
   '''
   VERBOSE levels
   0    nothing at all (only on log .ftmtk file)
   1    results on terminal
   2    steps and measurements 
   3    detailed measurements and steps  
   4    debugging stuff
   5    all graphics
   '''
   verbose = 2
   ## OUTPUT
   # output file format: 'CSV' or 'ASCII'
   outformat = 'CSV' 
   # what to save 'full' or 'brief'
   outlevel  = 'full' 

   ## STARMASKING (OUTDATED!!!!!!!!!!!!!)
   # how many sigmas above mean to mask out stars
   StarSigma = 10.0
   # Star are only searched outside StarRmin
   StarRmin  = 10.0 

   ## SEGMENTATION
   # routines to use in segmentation
   # 'mfmtk' or 'astropy'
   Segmentation='astropy'
   #
   # Region of Interest (if ALL or unsure leave an empty string '')
   #ROI = '[50:200, 50:200]'
   ROI = ''

   #How many sigmas over the bakground to select sources (usually 10 or greater)
   # normal galaxy: segK=20 segS=1/15.
   #          SDSS: segK=25 segS=1/18 
   #  small stamps: segK=25 segS=1/30
   # minimium size of the segmented region (in pixels)
   MinSegSz = 5**2
   # initial number of stddev  above median to segmentate (default 5)
   SegThreshold = 5
   # median fraction window size relative to of the image size (default 1/50)
   segS = 1/50.
   # galaxy center maximum offset allowed
   CenterOffset = 50
   # OffCenterThreshold
   # fraction of image diagonal size to consider QF OFFCENTER
   OffCenterThreshold = 0.1

   # segmentation region selection criteria
   # "position" selects the object closer to the stamp center
   # "size" the greater segemented region
   # "header" searches the Fits header for object position 
   selectby = 'position' # 'position' or 'size'  or 'header'

   # GalClean
   GCstdlevel=5.0

   # PHOTOMETRY
   # if consider ellipses center on the light PEAK center or CENTER-OF-LIGHT center
   photcenter = 'center'  # 'peak'  or 'center'
   # psf mask
   # how many PSF FWHM to take out at the center
   Npsf = 1.0
   PSFmethod = 'fit' # or momenta or anything except 'fit' :-)
   # pixel step in R for aperture photometry
   deltaR = 1.0


   # ASYMMETRY
   # size of the box for Asymmetry sky estimate
   #skyboxsz = 10
   # How many dispersions over median to use to segmentate Asymmetry and Smoothness maps
   #  AsySigma=5 and SmooSigma=3  are fine for most applications 
   AsySigma     = 5.00
   SmooSigma    = 5.00 
   AsyFitCenter = False  
   
   ## PETROSIAN REGION
   #How many Petrosian radii to use
   NRp = 2.0
   NRp_galpetro = NRp
   NRp_profile  = NRp
   #Petrosian Eta function value where Rp is defined
   # DO NOT TOUCH THIS, EVER :-) 
   PetroEta0 = 0.2



log_message(1,__doc__)
log_message(1,'    MFMTK G3 v.'+str(__version__)+'\n')



r'''
  _  _     _  _         _                            _  _     _  _   
 _| || |_ _| || |_   ___| |_ __ _ _ __ ___  _ __    _| || |_ _| || |_ 
|_  ..  _|_  ..  _| / __| __/ _` | '_ ` _ \| '_ \  |_  ..  _|_  ..  _|
|_      _|_      _| \__ \ || (_| | | | | | | |_) | |_      _|_      _|
  |_||_|   |_||_|   |___/\__\__,_|_| |_| |_| .__/    |_||_|   |_||_|  
                                           |_|                  
'''
class Stamp():
   '''
   Main class for OBJECT,GALAXY,IMAGE,STAMP and properties
   '''

   def __init__(self, galfilename, psffilename):

      log_message(2, "[ -------------------- STAMP -------------------- ]" )

      self.rootname = galfilename.replace('.fits', '') 
      self.basename = os.path.basename( self.rootname)

      # Edit here according to some pattern 
      # if not defined, use filename from command line
      if 'psfauto' in sys.argv:
          self.psffilename = self.psffilename_file()
      else:
          self.psffilename = psffilename

      self.read_data(galfilename)
      self.read_psf(self.psffilename)

      log_message(3, '   Estimating sky')
      self.skymed, self.skymad, self.skybg, self.skybgstd = self.skybg_iterative(self.gal0)



      if 'clean' in sys.argv:
         """ As GalClean is largely untested, this block defines
             a fallback for when it fails. The clean flag can
             be added to the output so one can know if GalClean
             was used or not. Particulary useful when running
             Morfometryka in bulk data where is costly to rerun
             it without the clean keyword.
         """
         try:
            self.gal0clean = galclean( self.gal0 - self.skybg, std_level=config.GCstdlevel, show=False, save=False)
            self.galOriginal = self.gal0.copy() - self.skybg
            self.gal0 = self.gal0clean
            self.clean = 1
         except Exception as e:
            print('GalClean Fail', e)
            self.gal0 = self.gal0 - self.skybg
            self.clean = 0
      else:
         self.gal0 =  self.gal0 - self.skybg


      
      self.segmentation()
      self.astrometry()
      self.geom_measures()
      self.peak_center()
      

      if config.verbose>1:
         print ('   OBJ file=%s' % self.basename+'.fits', '(%i,%i)' % (self.Mo,self.No), 'asecpix=%.3f' % self.asecpix)
         print ('   PSF file=%s' % os.path.basename(self.psffilename), '(%i,%i)' % self.psf.shape, 'FWHM=%.2f' % self.psffwhm, 'sigma=%.2f' % self.psfsigma)
         print ('   skybg=(%.2f,%.2f)'% (self.skybg, self.skybgstd)) 
         print ('   (x0,y0)_col=%.1f,%.1f'  % (self.x0col, self.y0col), '(x0,y0)_peak=%.1f,%.1f'  % (self.x0peak, self.y0peak)) 
         print ('   a,b=%.1f,%.1f' % (self.a,self.b), 'q=%.2f' % self.q,  'PA=%.2f' % self.PAdeg)  


   def geom_measures(self):
      'calculates a,b,q, PA from the image moments'


      if config.Segmentation == 'mfmtk':
         m00, m10, m01, m11, m20, m02, mu20, mu02, mu11 = self.momenta(self.gal0seg)
         # centroids
         x0  = m10/m00
         y0  = m01/m00
         # manor, minor and axis ratio
         lam1 = np.sqrt(abs(  (1/2.) * (mu20+mu02 + np.sqrt((mu20-mu02)**2 + 4*mu11**2))   )/m00)
         lam2 = np.sqrt(abs(  (1/2.) * (mu20+mu02 - np.sqrt((mu20-mu02)**2 + 4*mu11**2))   )/m00)
         a = max(lam1,lam2)
         b = min(lam1,lam2)
         PA = (1/2.) * np.arctan2(2*mu11,(mu20-mu02))

      elif config.Segmentation == 'astropy':
         x0  = self.photcat.xcentroid
         y0  = self.photcat.ycentroid
         a   = self.photcat.semimajor_sigma.value
         b   = self.photcat.semiminor_sigma.value
         PA  = self.photcat.orientation.value


      self.x0col = x0
      self.y0col = y0
      self.a  = a 
      self.b  = b
      self.q = self.b/self.a
      self.PAdeg = PA
      if self.PAdeg < 0:
         self.PAdeg = PA+180
      self.PArad = np.deg2rad(self.PAdeg)


   def psffilename_file(self):
      '''
      when the psf name can be created from the filename, use a function 
      like this in the read_psf function to give  psfname
      useful from makemySDSS stamps

      '''
      # leave run = '' if the stamps are not store in folders with SDSS run number
      # run=''
      # otherwise leave as:
      #run = self.rootname.split('/')[1] + '/x'

      #psfname = '../psf_r/' + run + 'psf_' +   self.basename + '.fits'
      psfname = 'psf_' +   self.basename + '.fits'

      return psfname



   def read_data(self,galfilename):
      '''
      Read Data from fits file
      '''
      # original galaxy, header and size
      self.gal0,self.hdr = pyfits.getdata(galfilename, header=1)

      if config.ROI != '':
         self.gal0 = eval( 'self.gal0' + config.ROI )

      self.Mo,self.No = self.gal0.shape
      #self.gal0 = 1e10 * self.gal0

      try:
         self.x0header = self.hdr['OBJXPIX']
         self.y0header = self.hdr['OBJYPIX']
      except:
         pass
         


   def read_psf(self,psfname):
      '''
      try to use argv[2] as psfsize or in case float() conversion fails,
      then argv[2]  is a psf filename
      
      in either case, defines:
         self.psf (image)
         psffwhm  (float)
      '''
      self.psf = pyfits.getdata(psfname)
      self.psf_hdr = pyfits.getheader(psfname)
      #self.psf =  mfmlib.trimborder(self.psf)[1]
      self.psf[ np.where( self.psf  < 0) ] = 0
      self.psf = self.psf/self.psf.sum()
      self.psffwhm = self.psf_params(self.psf)


   def psf_params(self, psf):
      '''Calculates the PSF full width half maximum 
      from its mean standard deviation across axis.

      Use image moments to evaluate center and then sigma
      (first and second moments)

      '''
      mm,nn = psf.shape

      y,x = np.indices(psf.shape)

      # center
      x0 = (x*psf).sum()/psf.sum()
      y0 = (y*psf).sum()/psf.sum()

      # width
      sigmax = np.sqrt( ( (x-x0)**2 * psf).sum() / psf.sum() )
      sigmay = np.sqrt( ( (y-y0)**2 * psf).sum() / psf.sum() )
      
      self.psfsigma = (sigmax + sigmay)/2.
      
      # FIT GAUSSIAN to PSF and make a Synthetic one 
      if config.PSFmethod == 'fit':
         params0 = [0, psf.max(), x0, y0, self.psfsigma]
         self.psf_fitparams, fitsucess = mfmlib.PSFfitgaussian(psf, params0)
      
         if fitsucess > 0:      
            self.psfsigma = np.abs(self.psf_fitparams[4])
            M = N = int(self.psfsigma * 11)
            self.psf      = mfmlib.gaussian2D(M/2., N/2., self.psfsigma, M, N )
            self.psf = self.psf/self.psf.sum()

      psffwhm = 2.35 * self.psfsigma
      return psffwhm


         
   def astrometry(self):
      '''Gets the astrometry from the Fits header.
      '''

      try:
         cd11 = self.hdr['CD1_1']
         cd12 = self.hdr['CD1_2']
         cd21 = self.hdr['CD2_1']
         cd22 = self.hdr['CD2_2'] 
         self.asecpix         =  3600*np.sqrt(cd11**2 + cd22**2)
         self.PAwcsrad        = np.arctan2(cd12, cd11)
         self.isImageMirrored = (cd11*cd22 - cd12*cd21) < 0  # determinant of the CD matrix
      except:
         try:
             cdelt1 = hdr['CDELT1']
             cdelt2 = hdr['CDELT2']
             self.asecpix =  3600*np.sqrt(cdelt1**2 + cdelt2**2)
         except:
             cdelt1 = 1.0
             cdelt2 = 1.0
             self.asecpix = 1.0      
             self.PAwcsrad = 0.0
      #self.asecpix = 1.2

      if config.verbose>2:
         print('PixelSZ = ', self.asecpix)
         print('PAwcs', np.degrees(self.PAwcsrad) )



   def skybg_iterative(self, data, max_iter=100, old_med=0, old_mad=0 ):
      '''
      1. Defines initial median and mad based on mask (initially the whole image). 
      2. Select those pixels below the limit;
      3. calculates new median and mad on 2.
      4. repeat until median and mad converges or max_iter is reached
      
      Based on idea and code from Leonardo Ferreira, 2015. 
      '''

      skymed = np.median(data)
      skymad = mfmlib.mad(data)

      if config.verbose > 3: 
         print('sky', skymed, skymad)


      if max_iter==0 or (skymed==old_med and skymad==old_mad):
         return (skymed, skymad, data.mean(), data.std())
      else:
         mask = (data < skymed + 3.*skymad)
         return self.skybg_iterative(data[mask], max_iter-1, skymed, skymad)


   def segmentation(self):
      if config.Segmentation == 'mfmtk':
         self.segmentation_mfmtk()
      elif config.Segmentation == 'astropy':
         self.segmentation_astropy()


   def segmentation_astropy(self):
      '''
         Photutils/Atropy segmentation. 
         ago 2024
      '''

      self.gal0fltr = nd.filters.gaussian_filter(self.gal0, config.segS*self.Mo)

      image_mean, image_median, image_stddev = sigma_clipped_stats(self.gal0, sigma=3)

      # threshold for segmenation
      threshold = config.SegThreshold * image_stddev

      #smallest object 
      npixels = config.MinSegSz

      segmap    = pu.segmentation.detect_sources(self.gal0fltr, threshold, npixels=npixels)
      while segmap is None:
         threshold = 0.9 * threshold
         npixels   = int(0.9 * npixels)  
         segmap    = pu.segmentation.detect_sources(self.gal0fltr, threshold, npixels=npixels)
         if config.verbose > 1:
            print('MFMTK: Lowering segmentation threshold')
            print(f'   threshold {threshold/image_stddev:.2f} stddev, npix = {npixels}')

      segmapdeb = pu.segmentation.deblend_sources(self.gal0fltr, segmap, npixels=npixels, progress_bar=False)
      cat = pu.segmentation.SourceCatalog(self.gal0, segmapdeb, convolved_data=self.gal0fltr)
      
      regioncenter  = ( (cat.xcentroid - self.Mo/2.)**2 + (cat.ycentroid-self.No/2)**2 )**(1/2.)  
      tgtlabel = np.argmin( regioncenter )
      self.photcat  = cat[tgtlabel]

      segoutros = segmapdeb.data
      outrasmask = np.logical_xor(segoutros.astype(bool),  (segoutros == (tgtlabel+1)) )
      self.segmask = np.logical_and( segoutros.astype(bool)  , np.logical_not(outrasmask.astype(bool)) )
      self.gal0seg = ma.array(self.gal0, mask=outrasmask)
      self.gal0fltrseg = self.gal0fltr * self.segmask

      if 'segment' in sys.argv:
         pyfits.writeto(self.rootname +'_galsegmask.fits', self.gal0seg.mask.astype('int'), overwrite=1)



   def segmentation_mfmtk(self):
      '''
      segments the image based on the median and median-absolute-deviation
      returns segmentation mask
      '''
      if config.verbose > 2: print('Segmentation: %.1f m.a.d. above median' % (config.segK))

      #  filtered galaxy image
      self.gal0fltr = nd.filters.gaussian_filter(self.gal0, config.segS*self.Mo)


      def segmenta(img, k):
         med  = np.median(img)
         mad  = mfmlib.mad(img)
         mask = np.zeros_like(img)
         mask[np.where(img>(med+k*mad))]=1.0
         return mask

      # segmentation mask
      self.segmask = segmenta(self.gal0fltr,k=config.segK)
            
      #while ~self.segmask.any():
      while len(np.where(self.segmask==1)[0]) < config.MinSegSz:
         print(len(np.where(self.segmask.shape==1)[0]))
         print('ERROR: No segemented region found, lowering config.segK')

         config.segK -= 1.
         print(config.segK)
         self.segmask = segmenta(self.gal0fltr, k=config.segK)
         if config.segK < 0:
            exit()


      #### CONNECTED COMPONENT LABELING
      # to select only the largest region
      # http://scipy-lectures.github.com/advanced/image_processing/index.html
      #
      # get the labelled regions from segmentation mask
      if config.verbose > 2:
         print('Selecting segmented regions')

      labelimg, labelnum  = nd.label(self.segmask)
      if labelnum > 1:
         if config.verbose >=2 : print('several segemented regions found')
         if config.selectby == 'size':
            if config.verbose>=2: print('choosing by size')
            # calculate the size of each region labelled
            regionsizes = np.array(nd.sum(segmask, labelimg, list(range(1,labelnum + 1))))
            # selects the index of the largest
            selectedlevel = regionsizes.argmax()
            # keep on the mask only the largest
         elif config.selectby == 'position':
            if config.verbose>=2: print('segmenting by position')
            #calculates center_of_mass of each region
            imagecenter   = np.array((self.Mo/2., self.No/2.))
            regioncenter  = np.array(nd.center_of_mass(self.segmask, labelimg, range(1,labelnum + 1)))
            # shift from center
            ddr = regioncenter-imagecenter
            #distance from center
            DD = np.sqrt(np.sum(ddr*ddr,axis=1))
            # select the closest to the center of image
            # must transpose before summing for the case of a single point
            selectedlevel = 1+np.argmin(DD)
         elif config.selectby == 'header':
            if config.verbose>=2: print('segmenting by header position')
            objpixcoords   = np.array((self.x0header, self.y0header ))
            regioncenter  = np.array(nd.center_of_mass(self.segmask, labelimg, range(1,labelnum + 1)))
            # shift from center
            ddr = regioncenter-objpixcoords
            #distance from center
            DD = np.sqrt(np.sum(ddr*ddr,axis=1))
            # select the closest to the center of image
            # must transpose before summing for the case of a single point
            selectedlevel = 1+np.argmin(DD)
         else:
            selectedlevel = 1
      else:
         selectedlevel = 1
         

      self.segmask = (labelimg==selectedlevel).astype(int)
      self.gal0seg     = self.gal0     * self.segmask
      self.gal0fltrseg = self.gal0fltr * self.segmask

      if 'segment' in sys.argv:
         pyfits.writeto(self.rootname +'_galseg.fits', self.gal0seg, overwrite=1)




   def peak_center(self):

      self.y0max, self.x0max = nd.maximum_position((self.gal0fltrseg))

      # size of peak region to consider in interpolarion for x_peak
      dp = 2
      peakimage = self.gal0fltrseg[self.y0max-dp:self.y0max+dp ,self.x0max-dp:self.x0max+dp]
      m00 = mfmlib.geo_mom(0,0,peakimage,centered=0,normed=0)
      m10 = mfmlib.geo_mom(1,0,peakimage,centered=0,normed=0)
      m01 = mfmlib.geo_mom(0,1,peakimage,centered=0,normed=0)
      
      self.x0peak = self.x0max + m10/m00 - dp
      self.y0peak = self.y0max + m01/m00 - dp 

      if ( np.isnan(self.x0peak) or np.isnan(self.y0peak) ) :
          self.x0peak = self.x0col
          self.y0peak = self.y0col

      # check if center is galaxy center, i.e., should be near the image center
      # otherwise apply a penalty to pixel value proportional to the center distance^2
      if np.sqrt((self.x0peak-self.No/2.)**2 + (self.y0peak-self.Mo/2.)**2) > config.CenterOffset:
         # define a penalty as we move from the center
         xx,yy  = np.meshgrid( np.arange(self.No)-self.No/2., np.arange(self.Mo)-self.Mo/2.)
         rr2 = xx**2 + yy**2
         self.y0peak, self.x0peak =  nd.maximum_position((self.gal0fltrseg/rr2))




   def momenta(self,image):
      '''
      Calculates center of mass, axis lengths and position angle
      '''
      if config.verbose > 2: print('Moments')
      
      m00 = mfmlib.geo_mom(0,0,image,centered=0,normed=0)
      m10 = mfmlib.geo_mom(1,0,image,centered=0,normed=0)
      m01 = mfmlib.geo_mom(0,1,image,centered=0,normed=0)
      m11 = mfmlib.geo_mom(1,1,image,centered=0,normed=0)
      m20 = mfmlib.geo_mom(2,0,image,centered=0,normed=0)
      m02 = mfmlib.geo_mom(0,2,image,centered=0,normed=0)

      mu20 = mfmlib.geo_mom(2,0,image,centered=1,normed=0)
      mu02 = mfmlib.geo_mom(0,2,image,centered=1,normed=0)
      mu11 = mfmlib.geo_mom(1,1,image,centered=1,normed=0)
      

      return (m00, m10, m01, m11, m20, m02, mu20, mu02, mu11)
      
      








r'''
   _  _          _           _                       _                  _  _   
 _| || |_  _ __ | |__   ___ | |_ ___  _ __ ___   ___| |_ _ __ _   _   _| || |_ 
|_  ..  _|| '_ \| '_ \ / _ \| __/ _ \| '_ ` _ \ / _ \ __| '__| | | | |_  ..  _|
|_      _|| |_) | | | | (_) | || (_) | | | | | |  __/ |_| |  | |_| | |_      _|
  |_||_|  | .__/|_| |_|\___/ \__\___/|_| |_| |_|\___|\__|_|   \__, |   |_||_|  
          |_|                                                 |___/            
'''
class Photometry():
   '''
   LUMINOSITY PROFILE  and other basic measurements
   '''
   #if config.verbose > 2 : print 'Luminosity profile'

   def __init__(self,S):
      log_message(2, "[ ------------------ PHOTOMETRY ----------------- ]" )

      self.S = S # instância local do stamp S passado como argumento 
      self._photometric_catalog()
      self.profile()
      self.cut_profile()
      self.kurvatureIR()
      self.kron()

   def _photometric_catalog(self):
    'reassigns several variables from photutils.SourceCatalog to class properties '

    pc = self.S.photcat.copy() # photometric catalog from Stamp
    self.P_xcentroid = pc.xcentroid
    self.P_ycentroid = pc.ycentroid
    self.P_area      = pc.area.value
    self.P_perimeter = pc.perimeter.value
    self.P_a_sigma   = pc.semimajor_sigma.value
    self.P_b_sigma   = pc.semiminor_sigma.value
    self.P_Requiv    = pc.equivalent_radius.value
    self.P_PA        = pc.orientation.value
    self.P_q         = 1/pc.elongation
    self.P_ellip     = pc.ellipticity
    #self.P_covar_l1  = pc.covariance_eigvals[0].value  sqrt(covar_l1) == a
    #self.P_covar_l2  = pc.covariance_eigvals[1].value
    self.P_gini      = pc.gini
    self.P_Fkron     = pc.kron_flux
    self.P_Rkronnorm = pc.kron_radius.value
    self.P_Rkron     = pc.kron_radius.value * np.sqrt(self.P_a_sigma*self.P_b_sigma)
    self.P_R10       = pc.fluxfrac_radius(0.1).value
    self.P_R20       = pc.fluxfrac_radius(0.2).value
    self.P_R30       = pc.fluxfrac_radius(0.3).value
    self.P_R40       = pc.fluxfrac_radius(0.4).value
    self.P_R50       = pc.fluxfrac_radius(0.5).value
    self.P_R60       = pc.fluxfrac_radius(0.6).value
    self.P_R70       = pc.fluxfrac_radius(0.7).value
    self.P_R80       = pc.fluxfrac_radius(0.8).value  
    self.P_R90       = pc.fluxfrac_radius(0.9).value
    self.P_R100      = pc.fluxfrac_radius(1.0).value

    self.P_C1        = np.log10(self.P_R80/self.P_R20)
    self.P_C2        = np.log10(self.P_R90/self.P_R50)

    if config.verbose >1:
         print(f'   P_Rkronnorm ={self.P_Rkronnorm:.3f} P_Rkron={self.P_Rkron:.3f}' )
         print(f'   R_frac {self.P_R10:.1f}, {self.P_R20:.1f}, {self.P_R30:.1f}, {self.P_R40:.1f}, {self.P_R50:.1f}, {self.P_R60:.1f}, {self.P_R70:.1f}, {self.P_R80:.1f}, {self.P_R90:.1f}, {self.P_R100:.1f}')
         print(f'   P_C1={self.P_C1:.2f} P_C2={self.P_C2:.2f}')
         print(f'   logFkron={np.log10(self.P_Fkron):.2f}   ')
         print( '   --')
  


   def profile(self):
      """
      LUMINOSITY PROFILES 
      * Measure at regular spaced ellipses with axis ratio 'q' and angle 'PA' 
        (Intensity at, mean intensity inside, integrated luminosity, ...)
      * masquerades stars
      * calculates Petrosian Radius
      * define galpetro, galnostars, galpetronostars
      """

      if config.verbose >2:
         print('profiling')

      xx,yy   = np.meshgrid(np.arange(self.S.No), np.arange(self.S.Mo))

      if (config.photcenter == 'peak'):
         ## using PEAK center
         xl =  (xx-self.S.x0peak)*np.cos(self.S.PArad) + (yy-self.S.y0peak)*np.sin(self.S.PArad) 
         yl = -(xx-self.S.x0peak)*np.sin(self.S.PArad) + (yy-self.S.y0peak)*np.cos(self.S.PArad)
      elif (config.photcenter == 'center'):
         ## using COL center
         xl =  (xx-self.S.x0col)*np.cos(self.S.PArad) + (yy-self.S.y0col)*np.sin(self.S.PArad) 
         yl = -(xx-self.S.x0col)*np.sin(self.S.PArad) + (yy-self.S.y0col)*np.cos(self.S.PArad)

      rl = np.sqrt(xl**2 + yl**2/self.S.q**2)

      # MAX RADIUS and INCREMENT BETWEEN ELLIPSES
      self.Rmax = np.sqrt(self.S.Mo**2 + self.S.No**2)
      
      # Pixel (linear) and Radius (squared) distance 
      self.Raios  = np.arange(1.,self.Rmax,config.deltaR)
      
      # dee: elliptical ring width
      dee = 0.5*np.gradient(self.Raios)

      self.IR     = np.zeros_like(self.Raios) # AT the ellipse 
      self.IRerr  = np.zeros_like(self.Raios) # IR error
      self.LR     = np.zeros_like(self.Raios) # TOTAL INSIDE ellipse
      self.LRerr  = np.zeros_like(self.Raios) # LRerr 
      self.IRM    = np.zeros_like(self.Raios) # MEAN INSIDE ellipse
      self.IRMerr = np.zeros_like(self.Raios) # IRM error
        

      i  = 0
      for sma in self.Raios:
         # pontos INTERNOS e no ANEL
         ellindxin   = np.where( rl <= sma)
         ellindxbrdr  = np.where( (rl>(sma-dee[i]))  & (rl<=(sma+dee[i])))
 
         # the S.gal0seg is a MaskedArray, has both 'data' and 'mask' properties
         # if needed, further masking can be done at S.gal0seg.mask

         ''' OUTDATED STAR MASKING ROUTINE REMOVE HERE   '''

         self.IR[i]    = np.mean(self.S.gal0seg[ellindxbrdr])
         self.IRerr[i] = np.std(self.S.gal0seg[ellindxbrdr])

         # elements inside the ellipse
         Nelemin  = len(np.ravel(ellindxin))/2.
         self.LR[i]  = np.sum(self.S.gal0seg[ellindxin])
         self.IRM[i] = self.LR[i] / Nelemin          

         if Nelemin <1:
            continue

         # PETROSIAN RADIUS
         try:
            self.Rp
            if sma > config.NRp*self.Rp:
               break
         except:
            if self.IR[i]/self.IRM[i] < config.PetroEta0:
               # when reaches PetroEta0 interpolates Petrosian position.
               # before, it only took the radius where it happened, 
               #          self.Rp = sma/config.ZOOM
               # but now interpolates linearly
               
               R1 = self.Raios[i-1]
               R2 = self.Raios[i]
               n1 = self.IR[i-1]/self.IRM[i-1]
               n2 = self.IR[i]/self.IRM[i]
               #interpolates linearly
               self.Rp = (R2-R1) * (config.PetroEta0 - n1)/(n2-n1) + R1

               if config.verbose>1:
                  print('   Rp=%.1f' % self.Rp)

         i += 1

         #------------------ END PROFILING ----------------------------#

      self.IRsnr = self.IR/self.IRerr
      self.MUR    = -2.5 * np.log10(self.IR)
      self.MURerr = (-2.5/self.IR) * self.IRerr

      self.petroRegionIdx  = np.where( rl < config.NRp_galpetro * self.Rp ) 

      self.petromask = np.zeros_like(self.S.gal0)
      self.petromask[self.petroRegionIdx] = 1.0

      self.galpetro  = self.S.gal0seg.data * self.petromask
      self.galmask   = np.logical_not(self.S.gal0seg.mask)


      if 'galpetro' in sys.argv:
         pyfits.writeto(self.S.rootname + '_galpetro.fits',  self.galpetro, overwrite=1)
         #pyfits.writeto(self.S.rootname + '_petromask.fits', self.petromask.astype('int'), overwrite=1)
         #pyfits.writeto(self.S.rootname + '_galmask.fits',   self.galmask.astype('int'), overwrite=1)


      if 'mask' in sys.argv:
         pyfits.writeto(self.S.rootname +'_mask.fits', self.galmask.astype('int'), overwrite=1)
         pyfits.writeto(self.S.rootname + '_gal+mask.fits',   S.gal0seg.data * np.logical_not(S.gal0seg.mask) , overwrite=1) 




   def cut_profile(self):
      '''
       DESCARTA pontos  de acordo com criterio
       '''
      if config.verbose > 2: 
         print('Discharging points beyond NRp Rp')
      
      if (self.Rp!=0.0 and config.NRp_profile*self.Rp < self.Rmax):

         # discards points beyond k*Rp
         try:
            lastpt1 = np.where(self.Raios >= config.NRp_profile * self.Rp)[0][0] 
         except:
            lastpt1 = np.inf


         try:
            lastpt2 = np.where( self.IR <= 0.0E00 )[0][0] 
         except:
            lastpt2 = np.inf

         if lastpt1==np.inf and lastpt2==np.inf:
            lastpt = len(self.IR)
         else:
            lastpt=min(lastpt1, lastpt2)

         if config.verbose>2:
            print('cutting profiles after', lastpt, 'point')


         self.Raios  = np.take(self.Raios, np.arange(lastpt))
         self.IR     = np.take(self.IR,    np.arange(lastpt))
         self.IRerr  = np.take(self.IRerr, np.arange(lastpt))
         self.LR     = np.take(self.LR,    np.arange(lastpt))
         self.LRerr  = np.take(self.LRerr, np.arange(lastpt))
         self.IRM    = np.take(self.IRM,   np.arange(lastpt))
         self.IRMerr = np.take(self.IRMerr,np.arange(lastpt))



         self.IRsnr = self.IR/self.IRerr
         self.MUR    = -2.5 * np.log10(self.IR)
         self.MURerr = (-2.5/self.IR) * self.IRerr
         
         if 'profile' in sys.argv:
            np.savetxt(self.S.rootname + '_IR.csv', \
                      np.transpose((self.Raios, self.IR, self.IRerr, self.LR, self.IRM)), \
                      delimiter=',',  header ='# Raios, IR, IRerr, LR, IRM ' )

   def kurvatureIR(self):
      '''New implementation calculated with the spline representation 
         of IR instead of the points directly.  
         FF, 10/set/2024
         '''

      from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline

      logIR = np.log10(self.IR)
      # we need this delta do avoid infty in the normalization
      delta = (logIR.max() - logIR.min()) / 100000
      logIRnorm =  (logIR - logIR.min()+delta) / (logIR.max() - logIR.min()) 

      #logIRspl = UnivariateSpline(R, logIRnorm, w=1/self.IRsnr , k=5, s=0.01)
      logIRspl = UnivariateSpline(self.Raios, logIRnorm , k=5)#, s=0.0020)
      self.logIRspl_d1 = logIRspl.derivative(n=1)
      self.logIRspl_d2 = logIRspl.derivative(n=2)
      self.IRkurv  = self.logIRspl_d2(self.Raios) / (1 + (self.logIRspl_d1(self.Raios))**2 )**(3/2.)
      self.IRkurvmedian = np.median(self.IRkurv)
      self.IRkurvmad    = mfmlib.mad(self.IRkurv)

      self.validkurvidx = self.IR/self.S.skybgstd > 1


      # save for plots later
      self.IRspl = UnivariateSpline(self.Raios, self.IR, w=1/self.IRsnr , k=5,s=0.1)(self.Raios)


      self.d2IdR2 = np.gradient( np.gradient(self.IR, self.Raios/self.Rp), self.Raios/self.Rp)


 
      def plotK():
         #print (logIspl(R))
         #pl.plot(R, logIRnorm  )
         #pl.plot(R, logIspl(R)  )
         pl.hist(self.IRkurv[self.validkurvidx])
         pl.show()
         sys.exit()
      #plotK()


   def kron(self):
      '''
      inputs: morphometrya stamp object S and photometry object P
      2024.06
      '''      

      PA = self.S.PArad
      x0 = self.S.x0col
      y0 = self.S.y0col
      xx,yy   = np.meshgrid(np.arange(self.S.No), np.arange(self.S.Mo))
      xl =  (xx-x0)*np.cos(PA) + (yy-y0)*np.sin(PA) 
      yl = -(xx-x0)*np.sin(PA) + (yy-y0)*np.cos(PA)

      rl = np.sqrt(xl**2 + yl**2/self.S.q**2)
      
      self.Rkron = (rl    * self.S.gal0seg).sum() / self.S.gal0seg.sum()
      #self.Rkron2 = (rl**2 * self.S.gal0seg).sum() / (rl * self.S.gal0seg).sum()

      self.Rpkron = (rl    * self.galpetro).sum()/self.galpetro.sum()
      #self.Rpkron2 = np.sum(rl**2 * self.galpetro)/np.sum(rl*self.galpetro)

      if config.verbose>1:
         print('   Rkron=%.1f Rpkron=%.1f' % (self.Rkron, self.Rpkron) )






r'''
   _  _                       _          _  _   
 _| || |_   ___  ___ _ __ ___(_) ___   _| || |_ 
|_  ..  _| / __|/ _ \ '__/ __| |/ __| |_  ..  _|
|_      _| \__ \  __/ |  \__ \ | (__  |_      _|
  |_||_|   |___/\___|_|  |___/_|\___|   |_||_|  
                            
'''
class Sersic():

   def __init__(self, S, P):
      log_message(2, "[ -------------------- SERSIC -------------------- ]" )

      self.S = S
      self.P = P

      #self.psffwhm = self.S.psffwhm
      self.firstpt = int(np.ceil( config.Npsf * self.S.psfsigma))
      
      self.MUR = -2.5*np.log10(self.P.IR)
      self.MURerr = -2.5/(np.log(10) * self.P.IR) * self.P.IRerr


      L2Rp = self.P.LR.max()
      LR50spl = scipy.interpolate.splrep(self.P.Raios, (self.P.LR-0.5*L2Rp), s=0)
      self.R50 = scipy.interpolate.sproot(LR50spl)[0]

      
      #self.fit1D(self.S,P)
      if config.verbose >2 : 
         print('Sersic fit 1D') 
      self.fit1D(self.S, self.P)

      if config.verbose >2 : 
         print('Sersic fit 2D') 
      self.fit2D(self.S, self.P)

      self.goodness1d(self.P)
      self.goodness2d(self.S, self.P)
      self.galpetroFit()

      #if config.verbose>1:
      #   print '   Sersic'
      #   print '   1D In=%.2f Rn=%.2f n=%.2f' % (self.InFit1D, self.RnFit1D, self.nFit1D)
      #   print '   2D In=%.2f Rn=%.2f n=%.1f PA=%.1f q=%.1f' % (self.InFit2D, self.RnFit2D, self.nFit2D, self.PAFit2D, self.qFit2D ) 


   def IRsersic(self, R, In, Rn, n):
      bn = 1.9992*n - 0.3271
      return In * np.exp(-bn*((R/Rn)**(1./n) - 1 ))


   def fit1D(self,S,P):

      # initial parameters
      IR0  = 10*np.mean(P.IR)
      n0   = 1.5

      Rn0 = self.R50

      p0Fit1D = [IR0, Rn0, n0]
      self.InFit1D, self.RnFit1D, self.nFit1D, self.sucFit1D \
         = mfmlib.SersicFit().fit1D(P.Raios[self.firstpt:], P.IR[self.firstpt:],  p0Fit1D, fittype='bounded')

      if config.verbose >1 : 
            print('   FIT1D In=%.1f Rn=%.1f  n=%.1f' % (self.InFit1D, self.RnFit1D, self.nFit1D))

      if self.sucFit1D not in [1,2,3,4]:
         print('FIT 1D ERROR', self.sucFit1D)
         InFit1D = MuFit1D = RnFit1D = nFit1D = 1.0



   def fit2D(self, S, P, mask=1, fittype='bounded'):
      '''Fit 2D Sersic profile 
      fittype = bounded or free'
      '''
      
      p0Fit2D = [S.x0peak,S.y0peak, self.InFit1D, self.R50, self.nFit1D, \
                 S.q, S.PAdeg]

      psfmask = Morphometry.psfmask(config.Npsf * S.psfsigma, *S.gal0.shape, S.x0peak,S.y0peak )
      
      self.fit2Dmask = psfmask * self.P.galmask
      
      self.x0Fit2D, self.y0Fit2D, \
      self.InFit2D, self.RnFit2D, self.nFit2D, \
      self.qFit2D, self.PAFit2D, self.sucFit2D \
           = mfmlib.SersicFit().fit2D(S.gal0seg, S.psf, p0Fit2D, self.fit2Dmask, fittype)
      

      if self.sucFit2D not in [1,2,3,4]:
         print('FIT 2D ERROR', self.sucFit1D)
         InFit1D = MuFit1D = RnFit1D = nFit1D = 1.0


      pFit2D = [self.x0Fit2D, self.y0Fit2D, self.InFit2D, \
                self.RnFit2D, self.nFit2D, self.qFit2D, self.PAFit2D ]

      if config.verbose >1 : 
            print('   FIT2D In=%.1f Rn=%.1f  n=%.1f   (x0,y0)=%.1f,%.1f  q=%.1f PA=%.1f' % \
               ( self.InFit2D, self.RnFit2D, self.nFit2D, self.x0Fit2D, self.y0Fit2D, self.qFit2D, self.PAFit2D))


      self.fitmodel =   mfmlib.SersicFit().sersicProfile2D(pFit2D, S.psf, *S.gal0.shape)

      self.fitresidual =  (S.gal0seg - self.fitmodel)
      
      self.fitresidualnorm = P.galmask * self.fitresidual/S.gal0seg


      self.fresid1D1 =  (self.P.IR[self.firstpt:]  -   self.IRsersic(self.P.Raios[self.firstpt:], self.InFit1D,  self.RnFit1D, self.nFit1D)  )
      self.fresid1D2 =  (self.P.IR[self.firstpt:]  -   self.IRsersic(self.P.Raios[self.firstpt:], self.InFit2D,  self.RnFit2D, self.nFit2D)  )

      self.fresid1D1_rss =  ((self.fresid1D1**2)**0.5).sum()
      self.fresid1D2_rss =  ((self.fresid1D2**2)**0.5).sum()


   def goodness1d(self,P):
      IRmodel  = self.IRsersic(P.Raios,self.InFit1D,self.RnFit1D,self.nFit1D)
      #MURmodel = mag(IRsersic(Raios,InFit2D,RnFit2D,nFit2D))

      Npix = len(IRmodel)

      #wi = P.IRsnr[self.firstpt:]
      #W  = wi.mean()#wi.sum()#len(wi)
      self.xsin   = np.sqrt( (1/Npix) * np.sum( ((P.IR[self.firstpt:]-IRmodel[self.firstpt:])/IRmodel[self.firstpt:])**2 ) )
      
      #xsi    = 100*xsif.mean()
      #self.xsin   = 100*np.mean((wi/W) * xsif)  

      if config.verbose > 2:
         print('Xsin =',self.xsin)


   def goodness2d(self,S,P):

      chiimage = (self.fitmodel - self.P.galpetro)**2/ self.P.galpetro**2
      chimean, chimedian, chistdev = sigma_clipped_stats(chiimage)
      chimask = chiimage < chimedian + 3*chistdev
      Npix = len(np.where(self.P.petromask ==1)[0])

      self.chi2 = np.sqrt(  (1/Npix) * np.ma.sum( chimask *  (self.fitmodel - self.P.galpetro)**2) / 
					np.ma.sum(chimask * self.P.galpetro**2 ) )
      
      if config.verbose > 1:
         print('   chi2 =',self.chi2)

   
   def galpetroFit(self):
      self.PAFit2Drad = np.deg2rad(self.PAFit2D)
      xx,yy   = np.meshgrid(np.arange(self.S.No), np.arange(self.S.Mo))
      xp =  (xx - self.x0Fit2D) * np.cos(self.PAFit2Drad) + (yy - self.y0Fit2D) * np.sin(self.PAFit2Drad) 
      yp = -(xx - self.x0Fit2D) * np.sin(self.PAFit2Drad) + (yy - self.y0Fit2D) * np.cos(self.PAFit2Drad)
      rp = np.sqrt(xp**2 + yp**2/self.qFit2D**2)

      self.petroRegionFitIdx  = np.where( rp < config.NRp_galpetro * self.P.Rp ) 

      # PETRO mask
      self.petromaskFit = np.zeros_like(self.S.gal0seg)
      self.petromaskFit[self.petroRegionFitIdx] = 1.0
      self.galpetroFit        = self.S.gal0seg * self.petromaskFit

      self.galmaskFit = (np.logical_and( self.petromaskFit, np.logical_not(self.S.gal0seg.mask) )).astype('int')

      #if 'galpetro' in sys.argv:
      #   pyfits.writeto(self.S.rootname + '_galpetroFit.fits', self.galpetroFit, overwrite=1)
      #   pyfits.writeto(self.S.rootname + '_petromaskFit.fits', self.petromaskFit, overwrite=1)
      #   pyfits.writeto(self.S.rootname + '_galmaskFit.fits', self.galmaskFit, overwrite=1)
   














r'''
   _  _                                _               _  _   
 _| || |_   _ __ ___   ___  _ __ _ __ | |__   ___    _| || |_ 
|_  ..  _| | '_ ` _ \ / _ \| '__| '_ \| '_ \ / _ \  |_  ..  _|
|_      _| | | | | | | (_) | |  | |_) | | | | (_) | |_      _|
  |_||_|   |_| |_| |_|\___/|_|  | .__/|_| |_|\___/    |_||_|  

'''
class Morphometry():
   '''
   '''
      
   def __init__(self, S, P, Ss):
      log_message(2, "[ ------------------ MORPHOMETRY  ----------------- ]" )
      self.S  = S
      self.P  = P
      self.Ss = Ss

      #make the standard galaxies
      self.gal0std  = \
            self.standartize(self.S.gal0seg, self.S.q, self.S.PArad, self.S.y0col, self.S.x0col)


      self.Wgalnostars = mfmlib.wlet2D(self.S.gal0seg, scales=(1,2))

      if 'stangal' in sys.argv:
         pyfits.writeto(self.S.rootname +'_stangal.fits', self.gal0std, overwrite=1)
      


      self._Concentration()
      self._Asymmetry()
      self._Gini()
      self._Smoothness()
      self._Moment20()
      self._Entropy()
      self._Gradindex()
      #self._Shear()

   @staticmethod
   def psfmask(size, M,N, x0,y0): 
      """"
      Creates a unitary mask with size (M,N) with a zero circular region of
      'size' centered at (x0,y0)
      """
      x,y = np.meshgrid(np.arange(N), np.arange(M))
      rr = np.sqrt( (x-x0)**2 + (y-y0)**2 )

      # PSF mask
      psfRegionIdx = np.where( rr >= size ) 
      psfmask      = np.zeros((M,N))
      psfmask[psfRegionIdx] = 1.0

      return psfmask


   def standartize(self, image, q, PArad,  x0,y0):
      """ make a standard galaxy, id est, PA=0, q=1
      arguments are 'image' to be standartized and  its 'S' stamp and P phot classes 
      """

      ##### rotate array
      R = np.array([[np.cos(PArad),np.sin(PArad)],[-np.sin(PArad),np.cos(PArad)]])

      ##### shear array
      S = np.diag([q, 1.])
      # SERSIC fit values
      #S = np.diag([self.Ss.qFit2D, 1.])

      # affine transform matrix, rotate then scale
      transform = np.dot(R,S)

      # where to transform about
      centro_i = (x0,y0)
      # contro_o: where to put center after 
      centro_o = np.array(image.shape)/2

      myoffset = centro_i - np.dot(transform, centro_o)
      bval     = np.mean(image[-2:])
      stangal  = nd.affine_transform(image, transform, offset=myoffset, order=2,cval=bval)

      return stangal




   def _Concentration(self):

      if config.verbose > 2: print('Concentration')

      self.LT = self.P.LR.max()


      def find_Rf(frac):
         'frac must be 0.1 0.2 ... 1.0'

         try:
            LRspl = scipy.interpolate.splrep(self.P.Raios, (self.P.LR - frac * self.LT), s=0)            
            RR    = scipy.interpolate.sproot(LRspl)[0]         
         except:
            RR = 1.0
         return RR

      self.R10  = find_Rf(0.1)
      self.R20  = find_Rf(0.2)
      self.R30  = find_Rf(0.3)
      self.R40  = find_Rf(0.4)
      self.R50  = find_Rf(0.5)
      self.R60  = find_Rf(0.6)
      self.R70  = find_Rf(0.7)
      self.R80  = find_Rf(0.8)
      self.R90  = find_Rf(0.9)
      self.R100 = find_Rf(0.9999999) # must be 99 instead of 100 :-( ?


      self.C1 = np.log10(self.R80/self.R20)
      self.C2 = np.log10(self.R90/self.R50)

      if config.verbose > 1:     
         print('   C radius: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f' % \
            (self.R10 , self.R20 , self.R30, self.R40, self.R50, self.R60, self.R70, self.R80, self.R90, self.R100))
         print('   C1=%.2f' % self.C1, 'C2=%.2f' % self.C2)
         print('   logLT=%.2f' % np.log10(self.LT)) 





   def _Asymmetry(self):


      if config.verbose > 2 :
         print('   Asymmetry')

      def rot180(imagem,x0,y0):
         R      = np.matrix([[-1,0],[0,-1]])
         img180 = nd.affine_transform(imagem, R, offset=(2.*(y0-0.5),2.*(x0-0.5)) )
         return img180

      def background_asymmetry(img, pre_clean=False):
         
         def measure_asymmetry_patch(pos, patch, img):
            (x0, y0) = pos
            rot_cell = rot180(patch, x0, y0)
            sub = patch - rot_cell            
            return np.sum(abs(sub))/np.sum(abs(self.Ss.petromaskFit * img))

         gridsize = self.S.Mo // 10 # 10% of the size of the image
         n_pix  = gridsize**2
         xcells = self.S.Mo // gridsize
         ycells = self.S.No // gridsize
         asymmetry_grid = np.zeros((xcells, ycells))
         gal_area = self.Ss.petromaskFit.sum()
         
         for xi in range(xcells):
            for yi in range(ycells):
               cell_mask = self.Ss.petromaskFit[xi*gridsize:(xi+1)*gridsize, yi*gridsize:(yi+1)*gridsize]
               
               if cell_mask.sum() > 0:
                  asymmetry_grid[xi, yi] = 0
                  continue

               cell_img = img[xi*gridsize:(xi+1)*gridsize, yi*gridsize:(yi+1)*gridsize]    
               x0, y0   = fmin(measure_asymmetry_patch, (gridsize // 2, gridsize //2), args=(cell_img, img), disp=0)
               asymmetry_grid[xi, yi] = (gal_area/n_pix) * measure_asymmetry_patch((x0, y0), cell_img, img)
               del cell_img

         linear = asymmetry_grid[np.where(asymmetry_grid != 0)].ravel()

         if len(linear) > 0:
             BGrandom = np.random.choice(linear, 1)[0]
             BGmedian = np.median(linear)
             BGmin    = linear.min()
             BGstd    = np.std(linear)
             position = np.where(asymmetry_grid==linear.min())
             x0       = position[1][0] * gridsize + gridsize//2
             y0       = position[0][0] * gridsize + gridsize//2


         elif (('clean' in sys.argv) & (pre_clean == False)):
             #measure background asymmetry with original pre-clean image if it fails for the clean one
             return background_asymmetry(self.S.galOriginal, pre_clean=True)
         else:
            '''
               This is a fallback for when something goes wrong with backgroun asymmetry estimates.
               It should also appear as a QF.
            '''
            BGrandom = 0
            BGmedian = 0
            BGmin = 0
            BGstd = 0
            x0 = 0
            y0 = 0

         return BGrandom, BGmedian, BGmin, BGstd, x0, y0


      def assimetria0(pos, img, box=False):
         (x0, y0) = pos
         psfmask = self.psfmask(config.Npsf * self.S.psfsigma, *img.shape, x0, y0)
         if(box):
             boxmask  = np.zeros_like(img)
             radii_px = np.ceil(self.P.Rp * 1.5)
             boxmask[int(x0-radii_px):int(x0+radii_px), int(y0-radii_px):int(y0+radii_px)] = 1
             imgorig  = boxmask * img
             imgsub   = boxmask * (img-rot180(img,x0,y0))
             A  = np.sum(abs(imgsub))/np.sum(abs(imgorig))
         else:         
             imgorig = self.Ss.petromaskFit  * img
             imgsub  = self.Ss.petromaskFit  * (img- rot180(img,x0,y0))
             A = np.sum(abs(imgsub))/np.sum(abs(imgorig))

         del psfmask, imgorig, imgsub
         return A

      def assimetria1(pos, img, image=0, sigmafilter=1):
         x0, y0 = pos
         A1img  =  np.abs(img - rot180(img, x0,y0))/(np.sum(np.abs(img)) )
         A1mask =  A1img > np.ma.median(A1img) + config.AsySigma * mfmlib.mad(A1img.data)

         if sigmafilter:
            A1 = np.sum( self.Ss.petromaskFit * A1mask * A1img)
         else:
            A1 = np.sum(A1img)

         if image:
            return (A1, A1img, A1mask)
         else:
            return (A1)

      def assimetria2(pos, img):
         x0, y0 = pos
         img = img * self.psfmask(config.Npsf * self.S.psfsigma, *img.shape, x0,y0)
         return -corrcoef4image(img, rot180(img,x0,y0))

      def assimetria3(pos, img):
         x0, y0 = pos
         #psfmask = self.psfmask(self.S.psfsigma, *img.shape, x0,y0)
         # segementation
         #return -spearmanr(img[self.P.petroRegionIdx].ravel(), \
         #                  (rot180(img,x0,y0))[self.P.petroRegionIdx].ravel())[0]
         #sersic fit
         return -spearmanr(img[self.Ss.petroRegionFitIdx].ravel(), \
                           (rot180(img,x0,y0))[self.Ss.petroRegionFitIdx].ravel())[0]

      def corrcoef4image(a,b):
         aa = a - a.mean()
         bb = b - b.mean()
         return np.sum( aa*bb ) / np.sqrt(  np.sum(aa**2) * np.sum(bb**2) )

      # remove first wavelet component (noise)
      galA     =   self.S.gal0seg - self.Wgalnostars.W[0]
      #x0A, y0A  =  self.S.x0col-borders[2], self.S.y0col-borders[0]
      x0A, y0A  =  self.S.x0col, self.S.y0col

      BGrandom, BGmedian, BGmin, BGstd, BGx0, BGy0 = background_asymmetry(self.S.gal0)

      if config.verbose > 2:
         print('fitting center and measuring Asymmetry')

      # fit center for A0 and check if need to do it for A1, A2, A3
      x0A0fit, y0A0fit  = fmin(assimetria0, (x0A,y0A), args=(self.S.gal0,), disp=0) 

      if config.AsyFitCenter:
         x0A1fit, y0A1fit  = fmin(assimetria1, (x0A,y0A), args=(self.S.gal0), disp=0)
         x0A2fit, y0A2fit  = fmin(assimetria2, (x0A,y0A), args=(self.S.gal0), disp=0)
         x0A3fit, y0A3fit  = fmin(assimetria3, (x0A,y0A), args=(self.S.gal0), disp=0)
      else:
         # does not fit x0,y0 for A1, A2, A3.
         # uses the same from A0 fit. 
         x0A1fit = x0A2fit = x0A3fit = x0A0fit
         y0A1fit = y0A2fit = y0A3fit = y0A0fit


      '''
         Error as estimated by the original CAS code. However, values measured by it are
         unusually low.
      '''
      asymmetry_error = (self.S.skybgstd / (self.LT + (self.S.gal0[self.Ss.petroRegionFitIdx].std())))


      
      self.BGmedian = BGmedian
      self.A_e   = asymmetry_error
      

      self.A0    = assimetria0((x0A0fit, y0A0fit), self.S.gal0) - self.BGmedian
      self.A1, self.A1img, self.A1mask    = assimetria1((x0A1fit,y0A1fit), galA, image=1, sigmafilter=1)
      self.A2    = 1+assimetria2((x0A2fit,y0A2fit), galA)
      self.A3    = 1+assimetria3((x0A3fit,y0A3fit), galA)
      self.A4    = 100 * np.sqrt((self.S.x0col - self.S.x0peak)**2 + \
                                 (self.S.y0col - self.S.y0peak)**2) / self.P.Rp

      self.A0Sersic = assimetria0((x0A0fit,y0A0fit), self.Ss.fitresidual)
      self.A1Sersic = assimetria1((x0A1fit,y0A1fit), self.Ss.fitresidual)

      # shape asymmetry (doi:10.1093/mnras/stv2878, Pawlik 2016)
      self.Ashape  = assimetria1((self.S.x0col,self.S.y0col), self.S.segmask.astype('int'), image=0, sigmafilter=0)
      

      # coorrects position for trimed border
      # borders= (yi,yf,xi,xf)

      self.x0BGfit = BGx0 #+ borders[2]
      self.y0BGfit = BGy0 #+ borders[0]

      self.x0A0fit = x0A0fit #+ borders[2]
      self.y0A0fit = y0A0fit #+ borders[0]

      self.x0A1fit = x0A1fit #+ borders[2]
      self.y0A1fit = y0A1fit #+ borders[0]

      self.x0A2fit = x0A2fit #+ borders[2]
      self.y0A2fit = y0A2fit #+ borders[0]
      
      self.x0A3fit = x0A3fit #+ borders[2]
      self.y0A3fit = y0A3fit #+ borders[0]


      if 'asymmetry' in sys.argv:
         pyfits.writeto(self.S.rootname+'_asymmetry.fits', self.A1img * self.A1mask, overwrite=1)


      if config.verbose>1:
         print('   A_bg std=%4.2f min=%4.2f random=%4.2f med=%4.2f' % (BGstd, BGmin, BGrandom, BGmedian ))
         print('   A_e=%4.3f ' % self.A_e)
         print('   A0=%4.2f' % self.A0, '    @(%.1f,%.1f)' % (self.x0A0fit,self.y0A0fit))
         print('   A1=%4.2f' % self.A1)
         print('   A2=%4.2f' % self.A2)
         print('   A3=%4.2f' % self.A3)
         print('   A4=%4.2f' % self.A4)
         print('   A0Sersic=%4.3f' % self.A0Sersic)
         print('   A1Sersic=%4.3f' % self.A1Sersic)
         print('   Ashape =%4.3f' % self.Ashape)




   def _Gini(self):

      if config.verbose >2 : print('Gini')
      
      def GG(imagem):
         pix=imagem.ravel()
         pix.sort()
         n = Npix = len(pix)
         #
         #G = np.sum(2*(arange(n)+1)*pix)/(n*pix.np.sum()) - (n+1)/n
         #
         # LOTZ 2004, eq.(6)
         G1 =   1./(abs(pix.mean() * n * (n-1)))  * np.sum( (2 * np.arange(1,n+1) - n - 1 ) * abs(pix) )
         
         # T.Lisker arXiv 0807.1531
         #i = np.arange(Npix)
         #G2 = np.sum( (2*i - Npix - 1 )*abs(pix)) / ((Npix-1)*np.sum(pix)  )
         
         return G1


      #naozero = np.where( self.P.galpetronostars !=0)
      #self.G = GG(self.P.galpetronostars[naozero])
      self.G = GG(self.S.gal0seg[self.Ss.petroRegionFitIdx])


      if config.verbose >1 : 
         print('   G=%.2f' % self.G)

   def _Moment20(self):
      """"
      2024 there was minor srrors still (ff)
      2020 new version (Leonardo)

      Old one had an error
      """
      if config.verbose>2:   print('M20')

      yy,xx = np.indices(self.S.gal0seg.shape)
      
      Mtot = np.nansum( (self.S.gal0seg * ((xx - self.S.x0peak)**2 + (yy - self.S.y0peak)**2))[self.Ss.petroRegionFitIdx] )


      sort_idx  = np.argsort((-self.S.gal0seg[self.Ss.petroRegionFitIdx]).ravel())

      total_flux   =   self.S.gal0seg[self.Ss.petroRegionFitIdx].sum()
      flux_cutoff  = 0.2 * total_flux
      cutouff_idxs = []
      fi_sum = 0
      for i, fi in enumerate(self.S.gal0seg[self.Ss.petroRegionFitIdx].ravel()[sort_idx]):
         fi_sum += fi
         if fi_sum > flux_cutoff:
            #this ignores the last pixel above cutoff, otherwise use i+1
            cutouff_idxs = np.arange(0, i)
            break

      Mi   = np.nansum((self.S.gal0seg * ((xx - self.S.x0peak)**2 + (yy - self.S.y0peak)**2))[self.Ss.petroRegionFitIdx].ravel()[sort_idx][cutouff_idxs])
     
      self.M20 = np.log10(Mi/Mtot)


   def _Smoothness(self):

      if config.verbose>2:   print('Smoothness')
      
      # FILTER width is 0.25 Rp
      hssize =  np.ceil(self.P.Rp/4.)      
      if hssize%2 ==0:
         hssize = hssize+1
      hs = sig.windows.hamming(int(hssize))
      sfilter = np.outer(hs,hs) 
      sfilter = sfilter/sfilter.sum()

      # removes first wavelet component (noise)
      SWgal  = self.S.gal0seg - self.Wgalnostars.W[0] 
      self.Sgal  =  sig.convolve2d(SWgal, sfilter, 'same', boundary='symm')

      self.S1map  = np.abs( SWgal - self.Sgal )
      self.S1mask = self.S1map > np.ma.median(self.S1map) + config.SmooSigma * mfmlib.mad(self.S1map)
      self.S1     = np.ma.sum( abs( self.Ss.petromaskFit * self.S1mask * self.S1map ) )/np.sum(self.Sgal)

      self.S3 = 1 - spearmanr( (self.S1mask*self.S.gal0seg).ravel(),  (self.S1mask * self.S1map).ravel())[0]
            
      if 'smooth' in sys.argv:
         pyfits.writeto(self.S.rootname+'_S1map.fits', self.S1map, overwrite=1)


      if config.verbose >1 : 
         print('   S1=%.2f' % self.S1)
         print('   S3=%.2f' % self.S3)


 


   def _Entropy(self):
      if config.verbose>2:   print('Entropy')
      #self.P.galnostars[np.isnan(self.P.galnostars)] = np.median(self.P.galnostars)

      self.H = mfmlib.entropy(self.S.gal0seg[self.Ss.petroRegionFitIdx], bins=200, normed=1)

      if config.verbose >1 : 
         print('   H=%.2f' % self.H)


   def _Gradindex(self):
      """
      Gradient Index
      Calculates an index based on the image gradient magnitude and orientation

      SGwindow and SGorder are Savitsky-Golay filter parameters
      F. Ferrari, 2014
      """
      if config.verbose>2:   print('Sigma_Psi')

      from scipy.stats import circmean,circstd


      # SAVITSKY-GOLAY parameters
      # polynom order
      self.SGorder  = 3

      # SG window is galaxy_size/10 and must be odd
      # CAN'T BE RELATIVE TO IMAGE SIZE... MUST BE RELATIVE TO GALAXY SIZE
      #SGwindow = int(self.S.Mo/10.)
      self.SGwindow = int(self.P.Rp/2.)
      if self.SGwindow % 2 ==0: 
         self.SGwindow = self.SGwindow +1

      def sigma_func(params):
         ''' calculates the sigma psi with different parameters
         called by the minimization routine '''
         (x0,y0,q, PA) = params
         #### creates standardized image
         # using segmentation geometric parameters
         # galnostarsstd = self.standartize(self.P.galnostars, self.S.q, self.S.PArad, self.S.y0col, self.S.x0col)
         # using Sersic geometric parameters
         galnostarsstd = self.standartize(self.S.gal0seg, q, PA, x0, y0)
         
         # creates polar imagem
         self.galpolar = self._polarim(galnostarsstd)

         #print '%.5f %.5f %.5f %.5f' % (x0,y0,q,PA)
         
         self.galpolarpetro = self.galpolar[  int(config.Npsf * self.S.psfsigma)   : int(config.NRp * self.P.Rp),  : ]
         
         if min(self.galpolarpetro.shape) <= self.SGwindow:
            self.SGorder -= 1

         try:
            dx,dy = mfmlib.savitzky_golay_2d(self.galpolarpetro, self.SGwindow, self.SGorder,'both')
         except:
            self.SGwindow = 3 
            dx,dy = mfmlib.savitzky_golay_2d(self.galpolarpetro, self.SGwindow, 1 ,'both')

         self.mag     = np.sqrt(dx**2 + dy**2)
         self.magmask = self.mag > (np.median(self.mag))
         ort          = np.arctan2(dy, dx)
         self.ortn    = (ort+np.pi) % (np.pi)

         self.psi       = circmean(self.ortn[self.magmask])
         self.sigma_psi = circstd(self.ortn[self.magmask])
         

         return self.sigma_psi


      self.x0sigma, self.y0sigma, self.qsigma, self.PAsigma = \
                    fmin(sigma_func, (self.Ss.x0Fit2D, self.Ss.y0Fit2D, self.Ss.qFit2D, np.deg2rad(self.Ss.PAFit2D)), ftol=0.1, xtol=1.0, disp=0)


      self.sigma_psi = sigma_func((self.x0sigma, self.y0sigma, self.qsigma, self.PAsigma)) 


      if 'ortn' in sys.argv:
         pyfits.writeto(self.S.rootname +'_ortn.fits', self.ortn, overwrite=1)

      if 'polar' in sys.argv:
         pyfits.writeto(self.S.rootname + '_polarim.fits', self.galpolarpetro, overwrite=1)

      if config.verbose >1 : 
         print('   sigma_psi=%.2f' % self.sigma_psi)




   def _polarim(self, image, origin=None, log=False ):
      """Reprojects a 2D numpy array ("image") into a polar coordinate system.
      "origin" is a tuple of (x0, y0) and defaults to the center of the image.
      http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
      refactored by FF, 2013-2014 (see transpolar.py)
      """

      if origin == None:
         origin = np.array(image.shape)/2.


      def cart2polar(x, y):
         r = np.sqrt(x**2 + y**2)
         theta = np.arctan2(y, x)
         return r, theta

      def polar2cart(r, theta):
         x = r * np.cos(theta)
         y = r * np.sin(theta)
         return x, y

      def cart2logpolar(x, y, M=1):
         alpha  = 0.01
         r = np.sqrt(x**2 + y**2)
         rho = M * np.log(r + alpha)
         theta = np.arctan2(y, x)
         return rho, theta

      def logpolar2cart(rho, theta, M=1):
         x = np.exp(rho/M) * np.cos(theta)
         y = np.exp(rho/M) * np.sin(theta)
         return x, y

        
      ny, nx = image.shape 
      if origin is None:
         x0, y0 = (nx // 2, ny // 2)
         origin = (x0,y0)
      else:
         x0, y0 = origin

      # Determine that the min and max r and theta coords will be...
      x, y = np.meshgrid( np.arange(nx) - x0, np.arange(ny) - y0 )

      r, theta = cart2polar(x, y)

      # Make a regular (in polar space) grid based on the min and max r & theta
      r_i     = np.linspace(r.min(),     r.max(),     nx )
      theta_i = np.linspace(theta.min(), theta.max(), ny)
      theta_grid, r_grid = np.meshgrid(theta_i, r_i)

      # Project the r and theta grid back into pixel coordinates
      xi, yi = polar2cart(r_grid, theta_grid)
      xi += origin[0] # We need to shift the origin back to 
      yi += origin[1] # back to the lower-left corner...
      xi, yi = xi.flatten(), yi.flatten()
      coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)
      
      zi = nd.map_coordinates(image, coords, order=1)
      galpolar      = zi.reshape((nx,ny))

      self.r_polar  = r_i
      self.theta_polar = theta_i


      return galpolar
      
   '''
   def _Shear(self):

      m00, m10, m01, m11, m20, m02, mu20, mu02, mu11 = self.S.momenta(self.P.galpetro)
      # Kaiser et al 1995 shear measurements (KSB method)
      #Qxx = mu20
      #Qyy = mu02
      #Qxy = mu11
      self.shear_e1 = (mu20 - mu02) / (mu20 + mu02)
      self.shear_e2 = (  2 * mu11) / (mu20 + mu02) 
      self.shear_R  = np.sqrt(mu20 + mu02)
      self.shear_q  = 1-np.sqrt(self.shear_e1**2 + self.shear_e2**2)

      if config.verbose > 1:
         print(f'   shear e1={self.shear_e1:.5f} e2={self.shear_e2:.5f} R={self.shear_R:.2f} q={self.shear_q:.2f}' )
   '''







r'''
 ____                       _   
|  _ \ ___ _ __   ___  _ __| |_ 
| |_) / _ \ '_ \ / _ \| '__| __|
|  _ <  __/ |_) | (_) | |  | |_ 
|_| \_\___| .__/ \___/|_|   \__|
          |_|                   
'''
class Report:

   def __init__(self,S,P,Ss,M):
      log_message(2, "[ -------------------- OUTPUT -------------------- ]" )
      self.QualityFlags()

      varlist = [ 
         'S.Mo', 'S.No',
         'S.psffwhm', 'S.asecpix',
         'S.skybg', 'S.skybgstd',
         'S.x0peak', 'S.y0peak', 'S.x0col', 'S.y0col',
         'S.a', 'S.b', 'S.PAdeg', 
         'Ss.InFit1D', 'Ss.RnFit1D', 'Ss.nFit1D', 'Ss.xsin', 'Ss.chi2', 
         'Ss.x0Fit2D', 'Ss.y0Fit2D', 'Ss.InFit2D', 'Ss.RnFit2D', 'Ss.nFit2D', 'Ss.qFit2D', 'Ss.PAFit2D', 'Ss.fresid1D1_rss', 'Ss.fresid1D1_rss',  
         'M.LT', 'M.R10', 'M.R20', 'M.R30', 'M.R40', 'M.R50', 'M.R60', 'M.R70', 'M.R80', 'M.R90', 'P.Rp', 'P.Rkron', 'P.Rpkron',
         'M.C1', 'M.C2',
         'M.BGmedian', 'M.A_e', 
         'M.A0', 'M.A1', 'M.A2', 'M.A3', 'M.A4', 'M.A0Sersic', 'M.A1Sersic', 'M.Ashape',
         'M.S1', 'M.S3', 'M.G', 'M.M20',
         'M.sigma_psi',
         'M.H', 
         'P.P_xcentroid', 'P.P_ycentroid', 'P.P_area', 'P.P_perimeter', 'P.P_a_sigma', 'P.P_b_sigma',   
         'P.P_Requiv', 'P.P_PA', 'P.P_q', 'P.P_ellip', 'P.P_gini',       
         'P.P_Fkron', 'P.P_Rkronnorm', 'P.P_Rkron', 'P.P_R10', 'P.P_R20', 'P.P_R30', 'P.P_R40', 'P.P_R50', 
         'P.P_R60','P.P_R70', 'P.P_R80', 'P.P_R90', 'P.P_R100', 'P.P_C1', 'P.P_C2', 'self.QF'        
      ]         

      if config.outformat =='CSV':
         sepchar = ','
         fmtS = '{0:s}' 
         fmtF = '{0:.8f}'
      elif config.outformat == 'ASCII':
         sepchar = ' '
         fmtS = '{0:>10s}' 
         fmtF = '{0:15.8f}'

         
      logrootname = '# objid%s' % str(__version__) 


      if 'nosave' not in sys.argv:
         #### FILE output
         logfile = open(S.rootname + '.mfmtk', 'w')

         logfile.write('{0:s}{1:s}'.format( '# objid%s' % str(__version__), sepchar ))
         #logfile.write(f'{logrootname},', end=' ') 

         for i,v in enumerate(varlist):
            logfile.write((fmtS+'{1:s}').format( varlist[i].split('.')[-1] , sepchar ))
         logfile.write('\b\n')

         logfile.write( '{0:s}{1:s}'.format( S.basename, sepchar ) )
         for i,v in enumerate(varlist):
            logfile.write( (fmtF+'{1:s}').format( eval(varlist[i]), sepchar ) )
         logfile.write('\b\n')

         logfile.close()




      #### SCREEN
      #print(('{0:<50s}{1:s}').format( '# rootname%s' % str(int(10*__version__)), sepchar ), end=' ')
      print( f'{logrootname},', end=' ' )

      for i,v in enumerate(varlist):
         print((fmtS+'{1:s}').format( varlist[i].split('.')[-1], sepchar), end=' ')
      print('\b\n', end='')

      print(('{0:<s}{1:s}').format( S.basename, sepchar), end=' ')
      for i,v in enumerate(varlist):
         print((fmtF+'{1:s}').format( eval(varlist[i]), sepchar ), end=' ')
      print('\b\n', end=' ')







   def QualityFlags(self):

      targetsize   = 1
      targetisstar = 2
      fit1Derror   = 4
      fit2Derror   = 8 
      crowded1     = 16
      crowded2     = 32
      crowded3     = 64
      offcenter    = 128

      self.QF = 0
      self.QFname = []

      # not enought spatial resolution
      if S.psffwhm > P.Rp:
         self.QF += targetsize
         self.QFname.append('TARGETSIZE')

      if Ss.RnFit2D <= S.psffwhm  and Ss.nFit2D <= 0.55 and S.q>0.8:
         self.QF += targetisstar 
         self.QFname.append('TARGETSTAR')

      if Ss.sucFit1D > 4 :
         self.QF += fit1Derror 
         self.QFname.append('FIT1DERROR')

      if Ss.sucFit2D > 4 :
         self.QF += fit2Derror 
         self.QFname.append('FIT2DERROR')

      if len(np.where( np.logical_not(P.galmask)*(Ss.petromaskFit)==1)[0]) /len(np.where(P.petromask==1)[0]) > 0.10:
         self.QF += crowded1
         self.QFname.append('CROWDED_STARS')

      if Ss.RnFit2D > 2*P.Rp:
         self.QF += crowded2
         self.QFname.append('CROWDED_Rn.GT.Rp')

      if M.A4 > 20:
         self.QF += crowded3
         self.QFname.append('CROWDED_A4') 

      # seriously offcenter
      _distance_from_center =   ((S.x0col - S.No/2.)**2 + (S.y0col - S.Mo/2.)**2)**(1/2.) 
      _image_diagonal = (S.Mo**2 + S.No**2)**(1/2.)
      #print (_distance_from_center / _image_diagonal)
      if  _distance_from_center / _image_diagonal > config.OffCenterThreshold :
         self.QF += offcenter
         self.QFname.append('OFFCENTER')

      if config.verbose > 1:
         print('   QF=', self.QF, self.QFname) 



   def C1teo(self,n):
      p =  [ 7.16838058,  0.48362306,  7.48752817]
      return p[0] * (n/p[2])**p[1]

   def C2teo(self,n):
      p = [  9.21227916,   0.51671021,  23.09334208]
      return p[0] * (n/p[2])**p[1]

   def Rpteo(self,n, Rn):
      Rmax, n0, a, alpha = [ 5.80920987, -1.03854724,  1.91461262,  0.7784799 ]
      return Rn * Rmax*((n-n0)/a)*np.exp(-((n-n0)/a)**alpha)










r'''    _       _       
 _ __ | | ___ | |_ ___ 
| '_ \| |/ _ \| __/ __|
| |_) | | (_) | |_\__ \
| .__/|_|\___/ \__|___/
|_|                    
'''
class Plots():

   def __init__(self, stamp,photometry,sersic,morphometry,report, show=1):
      self.plotparams()
      
      self.S  = stamp
      self.P  = photometry
      self.Ss = sersic
      self.M  = morphometry
      self.R  = report

      self.do_plot()

      if 'nosave' not in sys.argv:
         self.saveit()

      if show:
          pl.show()


   def plotparams(self):
      pl.rcParams['font.size']= 11.0
      pl.rcParams['text.usetex'] = False

      pl.rcParams['figure.subplot.bottom']= 0.1
      pl.rcParams['figure.subplot.hspace']= 0.15
      pl.rcParams['figure.subplot.left']  = 0.125
      pl.rcParams['figure.subplot.right'] = 0.95
      pl.rcParams['figure.subplot.top']   = 0.95
      pl.rcParams['figure.subplot.wspace']= 0.15

      pl.rcParams['xtick.labelsize'] = 8
      pl.rcParams['ytick.labelsize'] = 8

      pl.rcParams['axes.labelsize']=8.0

   

   def saveit(self):
      pl.savefig(self.S.rootname + '_mfmtk' + '.pdf',dpi=200, bbox_inches='tight')
      #pl.savefig(self.S.rootname + '_mfmtk' + '.eps',dpi=300, bbox_inches='tight')
      #pl.savefig(self.S.rootname + '_mfmtk' + '.png',dpi=300, bbox_inches='tight')


   def do_plot(self):
      #pl.rcParams["font.family"] = "monospace"

      fig = pl.figure(figsize=(12,6))
      import string
      from matplotlib.gridspec import GridSpec

      grids = GridSpec(3, 5)
      
      ##################################################
      ax1 = pl.subplot(grids[0,0], aspect='equal')

      from matplotlib.patches import Ellipse
      pl.gray()
      pl.title( self.S.basename, fontsize=10)
      mfmlib.imshow(np.arcsinh(self.S.gal0seg)) #,sigma=1.0)

      #if np.any(self.P.starmask):
      #   pl.contour (self.P.starmask,1,colors='y')

      pl.contour(self.S.segmask,1,colors='g')
      ax1.text(0, -0.2, 'Seg', color='g', transform=ax1.transAxes, fontsize=10)


      pl.text(self.S.x0col,   self.S.y0col,    'c', ha='center',va='center', color='k', fontsize=9)
      pl.text(self.S.x0peak,  self.S.y0peak,   'p', ha='center',va='center', color='r', fontsize=9)
      pl.text(self.Ss.x0Fit2D,self.Ss.y0Fit2D, 'n', ha='center',va='center', color='b', fontsize=9)
      #pl.text(self.M.x0A3fit, self.M.y0A3fit,  '3', ha='center',va='center', color='y', fontsize=9)
      #pl.text(self.M.x0A1fit, self.M.y0A1fit,  '1', ha='center',va='center', color='c', fontsize=9)

      ###  RnFit
      Ea = 2*self.Ss.RnFit2D
      Eb = 2*self.Ss.qFit2D * self.Ss.RnFit2D
      Elipse2 = Ellipse(xy=(self.Ss.x0Fit2D,self.Ss.y0Fit2D),width=Ea,height=Eb,angle=self.Ss.PAFit2D,\
                        facecolor='none', edgecolor='r',ls='solid', lw=1.0)
      ax1.add_artist(Elipse2)

      ax1.text(0.2, -0.2, 'Rn_2D', color='r', transform=ax1.transAxes,  fontsize=10)


      ###  Rkron ellipse
      Ea = 2*self.P.P_Rkron
      Eb = 2*self.P.P_Rkron * self.P.P_q
      Elipsek = Ellipse(xy=(self.S.x0col,self.S.y0col),width=Ea,height=Eb,angle=self.P.P_PA ,\
                        facecolor='none', edgecolor='y',ls='solid', lw=1.0)
      ax1.add_artist(Elipsek)

      ax1.text(0.5, -0.2, 'Rk', color='y', transform=ax1.transAxes)


      ### 2 Rp
      Ea = 2 * config.NRp * self.P.Rp
      Eb = 2 * config.NRp * self.S.q * self.P.Rp
      ElipsePetro = Ellipse(xy=(self.S.x0peak,self.S.y0peak),width=Ea,height=Eb,angle=self.S.PAdeg,\
                        facecolor='none',edgecolor='#08ff10', ls='dashed', lw=2.0)
      ax1.add_artist(ElipsePetro)

      Ea = 2 * config.NRp * self.P.Rp
      Eb = 2 * config.NRp * self.Ss.qFit2D * self.P.Rp
      ElipsePetroFit = Ellipse(xy=(self.Ss.x0Fit2D,self.Ss.y0Fit2D), width=Ea,height=Eb,angle=self.Ss.PAFit2D,\
                        facecolor='none',edgecolor='#08ff10', ls='dashed', lw=1.0)
      ax1.add_artist(ElipsePetroFit)

      ax1.text(0.7, -0.2, '2Rp', color='#08ff10', transform=ax1.transAxes)






      ##################################################
      ax2 = pl.subplot(grids[0,1], aspect='equal')
      pl.title('model', fontsize=10 )

      myfitmodel = self.Ss.fitmodel.copy()
      
      #PSF (to or not to show)
      MM,NN = self.S.psf.shape

      # if PSF is smaller then 1/4 of the original image, then shows
      if MM*NN < (1/4.)*S.Mo*S.No:
         mymax = myfitmodel.max()
         myfitmodel[:MM,:NN] = 10*mymax*self.S.psf
         pl.title('model and PSF', fontsize=10 )
      
      
      pl.vlines(NN, 0, MM, color='k', linestyle='-', linewidth=1)
      pl.hlines(MM, 0, NN, color='k', linestyle='-', linewidth=1)
      mfmlib.imshow(np.arcsinh(myfitmodel))



      ##################################################
      ax3 = pl.subplot(grids[0,2], aspect='equal')
      pl.title('residual', fontsize=10 )
      mfmlib.imshow(np.arcsinh(self.Ss.fitresidual))#,sigma=1)
      
      
      #colorbar(pad=0)
      #contour(self.S.gal0-self.Ss.fitmodel, [0.0], lw=0.5,colors='y')


      ##################################################
      ax4 = pl.subplot(grids[0,3], aspect='equal')
      pl.title('A1 map', fontsize=10)
      mfmlib.imshow(self.Ss.petromaskFit * np.abs(self.M.A1img))
      pl.contour(self.Ss.petromaskFit * self.M.A1mask, 1, linewidths=0.8)
      pl.contour(self.Ss.petromaskFit,1, linewidths=0.5)



      ##################################################
      ax5 = pl.subplot(grids[0,4], aspect='equal')
      pl.title('S1 map', fontsize=10)
      mfmlib.imshow(self.Ss.petromaskFit * np.abs( self.M.S1map))
      pl.contour(self.Ss.petromaskFit * self.M.S1mask, 1, linewidths=0.8)
      pl.contour(self.Ss.petromaskFit,1, linewidths=0.5)

     
      r'''
       _           _       
      | |_ _____ _| |_ ___ 
      |  _/ -_) \ /  _/ _ \
       \__\___/_\_\\__\___/
      '''
      props = dict(boxstyle='square', facecolor='white', edgecolor='none', alpha=1.0)
      mytextfontsize = 9       

      ##################################################
      ax6 = pl.subplot(grids[1:2, 0])

      # join all the strings with and \n
      textstr = '\n'.join((
         r'$(x_0,y_0)_{\rm col}  =(%6.1f,\ %6.1f)$' % (self.S.x0col, self.S.y0col),
         r'$(x_0,y_0)_{\rm peak} =(%6.1f,\ %6.1f)$' % (self.S.x0peak, self.S.y0peak),
         r'$(x_0,y_0)_{\rm fit}  =(%6.1f,\ %6.1f)$' % (self.Ss.x0Fit2D, self.Ss.y0Fit2D),
         r'$q_{\rm seg}=%6.2f\  PA_{\rm seg}=%6.2f$' % (self.S.q, self.S.PAdeg),
         r'$q_{\rm fit}=%6.2f\  PA_{\rm fit}=%6.2f$' % (self.Ss.qFit2D, self.Ss.PAFit2D),
         r'$\chi_{1D}=%6.2f$ ' % (self.Ss.xsin),
         r'$\log\chi_{2D}=%6.4f$ ' %  np.log10(self.Ss.chi2) ,
         r'$sky = %.3f \pm %.3f$ ' % (self.S.skybg, self.S.skybgstd), 
         r'$SNR = %6.1f$' % (self.Ss.InFit2D/self.S.skybgstd),
         r'$QF=%i$' % (self.R.QF),
         r'%s' % str.join('', [ i+'\n' for i in self.R.QFname])
      ))

      # place a text box in upper left in axes coords
      ax6.text(0.05, 0.85, textstr, transform=ax6.transAxes, fontsize=mytextfontsize,  verticalalignment='top', bbox=props)

      pl.xticks([])
      pl.yticks([])
      pl.box(on=0)


      ##################################################
      ax7 = pl.subplot(grids[1:2,1:2])
      
      # join all the strings with and \n
      textstr = '\n'.join((
         r'$In_{1D}=%6.3f \quad Rn_{1D}=%6.2f \quad n_{1D}=%6.2f$' % (self.Ss.InFit1D,self.Ss.RnFit1D, self.Ss.nFit1D),
         r'$In_{2D}=%6.3f \quad Rn_{2D}=%6.2f \quad n_{2D}=%6.2f$' % (self.Ss.InFit2D,self.Ss.RnFit2D, self.Ss.nFit2D),
         r'$A_{0,Sersic}=%6.3f \quad A_{1,Sersic}=%6.2f \quad  psf_{FWHM}=%6.2f$' % (self.M.A0Sersic,self.M.A1Sersic,  self.S.psffwhm ),
         r'$Rp=%6.2f  \quad Rpk=%6.2f \quad Rk=%6.2f \quad P\_Rk=%6.2f $' % (self.P.Rp, self.P.Rpkron, self.P.Rkron, self.P.P_Rkron),
         r'$C_1=%6.2f \quad C_2=%6.2f \quad P\_C_1=%6.2f \quad P\_C_2=%6.2f$' % (self.M.C1, self.M.C2, self.P.P_C1, self.P.P_C2 ),
         r'$A_0=%6.3f \quad A_1=%6.3f \quad A_3=%6.3f \quad A_s=%6.3f $' % (self.M.A0, self.M.A1,self.M.A3, self.M.Ashape),
         r'$S_1=%6.2f \quad S_3=%6.2f \quad G=%6.2f \quad P\_G=%6.2f  $' % (self.M.S1, self.M.S3, self.M.G, self.P.P_gini ),
         r'$M_{20}=%6.2f \quad \sigma_\psi=%6.2f\ \ H=%6.2f $' % (self.M.M20, self.M.sigma_psi, self.M.H),
         r'$\log L_{\rm kron}=%6.3f \quad \log L_{2R_p}=%6.3f$' % ( np.log10(self.P.P_Fkron), np.log10(self.M.LT) ),
         '\n\n',
         r'$\bigoplus$ ' + 'Morfometryka v'+str(__version__)+' '+str(__date__)  
      ))

      # place a text box in upper left in axes coords
      ax7.text(0.05, 0.85, textstr, transform=ax7.transAxes, fontsize=mytextfontsize,  verticalalignment='top', bbox=props)


      pl.xticks([])
      pl.yticks([])
      pl.box(on=0)









      ##################################################
      ax9 = pl.subplot(grids[1,3])

      pl.plot(self.P.Raios/self.P.Rp,self.P.MUR,'-w',lw=0)
      MURylim = pl.ylim()
      MURxlim = pl.xlim()

      #title('Mean Flux I(R) ' + self.stamp.rootname)
      pl.errorbar(self.P.Raios[::1]/self.P.Rp, self.P.MUR[::1], np.abs(self.P.MURerr[::1]), ms=0, lw=1, alpha=0.3, fmt='ok', ecolor='black')
      pl.plot(self.P.Raios[self.Ss.firstpt:]/self.P.Rp, mfmlib.mag(self.Ss.IRsersic(self.P.Raios[self.Ss.firstpt:], self.Ss.InFit1D,  self.Ss.RnFit1D, self.Ss.nFit1D)), '-y', lw=3, alpha=1)
      pl.plot(self.P.Raios[self.Ss.firstpt:]/self.P.Rp, mfmlib.mag(self.Ss.IRsersic(self.P.Raios[self.Ss.firstpt:], self.Ss.InFit2D,  self.Ss.RnFit2D, self.Ss.nFit2D)), '-r', lw=3, alpha=1)
      pl.plot(self.P.Raios[::1]/self.P.Rp, self.P.MUR[::1],'ok', ms=3)

      #pl.plot(self.P.Raios/self.P.Rp, mfmlib.mag(self.P.IRspl), 'y' )


      # ylimits to show points, no errors 
      dy = np.diff(MURylim) # size in y
      dx = np.diff(MURxlim) # size in x
      # set back limits of points, not error bars
      pl.ylim(MURylim[0]-0.1*dy, MURylim[1] + 0.1*dy)
      #pl.xlim(MURxlim[0], MURxlim[1] + 0.05*dx)
      pl.xlim(0,2)
      
      #reflete o eixo y
      pl.ylim(pl.ylim()[1], pl.ylim()[0])

      #pl.text(1.0,                           pl.ylim()[0] , r'$\!\!\!\!\!\!\downarrow$', color='#08ff10', fontsize=25)
      #pl.text(1.0*self.Ss.RnFit1D/self.P.Rp, pl.ylim()[0] , r'$\downarrow$', color='y', fontsize=25)
      #pl.text(1.0*self.Ss.RnFit2D/self.P.Rp, pl.ylim()[0] , r'$\downarrow$', color='r', fontsize=25)
      #pl.text(1.0*self.M.R50/self.P.Rp,      pl.ylim()[0] , r'$\downarrow$', color='b', fontsize=25)
 
      pl.ylabel(r'$\mu(R)$', fontsize=12)
      #pl.xlabel(r'$R\ \ {\rm  [pix]}$', fontsize=12)
      #self.draw_Rp()




      ##################################################
      ax10 = pl.subplot(grids[2,3])

 
      pl.plot(self.P.Raios[self.Ss.firstpt:]/self.P.Rp,   self.Ss.fresid1D1 , '.-y', lw=1, ms=3 )
      pl.plot(self.P.Raios[self.Ss.firstpt:]/self.P.Rp,   self.Ss.fresid1D2 , '.-r', lw=1, ms=3)


      pl.axhline(0.0, color='k', lw=1.0, alpha = 0.5)
   
      #print('######## SERSIC-residuo', Ss.fresid1D1_rss )
      #print('######## SERSIC-residuo', Ss.fresid1D2_rss )
      #print('######## d2IdR2-residuo', ((self.P.d2IdR2**2)**0.5).sum())

      pl.text(.9, .9, r"$\delta^1_{rss}=%.5f$" % Ss.fresid1D1_rss, ha='right', va='top', transform=ax10.transAxes, fontsize=8)
      pl.text(.9, .8, r"$\delta^2_{rss}=%.5f$" % Ss.fresid1D2_rss, ha='right', va='top', transform=ax10.transAxes, fontsize=8)


      pl.xlim(0, pl.xlim()[1])

      pl.ylabel(r'$\delta\ \mu(R)$', fontsize=12)
      pl.xlabel(r'$R/R_p$', fontsize=12)
      #self.draw_Rp()


 
      ##################################################
      axk = pl.subplot(grids[2,4])


      #pl.plot ((P.Raios/P.Rp)[P.validkurvidx], P.IRkurv[P.validkurvidx] ,'o-k', ms=3)
      #pl.plot ((P.Raios/P.Rp), P.IRkurv ,'pk', ms=3)

      #pl.plot(P.Raios/P.Rp ,     P.logIRspl_d2(P.Raios), '+y', ms=2)
      pl.plot(P.Raios/P.Rp, P.d2IdR2 , '-m'  )
   

      pl.axhline(P.IRkurvmedian, color='k', lw=0.75,alpha = 0.5)

      

      #pl.axhline(P.IRkurvmedian + 3* P.IRkurvmad, color='k', lw=0.5,alpha=0.5)
      #pl.axhline(P.IRkurvmedian - 3* P.IRkurvmad, color='k', lw=0.5,alpha=0.5)
      
      pl.axhline(0.0, color='k',lw=0.5)
      pl.axhline(0.01, color='y',lw=0.5)
      pl.axhline(-0.01, color='y',lw=0.5)
      
      pl.ylabel(r'$\kappa(R)$', fontsize=12)
      pl.xlabel(r'$R/R_p$', fontsize=12)

      #pl.ylim(-0.02,0.02)
      


      ##################################################
      ax10 = pl.subplot(grids[1,4])

      pl.title('polar', fontsize=10)
      mfmlib.imshow(self.M.galpolarpetro ,aspect='auto' )
      pl.contour(self.M.magmask, 1, colors='k')



def make_all_measurements(galfilename, psffilename, report=False, plot=False):
   '''all-in-one package for easy of use

   returns 4 instances of classes
   S stamp
   P photometry
   N sersic
   M morphometry

   S, P, N, M = make_all_measurements('n3379_lR.fits', 'psf.fits')
   
   '''
   S = Stamp(galfilename,psffilename)
   P = Photometry(S)
   N = Sersic(S,P)
   M = Morphometry(S,P,N)

   if report == True:
      Report(S,P,N,M)
   if plot == True:
      Plots(S,P,N,M, show=1)


   return S,P,N,M






if __name__ == '__main__':


   if len(sys.argv) < 3:
      print('USAGE')
      print(sys.argv[0] , \
         '''galaxy_filename   psf_filename
               [BPSM]      choose letter[s] for: Basic Photometry Sersic Morphometry
               [noshow]    do not open plots
               [galpetro]  saves image with Petrosian region 
               [mask]      saves masked galaxy and mask 
               [asymmetry] saves asymmetry image (used for A0, A1, ...)
               [polar]     saves polar image
               [smooth]    saves smooth image (used for S1, S3, ...)
               [stangal]   saves standard galaxy image (PA=0 and eps=1)
               [segment]   saves segmented image
               [ortn]      ?
               [profile]   saves light profile I(R) 
               [clean]     applies GalClean
               [rerun]     rerun on a previously run data
               [nosave]    do not save any output files (mfmtk and figs)
               [psfauto]   get psf filename from galaxy filename (usually prefix psf_)
               ''')
      exit()
   else:
      galfilename = sys.argv[1]
      psffilename  = sys.argv[2]
 

 
   if os.path.isfile( sys.argv[1].replace('fits','mfmtk')) and 'rerun' not in sys.argv:
      print('Already run on', sys.argv[1])
      print('put \'rerun\' on command line to repeat')
      exit()
 


   S  = Stamp(galfilename,psffilename)
   P  = Photometry(S)
   Ss = Sersic(S,P)
   M  = Morphometry(S,P,Ss)
   R  = Report(S,P,Ss,M)


   if 'noshow' in sys.argv:
      Plots(S,P,Ss,M,R, show=0)
   else:
      Plots(S,P,Ss,M,R, show=1)

   exit()

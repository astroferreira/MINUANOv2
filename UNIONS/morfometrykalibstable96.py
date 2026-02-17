#!/usr/bin/python
r"""
                          __   __  
                         (  \,/  ) 
    Morfometryka LIB      \_ | _/  
          G2              (_/ \_)
  
  Fabricio Ferrari, 2012-2015
"""



__author__  = 'Fabricio Ferrari'
__email__   = 'fabricio@ferrari.pro.br'
__date__    = '2025 04 28'
__version__ = 9.6


from  sys import argv
import numpy as np
import matplotlib.pyplot as pl
import sys
from astropy.io import fits as pyfits
import scipy.signal
import scipy.optimize as opt
import scipy.signal   as sig
import scipy.ndimage  as nd



def help():
   print(__doc__, file=sys.stderr)

def mag(x):
   return (-2.5*np.log10(x))

def mad(x):
   "median absolute deviation"
   return np.ma.median( np.ma.abs( x - np.ma.median(x)) )

def gaussian2D(x0,y0,sigma, M,N):
    x,y = np.meshgrid(np.arange(N)-x0, np.arange(M)-y0)
    r2 = (x)**2 + (y)**2
    mask = 1./np.sqrt(2.*np.pi*sigma**2.) * np.exp(-r2/(2.*sigma**2.))
    return mask

def lognormal(x,I0,mu,s):
   return (I0/(x*s*np.sqrt(2.0*np.pi)))*np.exp(-((np.log(x)-mu)**2.0)/(2.*s**2.))





def PSFfitgaussian(data, param0):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""



    def gaussian(base, height, center_x, center_y, width):
        """
        Returns a 2D gaussian function with the given parameters
        """    
        return lambda y,x: base+ \
            height*np.exp(-( ((x-center_x)/width)**2. + ((y-center_y)/width)**2. )/2.)


    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape))-data)

    p, success = opt.leastsq(errorfunction, param0)

    return (p,success)


def trimborder(gal):
   '''detects if the pixel border has only zeros,
   then walks pixel by pixel until find something,
   finally cuts a border around the image equal to the
   largest  border found (so object remains in center)
   (used to trim SDSS borders before processing with FTMTK)
   code elsewhere
   
   Returns     borders-(Ydown, Ytop, Xleft, Xright), unbordered-image
   '''
   M,N = gal.shape
   xi=yi=xf=yf=0.0
   FoundBorder = False

   if np.all(gal[0,:]==0):
       FoundBorder = True
       yi = 1
       while(all(gal[yi,:]==0)):
           yi+=1

   if np.all(gal[-1,:]==0):
       FoundBorder = True
       yf = 1
       while(all(gal[-yf,:]==0)):
           yf+=1

   if np.all(gal[:,0]==0):
       FoundBorder = True
       xi = 1
       while(all(gal[:, xi]==0)):
           xi += 1

   if np.all(gal[:,-1]==0):
       FoundBorder = True
       xf = 1
       while(all(gal[:, -xf]==0)):
           xf += 1

   if (FoundBorder==True):

       if yf==0:
           if xf!=0:
              gal = gal[yi:, xi:-xf]
           elif  xf==0:   
              gal = gal[yi:, xi:]
       elif yf!=0: 
           if xf!=0:
              gal = gal[yi:-yf, xi:-xf]
           elif  xf==0:   
              gal = gal[yi:-yf, xi:]

   return ((yi,yf,xi,xf),gal)




def savitzky_golay_2d ( z, window_size, order, derivative=None):
    """
    http://nbviewer.ipython.org/github/pv/SciPy-CookBook/blob/master/ipython/SavitzkyGolay.ipynb
    """

    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0
    
    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')
    
    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2
    
    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]
    
    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx  = np.repeat( ind, window_size )
    dy  = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])
        
    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band ) 
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z
    
    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band ) 
    
    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band ) 
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band ) 
    
    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')        
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')        
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
        if (window_size % 2) == 0:
           window_size += 1 
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = list(range(order+1))
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
    






def kurvature(I, dx=1.0, signed=1):
    '''
    calculates the curvature of the curve I
    https://en.wikipedia.org/wiki/Curvature
    '''
    if signed:
       curvatura = ( np.gradient(np.gradient(I,dx),dx))/ (  1 + (np.gradient(I,dx))**2. )**(3/2.) 
    else:
       curvatura = abs(curvatura) 

    return curvatura



def entropy(img, bins=100, normed=1):
   h = np.histogram(img.ravel(), bins=bins)[0]
   pimg = h/float(h.sum())
   S =  np.ma.sum([ -p*np.ma.log(p) for p in pimg if p!=0 ])
   if normed==1:
      N = np.ma.log(bins)
   else:
      N = 1.0
   return S/N



def geo_mom(p,q,I, centered=True, normed=True, complex=False, verbose=False):
   """return the central moment M_{p,q} of image I
   http://en.wikipedia.org/wiki/Image_moment
   F.Ferrari 2012, prior to 4th JPAS
   """

   M,N = I.shape
   x,y = np.meshgrid(np.arange(N), np.arange(M))

   M_00 = I.sum()
       
   if centered:
      # centroids
      x_c = (1/M_00) * np.sum(x * I ) 
      y_c = (1/M_00) * np.sum(y * I )
      
      x = x - x_c
      y = y - y_c

      if verbose:
         print('centroid  at', x_c,y_c) 
   
   if normed:
      NORM = M_00**(1+(p+q)/2.)
   else:
      NORM = 1.0
      
   if complex:
      XX = (x+y*1j)
      YY = (x-y*1j)
   else:
      XX = x
      YY = y
      
   M_pq = (1/NORM) * np.sum( XX**p * YY**q * I) 

   return M_pq



def imshow(img, sigma=3, contours=0, bar=None, aspect='equal', extent=None, vmin=None, vmax=None, use_median=False):
    """
    improved version of pl.imshow,

    shows image with limits sigma above and below mean.

    optionally show contours and colorbar
    """
    
    def mad(x):
       return np.median( np.abs( x - np.median(x)) )

    # deals with NaN and Infinity
    img[np.where(np.isnan(img))]=0.0
    img[np.where(np.isinf(img))]=0.0


    # wether to use median/mad or mean/stdev.
    # note that mad ~ 1.5 stdev
    if use_median==False:
      if vmin==None:
         vmin = img.mean() - sigma * img.std()
      if vmax==None:
         vmax = img.mean() + sigma * img.std()
    else:
      if vmin==None:
         vmin = np.median(img) - 1.5*sigma * mad(img)
      if vmax==None:
         vmax = np.median(img) + 1.5*sigma * mad(img)


    pl.imshow(img, vmin=vmin, vmax=vmax, origin='lower', aspect=aspect, extent=extent, interpolation=None)

    if bar != None:
        pl.colorbar(pad=0)

    if contours >0:
        pl.contour(img, contours, colors='k', linestyles='solid', aspect=aspect, extent=extent)





r'''____                _      
   / ___|  ___ _ __ ___(_) ___ 
   \___ \ / _ \ '__/ __| |/ __|
    ___) |  __/ |  \__ \ | (__ 
   |____/ \___|_|  |___/_|\___|
'''

"""
Ajuste Sersic 2D em galaxias
Fabricio Ferrari, ago 2013

 leastsq_bounds.py
http://stackoverflow.com/questions/9878558/scipy-optimize-leastsq-with-bound-constraints

IMPORTANT NOTE:
      bounded fit is approx 5x FASTER than unbounded.
"""




class SersicFit(object):

    def __init__(self):
        """ 
        galaxy    - image to fit  (array)
        p0        - [x00,y00,Ib0,In0,Rn0,n0, q0,PA0]
        PSF       - can be    1.  fwhm size  in pixels
                              2.  psf image
                              
        Example
        SF = SersicFit2D()
        x0, y0, Ib, In, Rn, n, q, PA, suc = SF.fit(galaxy, psf, [x00,y00,Ib0,In0,Rn0,n0, q0,PA0], 'meu_teste') 
        """



    def fit1D(self, R, IR, pars0, fittype='bounded'):

       
       self.R  = R
       self.IR = IR
       In0,Rn0,n0 = pars0

       ##### FREE FIT  
       if fittype=='free':
          parFit1, sucFit1 = opt.leastsq(self.error1D, pars0)
          Inffit,Rnffit,nffit = parFit1
          return np.array(( Inffit,Rnffit,nffit, sucFit2))

       #####  BOUNDED FIT
       if fittype=='bounded':
          #  LIMITES 
          Inmin=np.min(IR)
          Inmax=np.max(IR)
          Rnmin=1.0
          Rnmax=len(R)
          nmin=1./2.
          nmax=50.
          
          peso = 1000.

          vinculos = [[Inmin,Inmax], [Rnmin,Rnmax], [nmin,nmax]]

          parFit1, m1, m2, m3, sucFit1 = \
               self.leastsq_bounds(self.error1D, pars0, vinculos, boundsweight=peso,  full_output=1)
          
          Inbfit,Rnbfit,nbfit = parFit1

          return np.array(( Inbfit,Rnbfit,nbfit, sucFit1))




    def fit2D(self, galaxy, PSF, pars0, mask, fittype='bounded'):

        self.galaxy = galaxy
        self.PSF    = PSF
        self.mask   = mask  
        self.count=0

        x00,y00, In0,Rn0,n0, q0,PA0 = pars0

        ##### FREE FIT  
        if fittype=='free':
            parFit2, sucFit2 = opt.leastsq(self.error2D, pars0)

            x0ffit,y0ffit, Inffit,Rnffit,nffit, qffit,PAffit = parFit2
            PAffit = PAffit % 180

            return np.array(( x0ffit,y0ffit,Inffit,Rnffit,nffit, qffit,PAffit, sucFit2))


        #####  BOUNDED FIT
        if fittype=='bounded':
            #  LIMITES 
            x0min= 0.70*x00
            x0max= 1.30*x00
            y0min= 0.70*y00
            y0max= 1.30*y00
            Inmin= np.min( self.galaxy[np.where(mask!=0)] )
            Inmax= np.max( self.galaxy[np.where(mask!=0)] )
            Rnmin= 1.0
            Rnmax= np.sqrt(np.dot(self.galaxy.shape,self.galaxy.shape)) #  diagonal da imagem
            nmin=  0.1
            nmax=  21.
            qmin=  0.05
            qmax=  1.0
            PAmin=-3600.0
            PAmax= 3600.0

            peso = 1000.

            vinculos = [[x0min,x0max], [y0min,y0max], \
                        [Inmin,Inmax], [Rnmin,Rnmax], [nmin,nmax], \
                        [qmin,qmax], [PAmin,PAmax]]

            parFit2, m1, m2, m3, sucFit2 = \
                     self.leastsq_bounds(self.error2D, pars0, vinculos, boundsweight=peso,  full_output=1)

            x0bfit,y0bfit, Inbfit,Rnbfit,nbfit, qbfit,PAbfit = parFit2

            PAbfit = PAbfit % 180
            return np.array(( x0bfit,y0bfit,Inbfit,Rnbfit,nbfit, qbfit,PAbfit, sucFit2))




    def sersicProfile1D(self, p, R):
       In, Rn, n = p 
       bn = 1.9992*n - 0.3271
       return In * np.exp(-bn*((R/Rn)**(1./n) - 1 ))


    def sersicProfile2D(self, pars, psf, M,N):
       x0,y0, In, Rn,n, q, PA = pars

       x,y = np.meshgrid(np.arange(N), np.arange(M))

       (i,j) = self.rotation(PA, x0, y0, x, y)
       j = j / q
       r = (i**2 + j**2)**0.5
       bn = 1.9992 * n - 0.3271	

       # background Ib removed from equation
       G =  In * np.exp (-bn*((r/Rn)**(1./n) -1)) 
       G = sig.fftconvolve(G,psf, 'same')
       #G = sig.convolve(G,psf, 'same')
        
       return G


    def error2D(self,p):
       #pl.imshow(self.mask * (self.sersicProfile2D(p, self.PSF, self.galaxy.shape) - self.galaxy))
       #pl.show()
       return np.ravel(self.mask * (self.sersicProfile2D(p, self.PSF, *self.galaxy.shape) - self.galaxy))
       

    def error1D(self, p):
       return (self.sersicProfile1D(p,self.R)-self.IR)


    def rotation(self, PA,x0,y0, x, y):
       #convert to radians
       t = PA * np.pi/180.
       return ((x-x0)*np.cos(t) + (y-y0)*np.sin(t), -(x-x0)*np.sin(t) + (y-y0)*np.cos(t))

        
    def leastsq_bounds(self, func, x0, bounds, boundsweight=10, **kwargs ):
        """ 
        http://stackoverflow.com/questions/9878558/scipy-optimize-leastsq-with-bound-constraints

        leastsq with bound conatraints lo <= p <= hi
        run leastsq with additional constraints to minimize the sum of squares of
            [func(p) ...]
            + boundsweight * [max( lo_i - p_i, 0, p_i - hi_i ) ...]

        Parameters
        ----------
        func() : a function of parameters `p`
        bounds : an n x 2 list or array `[[lo_0,hi_0], [lo_1, hi_1] ...]`.
            Use e.g. [0, inf]; do not use NaNs.
            A bound e.g. [2,2] pins that x_j == 2.
        boundsweight : weights the bounds constraints
        kwargs : keyword args passed on to leastsq

        Returns
        -------
        exactly as for leastsq,
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html

        Notes
        -----
        The bounds may not be met if boundsweight is too small;
        check that with e.g. check_bounds( p, bounds ) below.

        To access `x` in `func(p)`, `def func( p, x=xouter )`
        or make it global, or `self.x` in a class.

        There are quite a few methods for box constraints;
        you'll maybe sing a longer song ...
        Comments are welcome, test cases most welcome.

        """
        # Example: test_leastsq_bounds.py

        if bounds is not None  and  boundsweight > 0:
            funcbox = lambda p: \
                np.hstack(( func(p), self._inbox( p, bounds, boundsweight )))
        else:
            funcbox = func
        return opt.leastsq( funcbox, x0, **kwargs )



    def _inbox(self, X, box, weight=1 ):
        """ -> [tub( Xj, loj, hij ) ... ]
            all 0  <=>  X in box, lo <= X <= hi
        """
        assert len(X) == len(box), \
            "len X %d != len box %d" % (len(X), len(box))
        return weight * np.array([
            np.fmax( lo - x, 0 ) + np.fmax( 0, x - hi )
                for x, (lo,hi) in zip( X, box )])

        # def tub( x, lo, hi ):
        #     """ \___/  down to lo, 0 lo .. hi, up from hi """
        #     return np.fmax( lo - x, 0 ) + np.fmax( 0, x - hi )


    def check_bounds( X, box ):
        """ print Xj not in box, loj <= Xj <= hij
            return nr not in
        """
        nX, nbox = len(X), len(box)
        assert nX == nbox, \
            "len X %d != len box %d" % (nX, nbox)
        nnotin = 0
        for j, x, (lo,hi) in zip( list(range(nX)), X, box ):
            if not (lo <= x <= hi):
                print("check_bounds: x[%d] %g is not in box %g .. %g" % (j, x, lo, hi))
                nnotin += 1
        return nnotin









#########
######## TIRAR FORA !!!!!!!!!!!!!! 22/08/2018
########

class wlet2D():
    r"""  
    ...................................................
    Discrete Wavelet Transform
    A Trous Algoritm    __   __  
           2D          (  \,/  ) 
                        \_ | _/  
                        (_/ \_)  
               Butterfly Software System 
 
    Fabricio Ferrari (c) 2009-2012 [www.ferrari.pro.br]
    ...................................................

    Class to perform the Bidimensional Discrete Wavelet Transform
    
    Can be call with arrays as arguments    wlet1D(img=array)  
    or specifying the filename              wlet1D(filename='filename')


    INPUTs
    img:         2D image array 
    filename:    reads from fits file
    scales:      array with scales. If None calculates as 2**Km where K is the
                 maximum possible scale. (must begin with 1!!!)                 
    wtype        'atrous' (default)  or 'gaussian' (uses gaussian low-pass filter)


    CALCULATES:
    C:           array cube of smooth  wavelets coefficients 
    W:           array cube of  wavelets transform coefficientes 
    N:           number of data points 
    """

    def __init__(self, img=None, filename=None, verbose=False, scales=None, wtype='atrous'):
        
        self.verbose = verbose

	# check if scale was specified 
        self.scales = scales
        
        self.ReadData(img,filename)

        if wtype   == 'atrous':
            self.WTransfATrous(scales)
        elif wtype == 'gaussian':
            self.WTransfGaussian(scales)


    def ReadData(self, img, filename):
        """ Reads the data from filename;
        first column is independent variable (x), 
        second is the dependent one (y)"""

        if   img is not None and filename is None:
            self.img = img
            self.outbasename =  'Buwlet2D--'
        elif img==None and filename!= None: 
            self.filename = filename 
            self.img, self.hdr = pf.getdata(filename, header=True)        
            self.outbasename =  self.filename.replace('.fits', '--' )
        else:
            print('Error loading data')
            exit()
      
        self.M   = self.img.shape[0]
        self.N   = self.img.shape[1]
        if self.verbose: 
            print('Reading image, shape MxN=',self.M,self.N)

        # estimation of the maxium possible scale
        hsize = 5  # filter size
        maxKy = int(1 + np.log10(self.M / (hsize - 1))/np.log10(2))
        maxKx = int(1 + np.log10(self.N / (hsize - 1))/np.log10(2))

        if self.scales==None:
            self.K = min(maxKx, maxKy)
            self.scales = 2.**np.arange(0,self.K)
        else:
            # must not begin with 0 scale
            if self.scales[0] == 0:
                self.scales = self.scales[1:]
	        
            self.K = len(self.scales)


        if self.verbose:
            print('Maximum wavelet scale: Kx=%2i  Ky=%2i' % (maxKx, maxKy), end=' ') 
            print('  using Kmax=', max(self.scales))
            print('scales ', self.scales)
            
        self.C = np.zeros((self.K+1,self.M,self.N), dtype='float32')
        self.W = np.zeros((self.K,self.M,self.N), dtype='float32')

        self.imgdenoised = np.zeros((self.M,self.N),dtype='float32')

        # Valores originais
        self.C[0] = self.img.copy()



    def WTransfATrous(self, scale):
        """Wavelet Transform the data with the a trous algorithm 
        Scale is the scale of the wavelet tranform
        """
        # coeficientes funcao escala
        h0 = np.array([1/16., 1/4., 3/8., 1/4., 1/16.])
        #hh = h * h.reshape((5,1))
        hh = np.outer(h0,h0)

        def filter_h_1D(k):
            Nelem = int(2**k * (len(h0)-1) + 1)
            hk = Nelem * [0]
            hk[::2**k] = h0
            return np.array(hk)

        def filterh(k):
            return np.outer(filter_h_1D(k), filter_h_1D(k))

        if self.verbose: print("Calculating the a-trous wavelet transform ")

        import scipy.signal as sig
        
        k = 0
        for k in range(self.K):
        #for s in self.scales:
            if self.verbose: print('k = ',k, '   scale =', 2**k, 'pix')

            filter = filterh(k)
            brdrsz =  2**(k+1)
            data   = self.border(self.C[k], brdrsz, 'constant')

            self.C[k+1] = sig.fftconvolve(data, filter, 'same')[brdrsz:-brdrsz, brdrsz:-brdrsz]
            self.W[k] = self.C[k] - self.C[k+1]
            k += 1

        self.W = np.concatenate((self.W, self.C[-1][np.newaxis,:,:]) )

            

    def WTransfGaussian(self, scales):
        """Wavelet Transform the data with a Gaussian lowpass filter 
        Scale is the scale of the wavelet tranform
        """
        import scipy.signal as sig
        print('IN TESTING PHASE!!!!!')
        if self.verbose: print("Calculating the wavelet transform ")

        def wfilter(sigma):
            # guarantees that filter border is ~zero
            # <1E-9 for 10x,  8000 for 7x
            size = 7.*sigma
            # avoid NaN
            if sigma == 0:
                sigma = 0.1
            h = sig.gaussian(size,sigma)
            hh = np.outer(h,h)
            hh = hh/hh.sum()
            return hh

        # border size, must be odd (fast FFT)
        brdrsz  = int(max(self.M, self.N)/20)
        if brdrsz % 2 != 0:
            brdrsz +=1
            
        imgbrdr = self.border(self.img, brdrsz)

        if self.verbose: print(self.scales)
        k = 0 
        for s in self.scales:
            if self.verbose: print('k = ',k, '   scale =', s, 'pix')
            self.C[k+1] = sig.fftconvolve(imgbrdr, wfilter(s), 'same')[brdrsz:-brdrsz, brdrsz:-brdrsz ]
            k += 1 


        for k in range(0,self.K):
            self.W[k] = self.C[k] - self.C[k+1]

        # add wC so self.W is the complete wavelet transform
        self.W = np.concatenate((self.W, self.C[-1][np.newaxis,:,:]) )

        if self.verbose: print('W[] calculated') 

    '''
    def power_spectrum(self, normed=True):
        """Calculates the POWER SPECTRUM of W
        return scales and Spectrum """
        S = np.zeros(self.K)
        for i in range(self.K):
            S[i] =  abs((self.W[i])**2).sum()
        if normed:
            S = S/S.sum()
        return self.scales, S

    def power_spectrum_entropy(self):
        """Calculates the PATTERN SPECTRUM ENTROPY of W
        return scales and Spectrum
        TESTING PURPOSES
        MAYBE REMOVED LATER"""
        S = self.power_spectrum(normed=1)[1]
        return (-sum(S*np.log(S)))

    def power_spectrum_meansize(self):
        """Calculates the POWER SPECTRUM MEANSIZE of W
        TESTING PURPOSES
        MAYBE REMOVED LATER"""
        lam, S = self.power_spectrum(normed=1)
        return (sum(lam*S))

    def pattern_spectrum(self, normed=True):
        """Calculates the PATTERN SPECTRUM of W
        return scales and Spectrum
        TESTING PURPOSES
        MAYBE REMOVED LATER"""
        S = np.zeros(self.K)
        for i in range(self.K):
            S[i] =  abs(self.W[i]).sum()
        if normed:
            S = S/S.sum()
        return self.scales, S

    def power_spectrum_entropy(self):
        """Calculates the PATTERN SPECTRUM ENTROPY of W
        return scales and Spectrum
        TESTING PURPOSES
        MAYBE REMOVED LATER"""
        S = self.pattern_spectrum(normed=1)[1]
        return (-sum(S*np.log(S)))

    def pattern_spectrum_meansize(self):
        """Calculates the PATTERN SPECTRUM MEANSIZE of W
        return scales and Spectrum
        TESTING PURPOSES
        MAYBE REMOVED LATER"""
        lam, S = self.pattern_spectrum(normed=1)
        return (sum(lam*S))

    '''

    def border(self, x, bsize, type='constant'):
        """
        Extends the original data by adding borders
        x 1D ou 2D
        type:   mirror 
        """
        if bsize == 0:
            return x
        
        bsize = int(bsize)
        
        if type=='mirror':
            xtmp = np.hstack((x[:,bsize:0:-1], x, x[:,-2:-bsize-2:-1]))    
            xnew = np.vstack((xtmp[bsize:0:-1, :], xtmp, xtmp[-2:-bsize-2:-1, :]))

        if type=='constant':
            # lado vertical esquerdo e direito
            ladove = x[:,0]
            ladovd = x[:,-1]
            for m in range(bsize-1):
                ladove =  np.vstack((ladove,x[:,0]))
                ladovd =  np.vstack((ladovd,x[:,-1]))
            xnew = np.hstack((ladove.T,x,ladovd.T))

            ladohe = xnew[0]
            ladohd = xnew[-1]
            for m in range(bsize-1):
                ladohe =  np.vstack((ladohe,xnew[0]))
                ladohd =  np.vstack((ladohd,xnew[-1]))
            xnew = np.vstack((ladohe,xnew,ladohd))

            
        return xnew


    def getWs(self):
        "return the array with the wavelet transform"
        return self.W

    def getOriginal(self):
        "return the original data"
        return self.C[0]

    def saveWs(self):
        "Saves the Wavelets coefficients  in a file"
        datafileW = self.outbasename + 'Ws.fits'

        try:
            pf.writeto(datafileW,self.W, clobber=1)
            print("Saved wavelets coeffs in ", datafileW) 
        except:
            print('Could not save Ws and C data files')

'''          END CLASS WLET2D                       '''











'''
  mmm    mm   m        mmm  m      mmmmmm   mm   mm   m
 m"   "   ##   #      m"   " #      #        ##   #"m  #
 #   mm  #  #  #      #      #      #mmmmm  #  #  # #m #
 #    #  #mm#  #      #      #      #       #mm#  #  # #
  "mmm" #    # #mmmmm  "mmm" #mmmmm #mmmmm #    # #   ##

'''




'''

    GalClean (https://github.com/astroferreira/galclean)

    This simple tool was developed to remove bright sources
    other than the central galaxy in EFIGI [1] stamps. It uses
    Astropy's Photutils detect_sources alongside some transformations
    with binary masks. It can be used as a standalone
    tool or as a module through the galclean function (see .ipynb
    for an example). The first version of galclean was used in
    10.1093/mnras/stx2266 [2].

    usage:
        python galclean.py <file_path_to_fits>
                [--siglevel SIGLEVEL] [--min_size MIN_SIZE]

        siglevel : float
            float with the number of std deviations
            above the sky background in which to use in the detection step.

        min_size: float
            minimum size in fraction of the original img size to detect
            external sources. If min_size = 0.01, this means 1% of
            the image size. For a image of NxN size this would mean
            (N*0.01)^2 pixels of the upscaled image.

    output:
        segmented fits: <original_name>_seg.fits
        plot with segmentation and residuals: segmentation.png

    Author: Leonardo de Albernaz Ferreira
            leonardo.ferreira@nottingham.ac.uk
            (https://github.com/astroferreira)

    References:
        [1] https://www.astromatic.net/projects/efigi
        [2] http://adsabs.harvard.edu/abs/2018MNRAS.473.2701D

'''
import sys
import argparse
import warnings

import numpy as np
np.round_ = np.round

import numpy.ma as ma

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plt

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.stats import biweight_midvariance, mad_std, sigma_clipped_stats
from astropy.utils.exceptions import AstropyDeprecationWarning
from photutils.segmentation import detect_sources, detect_threshold

from scipy.ndimage import binary_dilation, zoom


def measure_background(data, iterations, mask):
    '''
        Measure background mean, median and std using an
        recursive/iteractive function.

        This is a little different from examples in
        Photutils for Background estimation, here I use
        dilation on sources masks and feed the new mask
        to the next iteration.

        Parameters
        ----------
        data : array_like
            2D array of the image.
        iterations : int
            Number of iteractions until returning the
            measurement of the sky background. This is
            a recursive function. When ``iterations`` = 0
            it will return the mean, median, and std.
        mask :  array_like
            2D segmentation array of the sources found
            in previous interaction.
        Returns
        -------
        (mean, median, std) : tuple of floats
            The mean, median and standard deviation of
            the sky background measured in ``data``.
    '''

    if(mask.sum() > 0):
        mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    else:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    if(iterations == 0):
        return mean, median, std
    else:
        threshold = median + (std * 2)
        segm_img = detect_sources(data, threshold, npixels=5)

        circ_mask = generate_circular_kernel(5)
        next_mask = binary_dilation(segm_img, circ_mask)
        return measure_background(data, iterations-1, next_mask)


def generate_circular_kernel(d):
    '''
        Generates a circular kernel to be used with
        dilation transforms used in GalClean. This
        is done with an 2D matrix of 1s, flipping to
        0 when x^2 + y^2 > r^2.

        Parameters
        ----------
        d : int
            The diameter of the kernel. This should be an odd
            number, but the function handles it in either case.

        Returns
        -------
        circular_kernel : 2D numpy.array bool
            A 2D boolean array with circular shape.
    '''

    d = int(d)

    if (d % 2) == 0:
        d = d + 1

    mask = np.ones((d, d))
    r = np.round(d/2)
    x0 = r
    y0 = r
    for i in range(0, d, 1):
        for j in range(0, d, 1):
            xx = (i-x0)**2
            yy = (j-y0)**2
            rr = r**2
            if xx + yy > rr:
                mask[i][j] = 0

    return mask


def segmentation_map(data, threshold, min_size=0.01):
    '''
        Generates the segmentation map used to segment
        the galaxy image. This function handles the
        detection of sources, selection of the galaxy
        mask and the manipulations necessary to remove
        the galaxy from the segmentation map. Dilation
        transforms are applied to the external sources
        mask and to the galaxy mask. This is done to make
        the segmentation slightly bigger than what is
        produced by detect_sources. This procedure enable
        galclean to extract the outskirts of sources that
        are bellow the sky background threshold (which is
        generally the case with most sources).

        Parameters
        ----------
        data :  array_like
            The 2D array of the image.
        threshold : float
            Threshold above the sky background in which to
            use as the detection level in detect_sources.
        min_size : float
            Minimum fraction size as a fraction of the image size
            to convert to number of pixels in which is used
            to detect sources. If the source is below
            this size it will not be detected.
        Returns
        -------
        seg_map : array_like
            2D segmentation map of the external sources
            excluding the galaxy in the centre.
    '''

    zp = int(np.round(data.shape[0]/2))

    npixels = int((data.shape[0]*min_size)**2)
    seg_map = detect_sources(data, threshold, npixels=npixels).data

    gal_mask = np.zeros_like(seg_map)

    gal_mask[np.where(seg_map == seg_map[zp, zp])] = 1

    # binary dilation with gal_mask, to make galmask bigger
    gal_mask = binary_dilation(gal_mask, generate_circular_kernel(zp/10))

    background_pixels = data[seg_map == 0]

    seg_map[seg_map == seg_map[zp, zp]] = 0
    seg_map[seg_map > 0] = 1

    seg_map = seg_map - gal_mask

    seg_map[seg_map < 0] = 0
    seg_map[seg_map > 0] = 1

    # binary dilation for sources segmentation map,
    #  zp/20 ~ 2.5% of the image galaxy size
    seg_map = binary_dilation(seg_map, generate_circular_kernel(zp/20))

    return seg_map, background_pixels


def rescale(data, scale_factor):
    '''
        This is simply a wrapper to the zoom
        function of scipy in order to avoid
        oversampling.

        Parameters
        ----------
        data : array_like
            2D array of the image
        scale_factor : float
            Scale factor to apply in the zoom.
        Returns
        -------
        rescaled_data : array_like
            scaled version of ``data`` in which its
            size was changed by ``scale_factor``.
    '''

    if(data.shape[0]*scale_factor > 2000):
        scale_factor = 2000/data.shape[0]

    return zoom(data, scale_factor, prefilter=True)


def galclean(ori_img, std_level=4, min_size=0.01, show=False, save=True):
    '''
        Galclean measures the sky background, upscales
        the galaxy image, find the segmentation map of
        sources above the threshold

            threshold = sky median + sky std * level

        where level is usually 3 but can be passed as
        argument in the function. <min_size> is the
        minimum size (as a fraction of the image) for
        the detection of a source. It is converted in the
        segmentation map step to the correspondent number
        of pixels a source must have to be considered for
        detection. It then removes the galaxy from the center,
        apply the segmentation map to the upscaled image,
        replaces external sources with the sky background
        median and then downscale it to the original scale.

        Parameters
        ----------
        ori_img : array_like
            2D array of the image
        std_level : float
            Number of standard deviations above the sky median
            used to define the threshold for detection.
        min_size : float
            Minimum fraction size as a fraction of the image size
            to convert to number of pixels in which is used
            to detect sources. If the source is below
            this size it will not be detected.
        show: bool
            Used as an option to show or not the inspection
            PNG after segmentation.
        Returns
        -------
        segmented_img : array_like
            2D array of the image after segmentation.

    '''
    mean, median, std = measure_background(ori_img, 2, np.zeros_like(ori_img))
    threshold = median + (std_level*std)

    # upscale the image. It is easier to segment larger sources.
    scaled_img = rescale(ori_img, 4)

    seg_map, background_pixels = segmentation_map(scaled_img, threshold, min_size=min_size)

    
    # apply segmentation map to the image. Replace segmented regions
    # with sky median
    segmented_img = np.zeros_like(scaled_img)
    segmented_img[seg_map == 0] = scaled_img[seg_map == 0]

    n_pix_to_replace = segmented_img[seg_map == 1].shape[0]
    segmented_img[seg_map == 1] = np.random.choice(background_pixels,
                                                   n_pix_to_replace)

    downscale_factor = ori_img.shape[0]/segmented_img.shape[0]
    segmented_img = rescale(segmented_img, downscale_factor)

    #plot_result(ori_img, segmented_img, seg_map, show=show, save=save)
    
    return segmented_img


def galshow(data, ax=None, vmax=99.5, vmin=None):
    '''
        Wrapper to the imshow function of matplotlib
        pyplot. This is just to apply the look and feel
        I generally use with galaxy images representations.

        Parameters
        ----------
        data : array_like
            2D array of the image
        ax : Axis
            Axis in which to plot the image. If ``None``, it
            creates a new Axis.
        vmax: float
            Top fraction in which to use clip the image when plotting
            it. It is used in numpy.percentile.
        vmin: flot
            Bottom fraction in which to clip the image when plotting
            it. It is used in numpy.percentile.
        Returns
        -------
        ax : AxisImage
            The Axis provided as argument after plot or the one
            created for it.
    '''

    if(vmax is not None):
        vmax = np.percentile(data, vmax)

    if(vmin is not None):
        vmin = np.percentile(data, vmin)

    if(ax is None):
        f, ax = plt.subplots(1, 1)

    ax.set_xticks([])
    ax.set_yticks([])

    return ax.imshow(data, vmax=vmax, vmin=vmin, cmap='gist_gray_r',
                     origin='lower')


def plot_result(ori_img, segmented_img, seg_map, show=False, save=False):
    '''
        Plot the original image, segmented and
        the residual.

        Parameters
        ----------
        ori_img : array_like
            2D array for the original image (before)
            segmentation.
        segmented_img : array_like
            2D array for the segmented image.
        save : bool
            Option used to determine to save the
            ouput image or not.

    '''
    residual = ori_img - segmented_img
    
    if(show):
        fig, axs = plt.subplots(1, 3, figsize=(10, 4))

        axs[0].set_title('Original Image')
        galshow(ori_img, axs[0])

        axs[1].set_title('Segmented Image')
        galshow(segmented_img, axs[1])

        axs[2].set_title('Original - Segmented')
        galshow(seg_map, axs[2])

        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()

        if(save):
            fig.savefig('segmentation.png')
            print('Output Inspection PNG {}'.format('segmentation.png'))
            
    if(save):
        np.save('segmentation_map', seg_map)
        print('Output Segmap Mask {}'.format('segmentation_map.npy'))
        


def __handle_input(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path',
                        help='Path to the galaxy image fits file'
                        )
    parser.add_argument('--siglevel',
                        help='STD above skybackground level \
                              to detect sources. Default: 4',
                        type=float,
                        default=4)
    parser.add_argument('--min_size',
                        help='Minimum size of the sources to be \
                              extracted in fraction of the \
                              image total size (e.g. 0.01 for 1 \
                              per cent of the image size). Default: 0.01',
                        type=float,
                        default=0.01)
    
    parser.add_argument('--show', nargs='?', default=False, const=True)
    parser.add_argument('--save', nargs='?', default=False, const=True)

    args = parser.parse_args()

    print('Running GalClean with siglevel {} and min_size {}'
          .format(args.siglevel, args.min_size))

    return args


#EOF

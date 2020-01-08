# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u
import scipy.stats as stats
from scipy import signal, linalg

from iturpropag.utils import prepare_quantity


class __ITU1853():
    """Tropospheric attenuation time series synthesis

    Available versions include:
    * P.1853-0 (10/09) (Superseded)
    * P.1853-1 (02/12) (Superseded)
    * P.1853-2 (08/2019) (Current version)
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.1853 recommendation.

    def __init__(self, version=2):
        if version == 2:
            self.instance = _ITU1853_2()
        elif version == 1:
            self.instance = _ITU1853_1()
        elif version == 0:
            self.instance = _ITU1853_0()
        else:
            raise ValueError(
                'Version {0} is not implemented for the ITU-R P.1853 model.'
                .format(version))


    @property
    def __version__(self):
        return self.instance.__version__

    def scintillation_attenuation_synthesis(self, Ns, f_c=0.1, Ts=1):
        return self.instance.scintillation_attenuation_synthesis(
                Ns, f_c=f_c, Ts=Ts)


class _ITU1853_2():

    def __init__(self):
        self.__version__ = 2
        self.year = 2019
        self.month = 8
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en'

    
    def yulewalk(self, na, ff, aa):
        """
        YULEWALK Recursive filter design using a least-squares method.
        `[b,a] = yulewalk(na,ff,aa)` finds the `na-th` order recursive filter
        coefficients `b` and `a` such that the filter::

                                -1              -M
                    b[0] + b[1]z  + ... + b[M] z
            Y(z) = -------------------------------- X(z)
                                -1              -N
                    a[0] + a[1]z  + ... + a[N] z

        matches the magnitude frequency response given by vectors `ff` and `aa`.
        Vectors `ff` and `aa` specify the frequency and magnitude breakpoints for
        the filter such that `plot(ff,aa)` would show a plot of the desired
        frequency response.
        
        Parameters
        -----------
        - na : integer scalar.
                order of the recursive filter
        - ff : 1-D array
                frequencies sampling which must be between 0.0 and 1.0,
                with 1.0 corresponding to half the sample rate. They must be in
                increasing order and start with 0.0 and end with 1.0.
        
        Returns
        -----------
        - b : 1-D array
                numerator coefficients of the recursive filter
        - a : 1-D array
                denumerator coefficients of the recursive filter
        
        References
        ------------
        [1] Friedlander, B., and Boaz Porat. "The Modified Yule-Walker Method of 
        ARMA Spectral Estimation." IEEE® Transactions on Aerospace Electronic 
        Systems. Vol. AES-20, Number 2, 1984, pp. 158–173.

        [2] Matlab R2016a `yulewalk` function
        """
        npt = 512
        lap = np.fix(npt/25)

        npt = npt + 1
        Ht = np.zeros(npt)

        nint = np.size(ff) - 1
        df = np.diff(ff)

        nb = 1
        Ht[0] = aa[0]
        for ii in np.arange(nint):

            if df[ii] == 0:
                nb = int(nb - lap/2)
                ne = int(nb + lap)
            else:
                ne = int(np.fix(ff[ii+1] * npt))
            
            jj = np.arange(nb, ne + 1)
            if ne == nb:
                inc = 0
            else:
                inc = (jj - nb) / (ne - nb); 
            
            Ht[nb -1 : ne] = inc * aa[ii + 1] + (1 - inc) * aa[ii]
            nb = int(ne + 1)
        
        Ht = np.append(Ht, Ht[-2:0:-1])
        n = np.size(Ht)
        n2 = int(np.fix((n+1) / 2))
        nb = na
        nr = 4 * na
        nt = np.arange(nr)

        R = np.real( np.fft.ifft(Ht * Ht) )
        R = R[:nr] * (0.54 + 0.46 * np.cos(np.pi * nt/(nr-1) ))

        Rwindow = np.append(0.5, np.ones(n2 - 1))
        Rwindow = np.append( Rwindow, np.zeros(n - n2) )

        A = self.polystab( self.denf(R, na) )

        R = R[:nr]
        R[0] = R[0]/2
        Qh = self.numf(R, A, na)

        _, Ss = 2* np.real(signal.freqz(Qh, A, n , whole=True))
        var1 = np.log( Ss.astype('complex') )
        var2 = np.fft.ifft(var1)
        hh = np.fft.ifft( np.exp( np.fft.fft(Rwindow * var2) ) )
        B = np.real( self.numf(hh[:nr], A, nb ))

        return B, A

    def polystab(self, a):
        """
        Polynomial stabilization.
        polystab(a), where a is a vector of polynomial coefficients,
        stabilizes the polynomial with respect to the unit circle;
        roots whose magnitudes are greater than one are reflected
        inside the unit circle.

        Parameters
        ----------
        - a : 1-D numpy.array.
                vector of polynomial coefficients
        
        Returns
        -------
        - b : 1-D numpy.array

        References
        ----------
        [1] Matlab R2016a `polystab` function 
        """
        if np.size(a) <= 1:
            return a
        ## Actual process
        v = np.roots(a)
        ii = np.where(v != 0)[0]
        vs = 0.5 * (np.sign( np.abs( v[ ii ] ) - 1 ) + 1)
        v[ii] = (1 - vs) * v[ii] + vs / np.conj(v[ii])
        ind = np.where(a != 0)[0]
        b =  a[ ind[0] ] * np.poly(v)
        ## Security
        if not (np.any(np.imag(a))):
            b = np.real(b)
        return b

    def numf(self, h, a, nb):
        """
        Find numerator B given impulse-response h of B/A and denominator a

        Parameters
        ----------
        - h : real 1D array
                impulse-response.   
        - a : real 1D array
                denominator of the estimated filter.
        - nb : integer scalar
                numerator order.
            
        Returns
        -------
        - b : real 1D array.
                numerator of the estimated filter.
            
        References
        ----------
        [1] Matlab R2016a `yulewalk` function
        """
        nh = np.max(np.shape(h))
        impr = signal.lfilter([1.0], a, np.append(1.0, np.zeros(nh - 1)))
        b = np.matmul(h, linalg.pinv( linalg.toeplitz(impr, np.append(1.0, np.zeros(nb))).T ) )
        return b

    def denf(self, R, na):
        """
        Compute filter denominator from covariances.
        A = denf(R,na) computes order na denominator A from covariances 
        R(0)...R(nr) using the Modified Yule-Walker method.  
        This function is used by yulewalk.

        Parameters
        ----------
        - R : real 1D array.
                Covariances.
            
        - na : integer scalar.
                Order of the denominator.
                
        Returns
        -------
        - A : real 1d array. 
                Denominator of the estimated filter.
        
        References
        ----------
        [1] Matlab R2016a `yulewalk` function
        """
        nr = np.max(np.shape(R))
        Rm = linalg.toeplitz(R[na:nr - 1], R[na:0:-1])
        Rhs = -R[na + 1:nr + 1]
        A = np.matmul(Rhs, linalg.pinv(Rm.T))
        return np.append([1], A)


    def scintillation_attenuation_synthesis(self, Ns, f_c=0.1, Ts=1):
        """
        calculate the scintillation attenuation filter
        """
        t_disc = int(2e5)  # discarding samples
        # white gaussian noise
        n = np.random.normal(0, 1, int(Ns * Ts + t_disc))

        # create frequency in range [0, fs/2]
        freqs = np.linspace(0, 1 / (2*Ts), num=201)

        # create the Low pass filter with f^-8/3 roll-off
        # and cut-off frequency fc=0.1 Hz
        # Hf = 1 / (1 + np.abs(freqs/fc))**(8/3)   # normal low-pass filter
        with np.errstate(divide='ignore'):
            Hf = np.where(freqs <= f_c, 1, (freqs/f_c)**(-8/3))  # ideal low-pass filter

        # normalize the frequency [0,1] to be used in yulewalk function
        fr = freqs / np.max(freqs)  

        b, a = self.yulewalk(12, fr, Hf)

        zi = signal.lfilter_zi(b, a)
        sci_t, _ = signal.lfilter(b, a, n, zi=zi*n[0])

        return sci_t[np.ceil(t_disc/Ts).astype(int):].flatten()

class _ITU1853_1():

    def __init__(self):
        self.__version__ = 1
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-1-201202-I/en'

    def scintillation_attenuation_synthesis(self, *args, **kwargs):
        """
        For Earth-space paths, the time series synthesis method is valid for
        frequencies between 4 GHz and 55 GHz and elevation angles between
        5 deg and 90 deg.

        The scintillation model in version ITU-R P.1853-1 is the same 
        as version ITU-R P.1853-2
        """
        return _ITU1853_2().scintillation_attenuation_synthesis(*args, **kwargs)


class _ITU1853_0():

    def __init__(self):
        self.__version__ = 0
        self.year = 2009
        self.month = 10
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-0-200910-I/en'

    def scintillation_attenuation_synthesis(self, *args, **kwargs):
        return _ITU1853_2().scintillation_attenuation_synthesis(*args, **kwargs)


__model = __ITU1853()


def change_version(new_version):
    """
    Change the version of the ITU-R P.1853 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.1853-1 (02/12)
            * P.1853-2 (08/2019) (Current version)
    """
    global __model
    __model = __ITU1853(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.1853 recommendation currently being used.
    """
    global __model
    return __model.__version__


def scintillation_attenuation_synthesis(Ns, f_c=0.1, Ts=1):
    """
    A method to generate a synthetic time series of scintillation attenuation
    values.


    Parameters
    ----------
    - Ns : int
            Number of samples
    - f_c : float
            Cut-off frequency for the low pass filter
    - Ts : int
            Time step between consecutive samples (seconds)

    Returns
    -------
    - sci_att: numpy.ndarray
            Synthesized scintilation attenuation time series (dB)


    References
    ----------
    [1] Characteristics of precipitation for propagation modelling
    https://www.itu.int/rec/R-REC-P.1853/en
    """
    global __model

    val = __model.scintillation_attenuation_synthesis(Ns, f_c=f_c, Ts=Ts)
    return val * u.dB

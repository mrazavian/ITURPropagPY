# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u
import scipy.stats as stats
from scipy.signal import lfilter

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


    def scintillation_attenuation_synthesis(self, Ns, f_c=0.1, Ts=1):
        """
        calculate the scintillation attenuation filter
        """
        t_disc = int(2e5)  # discarding samples
        # white gaussian noise and fourier transform
        n = np.random.normal(0, 1, 2 * int(Ns + t_disc) + 1)
        nf = np.fft.fft(n, norm='ortho')

        # create frequency in range [-Ts/2, Ts/2] 
        # with zero frequency at centered
        freqs = np.fft.fftfreq(n.shape[-1], 1 / Ts)
        freqs = np.fft.fftshift(freqs)

        # create the Low pass filter with f^-8/3 roll-off
        # and cut-off frequency fc=0.1 Hz
        # Hf = 1 / (1 + np.abs(freqs/fc))**(8/3)   # normal low-pass filter
        with np.errstate(divide='ignore'):
            Hf = np.where(np.abs(freqs) <= f_c, 1, np.abs(freqs/f_c)**(-8/3))  # ideal low-pass filter

        sci_f = nf * Hf
        sci_t = np.fft.ifft(sci_f, norm='ortho').real
        return sci_t[t_disc:int(Ns + t_disc)].flatten()

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

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u
import scipy.stats as stats
from scipy.signal import lfilter

from iturpropag.models.iturp836.total_water_vapour_content import total_water_vapour_content
from iturpropag.models.iturp1853.integrated_water_vapour_coefficients import integrated_water_vapour_coefficients
from iturpropag.utils import prepare_quantity


class __ITU1853():
    """Tropospheric attenuation time series synthesis

    Available versions include:
    * P.1853-0 (10/09) (Superseded)
    * P.1853-1 (02/12) (Current version)
    * P.1853-2 (08/2019) (N/A)
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.1853 recommendation.

    def __init__(self, version=1):
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

    def set_seed(self, seed):
        np.random.seed(seed)

    @property
    def __version__(self):
        return self.instance.__version__

    def integrated_water_vapour_synthesis(self, lat, lon, Ns, Ts=1, n=None):
        return self.instance.integrated_water_vapour_synthesis(
                lat, lon, Ns, Ts=Ts, n=n)

class _ITU1853_2():

    def __init__(self):
        self.__version__ = 2
        self.year = 2019
        self.month = 8
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en'

    def integrated_water_vapour_synthesis(self, *args, **kwargs):
        raise NotImplementedError(
            "Recommendation ITU-R P.1853-2 does not specify a method to compute "
            "time series for the water vapour content.")

class _ITU1853_1():

    def __init__(self):
        self.__version__ = 1
        self.year = 2012
        self.month = 2
        self.link = 'https://www.p.int/rec/R-REC-P.1853-1-201202-I/en'

    def integrated_water_vapour_synthesis(self, lat, lon, Ns, Ts=1, n=None):
        # A Estimation of κ and λ
        kappa, lambd = integrated_water_vapour_coefficients(lat, lon, None)

        # B Low-pass filter parameter
        beta_V = 3.24e-6

        # Step C: Time series synthesis
        # Step C1: Synthesize a white Gaussian noise time series
        if n is None:
            n = np.random.normal(0, 1, (Ns * Ts + 5e6))[::Ts]
            discard_samples = True
        else:
            discard_samples = False

        # Step C3: Filter the noise time series, with two recursive low-pass
        # filters
        rho = np.exp(-beta_V * Ts)
        G_v = lfilter([np.sqrt(1 - rho**2)], [1, -rho], n, 0)
        # Step C4: Compute Compute V(kTs),
        V = lambd * (- np.log10(stats.norm.sf(G_v)))**(1 / kappa)
        # Step C5: Discard the first 5 000 000 samples from the synthesized
        if discard_samples:
            V = V[np.ceil(5e6/Ts).astype(int):]

        return V.flatten()


class _ITU1853_0():

    def __init__(self):
        self.__version__ = 0
        self.year = 2009
        self.month = 10
        self.link = 'https://www.p.int/rec/R-REC-P.1853-0-200910-I/en'

    def integrated_water_vapour_synthesis(self, *args, **kwargs):
        raise NotImplementedError(
            "Recommendation ITU-R P.1853-0 does not specify a method to compute "
            "time series for the water vapour content.")


__model = __ITU1853()


def change_version(new_version):
    """
    Change the version of the ITU-R P.1853 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.1853-1 (02/12) (Current version)
    """
    global __model
    __model = __ITU1853(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.1853 recommendation currently being used.
    """
    global __model
    return __model.__version__


def integrated_water_vapour_synthesis(lat, lon, Ns, Ts=1, n=None):
    """ The time series synthesis method generates a time series that
    reproduces the spectral characteristics and the distribution of water
    vapour content.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - Ns : int
            Number of samples
    - Ts : int
            Time step between consecutive samples (seconds)
    - n : list, np.array, optional
            Additive White Gaussian Noise used as input for the

    Returns
    -------
    - V: numpy.ndarray
            Synthesized water vapour content time series (kg/m**2)


    References
    ----------
    [1] Characteristics of precipitation for propagation modelling
    https://www.itu.int/rec/R-REC-P.1853-1-201202-S/en
    """
    global __model

    lon = np.mod(lon, 360)
    val = __model.integrated_water_vapour_synthesis(lat, lon, Ns, Ts=Ts, n=n)
    return val * u.kg / u.m**2

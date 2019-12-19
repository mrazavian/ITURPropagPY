# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u
import scipy.stats as stats
from scipy.signal import lfilter

from iturpropag.models.iturp840.lognormal_approximation_coefficient import lognormal_approximation_coefficient
from iturpropag.utils import prepare_quantity


class __ITU1853():
    """Tropospheric attenuation time series synthesis

    Available versions include:
    * P.1853-0 (10/09) (Superseded)
    * P.1853-1 (02/12) (Current version)
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


    @property
    def __version__(self):
        return self.instance.__version__

    def cloud_liquid_water_synthesis(self, lat, lon, Ns, Ts=1, n=None):
        return self.instance.cloud_liquid_water_synthesis(
                lat, lon, Ns, Ts=Ts, n=n)


class _ITU1853_2():

    def __init__(self):
        self.__version__ = 2
        self.year = 2019
        self.month = 8
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en'

    def cloud_liquid_water_synthesis(self, *args, **kwargs):
        raise NotImplementedError(
            "Recommendation ITU-R P.1853-2 does not specify a method to compute "
            "time series for the cloud liquid water content.")


class _ITU1853_1():

    def __init__(self):
        self.__version__ = 1
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-1-201202-I/en'

    def cloud_liquid_water_synthesis(self, lat, lon, Ns, Ts=1, n=None):
        """
        """

        # Step A: Estimation of m, sigma and Pcwl
        m, sigma, Pcwl = lognormal_approximation_coefficient(lat, lon)
        m = m.value
        sigma = sigma.value
        Pcwl = Pcwl.value / 100

        # Step B: Low pass filter parameters
        beta_1 = 7.17e-4
        beta_2 = 2.01e-5
        gamma_1 = 0.349
        gamma_2 = 0.830

        # Step C: Truncation threshold
        alpha = stats.norm.ppf(1 - Pcwl)

        # Step D: Time series synthesis
        # Step D1: Synthesize a white Gaussian noise time series
        if n is None:
            n = np.random.normal(0, 1, int(Ns * Ts + 5e5))[::Ts]
            discard_samples = True
        else:
            discard_samples = False

        # Step D3: Filter the noise time series, with two recursive low-pass
        # filters
        rho_1 = np.exp(-beta_1 * Ts)
        X_1 = lfilter([np.sqrt(1 - rho_1**2)], [1, -rho_1], n, 0)
        rho_2 = np.exp(-beta_2 * Ts)
        X_2 = lfilter([np.sqrt(1 - rho_2**2)], [1, -rho_2], n, 0)
        # Step D4: Compute Gc(kTs),
        G_c = gamma_1 * X_1 + gamma_2 * X_2
        # Step D5: Compute L(kTs) (dB)
        L = np.where(G_c > alpha, np.exp(m + sigma * stats.norm.ppf(
            1 - 1 / Pcwl * stats.norm.sf(G_c))), 0)

        # D6: Discard the first 500 000 samples from the synthesized
        if discard_samples:
            L = L[np.ceil(5e5/Ts).astype(int):]

        return L.flatten()


class _ITU1853_0():

    def __init__(self):
        self.__version__ = 0
        self.year = 2009
        self.month = 10
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-0-200910-I/en'

    def cloud_liquid_water_synthesis(self, *args, **kwargs):
        raise NotImplementedError(
            "Recommendation ITU-R P.1853-0 does not specify a method to compute "
            "time series for the cloud liquid water content.")


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


def cloud_liquid_water_synthesis(lat, lon, Ns, Ts=1, n=None):
    """ The time series synthesis method generates a time series that
    reproduces the spectral characteristics, rate of change and duration
    statistics of cloud liquid content events.

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
    - L: numpy.ndarray
            Synthesized cloud liquid water time series (kg / m**2)


    References
    ----------
    [1] Characteristics of precipitation for propagation modelling
    https://www.itu.int/rec/R-REC-P.1853/en
    """
    global __model

    lon = np.mod(lon, 360)
    val = __model.cloud_liquid_water_synthesis(lat, lon, Ns, Ts, n)
    return val * u.kg / u.m**2

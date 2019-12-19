# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u
import scipy.stats as stats
from scipy.signal import lfilter

from iturpropag.models.iturp840.lognormal_approximation_coefficient import lognormal_approximation_coefficient
from iturpropag.models.iturp840.specific_attenuation_coefficients import specific_attenuation_coefficients
from iturpropag.models.iturp1853.cloud_liquid_water_synthesis import cloud_liquid_water_synthesis
from iturpropag.utils import prepare_quantity, prepare_input_array,\
                    compute_distance_earth_to_earth


class __ITU1853():
    """Tropospheric attenuation time series synthesis

    Available versions include:
    * P.1853-0 (10/09) (Superseded)
    * P.1853-1 (02/12) 
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

    def cloud_attenuation_synthesis(self, lat, lon, f, el, Ns, Ts=1, n=None,\
                            rain_contribution=False):
        return self.instance.cloud_attenuation_synthesis(
                            lat, lon, f, el, Ns, Ts=Ts, n=n,\
                            rain_contribution=rain_contribution)


class _ITU1853_2():

    def __init__(self):
        self.__version__ = 2
        self.year = 2019
        self.month = 8
        self.link = 'https://www.itu.int/rec/R-REC-P.1853/en'


    def single_site_cloud_attenuation_synthesis(self, lat, lon, f,\
                                el, n, beta_1, beta_2, gamma_1, gamma_2, Ts=1):
        # Step A: 
        # SS_CL_1~4: Estimation of m, sigma and Pcwl
        m, sigma, Pcwl = lognormal_approximation_coefficient(lat, lon)
        Kl = specific_attenuation_coefficients(f, T=0)

        m = m.value + np.log(Kl / np.sin(np.deg2rad(el)))
        sigma = sigma.value
        Pc = Pcwl.value

        # Step C
        # SS_CL_6: Truncation threshold
        alpha = stats.norm.ppf(1 - Pc/100)

        # Step D: 
        # SS_CL_8~9: Filter the noise time series, with two recursive low-pass
        # filters
        rho_1 = np.exp(-beta_1 * Ts)
        rho_2 = np.exp(-beta_2 * Ts)

        X_1 = lfilter([np.sqrt(1 - rho_1**2)], [1, -rho_1], n, 0)
        X_2 = lfilter([np.sqrt(1 - rho_2**2)], [1, -rho_2], n, 0)

        # SS_CL_10: Compute Gc(kTs),
        G_c = gamma_1 * X_1 + gamma_2 * X_2
        # SS_CL_11: Compute Ac(kTs) (dB)
        arg1 = 100 * stats.norm.sf(G_c) / Pc
        arg2 = stats.norm.ppf(1 - arg1)

        Y_cloud = np.exp(m + sigma * arg2)
        A_cloud = np.where(G_c > alpha, Y_cloud, 0)

        return A_cloud.flatten()
    
    def cloud_attenuation_synthesis(self, lat, lon, f, el, Ns, Ts=1, n=None,\
                                    rain_contribution=False):
        # Step B:
        # MS_CL_2: Low-pass filter parameters
        if rain_contribution: # page 26
            beta_1 = 9.0186e-4
            beta_2 = 5.099e-5
            gamma_1 = 0.3746
            gamma_2 = 0.7738
        else: # page 12
            beta_1 = 5.7643e-4
            beta_2 = 1.7663e-5
            gamma_1 = 0.4394
            gamma_2 = 0.7613

        # Step D: 
        # MS_CL_4: Synthesize a white Gaussian noise time series page=12
        if n is None:
            n = np.random.normal(0, 1, \
                (np.size(lat), int(Ns * Ts + 5e6)))[::Ts]
            discard_samples = True
        else:
            discard_samples = False

        # MS_CL_5: calculate the matrix Rn=[rnij]
        rho_1 = np.exp(-beta_1 * Ts)
        rho_2 = np.exp(-beta_2 * Ts)

        D = np.zeros( (np.size(lat), np.size(lat)) )
        for ii,_ in enumerate(lat):
            D[ii,:] = compute_distance_earth_to_earth(lat[ii], lon[ii], lat, lon)

        if rain_contribution: # page 26
            rG = 0.59 * np.exp(-D/31) + 0.41 * np.exp(-D/800)  # eq. 40
        else:  # page 13
            rG = 0.55 * np.exp(-D/24) + 0.45 * np.exp(-D/700) # eq. 21

        Rn = rG / (gamma_1**2 + gamma_2**2 +\
            2 *gamma_1 *gamma_2 *np.sqrt(1-rho_1**2) *np.sqrt(1-rho_2**2) /(1-rho_1*rho_2))

        # MS_CL_6~7: calculate cholesky factorization
        CR = np.linalg.cholesky(Rn)

        n =  np.matmul(CR, n)
        # MS_CL_8: single-site cloud attenuation time-series
        Acloud = np.zeros_like(n)
        for ii,_ in enumerate(lat):
            Acloud[ii,:] = self.single_site_cloud_attenuation_synthesis(lat[ii], lon[ii],\
                                                f[ii], el[ii], n[ii,:], beta_1, beta_2, 
                                                gamma_1, gamma_2, Ts=Ts)

        # SS_CL_12: Discard the first 5 000 000 samples
        # from the synthesized time-series (page=10)
        if discard_samples:
            Acloud = Acloud[:, np.ceil(5e6/Ts).astype(int):]

        
        return Acloud


class _ITU1853_1():

    def __init__(self):
        self.__version__ = 1
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-1-201202-I/en'

    
    def cloud_attenuation_synthesis(self, lat, lon, f, el, Ns, Ts=1, n=None,
                                    **kwargs):
        """
        """

        L = cloud_liquid_water_synthesis(lat, lon, Ns, Ts=Ts, n=n).value

        Ac = L * \
            specific_attenuation_coefficients(f, T=0) / np.sin(np.deg2rad(el))
        
        return Ac.flatten()


class _ITU1853_0():

    def __init__(self):
        self.__version__ = 0
        self.year = 2009
        self.month = 10
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-0-200910-I/en'

    def cloud_attenuation_synthesis(self, *args, **kwargs):
        raise NotImplementedError(
            "Recommendation ITU-R P.1853 does not specify a method to compute "
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
            * P.1853-2 (08/2019) (current)
            * P.1853-1 (02/2012)
    """
    global __model
    __model = __ITU1853(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.1853 recommendation currently being used.
    """
    global __model
    return __model.__version__


def cloud_attenuation_synthesis(lat, lon, f, el, Ns, Ts=1, n=None,
                                rain_contribution=False):
    """ The time series synthesis method generates a time series that
    reproduces the spectral characteristics, rate of change and duration
    statistics of cloud attenuation events.

    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - f : number or Quantity
            Frequency (GHz)
    - el : number, sequence, or numpy.ndarray
            Elevation angle (degrees)
    - Ns : int
            Number of samples
    - Ts : int
            Time step between consecutive samples (seconds)
    - n : list, np.array, optional
            Additive White Gaussian Noise used as input for the
    - rain_contribution: bool, optional
            Determines whether rain effect is considered in cloud
            attenuation or not. default value is False. when the
            value is True the effect of rain is considered in cloud attenuation.


    Returns
    -------
    - cloud_att: numpy.ndarray
            Synthesized cloud attenuation time series (dB)


    References
    ----------
    [1] P.1853 : Time series synthesis of tropospheric impairments
    https://www.itu.int/rec/R-REC-P.1853/en
    """
    global __model

    # prepare the input array
    lon = np.mod(lon, 360)
    lat = prepare_input_array(lat).flatten()
    lon = prepare_input_array(lon).flatten()
    f = prepare_input_array(f).flatten()
    el = prepare_input_array(el).flatten()

    # prepare quantity
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(el, u.deg, 'Elevation angle')
    Ts = prepare_quantity(Ts, u.second, 'Time step between samples')
    # calculate the output
    val = __model.cloud_attenuation_synthesis(lat, lon, f, el, Ns, Ts, n,\
                                            rain_contribution)
    return val * u.dB

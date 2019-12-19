# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u
import scipy.stats as stats
from scipy.signal import lfilter, lfilter_zi

from iturpropag.models.iturp618.rain_attenuation import rain_attenuation
from iturpropag.models.iturp837.rainfall_probability import rainfall_probability
from iturpropag.utils import prepare_quantity, compute_distance_earth_to_earth,\
                             prepare_input_array


class __ITU1853():
    """Tropospheric attenuation time series synthesis

    Available versions include:
    * P.1853-0 (10/09)
    * P.1853-1 (02/2012)
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

    def set_seed(self, seed):
        np.random.seed(seed)

    @property
    def __version__(self):
        return self.instance.__version__

    def rain_attenuation_synthesis(self, lat, lon, f, el, tau, Ns,
                                   hs=None, Ts=1, n=None):
        return self.instance.rain_attenuation_synthesis(
                lat, lon, f, el, tau, Ns, hs=hs, Ts=Ts, n=n)


class _ITU1853_2():

    def __init__(self):
        self.__version__ = 2
        self.year = 2019
        self.month = 8
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en'

        # Step B:
        # SS_RA_5: Set the low-pass filter parameter page=15
        self.beta1 = 9.0186e-4
        self.beta2 = 5.0990e-5
        self.gamma1 = 0.3746
        self.gamma2 = 0.7738

    def single_site_rain_attenuation_synthesis(self, lat, lon, f, el, tau,
                                    n, hs=None, Ts=1):
        """
        For Earth-space paths, the time series synthesis method is valid for
        frequencies between 4 GHz and 55 GHz and elevation angles between
        5 deg and 90 deg.
        """
        # Step A: 
        # SS_RA_1: Determine Prain (% of time), the probability of rain on the
        # path. Prain can be well approximated as P0(lat, lon)
        P_rain = rainfall_probability(lat, lon).to(u.pct).value
 
        # SS_RA_2: Construct the set of pairs [Pi, Ai] where Pi (% of time) is
        # the probability the attenuation Ai (dB) is exceeded where Pi < P_R
        p_i = np.array([0.01, 0.02, 0.03, 0.05,
                        0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10])
        Pi = np.array([p for p in p_i if p < P_rain], dtype=np.float)
        Ai = np.array([0 for p in p_i if p < P_rain], dtype=np.float)

        for i, p in enumerate(Pi):
            Ai[i] = rain_attenuation(lat, lon, f, el, p, tau, hs=hs).value

        # SS_RA_3: Transform the set of pairs [Pi, Ai] to [Q^{-1}(Pi/P_k),
        # ln(Ai)]
        Q = stats.norm.ppf(1 - (Pi / P_rain))
        lnA = np.log(Ai)

        # SS_RA_4: Determine the variables sigma_lna, m_lna by performing a
        # least-squares fit to lnAi = sigma_lna Q^{-1}(Pi/P_R) + m_lna
        m, sigma = np.linalg.lstsq(np.vstack([np.ones(len(Q)), Q]).T,
                                   lnA, rcond=None)[0]

        # Step C:
        # SS_RA_6: Truncation Threshold
        alpha = stats.norm.ppf(1 - P_rain/100)

        # SS_RA_8~9 : Filter the noise time series with a recursive low-pass
        # filter
        rho1 = np.exp(-self.beta1 * Ts)
        rho2 = np.exp(-self.beta2 * Ts)

        X1, _ = lfilter([np.sqrt(1 - rho1**2)], [1, -rho1], n, zi=np.array([0]))
        X2, _ = lfilter([np.sqrt(1 - rho2**2)], [1, -rho2], n, zi=np.array([0]))

        # SS_RA_10
        G = self.gamma1 * X1 + self.gamma2 * X2

        # SS_CL_11: Compute A_rain
        arg1 = 100 * stats.norm.sf(G) / P_rain
        arg2 = stats.norm.ppf(1 - arg1)

        Y_rain = np.exp(m + sigma * arg2)
        A_rain = np.where(G > alpha, Y_rain, 0)

        return A_rain.flatten()
    
    def rain_attenuation_synthesis(self, lat, lon, f, el, tau, Ns,
                                    hs=None, Ts=1, n=None):
        # Step D: 
        # MS_RA_4: Synthesize a white Gaussian noise time series page=18
        if n is None:
            n = np.random.normal(0, 1, \
                (np.size(lat), int(Ns * Ts + 5e6)))[::Ts]
            discard_samples = True
        else:
            discard_samples = False

        # MS_RA_5: calculate the matrix Rn=[rnij]
        rho1 = np.exp(-self.beta1 * Ts)
        rho2 = np.exp(-self.beta2 * Ts)

        D = np.zeros( (np.size(lat), np.size(lat)) )
        for ii,_ in enumerate(lat):
            D[ii,:] = compute_distance_earth_to_earth(lat[ii], lon[ii], lat, lon)

        rG = 0.59 * np.exp(-D/31) + 0.41 * np.exp(-D/800)

        Rn = rG / (self.gamma1**2 + self.gamma2**2 +\
            2 *self.gamma1 *self.gamma2 *np.sqrt(1-rho1**2) *np.sqrt(1-rho2**2) /(1-rho1*rho2))

        # MS_RA_6: calculate cholesky factorization
        CR = np.linalg.cholesky(Rn)

        n =  np.matmul(CR, n)

        Arain = np.zeros_like(n)
        for ii,_ in enumerate(lat):
            Arain[ii,:] = self.single_site_rain_attenuation_synthesis(lat[ii], lon[ii],\
                                                f[ii], el[ii], tau[ii], n[ii,:], hs=hs, Ts=Ts)

        # D6: Discard the first 5 000 000 samples from the synthesized
        if discard_samples:
            Arain = Arain[:, np.ceil(5e6/Ts).astype(int):]
                                               
        return Arain


class _ITU1853_1():

    def __init__(self):
        self.__version__ = 1
        self.year = 2012
        self.month = 2
        self.link = 'https://www.p.int/rec/R-REC-P.1853-1-201202-I/en'

    def rain_attenuation_synthesis(self, lat, lon, f, el, tau, Ns,
                                   hs=None, Ts=1, n=None):
        """
        For Earth-space paths, the time series synthesis method is valid for
        frequencies between 4 GHz and 55 GHz and elevation angles between
        5 deg and 90 deg.
        """
        # Step A1: Determine Prain (% of time), the probability of rain on the
        # path. Prain can be well approximated as P0(lat, lon)
        P_rain = rainfall_probability(lat, lon).\
            to(u.dimensionless_unscaled).value

        # Step A2: Construct the set of pairs [Pi, Ai] where Pi (% of time) is
        # the probability the attenuation Ai (dB) is exceeded where Pi < P_K
        p_i = np.array([0.01, 0.02, 0.03, 0.05,
                        0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10])
        Pi = np.array([p for p in p_i if p < P_rain * 100], dtype=np.float)
        Ai = np.array([0 for p in p_i if p < P_rain * 100], dtype=np.float)

        for i, p in enumerate(Pi):
            Ai[i] = rain_attenuation(lat, lon, f, el, p, tau, hs=hs).value

        # Step A3: Transform the set of pairs [Pi, Ai] to [Q^{-1}(Pi/P_k),
        # ln(Ai)]
        Q = stats.norm.ppf(1 - (Pi / 100))
        lnA = np.log(Ai)

        # Step A4: Determine the variables sigma_lna, m_lna by performing a
        # least-squares fit to lnAi = sigma_lna Q^{-1}(Pi/P_k) + m_lna
        m, sigma = np.linalg.lstsq(np.vstack([np.ones(len(Q)), Q]).T,
                                   lnA, rcond=None)[0]
        
        # Step B: Set the low-pass filter parameter
        beta = 2e-4
        # Step C: compute the attenuation offset
        A_offset = np.exp(m + sigma * stats.norm.ppf(1-P_rain))

        # Step D: Time series synthesis
        # D1: Synthesize a white Gaussian noise time series
        if n is None:
            n = np.random.normal(0, 1, int(Ns * Ts + 2e5))[::Ts]
            discard_samples = True
        else:
            discard_samples = False

        # D2, D3 : Filter the noise time series with a recursive low-pass
        # filter
        rho = np.exp(-beta * Ts)
        X = lfilter([np.sqrt(1 - rho**2)], [1, -rho], n, 0)
        # D4: Compute Y_rain
        Y_rain = np.exp(m + sigma * X)
        # D5: Compute Arain
        A_rain = np.maximum(Y_rain - A_offset, 0)

        # D6: Discard the first 200 000 samples from the synthesized
        if discard_samples:
            A_rain = A_rain[np.ceil(200000/Ts).astype(int):]

        return A_rain.flatten()


class _ITU1853_0():

    def __init__(self):
        self.__version__ = 0
        self.year = 2009
        self.month = 10
        self.link = 'https://www.p.int/rec/R-REC-P.1853-0-200910-I/en'

    def rain_attenuation_synthesis(self, *args, **kwargs):
        return _ITU1853_1.rain_attenuation_synthesis(*args, **kwargs)


__model = __ITU1853()


def change_version(new_version):
    """
    Change the version of the ITU-R P.1853 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.1853-2 (08/2019) (Current version)
            * P.1853-1 (02/2012)
            * P.1853-0
    """
    global __model
    __model = __ITU1853(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.1853 recommendation currently being used.
    """
    global __model
    return __model.__version__


def rain_attenuation_synthesis(lat, lon, f, el, tau, Ns, hs=None, Ts=1, n=None):
    """
    A method to generate a synthetic time series of rain attenuation values.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - f : number or Quantity
            Frequency (GHz)
    - el : sequence, or number
            Elevation angle (degrees)
    - hs : number, sequence, or numpy.ndarray, optional
            Heigh above mean sea level of the earth station (km). If local data for
            the earth station height above mean sea level is not available, an
            estimate is obtained from the maps of topographic altitude
            given in Recommendation ITU-R P.1511.
    - Ns : int
            Number of samples
    - Ts : int
            Time step between consecutive samples (seconds)
    - tau : number, optional
            Polarization tilt angle relative to the horizontal (degrees)
            (tau = 45 deg for circular polarization). Default value is 45
    - n : list, np.array, optional
            Additive White Gaussian Noise used as input for the


    Returns
    -------
    - rain_att: numpy.ndarray
            Synthesized rain attenuation time series (dB)


    References
    ----------
    [1] Characteristics of precipitation for propagation modelling
    https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en
    """
    global __model
    
    # prepare the input array
    lon = np.mod(lon, 360)
    lat = prepare_input_array(lat).flatten()
    lon = prepare_input_array(lon).flatten()
    f = prepare_input_array(f).flatten()
    el = prepare_input_array(el).flatten()
    tau = prepare_input_array(tau).flatten()

    # prepare quantity
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(el, u.deg, 'Elevation angle')
    hs = prepare_quantity(
        hs, u.km, 'Heigh above mean sea level of the earth station')
    Ts = prepare_quantity(Ts, u.second, 'Time step between samples')
    # calculate the output
    val = __model.rain_attenuation_synthesis(lat, lon, f, el, tau, Ns, hs=hs,
                                              Ts=Ts, n=n)
    return val * u.dB

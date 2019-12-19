# -*- coding: utf-8 -*-
# checked by <TP>: 2019-06-07
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as stats
import scipy.special
import scipy.integrate
from astropy import units as u
import warnings

from iturpropag.models.iturp618.rain_attenuation import rain_attenuation
from iturpropag.models.iturp837.rainfall_probability import rainfall_probability
from iturpropag.utils import prepare_input_array, prepare_output_array,\
    prepare_quantity, compute_distance_earth_to_earth, memory



class _ITU618():
    """
    Propagation data and prediction methods required for the design of
    Earth-space telecommunication systems.

    Available versions include:
       * P.618-13 (12/17) (Current version)
       * P.618-12 (07/15) (Superseded)

    Versions that need to be implemented
       * P.618-11
       * P.618-10
       * P.618-09
       * P.618-08
       * P.618-07
       * P.618-06
       * P.618-05
       * P.618-04
       * P.618-03
       * P.618-02
       * P.618-01

    Recommendation ITU-R P.618 provides methods to estimate the propagation
    loss on an Earth-space path, relative to the free-space loss. This value
    is the sum of different contributions as follows:
    * attenuation by atmospheric gases;
    * attenuation by rain, other precipitation and clouds;
    * focusing and defocusing;
    * decrease in antenna gain due to wave-front incoherence;
    * scintillation and multipath effects;
    * attenuation by sand and dust storms.
    Each of these contributions has its own characteristics as a function of
    frequency, geographic location and elevation angle. As a rule, at elevation
    angles above 10°, only gaseous attenuation, rain and cloud attenuation and
    possibly scintillation will be significant, depending on propagation
    conditions.
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.618 recommendation.

    def __init__(self, version=13):
        if version == 13:
            self.instance = _ITU618_13()
        elif version == 12:
            self.instance = _ITU618_12()
        else:
            err_msg = 'Version {0} is not implemented for the ITU-R P.618 model.'
            raise ValueError(err_msg.format(version))

    @property
    def __version__(self):
        return self.instance.__version__

    def fit_rain_attenuation_to_lognormal(self, lat, lon, f, el, tau, hs, P_k):
        fcn = np.vectorize(self.instance.fit_rain_attenuation_to_lognormal)
        return fcn(lat, lon, f, el, tau, hs, P_k)


class _ITU618_13():

    def __init__(self):
        self.__version__ = 13

    def fit_rain_attenuation_to_lognormal(self, lat, lon, f, el, tau, hs, P_k):
        
        # calculate the rain probability from ITU-R P.837
        if P_k is None:
            P_k = rainfall_probability(lat, lon).to(u.pct).value

        # Performs the log-normal fit of rain attenuation vs. probability of
        # occurrence for a particular path

        # Step 1: Construct the set of pairs [Pi, Ai] where Pi (% of time) is
        # the probability the attenuation Ai (dB) is exceeded where Pi < P_K
        p_i = np.array([0.01, 0.02, 0.03, 0.05,
                        0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10])
        Pi = np.array([p for p in p_i if p < P_k], dtype=np.float)
        Ai = np.array([0 for p in p_i if p < P_k], dtype=np.float)

        for i, p in enumerate(Pi):
            Ai[i] = rain_attenuation(lat, lon, f, el, p, tau, hs=hs).value

        # Step 2: Transform the set of pairs [Pi, Ai] to [Q^{-1}(Pi/P_k),
        # ln(Ai)]
        Q = stats.norm.ppf(1 - (Pi / P_k))                      # Eq. 34
        lnA = np.log(Ai)                                        # Eq. 33

        # Step 3: Determine the variables sigma_lna, m_lna by performing a
        # least-squares fit to lnAi = sigma_lna Q^{-1}(Pi/P_k) + m_lna
        m_lna, sigma_lna = np.linalg.lstsq(np.vstack([np.ones(len(Q)), Q]).T,
                                           lnA, rcond=None)[0]

        return sigma_lna, m_lna


class _ITU618_12():

    def __init__(self):
        self.__version__ = 12

    def fit_rain_attenuation_to_lognormal(self, *args, **kwargs):
        return _ITU618_13.fit_rain_attenuation_to_lognormal(*args, **kwargs)


__model = _ITU618()


def change_version(new_version):
    global __model
    __model = _ITU618(new_version)
    memory.clear()


def get_version():
    global __model
    return __model.__version__


def fit_rain_attenuation_to_lognormal(lat, lon, f, el, tau, hs=None, P_k=None):
    """
    Compute the log-normal fit of rain attenuation vs. probability of
    occurrence.


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
    - tau : number
            Polarization tilt angle relative to the horizontal (degrees)
            (tau = 45 deg for circular polarization).
    - hs : number, sequence, or numpy.ndarray, optional
            Heigh above mean sea level of the earth station (km). If local data for
            the earth station height above mean sea level is not available, an
            estimate is obtained from the maps of topographic altitude
            given in Recommendation ITU-R P.1511.
    - P_k : number, sequence, or numpy.ndarray, optional
            Rain probability on k-th path (percent [%]). if the local data for the
            earth station is not available, an estimate is obtained from the maps
            given in Recommendation ITU-R P.837.

    Returns
    -------
    - sigma_lna:
            Standar deviation of the lognormal distribution
    - m_lna:
            Mean of the lognormal distribution

    References
    ----------
    [1] Propagation data and prediction methods required for the design of
    Earth-space telecommunication systems:
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-12-201507-I!!PDF-E.pdf

    """
    global __model

    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(prepare_input_array(el), u.deg, 'Elevation angle')
    hs = prepare_quantity(
        hs, u.km, 'Heigh above mean sea level of the earth station')
    P_k = prepare_quantity(
        P_k, u.pct, 'Rain probability on k-th path')
    tau = prepare_quantity(tau, u.one, 'Polarization tilt angle')
    sigma_lna, m_lna = __model.fit_rain_attenuation_to_lognormal(
        lat, lon, f, el, tau, hs, P_k)
    return sigma_lna, m_lna

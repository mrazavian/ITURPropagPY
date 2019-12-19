# -*- coding: utf-8 -*-
# inspected: <TP> 2019-06-04
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as stats
import scipy.special
import scipy.integrate
from astropy import units as u
import warnings

from iturpropag.models.iturp837.rainfall_probability import rainfall_probability
from iturpropag.models.iturp839.rain_height import rain_height
from iturpropag.models.iturp1511.topographic_altitude import topographic_altitude
from iturpropag.utils import prepare_input_array, prepare_output_array,\
    prepare_quantity, compute_distance_earth_to_earth, memory



def __CDF_bivariate_normal__(alpha_x, alpha_y, rho):
    # This function calculates the complementary bivariate normal
    # distribution with limits alpha_x, alpha_y and correlation factor rho
    def CDF_bivariate_normal_fcn(x, y, rho):                # Eq. 13
        return np.exp(- (x**2 - 2 * rho * x * y + y**2) /
                      (2. * (1 - rho**2)))

    def CDF_bivariate_normal_int(alpha, y, rho):
        return scipy.integrate.quad(
            CDF_bivariate_normal_fcn, alpha, np.inf, args=(y, rho))[0]

    return 1 / (2 * np.pi * np.sqrt(1 - rho**2)) * scipy.integrate.quad(
        lambda y: CDF_bivariate_normal_int(alpha_x, y, rho),
        alpha_y,
        np.inf)[0]


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

    def rain_attenuation_probability(self, lat, lon, el, hs=None, P0=None):
        fcn = np.vectorize(self.instance.rain_attenuation_probability,
                           excluded=[0, 1, 2], otypes=[np.ndarray])
        return np.array(fcn(lat, lon, el, hs, P0).tolist())


class _ITU618_13():

    def __init__(self):
        self.__version__ = 13

    def rain_attenuation_probability(self, lat, lon, el, hs=None, P0=None):
        Re = 8500.   # Effective radius of the Earth (km)
        if hs is None:
            hs = topographic_altitude(lat, lon).to(u.km).value # from P.1511

        # Step 1: Estimate the probability of rain, at the earth station either
        # from Recommendation ITU-R P.837 or from local measured rainfall
        # rate data
        if P0 is None:
            P0 = rainfall_probability(lat, lon).\
                    to(u.dimensionless_unscaled).value # from P.837 Eq. 3
        else:   # from local rainfall rate
            P0 = P0 / 100. # % to absolute value
        
        # Step 2: Calculate the parameter alpha using the inverse of the
        # Q-function alpha = Q^{-1}(P0)  -> Q(alpha) = P0
        # Eq. 9
        alpha = stats.norm.ppf(1 - P0)                  # Eq. 9,10 <TP> TBC
        #alpha = np.sqrt(2) * scipy.special.erfinv(1-2*P0)
        # Step 3: Calculate the spatial correlation function, rho:
        hr = rain_height(lat, lon).value # from P.839

        # <TP> formulas simplified in Eq.1 and Eq. 2
        Ls = np.where(el >= 5,
                (hr - hs) / np.sin(np.deg2rad(el)),                 # Eq. 1
                2* (hr - hs) / ((np.sin(np.deg2rad(el))**2 +        # Eq. 2
                2* (hr - hs) / Re)**0.5 + np.sin(np.deg2rad(el))))

        d = Ls * np.cos(np.deg2rad(el))                             # Eq. 12
        # Eq. 11
        rho = 0.59 * np.exp(-abs(d) / 31.) + 0.41 * np.exp(-abs(d) / 800.)
        
        # Step 4: Calculate the complementary bivariate normal distribution
        biva_fcn = np.vectorize(__CDF_bivariate_normal__)
        c_B = biva_fcn(alpha, alpha, rho)                 # Eq. 13  <TP> TBC
        
        # Step 5: Calculate the probability of rain attenuation on the slant
        # path: conversion of probability to % done in prepare_output_array
        P = 1 - (1 - P0) * ((c_B - P0**2) / (P0 * (1 - P0)))**P0  # Eq. 14
        return P  # 0..1, conversion to % done at the end


class _ITU618_12():

    def __init__(self):
        self.__version__ = 12

    def rain_attenuation_probability(self, *args, **kwargs):
        return _ITU618_13.rain_attenuation_probability(*args, **kwargs)
        

__model = _ITU618()


def change_version(new_version):
    global __model
    __model = _ITU618(new_version)
    memory.clear()


def get_version():
    global __model
    return __model.__version__


def rain_attenuation_probability(lat, lon, el, hs=None, P0=None):
    """
    The following procedure computes the probability of non-zero rain
    attenuation on a given slant path Pr(Ar > 0).

    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - el : sequence, or number
            Elevation angle (degrees)
    - hs : number, sequence, or numpy.ndarray, optional
            Heigh above mean sea level of the earth station (km). If local data for
            the earth station height above mean sea level is not available, an
            estimate is obtained from the maps of topographic altitude
            given in Recommendation ITU-R P.1511.
    - P0 : number, sequence, or numpy.ndarray, optional
            Probability of rain at the earth station, (%)



    Returns
    -------
    - p: Quantity
            Probability of rain attenuation on the slant path (%)


    References
    ----------
    [1] Propagation data and prediction methods required for the design of
    Earth-space telecommunication systems:
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-12-201507-I!!PDF-E.pdf
    """
    global __model
    type_output = type(lat)

    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)

    lon = np.mod(lon, 360)
    el = prepare_quantity(prepare_input_array(el), u.deg, 'Elevation angle')
    hs = prepare_quantity(
        hs, u.km, 'Heigh above mean sea level of the earth station')
    P0 = prepare_quantity(P0, u.pct, 'Point rainfall rate')

    val = __model.rain_attenuation_probability(lat, lon, el, hs, P0)

    return prepare_output_array(val, type_output) * 100 * u.pct

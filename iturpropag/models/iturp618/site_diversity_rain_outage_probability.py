# -*- coding: utf-8 -*-
# inspected: <TP> 2019-06-07
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
from iturpropag.models.iturp618.fit_rain_attenuation_to_lognormal import fit_rain_attenuation_to_lognormal
from iturpropag.utils import prepare_input_array, prepare_output_array,\
    prepare_quantity, compute_distance_earth_to_earth, memory



def __CDF_bivariate_normal__(alpha_x, alpha_y, rho):
    # This function calculates the complementary bivariate normal
    # distribution with limits alpha_x, alpha_y and correlation factor rho
    def CDF_bivariate_normal_fcn(x, y, rho):                # Eq. 13
        return np.exp(- (x**2 - 2. * rho * x * y + y**2) /
                      (2. * (1. - rho**2)))

    def CDF_bivariate_normal_int(alpha, y, rho):
        return scipy.integrate.quad(
            CDF_bivariate_normal_fcn, alpha, np.inf, args=(y, rho))[0]

    return 1. / (2. * np.pi * np.sqrt(1. - rho**2)) * scipy.integrate.quad(
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

    def site_diversity_rain_outage_probability(self, lat1, lon1, a1, el1,
                                               lat2, lon2, a2, el2, f, tau,
                                               hs1=None, hs2=None):
        fcn = np.vectorize(
                self.instance.site_diversity_rain_outage_probability)
        return np.array(fcn(lat1, lon1, a1, el1,
                            lat2, lon2, a2, el2,
                            f, tau, hs1, hs2).tolist())


class _ITU618_13():

    def __init__(self):
        self.__version__ = 13

    def site_diversity_rain_outage_probability(self, lat1, lon1, a1, el1, lat2,
                                               lon2, a2, el2, f, tau,
                                               hs1=None, hs2=None):
        # The diversity prediction method assumes a log-normal distribution of
        # rain intensity and rain attenuation. This method predicts
        # Pr(A1 > a1, A2 > a2), the joint probability (%) that the attenuation
        # on the path to the first site is greater than a1 and the attenuation
        # on the path to the second site is greater than a2.
        d = compute_distance_earth_to_earth(lat1, lon1, lat2, lon2)
        rho_r = 0.7 * np.exp(-d / 60.) + 0.3 * np.exp(-(d / 700.)**2)  # Eq. 27
        # from P.837
        P_1 = rainfall_probability(lat1, lon1).\
            to(u.dimensionless_unscaled).value
        # from P.837
        P_2 = rainfall_probability(lat2, lon2).\
            to(u.dimensionless_unscaled).value
        
        R_1 = stats.norm.ppf(1 - P_1)
        R_2 = stats.norm.ppf(1 - P_2)
        
        biva_fcn = np.vectorize(__CDF_bivariate_normal__)
        P_r = biva_fcn(R_1, R_2, rho_r)
        
        sigma_lna1, m_lna1 = fit_rain_attenuation_to_lognormal(
            lat1, lon1, f, el1, tau, hs1, P_1 * 100)

        sigma_lna2, m_lna2 = fit_rain_attenuation_to_lognormal(
            lat2, lon2, f, el2, tau, hs2, P_2 * 100)
       
        rho_a = 0.94 * np.exp(-d / 30) + 0.06 * np.exp(-(d / 500)**2) # Eq. 29
        lim_1 = (np.log(a1) - m_lna1) / sigma_lna1
        lim_2 = (np.log(a2) - m_lna2) / sigma_lna2
        
        P_a = biva_fcn(lim_1, lim_2, rho_a)
        
        return 100 * P_r * P_a


class _ITU618_12():

    def __init__(self):
        self.__version__ = 12

    def site_diversity_rain_outage_probability(self, *args, **kwargs):
        return _ITU618_13.site_diversity_rain_outage_probability(*args,
                                                                 **kwargs)


__model = _ITU618()


def change_version(new_version):
    global __model
    __model = _ITU618(new_version)
    memory.clear()


def get_version():
    global __model
    return __model.__version__


def site_diversity_rain_outage_probability(lat1, lon1, a1, el1, lat2,
                                           lon2, a2, el2, f, tau, hs1=None,
                                           hs2=None):
    """
    Calculate the link outage probability in a diversity based scenario (two
    ground stations) due to rain attenuation. This method is valid for
    frequencies below 20 GHz, as at higher frequencies other impairments might
    affect affect site diversity performance.

    This method predicts Pr(A1 > a1, A2 > a2), the joint probability (%) that
    the attenuation on the path to the first site is greater than a1 and the
    attenuation on the path to the second site is greater than a2.


    Parameters
    ----------
    - lat1 : number or Quantity
            Latitude of the first ground station (deg)
    - lon1 : number or Quantity
            Longitude of the first ground station (deg)
    - a1 : number or Quantity
            Maximum admissible attenuation of the first ground station (dB)
    - el1 : number or Quantity
            Elevation angle to the first ground station (deg)
    - lat2 : number or Quantity
            Latitude of the second ground station (deg)
    - lon2 : number or Quantity
            Longitude of the second ground station (deg)
    - a2 : number or Quantity
            Maximum admissible attenuation of the second ground station (dB)
    - el2 : number or Quantity
            Elevation angle to the second ground station (deg)
    - f : number or Quantity
            Frequency (GHz)
    - tau : number, optional
            Polarization tilt angle relative to the horizontal (degrees)
            (tau = 45 deg for circular polarization).
    - hs1 : number or Quantity, optional
            Altitude over the sea level of the first ground station (km). If not
            provided, uses Recommendation ITU-R P.1511 to compute the toporgraphic
            altitude
    - hs2 : number or Quantity, optional
            Altitude over the sea level of the first ground station (km). If not
            provided, uses Recommendation ITU-R P.1511 to compute the toporgraphic
            altitude


    Returns
    -------
    - probability: Quantity
            Joint probability (%) that the attenuation on the path to the first
            site is greater than a1 and the attenuation on the path to the second
            site is greater than a2


    References
    ----------
    [1] Propagation data and prediction methods required for the design of
    Earth-space telecommunication systems:
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-12-201507-I!!PDF-E.pdf
    """
    global __model
    type_output = type(lat1)
    lon1 = np.mod(lon1, 360)
    lat1 = prepare_quantity(lat1, u.deg, 'Latitude in ground station 1')
    lon1 = prepare_quantity(lon1, u.deg, 'Longitude in ground station 1')
    a1 = prepare_quantity(a1, u.dB, 'Attenuation margin in ground station 1')
    el1 = prepare_quantity(el1, u.deg, 'Elevation angle in ground station 1')

    lon2 = np.mod(lon2, 360)
    lat2 = prepare_quantity(lat2, u.deg, 'Latitude in ground station 2')
    lon2 = prepare_quantity(lon2, u.deg, 'Longitude in ground station 2')
    a2 = prepare_quantity(a2, u.dB, 'Attenuation margin in ground station 2')
    el2 = prepare_quantity(el2, u.deg, 'Elevation angle in ground station 2')

    f = prepare_quantity(f, u.GHz, 'Frequency')
    tau = prepare_quantity(tau, u.one, 'Polarization tilt angle')
    
    hs1 = prepare_quantity(
        hs1, u.km, 'Altitude over the sea level for ground station 1')
    hs2 = prepare_quantity(
        hs2, u.km, 'Altitude over the sea level for ground station 2')

    val = __model.site_diversity_rain_outage_probability(
        lat1, lon1, a1, el1, lat2, lon2, a2, el2, f, tau, hs1=hs1, hs2=hs2)

    return prepare_output_array(val, type_output) * u.pct

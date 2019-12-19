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

from iturpropag.models.iturp618.scintillation_attenuation_sigma import scintillation_attenuation_sigma
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

    def scintillation_attenuation(self, lat, lon, f, el, p, D, eta,
                                  T, H, P, hL):
        fcn = np.vectorize(self.instance.scintillation_attenuation,
                           excluded=[0, 1, 3, 7, 8, 9], otypes=[np.ndarray])
        return np.array(fcn(lat, lon, f, el, p, D, eta, T, H, P, hL).tolist())


class _ITU618_13():

    def __init__(self):
        self.__version__ = 13

    def scintillation_attenuation(self, lat, lon, f, el, p, D, eta, T=None,
                                  H=None, P=None, hL=1000):
        # Step 1 - 7: Calculate the standard deviation of the signal for the
        # applicable period and propagation path
    	sigma = scintillation_attenuation_sigma(lat, lon, f, el,
                                                D, eta, T=T, H=H, P=P, hL=hL).value                                               
        # Step 8: Calculate the time percentage factor, a(p), for the time
        # percentage, p, in the range between 0.01% < p < 50%
    	a = -0.061 * np.log10(p)**3 + 0.072 * np.log10(p)**2 - 1.71 * np.log10(p) + 3
        # Step 9: Calculate the fade depth, A(p), exceeded for p% of the time:
    	A_s = a * sigma  # Eq. 49   [dB]
        
    	return A_s


class _ITU618_12():

    def __init__(self):
        self.__version__ = 12

    def scintillation_attenuation(self, *args, **kwargs):
        return _ITU618_13.scintillation_attenuation(*args, **kwargs)


__model = _ITU618()


def change_version(new_version):
    global __model
    __model = _ITU618(new_version)
    memory.clear()


def get_version():
    global __model
    return __model.__version__


def scintillation_attenuation(lat, lon, f, el, p, D, eta, T=None,
                              H=None, P=None, hL=1000):
    """
	Calculation of monthly and long-term statistics of amplitude scintillations
    at elevation angles greater than 5° and frequencies up to 20 GHz.


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
    - p : number
            Percetage of the time the scintillation attenuation value is exceeded.
    - D: number
            Physical diameter of the earth-station antenna (m)
    - eta: number
            Antenna efficiency.
    - T: number, sequence, or numpy.ndarray, optional
            Average surface ambient temperature (°C) at the site. If None, uses the
            ITU-R P.453 to estimate the wet term of the radio refractivity.
    - H: number, sequence, or numpy.ndarray, optional
            Average surface relative humidity (%) at the site. If None, uses the
            ITU-R P.453 to estimate the wet term of the radio refractivity.
    - P: number, sequence, or numpy.ndarray, optional
            Average surface pressure (hPa) at the site. If None, uses the
            ITU-R P.453 to estimate the wet term of the radio refractivity.
    - hL : number, optional
            Height of the turbulent layer (m). Default value 1000 m


    Returns
    -------
    - attenuation: Quantity
            Attenuation due to scintillation (dB)


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
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(prepare_input_array(el), u.deg, 'Elevation angle')
    D = prepare_quantity(D, u.m, 'Antenna diameter')
    eta = prepare_quantity(eta, u.one, 'Antenna efficiency')
    T = prepare_quantity(T, u.deg_C, 'Average surface temperature')
    H = prepare_quantity(H, u.percent, 'Average surface relative humidity')
    P = prepare_quantity(P, u.hPa, 'Average surface pressure')
    hL = prepare_quantity(hL, u.m, 'Height of the turbulent layer')

    val = __model.scintillation_attenuation(lat, lon, f, el, p, D, 
                                        eta, T=T, H=H, P=P, hL=hL)
    return prepare_output_array(val, type_output) * u.dB

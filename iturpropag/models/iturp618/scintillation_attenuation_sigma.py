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


from iturpropag.models.iturp453.water_vapour_pressure import water_vapour_pressure
from iturpropag.models.iturp453.wet_term_radio_refractivity import wet_term_radio_refractivity
from iturpropag.models.iturp453.map_wet_term_radio_refractivity import map_wet_term_radio_refractivity
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

    def scintillation_attenuation_sigma(self, lat, lon, f, el, D, eta,
                                  T, H, P, hL):
        #fcn = np.vectorize(self.instance.scintillation_attenuation_sigma,
        #                   excluded=[0, 1, 3, 6, 7, 8], otypes=[np.ndarray])
        #return np.array(fcn(lat, lon, f, el, D, eta, T, H, P, hL).tolist())
        return self.instance.scintillation_attenuation_sigma(
                        lat, lon, f, el, D, eta, T, H, P, hL)


class _ITU618_13():

    def __init__(self):
        self.__version__ = 13

    def scintillation_attenuation_sigma(self, lat, lon, f, el, D, eta,
                                        T=None, H=None, P=None, hL=1000):
        # Step 1: For the value of t, calculate the saturation water vapour
        # pressure, es, (hPa), as specified in Recommendation ITU-R P.453.
        if T is not None and H is not None and P is not None:
            _, ew = water_vapour_pressure(T, P, H)

            # Step 2: Compute the wet term of the radio refractivity, Nwet,
            # corresponding to es, t and H as given in Recommendation ITU-R
            # P.453.
            N_wet = wet_term_radio_refractivity(ew.value, T+273.15).value
        else:
            N_wet = map_wet_term_radio_refractivity(lat, lon, 50).value
            
        # Step 3: Calculate the standard deviation of the reference signal
        # amplitude:
        sigma_ref = 3.6e-3 + 1.e-4 * N_wet              # Eq. 40   [dB]

        # Step 4: Calculate the effective path length L:  Eq. 41
        L = 2. * hL / (np.sqrt(np.sin(np.deg2rad(el))**2 + 2.35e-4) +
                      np.sin(np.deg2rad(el)))           # Eq. 44   [m]

        # Step 5: Estimate the effective antenna diameter, Deff
        D_eff = np.sqrt(eta) * D                        # Eq. 42   [m]

        # Step 6: Step 6: Calculate the antenna averaging factor
        x = 1.22 * (D_eff**2) * f / L                   # Eq. 43a
        g = np.where(x >= 7.0, 0,
                     np.sqrt(3.86 * ((x**2 + 1)**(11. / 12.)) *
                             np.sin(11. / 6. * np.arctan2(1, x)) -
                             7.08 * x**(5. / 6.)))      # Eq. 43    [-]

        # Step 7: Calculate the standard deviation of the signal for the
        # applicable period and propagation path:         Eq. 44
        sigma = sigma_ref * f**(7. / 12.) * g / (np.sin(np.deg2rad(el))**1.2)
        return sigma


class _ITU618_12():

    def __init__(self):
        self.__version__ = 12

    def scintillation_attenuation_sigma(self, *args, **kwargs):
        return _ITU618_13.scintillation_attenuation_sigma(*args, **kwargs)


__model = _ITU618()


def change_version(new_version):
    global __model
    __model = _ITU618(new_version)
    memory.clear()


def get_version():
    global __model
    return __model.__version__


def scintillation_attenuation_sigma(lat, lon, f, el, D, eta, T=None,
                                    H=None, P=None, hL=1000):
    """
    Calculation of the standard deviation of the amplitude of the
    scintillations attenuation at elevation angles greater than 5° and
    frequencies up to 20 GHz.


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
    - D: number
            Physical diameter of the earth-station antenna (m)
    - eta: number, optional
            Antenna efficiency. Default value 0.5 (conservative estimate)
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

    lat = prepare_input_array(lat).flatten()
    lon = prepare_input_array(lon).flatten()

    type_output = type(lat)

    lon = np.mod(lon, 360)
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(prepare_input_array(el), u.deg, 'Elevation angle')
    D = prepare_quantity(D, u.m, 'Antenna diameter')
    eta = prepare_quantity(eta, u.one, 'Antenna efficiency')
    T = prepare_quantity(T, u.deg_C, 'Average surface temperature')
    H = prepare_quantity(H, u.percent, 'Average surface relative humidity')
    P = prepare_quantity(P, u.hPa, 'Average surface pressure')
    hL = prepare_quantity(hL, u.m, 'Height of the turbulent layer')

    val = __model.scintillation_attenuation_sigma(
        lat, lon, f, el, D, eta, T=T, H=H, P=P, hL=hL)

    return prepare_output_array(val, type_output) * u.dB 
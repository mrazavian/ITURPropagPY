# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u
import warnings

from iturpropag.utils import prepare_input_array, prepare_quantity, load_data,\
    prepare_output_array, dataset_dir


class __ITU453():
    """ Implementation of the methods in Recommendation ITU-R P.453
    "The radio refractive index: its formula and refractivity data"

    Available versions:
       * P.453-13 (12/17)
       * P.453-12 (07/15)

    Recommendation ITU-R P.453 provides methods to estimate the radio
    refractive index and its behaviour for locations worldwide; describes both
    surface and vertical profile characteristics; and provides global maps for
    the distribution of refractivity parameters and their statistical
    variation.
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.453 recommendation.

    def __init__(self, version=13):
        if version == 13:
            self.instance = _ITU453_13()
        elif version == 12:
            self.instance = _ITU453_12()
        else:
            raise ValueError(
                'Version {0} is not implemented for the ITU-R P.453 model.'
                .format(version))

    @property
    def __version__(self):
        return self.instance.__version__

    def saturation_vapour_pressure(self, T, P):
        return self.instance.saturation_vapour_pressure(T, P)


class _ITU453_13():

    def __init__(self):
        self.__version__ = 13
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.453-13-201712-I/en'

    def es_water(self, T, P):
        EF = 1. + 1.e-4 * (7.2 + P * (0.0320 + 5.9e-6 * T**2))
        a = 6.1121
        b = 18.678
        c = 257.14
        d = 234.5

        es = EF * a * np.exp((b - T / d) * T / (T + c)) # Eq.9
        return es
    
    def es_ice(self, T, P):
        EF = 1. + 1.e-4 * (2.2 + P * (0.0383 + 6.4e-6 * T**2))
        a = 6.1115
        b = 23.036
        c = 279.82
        d = 333.7

        es = EF * a * np.exp((b - T / d) * T / (T + c)) # Eq.9
        return es

    def saturation_vapour_pressure(self, T, P):
        if np.logical_or(T < -40, 50 < T):
            warning_text = 'The method for calcultaion of saturation vapour '+\
                            'pressure for water in ITU-R P.453-13 is only valid for temperature'+\
                            ' -40 <= T °C <= 50. Current temperature is '+ str(T) + ' °C'
            warnings.warn(RuntimeWarning(warning_text))
        
        if np.logical_or(T < -80, 0 < T):
            warning_text = 'The method for calcultaion of saturation vapour '+\
                            'pressure for ice in ITU-R P.453-13 is only valid for temperature'+\
                            ' -80 <= T °C <= 0 . Current temperature is '+ str(T) + ' °C'
            warnings.warn(RuntimeWarning(warning_text))
        
        es_ice = np.where(np.logical_and(-80 <= T, T <= 0), self.es_ice(T, P), np.nan)

        es_water = np.where(np.logical_and(-40 <= T, T <= 50), self.es_water(T, P), np.nan)
            
        return es_ice, es_water


class _ITU453_12():

    def __init__(self):
        self.__version__ = 12
        self.year = 2016
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.453-12-201609-I/en'

    def saturation_vapour_pressure(self, T, P):
        return _ITU453_13.saturation_vapour_pressure(T, P)


__model = __ITU453()


def change_version(new_version):
    """
    Change the version of the ITU-R P.453 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.453-13 (02/12) (Current version)
            * p.453-12
    """
    global __model
    __model = __ITU453(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.453 recommendation currently being used.
    """
    global __model
    return __model.__version__


def saturation_vapour_pressure(T, P):
    """
    Method to determine the saturation water vapour pressure


    Parameters
    ----------
    - T : number or Quantity
            temperature (°C)
    - P : number or Quantity
            Total atmospheric pressure (hPa)


    Returns
    -------
    - es_ice: Quantity
            Saturation vapour pressure for ice (hPa)
    - es_water: Quantity
            Saturation vapour pressure for water (hPa)


    References
    ----------
    [1] The radio refractive index: its formula and refractivity data
    https://www.itu.int/rec/R-REC-P.453/en

    """
    global __model
    T = prepare_quantity(T, u.deg_C, 'Temperature')
    P = prepare_quantity(P, u.hPa, 'Total atmospheric pressure')
    es_ice, es_water = __model.saturation_vapour_pressure(T, P)
    return es_ice * u.hPa, es_water * u.hPa


# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u

from iturpropag.models.iturp453.saturation_vapour_pressure import saturation_vapour_pressure
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

    def water_vapour_pressure(self, T, P, H):
        return self.instance.water_vapour_pressure(T, P, H)


class _ITU453_13():

    def __init__(self):
        self.__version__ = 13
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.453-13-201712-I/en'

    def water_vapour_pressure(self, T, P, H):
        es_ice, es_water = saturation_vapour_pressure(T, P)

        e_ice = H * es_ice.value / 100.
        e_water = H * es_water.value / 100.
        return e_ice, e_water   # Eq. 8


class _ITU453_12():

    def __init__(self):
        self.__version__ = 12
        self.year = 2016
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.453-12-201609-I/en'

    def water_vapour_pressure(self, T, P, H):
        return _ITU453_13().water_vapour_pressure(T, P, H)

__model = __ITU453()


def change_version(new_version):
    """
    Change the version of the ITU-R P.453 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.453-12 (02/12) (Current version)
            * P.453-12
    """
    global __model
    __model = __ITU453(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.453 recommendation currently being used.
    """
    global __model
    return __model.__version__


def water_vapour_pressure(T, P, H):
    """
    Method to determine the water vapour pressure


    Parameters
    ----------
    - T : number or Quantity
            Temperature (°C)
    - P : number or Quantity
            Total atmospheric pressure (hPa)
    - H : number or Quantity
            Relative humidity (%)


    Returns
    -------
    - e_ice: Quantity
            Water vapour pressure for ice (hPa)
    - e_water: Quantity
            Water vapour pressure for water (hPa)


    References
    ----------
    [1] The radio refractive index: its formula and refractivity data
    https://www.itu.int/rec/R-REC-P.453/en

    """
    global __model
    T = prepare_quantity(T, u.deg_C, 'Temperature')
    P = prepare_quantity(P, u.hPa, 'Total atmospheric pressure')
    H = prepare_quantity(H, u.percent, 'Total atmospheric pressure')
    e_ice, e_water = __model.water_vapour_pressure(T, P, H)
    return e_ice * u.hPa, e_water * u.hPa


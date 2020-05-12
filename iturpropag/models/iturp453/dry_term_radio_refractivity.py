# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u

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

    def dry_term_radio_refractivity(self, P, e, T):
        return self.instance.dry_term_radio_refractivity(P, e, T)


class _ITU453_13():

    def __init__(self):
        self.__version__ = 13
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.453-13-201712-I/en'

    def dry_term_radio_refractivity(self, P, e, T):
        Pd = P - e
        N_dry = 77.6 * Pd / T  # Eq. 3
        return N_dry


class _ITU453_12():

    def __init__(self):
        self.__version__ = 12
        self.year = 2016
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.453-12-201609-I/en'

    def dry_term_radio_refractivity(self, P, e, T):
        return _ITU453_13().dry_term_radio_refractivity(P, e, T)


__model = __ITU453()


def change_version(new_version):
    """
    Change the version of the ITU-R P.453 recommendation currently being used.

    Parameters
    ----------
    new_version : int
        Number of the version to use.
        Valid values are:
           * P.453-13 (02/12) (Current version)
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


def dry_term_radio_refractivity(P, e, T):
    """
    Method to determine the dry term of the radio refractivity


    Parameters
    ----------
    - P : number or Quantity
            Total atmospheric pressure (hPa)
    - T : number or Quantity
            Absolute temperature (K)
    - e : number or Quantity
            water vapour partial pressure (hPa)


    Returns
    -------
    - N_dry: Quantity
            Dry term of the radio refractivity (-)


    References
    ----------
    [1] The radio refractive index: its formula and refractivity data
    https://www.itu.int/rec/R-REC-P.453/en

    """
    global __model
    P = prepare_quantity(P, u.hPa, 'Total atmospheric pressure')
    T = prepare_quantity(T, u.K, 'Absolute temperature')
    e = prepare_quantity(e, u.hPa, 'Water vapour partial pressure')
    val = __model.dry_term_radio_refractivity(P, e, T)
    return val * u.dimensionless_unscaled


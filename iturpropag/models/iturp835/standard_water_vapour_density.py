# -*- coding: utf-8 -*-
# checked by <TP> 2019-06-06
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u

from iturpropag.utils import prepare_input_array, prepare_output_array,\
    prepare_quantity


class __ITU835():
    """Reference Standard Atmospheres

    Available versions:
       * P.835-6 (12/17) (Current version)
       * P.835-5 (02/12) (Superseded)


    The procedures to compute the reference standard atmosphere parameters
    pressented in these versions are identical to those included in version
    ITU_T P.835-5.
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.835 recommendation.

    def __init__(self, version=6):
        if version == 6:
            self.instance = _ITU835_6()
        elif version == 5:
            self.instance = _ITU835_5()
        else:
            raise ValueError(
                'Version ' +
                str(version) +
                ' is not implemented' +
                ' for the ITU-R P.835 model.')

    @property
    def __version__(self):
        return self.instance.__version__

    def standard_water_vapour_density(self, h, h_0, rho_0):
        return self.instance.standard_water_vapour_density(h, h_0, rho_0)


class _ITU835_6():

    def __init__(self):
        self.__version__ = 6
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.835-6-201712-I/en'

    #  Mean annual global reference atmosphere (Section 1 of ITU-R P.835-6)  #
    def standard_water_vapour_density(self, h, h_0=2., rho_0=7.5):  # Eq. 7
        """
        section 1.2 of ITU-R P.835-6 
        """
        return rho_0 * np.exp(-h / h_0)                             # Eq. 6


class _ITU835_5():

    def __init__(self):
        self.__version__ = 5
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.835-5-201202-I/en'

    #  Mean annual global reference atmosphere (Section 1 of ITU-R P.835-5)  #
    def standard_water_vapour_density(self, h, h_0=2., rho_0=7.5):  # Eq. 7
        """
        section 1.2 of ITU-R P.835-5
        """
        return rho_0 * np.exp(-h / h_0)                             # Eq. 6


__model = __ITU835()


def change_version(new_version):
    """
    Change the version of the ITU-R P.835 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.835-6 (12/17) (Current version)
            * P.835-5 (02/12) (Superseded)
            
    """
    global __model
    __model = __ITU835(new_version)



def get_version():
    """
    Obtain the version of the ITU-R P.835 recommendation currently being used.
    """
    global __model
    return __model.__version__


def standard_water_vapour_density(h, h_0=2., rho_0=7.5):
    """
    Method to compute the water vapour density of an standard atmosphere at
    a given height.

    The reference standard atmosphere is based on the United States Standard
    Atmosphere, 1976, in which the atmosphere is divided into seven successive
    layers showing linear variation with temperature.


    Parameters
    ----------
    - h : number or Quantity
            Height (km)
    - h_0 : number or Quantity
            Scale height (km)
    - rho_0 : number or Quantity
            Surface water vapour density (g/m^3)


    Returns
    -------
    - rho: Quantity
            Water vapour density (g/m^3)


    References
    ----------
    [1] Reference Standard Atmospheres
    https://www.itu.int/rec/R-REC-P.835/en
    """
    global __model

    h = prepare_quantity(h, u.km, 'Height')
    h_0 = prepare_quantity(h_0, u.km, 'Scale height')
    rho_0 = prepare_quantity(
        rho_0,
        u.g / u.m**3,
        'Surface water vapour density')
    return __model.standard_water_vapour_density(h, h_0, rho_0) * u.g / u.m**3

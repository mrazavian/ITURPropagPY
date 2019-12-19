# -*- coding: utf-8 -*-
# checked by <TP> 2019-06-04
# remark the error in conjunction with the use of the numpy.sqrt function could not be identified
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

    Not available versions:
       * P.835-1 (08/94) (Superseded)
       * P.835-2 (08/97) (Superseded)
       * P.835-3 (10/99) (Superseded)
       * P.835-4 (03/05) (Superseded)

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

    def standard_temperature(self, h, T_0):
        return self.instance.standard_temperature(h, T_0)


class _ITU835_6():

    def __init__(self):
        self.__version__ = 6
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.835-6-201712-I/en'

    #  Mean annual global reference atmosphere (Section 1 of ITU-R P.835-6)  #
    def standard_temperature(self, h, T_0=288.15):
        """Section 1.1 of Recommendation ITU-R P.835-6
        """
        
        h_p = 6356.766 * h / (6356.766 + h)                 # Eq. 1a
        arg1 = 1. - ((h - 91.)/19.9429)**2.
        arg1 = np.where(arg1 < 0., 0., 1. - ((h - 91.)/19.9429)**2.) # avoid sqrt error
       
        # First height regime
        T = np.where(np.logical_and(0 <= h_p, h_p <= 11),
                288.15 - 6.5 * h_p,                         # Eq. 2a
            np.where(np.logical_and(11 < h_p, h_p <= 20),
                216.65,                                     # Eq. 2b
            np.where(np.logical_and(20 < h_p, h_p <= 32),
                216.65 + (h_p - 20),                        # Eq. 2c
            np.where(np.logical_and(32 < h_p, h_p <= 47),
                228.65 + 2.8 * (h_p - 32),                  # Eq. 2d
            np.where(np.logical_and(47 < h_p, h_p <= 51),
                270.65,                                     # Eq. 2e
            np.where(np.logical_and(51 < h_p, h_p <= 71),
                270.65 - 2.8 * (h_p - 51),                  # Eq. 2f
            np.where(np.logical_and(71 < h_p, h_p <= 84.852),
                214.65 - 2.0 * (h_p - 71),                  # Eq. 2g
        # Second height regime
            np.where(np.logical_and(86 <= h, h <= 91),
                186.8673,                                   # Eq. 4a
            np.where(np.logical_and(91 < h, h <= 100),
                263.1905 - 76.3232 * np.sqrt(arg1),         # Eq. 4b
            195.08134433524688)))))))))

        return T

    
class _ITU835_5():

    def __init__(self):
        self.__version__ = 5
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.835-5-201202-I/en'

    #  Mean annual global reference atmosphere (Section 1 of ITU-R P.835-5)  #
    def standard_temperature(self, h, T_0=288.15):
        """
        section 1.1 of ITU-R P.835-5
        """
        # Table 1
        H = np.array([0., 11., 20., 32., 47., 51., 71., 85.])
        # Figure 1
        T = np.array([0., -71.5, -71.5, -59.5, -17.5, -17.5, -73.5, -101.5]) + T_0

        return np.interp(h, H, T)


__model = __ITU835()


def change_version(new_version):
    """
    Change the version of the ITU-R P.835 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.835-6 (02/12) (Current version)
            * P.835-5 
           
    """
    global __model
    __model = __ITU835(new_version)



def get_version():
    """
    Obtain the version of the ITU-R P.835 recommendation currently being used.
    """
    global __model
    return __model.__version__


def standard_temperature(h, T_0=288.15):
    """
    Method to compute the temperature of an standard atmosphere at
    a given height.

    The reference standard atmosphere is based on the United States Standard
    Atmosphere, 1976, in which the atmosphere is divided into seven successive
    layers showing linear variation with temperature.


    Parameters
    ----------
    - h : number or Quantity
            Height (km)
    - T_0 : number or Quantity
            Surface absolute temperature (K)


    Returns
    -----------
    - T: Quantity
            Absolute Temperature (K)


    References
    ----------
    [1] Reference Standard Atmospheres
    https://www.itu.int/rec/R-REC-P.835/en
    """
    global __model

    h = prepare_quantity(h, u.km, 'Height')
    T_0 = prepare_quantity(T_0, u.Kelvin, 'Surface temperature')
    return __model.standard_temperature(h, T_0) * u.Kelvin

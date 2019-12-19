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

    def water_vapour_density(self, lat, h, season):
        return self.instance.water_vapour_density(lat, h, season)

    
class _ITU835_6():

    def __init__(self):
        self.__version__ = 6
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.835-6-201712-I/en'

    #  Low latitude standard atmosphere functions  (Section 2 of ITU-R P.835-6)  #
    def low_latitude_water_vapour(self, h):
        """Section 2 of Recommendation ITU-R P.835-6
        """
        return np.where(np.logical_and((0 <= h), (h <= 15)),
                    19.6542 * np.exp(-0.2313 * h - 0.1122 * h**2 +
                        0.01351 * h**3 - 0.0005923 * h**4),
                    0.)

    # Mid latitude standard atmosphere functions  (Section 3 of ITU-R P.835-6)
    def mid_latitude_water_vapour_summer(self, h):
        """Section 3.1 of Recommendation ITU-R P.835-6
        """
        return np.where(np.logical_and((0 <= h), (h <= 15)),
                    14.3542 * np.exp(-0.4174 * h - 0.02290 * h**2 +
                        0.001007 * h**3),
                    0.)

    def mid_latitude_water_vapour_winter(self, h):
        """Section 3.2 of Recommendation ITU-R P.835-6
        """
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    3.4742 * np.exp(-0.2697 * h - 0.03604 * h**2 +
                        0.0004489 * h**3),
                    0.)

    #  High latitude standard atmosphere functions  (Section 4 of ITU-R P.835-6)  #
    def high_latitude_water_vapour_summer(self, h):
        """Section 4.1 of Recommendation ITU-R P.835-6
        """
        return np.where(np.logical_and((0 <= h), (h <= 15)),
                    8.988 * np.exp(-0.3614 * h - 0.005402 * h**2 -
                        0.001955 * h**3),
                    0.)

    def high_latitude_water_vapour_winter(self, h):
        """Section 4.2 of Recommendation ITU-R P.835-6
        """
        return np.where(np.logical_and(0 <= h, h <= 10),
                    1.2319 * np.exp(0.07481 * h - 0.0981 * h**2 +
                        0.00281 * h**3),
                    0.)

    def water_vapour_density(self, lat, h, season):
        """ Section 2 of Recommendation ITU-R P.835-6
        """
        if season == 'summer':
            return np.where(
                np.abs(lat) < 22, self.low_latitude_water_vapour(h),
                np.where(
                    np.abs(lat) < 45, self.mid_latitude_water_vapour_summer(h),
                    self.high_latitude_water_vapour_summer(h)))
        elif season == 'winter':
            return np.where(
                np.abs(lat) < 22, self.low_latitude_water_vapour(h),
                np.where(
                    np.abs(lat) < 45, self.mid_latitude_water_vapour_winter(h),
                    self.high_latitude_water_vapour_winter(h)))
        else:
            raise NameError("The season {} is not correct. possible\
                            choices are \'winter\' or \'summer\'".format(season))

class _ITU835_5():

    def __init__(self):
        self.__version__ = 5
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.835-5-201202-I/en'

    #  Low latitude standard atmosphere functions  (Section 2 of ITU-R P.835-5)  #
    def low_latitude_water_vapour(self, h):
        """Section 2 of Recommendation ITU-R P.835-5
        """
        return np.where(np.logical_and((0 <= h), (h <= 15)),
                    19.6542 * np.exp(-0.2313 * h - 0.1122 * h**2 +
                        0.01351 * h**3 - 0.0005923 * h**4),
                    0.)

    # Mid latitude standard atmosphere functions  (Section 3 of ITU-R P.835-5)  #
    def mid_latitude_water_vapour_summer(self, h):
        """Section 3.1 of Recommendation ITU-R P.835-5
        """
        return np.where(np.logical_and((0 <= h), (h <= 15)),
                    14.3542 * np.exp(-0.4174 * h - 0.02290 * h**2 +
                        0.001007 * h**3),
                    0.)
 
    def mid_latitude_water_vapour_winter(self, h):
        """Section 3.2 of Recommendation ITU-R P.835-5
        """
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    3.4742 * np.exp(- 0.2697 * h - 0.03604 * h**2 +
                        0.0004489 * h**3),
                    0.)
 
    #  High latitude standard atmosphere functions  (Section 4 of ITU-R P.835-5)  #
    def high_latitude_water_vapour_summer(self, h):
        """Section 4.1 of Recommendation ITU-R P.835-5
        """
        return np.where(np.logical_and((0 <= h), (h <= 15)),
                    8.988 * np.exp(- 0.3614 * h - 0.005402 * h**2 -
                        0.001955 * h**3),
                    0.)

    def high_latitude_water_vapour_winter(self, h):
        """Section 4.2 of Recommendation ITU-R P.835-5
        """
        return np.where(np.logical_and(0 <= h, h <= 10),
                    1.2319 * np.exp(0.07481 * h - 0.0981 * h**2 +
                        0.00281 * h**3),
                    0.)


    def water_vapour_density(self, lat, h, season='summer'):
        """ Section 2 of Recommendation ITU-R P.835
        """
        if season == 'summer':
            return np.where(
                np.abs(lat) < 22, self.low_latitude_water_vapour(h),
                np.where(
                    np.abs(lat) < 45, self.mid_latitude_water_vapour_summer(h),
                    self.high_latitude_water_vapour_summer(h)))
        else:
            return np.where(
                np.abs(lat) < 22, self.low_latitude_water_vapour(h),
                np.where(
                    np.abs(lat) < 45, self.mid_latitude_water_vapour_winter(h),
                    self.high_latitude_water_vapour_winter(h)))


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


def water_vapour_density(lat, h, season='summer'):
    """
    Method to determine the water-vapour density as a
    function of altitude and latitude, for calculating gaseous attenuation
    along an Earth-space path. This method is recommended when more reliable
    local data are not available.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - h : number or Quantity
            Height (km)
    - season : string
            Season of the year (available values, 'summer', and 'winter').
            Default 'summer'


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
    type_output = type(lat)
    lat = prepare_input_array(lat)
    h = prepare_quantity(h, u.km, 'Height')
    val = __model.water_vapour_density(lat, h, season)
    return prepare_output_array(val, type_output) * u.g / u.m**3

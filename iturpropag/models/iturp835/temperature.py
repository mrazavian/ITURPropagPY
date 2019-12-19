# -*- coding: utf-8 -*-
# checked by <TP>: 2019-06-06
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

    def temperature(self, lat, h, season):
        #
        return self.instance.temperature(lat, h, season)

    
class _ITU835_6():

    def __init__(self):
        self.__version__ = 6
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.835-6-201712-I/en'

    #  Low latitude standard atmosphere functions  (Section 2 of ITU-R P.835-6)  #
    def low_latitude_temperature(self, h):
        """Section 2 of Recommendation ITU-R P.835-6
        """
        return np.where(np.logical_and((0 <= h), (h < 17)),
                    300.4222 - 6.3533 * h + 0.005886 * h**2,
               np.where(np.logical_and((17 <= h), (h < 47)),
                    194. + (h - 17.) * 2.533,
               np.where(np.logical_and((47 <= h), (h < 52)),
                    270.,
               np.where(np.logical_and((52 <= h), (h < 80)),
                    270. - (h - 52.) * 3.0714,
               np.where(np.logical_and((80 <= h), (h <= 100)),
                    184.,
                    np.nan)))))

    # Mid latitude standard atmosphere functions  (Section 3 of ITU-R P.835-6)
    def mid_latitude_temperature_summer(self, h):
        """Section 3.1 of Recommendation ITU-R P.835-6
        """
        return np.where(np.logical_and((0 <= h), (h < 13)),
                    294.9838 - 5.2159 * h - 0.07109 * h**2,
               np.where(np.logical_and((13 <= h), (h < 17)),
                    215.15,
               np.where(np.logical_and((17 <= h), (h < 47)),
                    215. * np.exp((h - 17.) * 0.008128),
               np.where(np.logical_and((47 <= h), (h < 53)),
                    275.,
               np.where(np.logical_and((53 <= h), (h < 80)),
                    275. + (1 - np.exp((h - 53.) * 0.06)) * 20.,
               np.where(np.logical_and((80 <= h), (h <= 100)),
                    175.,
                    np.nan))))))

    def mid_latitude_temperature_winter(self, h):
        """Section 3.2 of Recommendation ITU-R P.835-6
        """
        return np.where(np.logical_and((0 <= h), (h < 10)),
                    272.7241 - 3.6217 * h - 0.1759 * h ** 2 ,
               np.where(np.logical_and((10 <= h), (h < 33)),
                    218.,
               np.where(np.logical_and((33 <= h), (h < 47)),
                    218. + (h - 33.) * 3.3571,
               np.where(np.logical_and((47 <= h), (h < 53)),
                    265.,
               np.where(np.logical_and((53 <= h), (h < 80)),
                    265. - (h - 53.) * 2.0370, 
               np.where(np.logical_and((80 <= h), (h <= 100)),
                    210.,
                    np.nan))))))

    #  High latitude standard atmosphere functions  (Section ITU-R P.835-6)  #
    def high_latitude_temperature_summer(self, h):
        """Section 4.1 of Recommendation ITU-R P.835-6
        """
        return np.where(np.logical_and((0 <= h), (h < 10)),
                    286.8374 - 4.7805 * h - 0.1402 * h**2,
               np.where(np.logical_and((10 <= h), (h < 23)),
                    225.,
               np.where(np.logical_and((23 <= h), (h < 48)),
                    225. * np.exp((h - 23.) * 0.008317),
               np.where(np.logical_and((48 <= h), (h < 53)),
                    277.,
               np.where(np.logical_and((53 <= h), (h < 79)),
                    277. - (h - 53.) * 4.0769,
               np.where(np.logical_and((79 <= h), (h <= 100)),
                    171.,
                    np.nan))))))

    def high_latitude_temperature_winter(self, h):
        """Section 4.2 of Recommendation ITU-R P.835
        """
        return np.where(np.logical_and((0 <= h), (h < 8.5)),
                    257.4345 + 2.3474 * h - 1.5479 * h**2 + 0.08473 * h**3,
               np.where(np.logical_and((8.5 <= h), (h < 30)),
                    217.5,
               np.where(np.logical_and((30 <= h), (h < 50)),
                    217.5 + (h - 30.) * 2.125,
               np.where(np.logical_and((50 <= h), (h < 54)),
                    260.,
               np.where(np.logical_and((54 <= h), (h <= 100)),
                    260. - (h - 54.) * 1.667,
                    np.nan)))))

    def temperature(self, lat, h, season):
        """ Section 2,3,4 of Recommendation ITU-R P.835-6
        """
        if season == 'summer':
            return np.where(
                np.abs(lat) < 22, self.low_latitude_temperature(h),
                np.where(
                    np.abs(lat) <= 45, self.mid_latitude_temperature_summer(h),
                    self.high_latitude_temperature_summer(h)))
        elif season == 'winter':
            return np.where(
                np.abs(lat) < 22, self.low_latitude_temperature(h),
                np.where(
                    np.abs(lat) <= 45, self.mid_latitude_temperature_winter(h),
                    self.high_latitude_temperature_winter(h)))
        else:
            raise NameError("The season {} is not correct. possible choices\
                             are \'winter\' or \'summer\'".format(season))

class _ITU835_5():

    def __init__(self):
        self.__version__ = 5
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.835-5-201202-I/en'

    #  Low latitude standard atmosphere functions  (Section 2 of ITU-R P.835-5)  #
    def low_latitude_temperature(self, h):
        """Section 2 of Recommendation ITU-R P.835-5
        """
        return np.where(np.logical_and((0 <= h), (h < 17)),
                    300.4222 - 6.3533 * h + 0.005886 * h**2,
               np.where(np.logical_and((17 <= h), (h < 47)),
                    194. + (h - 17.) * 2.533,
               np.where(np.logical_and((47 <= h), (h < 52)),
                    270.,
               np.where(np.logical_and((52 <= h), (h < 80)),
                    270. - (h - 52.) * 3.0714,
               np.where(np.logical_and((80 <= h), (h <= 100)),
                    184.,
                    np.nan)))))

    # Mid latitude standard atmosphere functions  (Section 3 of ITU-R P.835-5)  #
    def mid_latitude_temperature_summer(self, h):
        """Section 3.1 of Recommendation ITU-R P.835-5
        """
        return np.where(np.logical_and((0 <= h), (h < 13)),
                    294.9838 - 5.2159 * h - 0.07109 * h**2,
               np.where(np.logical_and((13 <= h), (h < 17)),
                    215.15,
               np.where(np.logical_and((17 <= h), (h < 47)),
                    215.15 * np.exp((h - 17.) * 0.008128),
               np.where(np.logical_and((47 <= h), (h < 53)),
                    275.,
               np.where(np.logical_and((53 <= h), (h < 80)),
                    275. - 20. * (1. - np.exp((h - 53.) * 0.06)),
               np.where(np.logical_and((80 <= h), (h <= 100)),
                    175.,
                    np.nan))))))

    def mid_latitude_temperature_winter(self, h):
        """Section 3.2 of Recommendation ITU-R P.835-5
        """
        return np.where(np.logical_and((0 <= h), (h < 10)),
                    272.7241 - 3.6217 * h - 0.1759 * h**2,
               np.where(np.logical_and((10 <= h), (h < 33)),
                    218.,
               np.where(np.logical_and((33 <= h), (h < 47)),
                    218. + (h - 33.) * 3.3571,
               np.where(np.logical_and((47 <= h), (h < 53)),
                    265.,
               np.where(np.logical_and((53 <= h), (h < 80)),
                    265. - (h - 53.) * 2.0370, 
               np.where(np.logical_and((80 <= h), (h <= 100)),
                    210.,
                    np.nan))))))

    #  High latitude standard atmosphere functions  (Section 4 of ITU-R P.835-5)  #
    def high_latitude_temperature_summer(self, h):
        """Section 4.1 of Recommendation ITU-R P.835-5
        """
        return np.where(np.logical_and((0 <= h), (h < 10)),
                    286.8374 - 4.7805 * h - 0.1402 * h**2,
               np.where(np.logical_and((10 <= h), (h < 23)),
                    225.,
               np.where(np.logical_and((23 <= h), (h < 48)),
                    225. * np.exp((h - 23.) * 0.008317),
               np.where(np.logical_and((48 <= h), (h < 53)),
                    277.,
               np.where(np.logical_and((53 <= h), (h < 79)),
                    277. - (h - 53.) * 4.0769,
               np.where(np.logical_and((79 <= h), (h <= 100)),
                    171.,
                    np.nan))))))

    def high_latitude_temperature_winter(self, h):
        """Section 4.2 of Recommendation ITU-R P.835
        """
        return np.where(np.logical_and((0 <= h), (h < 8.5)),
                    257.4345 + 2.3474 * h - 1.5479 * h**2 + 0.08473 * h**3,
               np.where(np.logical_and((8.5 <= h), (h < 30)),
                    217.5,
               np.where(np.logical_and((30 <= h), (h < 50)),
                    217.5 + (h - 30.) * 2.125,
               np.where(np.logical_and((50 <= h), (h < 54)),
                    260.,
               np.where(np.logical_and((54 <= h), (h <= 100)),
                    260. - (h - 54.) * 1.667,
                    np.nan)))))

    def temperature(self, lat, h, season='summer'):
        """ Section 2 of Recommendation ITU-R P.835
        """
        if season == 'summer':
            return np.where(
                np.abs(lat) < 22, self.low_latitude_temperature(h),
                np.where(
                    np.abs(lat) <= 45, self.mid_latitude_temperature_summer(h),
                    self.high_latitude_temperature_summer(h)))
        else:
            return np.where(
                np.abs(lat) < 22, self.low_latitude_temperature(h),
                np.where(
                    np.abs(lat) <= 45, self.mid_latitude_temperature_winter(h),
                    self.high_latitude_temperature_winter(h)))


__model = __ITU835()


def change_version(new_version):
    """
    Change the version of the ITU-R P.835 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.835-6  (Current version)
            * P.835-5  (Superseded)
            
    """
    global __model
    __model = __ITU835(new_version)



def get_version():
    """
    Obtain the version of the ITU-R P.835 recommendation currently being used.
    """
    global __model
    return __model.__version__


def temperature(lat, h, season='summer'):
    """
    Method to determine the temperature as a function of altitude and latitude,
    for calculating gaseous attenuation along an Earth-space path. This method
    is recommended when more reliable local data are not available.


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
    - T: Quantity
            Absolute Temperature (K)


    References
    ----------
    [1] Reference Standard Atmospheres
    https://www.itu.int/rec/R-REC-P.835/en

    """
    global __model
    type_output = type(lat)
    lat = prepare_input_array(lat)
    h = prepare_quantity(h, u.km, 'Height')
    val = __model.temperature(lat, h, season)
    return prepare_output_array(val, type_output) * u.Kelvin

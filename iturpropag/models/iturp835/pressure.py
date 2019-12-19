# -*- coding: utf-8 -*-
# checked by <TP>: 2019-06-06
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u

from iturpropag.models.iturp835.standard_pressure import standard_pressure
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

    def pressure(self, lat, h, season='summer'):
        return self.instance.pressure(lat, h, season)


class _ITU835_6():

    def __init__(self):
        self.__version__ = 6
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.835-6-201712-I/en'

    #  Low latitude standard atmosphere functions  (Section 2 of ITU-R P.835-6)  #
    def low_latitude_pressure(self, h):
        """Section 2 of Recommendation ITU-R P.835-6
        """

        P10 = standard_pressure(10).to(u.hPa).value
        P72 = standard_pressure(72).to(u.hPa).value
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    1012.0306 - 109.0338 * h + 3.6316 * h**2,
               np.where(np.logical_and((10 < h), (h <= 72)),
                    P10 * np.exp(-0.147 * (h - 10.)),
               np.where(np.logical_and((72 < h), (h <= 100)),
                    P72 * np.exp(-0.165 * (h - 72.)), np.nan)))

    # Mid latitude standard atmosphere functions  (Section 3 of ITU-R P.835-6)
    # ##
    def mid_latitude_pressure_summer(self, h):
        """Section 3.1 of Recommendation ITU-R P.835-6
        """
        P10 = standard_pressure(10).to(u.hPa).value
        P72 = standard_pressure(72).to(u.hPa).value
        return np.where(
            np.logical_and((0 <= h), (h <= 10)),
                    1012.8186 - 111.5569 * h + 3.8646 * h**2, np.where(
                np.logical_and((10 < h), (h <= 72)),
                    P10 * np.exp(-0.147 * (h - 10.)),
                np.where(
                    np.logical_and((72 < h), (h <= 100)),
                    P72 * np.exp(-0.165 * (h - 72.)),
                    np.nan)))

    def mid_latitude_pressure_winter(self, h):
        """Section 3.2 of Recommendation ITU-R P.835-6
        """
        P10 = standard_pressure(10).to(u.hPa).value
        P72 = standard_pressure(72).to(u.hPa).value
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    1018.8627 - 124.2954 * h + 4.8307 * h**2,
               np.where(np.logical_and((10 < h), (h <= 72)),
                    P10 * np.exp(-0.147 * (h - 10.)),
               np.where(np.logical_and((72 < h), (h <= 100)),
                    P72 * np.exp(-0.155 * (h - 72.)), np.nan)))

    #  High latitude standard atmosphere functions  (Section 4 of ITU-R P.835-6)  #
    def high_latitude_pressure_summer(self, h):
        """Section 4.1 of Recommendation ITU-R P.835-6
        """
        P10 = standard_pressure(10).to(u.hPa).value
        P72 = standard_pressure(72).to(u.hPa).value
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    1008.0278 - 113.2494 * h + 3.9408 * h**2,
               np.where(np.logical_and((10 < h), (h <= 72)),
                    P10 * np.exp(-0.140 * (h - 10.)),
               np.where(np.logical_and((72 < h), (h <= 100)),
                    P72 * np.exp(-0.165 * (h - 72.)), np.nan)))

    def high_latitude_pressure_winter(self, h):
        """Section 4.2 of Recommendation ITU-R P.835-6
        """
        P10 = standard_pressure(10).to(u.hPa).value
        P72 = standard_pressure(72).to(u.hPa).value
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    1010.8828 - 122.2411 * h + 4.554 * h**2,
               np.where(np.logical_and((10 < h), (h <= 72)),
                    P10 * np.exp(-0.147 * (h - 10.)),
               np.where(np.logical_and((72 < h), (h <= 100)),
                    P72 * np.exp(-0.150 * (h - 72.)), np.nan)))

    def pressure(self, lat, h, season='summer'):
        """ Section 2,3,4 of Recommendation ITU-R P.835
        """
        if season == 'summer':
            return np.where(
                np.abs(lat) < 22, self.low_latitude_pressure(h),
                np.where(
                    np.abs(lat) <= 45, self.mid_latitude_pressure_summer(h),
                    self.high_latitude_pressure_summer(h)))
        elif season == 'winter':
            return np.where(
                np.abs(lat) < 22, self.low_latitude_pressure(h),
                np.where(
                    np.abs(lat) <= 45, self.mid_latitude_pressure_winter(h),
                    self.high_latitude_pressure_winter(h)))
        else:
            raise NameError("The season {} is not correct. possible choices are\
                            \'winter\' or \'summer\' ".format(season))

class _ITU835_5():

    def __init__(self):
        self.__version__ = 5
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.835-5-201202-I/en'

    #  Low latitude standard atmosphere functions  (Section 2 of ITU-R P.835-5)  #
    def low_latitude_pressure(self, h):
        """Section 2 of Recommendation ITU-R P.835-5
        """
        P10 = standard_pressure([10]).to(u.hPa).value
        P72 = standard_pressure([72]).to(u.hPa).value
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    1012.0306 - 109.0338 * h + 3.6316 * h**2,
               np.where(np.logical_and((10 < h), (h <= 72)),
                    P10 * np.exp(-0.147 * (h - 10.)),
               np.where(np.logical_and((72 < h), (h <= 100)),
                    P72 * np.exp(-0.165 * (h - 72.)), np.nan)))

    # Mid latitude standard atmosphere functions  (Section 3 of ITU-R P.835-5)  #
    def mid_latitude_pressure_summer(self, h):
        """Section 3.1 of Recommendation ITU-R P.835-5
        """
        P10 = standard_pressure([10]).to(u.hPa).value
        P72 = standard_pressure([72]).to(u.hPa).value
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    1012.8186 - 111.5569 * h + 3.8646 * h**2,
               np.where(np.logical_and((10 < h), (h <= 72)),
                    P10 * np.exp(-0.147 * (h - 10.)),
               np.where(np.logical_and((72 < h), (h <= 100)),
                    P72 * np.exp(-0.165 * (h - 72.)),
                    np.nan)))

    def mid_latitude_pressure_winter(self, h):
        """Section 3.2 of Recommendation ITU-R P.835-5
        """
        P10 = standard_pressure([10]).to(u.hPa).value
        P72 = standard_pressure([72]).to(u.hPa).value
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    1018.8627 - 124.2954 * h + 4.8307 * h**2,
               np.where(np.logical_and((10 < h), (h <= 72)),
                    P10 * np.exp(-0.147 * (h - 10.)),
               np.where(np.logical_and((72 < h), (h <= 100)),
                    P72 * np.exp(-0.155 * (h - 72.)), np.nan)))

    #  High latitude standard atmosphere functions  (Section 4 of ITU-R P.835-5)  #
    def high_latitude_pressure_summer(self, h):
        """Section 4.1 of Recommendation ITU-R P.835-5
        """
        P10 = standard_pressure([10]).to(u.hPa).value
        P72 = standard_pressure([72]).to(u.hPa).value
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    1008.0278 - 113.2494 * h + 3.9408 * h**2,
               np.where(np.logical_and((10 < h), (h <= 72)),
                    P10 * np.exp(-0.140 * (h - 10.)),
               np.where(np.logical_and((72 < h), (h <= 100)),
                    P72 * np.exp(-0.165 * (h - 72.)), np.nan)))

    def high_latitude_pressure_winter(self, h):
        """Section 4.2 of Recommendation ITU-R P.835-5
        """
        P10 = standard_pressure([10]).to(u.hPa).value
        P72 = standard_pressure([72]).to(u.hPa).value
        return np.where(np.logical_and((0 <= h), (h <= 10)),
                    1010.8828 - 122.2411 * h + 4.554 * h**2,
               np.where(np.logical_and((10 < h), (h <= 72)),
                    P10 * np.exp(-0.147 * (h - 10.)),
               np.where(np.logical_and((72 < h), (h <= 100)),
                    P72 * np.exp(-0.150 * (h - 72.)), np.nan)))

    def pressure(self, lat, h, season='summer'):
        """ Section 2,3,4 of Recommendation ITU-R P.835
        """
        if season == 'summer':
            return np.where(
                np.abs(lat) < 22, self.low_latitude_pressure(h),
                np.where(
                    np.abs(lat) <= 45, self.mid_latitude_pressure_summer(h),
                    self.high_latitude_pressure_summer(h)))
        else:
            return np.where(
                np.abs(lat) < 22, self.low_latitude_pressure(h),
                np.where(
                    np.abs(lat) <= 45, self.mid_latitude_pressure_winter(h),
                    self.high_latitude_pressure_winter(h)))


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


def pressure(lat, h, season='summer'):
    """
    Method to determine the pressure as a function of altitude and latitude,
    for calculating gaseous attenuation along an Earth-space path.
    This method is recommended when more reliable local data are not available.


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
    - P: Quantity
            Pressure (hPa)


    References
    ----------
    [1] Reference Standard Atmospheres
    https://www.itu.int/rec/R-REC-P.835/en
    """
    global __model
    type_output = type(lat)
    lat = prepare_input_array(lat)
    h = prepare_quantity(h, u.km, 'Height')
    val = __model.pressure(lat, h, season)
    return prepare_output_array(val, type_output) * u.hPa

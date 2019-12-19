# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u

from iturpropag.models.iturp1144.bilinear_2D_interpolator import bilinear_2D_interpolator
from iturpropag.utils import load_data, dataset_dir, prepare_input_array,\
    prepare_output_array, memory


class __ITU1853():
    """Annual mean surface water vapour density

    Available versions include:
    * P.1853-2 (08/2019) (Current version)
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.1853 recommendation.

    def __init__(self, version=2):
        if version == 2:
            self.instance = _ITU1853_2()
        elif version == 1:
            self.instance = _ITU1853_1()
        else:
            raise ValueError('Version ' + str(version) + ' is not implemented'
                             ' for the ITU-R P.1853 model.')

    @property
    def __version__(self):
        return self.instance.__version__

    def surface_mean_water_vapour_density(self, lat, lon):
        # Abstract method to compute the surface mean pressure
        return self.instance.surface_mean_water_vapour_density(lat, lon)


class _ITU1853_2():

    def __init__(self):
        self.__version__ = 2
        self.year = 2019
        self.month = 8
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en'

        self._wvd = {}

    def water_vapour_density(self, lat, lon):
        if not self._wvd:
            vals = load_data(os.path.join(dataset_dir, 'p1853/v2_WV_Annual.h5'))
            lats = load_data(os.path.join(dataset_dir, 'p1853/v2_Lat.h5'))
            lons = load_data(os.path.join(dataset_dir, 'p1853/v2_Lon.h5'))
            self._wvd = bilinear_2D_interpolator(
                    np.flipud(lats), lons, np.flipud(vals))

        # In this recommendation the longitude is encoded with format -180 to
        # 180 whereas we always use 0 - 360 encoding
        lon[lon > 180] = lon[lon > 180] - 360
        return self._wvd(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def surface_mean_water_vapour_density(self, lat, lon):
        """
        Method to compute the annual mean surface water vapour density (g/m**3) 
        """
        return self.water_vapour_density(lat, lon)


class _ITU1853_1():

    def __init__(self):
        self.__version__ = 1
        self.year = 2019
        self.month = 8
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en'

    def surface_mean_water_vapour_density(self, *args, **kwargs):
        raise NotImplementedError(
            "Recommendation ITU-R P.1853-1 does not specify a method to compute "
            "mean surface water vapour density")


__model = __ITU1853()


def change_version(new_version):
    """
    Change the version of the ITU-R P.1853 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            *version 2: P.1853-2 (08/2019) (Current version)
    """
    global __model
    __model = __ITU1853(new_version)
    memory.clear()


def get_version():
    """
    Obtain the version of the ITU-R P.1853 recommendation currently being used.
    """
    global __model
    return __model.__version__


@memory.cache
def surface_mean_water_vapour_density(lat, lon):
    """
    A method to estimate the annual mean surface water vapour density (g/m^3)


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points


    Returns
    -------
    - water_vapour_density: numpy.ndarray
            Annual mean surface water vapour density (g/m^3)


    References
    ----------
    [1] Time series synthesis of tropospheric impairments:
    https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en

    """
    global __model
    type_output = type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.surface_mean_water_vapour_density(lat, lon)
    return prepare_output_array(val, type_output) * u.g / u.m**3

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u

from iturpropag.models.iturp1144.bilinear_2D_interpolator import bilinear_2D_interpolator
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

    def DN65(self, lat, lon, p):
        fcn = np.vectorize(self.instance.DN65, excluded=[0, 1],
                           otypes=[np.ndarray])
        return np.array(fcn(lat, lon, p).tolist())


class _ITU453_13():

    def __init__(self):
        self.__version__ = 13
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.453-13-201712-I/en'

        self._DN65 = {}
        

    def DN65(self, lat, lon, p):
        if not self._DN65:
            ps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80,
                  90, 95, 98, 99, 99.5, 99.8, 99.9]
            d_dir = os.path.join(dataset_dir, 'p453/v13_DN65m_%02dd%02d_v1.txt')
            lats = load_data(os.path.join(dataset_dir, 'p453/v13_lat0d75.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p453/v13_lon0d75.txt'))
            for p_loads in ps:
                int_p = p_loads // 1
                frac_p = round((p_loads % 1.0) * 100)
                vals = load_data(d_dir % (int_p, frac_p))
                self._DN65[float(p_loads)] = bilinear_2D_interpolator(
                    lats, lons, vals)

        return self._DN65[float(p)](
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)


class _ITU453_12():

    def __init__(self):
        self.__version__ = 12
        self.year = 2016
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.453-12-201609-I/en'

        self._DN65 = {}
       

    def DN65(self, lat, lon, p):
        if not self._DN65:
            ps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80,
                  90, 95, 98, 99, 99.5, 99.8, 99.9]
            d_dir = os.path.join(dataset_dir, 'p453/v12_DN65m_%02dd%02d_v1.txt')
            lats = load_data(os.path.join(dataset_dir, 'p453/v12_lat0d75.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p453/v12_lon0d75.txt'))
            for p_loads in ps:
                int_p = p_loads // 1
                frac_p = round((p_loads % 1.0) * 100)
                vals = load_data(d_dir % (int_p, frac_p))
                self._DN65[float(p_loads)] = bilinear_2D_interpolator(
                    lats, lons, vals)

        return self._DN65[float(p)](
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)


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


def DN65(lat, lon, p):
    """
    Method to determine the statistics of the vertical gradient of radio
    refractivity in the lowest 65 m from the surface of the Earth.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - p : number
            Percentage of time exceeded for p% of the average year


    Returns
    -------
    - DN65_p: Quantity
            Vertical gradient of radio refractivity in the lowest 65 m from the
            surface of the Earth exceeded for p% of the average year



    References
    ----------
    [1] The radio refractive index: its formula and refractivity data
    https://www.itu.int/rec/R-REC-P.453/en

    """
    global __model
    type_output = type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    p = prepare_quantity(p, units=u.pct, name_val='percentage of time exceeded')
    val = __model.DN65(lat, lon, p)
    return prepare_output_array(val, type_output) * u.one


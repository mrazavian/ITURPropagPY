# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u
from scipy.optimize import bisect
import scipy.stats as stats

from iturpropag.models.iturp1510.surface_month_mean_temperature import surface_month_mean_temperature
from iturpropag.models.iturp1144.bilinear_2D_interpolator import bilinear_2D_interpolator
from iturpropag.utils import load_data, dataset_dir, prepare_input_array,\
    prepare_output_array, memory


class __ITU837():
    """Characteristics of precipitation for propagation modelling

    Available versions include:
    * P.837-6 (02/12) (Superseded)
    * P.837-7 (12/17) (Current version)
    Not-available versions:
    * P.837-1 (08/94) (Superseded)
    * P.837-2 (10/99) (Superseded)
    * P.837-3 (02/01) (Superseded)
    * P.837-4 (04/03) (Superseded)
    * P.837-5 (08/07) (Superseded)

    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.837 recommendation.

    def __init__(self, version=7):
        if version == 7:
            self.instance = _ITU837_7()
        elif version == 6:
            self.instance = _ITU837_6()
        else:
            raise ValueError(
                'Version {0} is not implemented for the ITU-R P.837 model.'
                .format(version))

        self._Pr6 = {}
        self._Mt = {}
        self._Beta = {}
        self._R001 = {}

    @property
    def __version__(self):
        return self.instance.__version__

    def rainfall_probability(self, lat, lon):
        # Abstract method to compute the rain height
        return self.instance.rainfall_probability(lat, lon)

    
class _ITU837_7():

    def __init__(self):
        self.__version__ = 7
        self.year = 2017
        self.month = 6
        self.link = 'https://www.p.int/rec/R-REC-P.837-7-201706-I/en'

        self.months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self._Mt = {}

    def Mt(self, lat, lon, m):
        if not self._Mt:
            lats = load_data(os.path.join(dataset_dir, 'p837/v7_LAT_MT.h5'))
            lons = load_data(os.path.join(dataset_dir, 'p837/v7_LON_MT.h5'))
            
            for _m in self.months:
                vals = load_data(os.path.join(dataset_dir,
                                              'p837/v7_MT_Month{0:02d}.h5')
                                 .format(_m))
                self._Mt[_m] = bilinear_2D_interpolator(np.flipud(lats), lons,
                                                        np.flipud(vals))
        # In this recommendation the longitude is encoded with format -180 to
        # 180 whereas we always use 0 - 360 encoding
        lon = np.array(lon)
        lon[lon > 180] = lon[lon > 180] - 360
        return self._Mt[m](
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def rainfall_probability(self, lat_d, lon_d):
        """

        """
        lat_f = lat_d.flatten()
        lon_f = lon_d.flatten()

        Nii = np.array([[31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]])

        # Step 2: For each month, determine the monthly mean surface
        # temperature
        Tii = surface_month_mean_temperature(lat_f, lon_f, self.months).value.T
        
        # Step 3: For each month, determine the monthly mean total rainfall
        MTii = np.array([self.Mt(lat_f, lon_f, m) for m in self.months]).T
        
        # Step 4: For each month, determine the monthly mean total rainfall
        tii = Tii - 273.15

        # Step 5: For each month number, calculate rii
        rii = np.where(tii >= 0, 0.5874 * np.exp(0.0883 * tii), 0.5874)  # Eq.1

        # Step 6a For each month number, calculate the probability of rain:
        P0ii = 100 * MTii / (24 * Nii * rii)  # Eq. 2

        # Step 7b:
        rii = np.where(P0ii > 70, 100/70. * MTii / (24 * Nii), rii)
        P0ii = np.where(P0ii > 70, 70, P0ii)

        # Step 7: Calculate the annual probability of rain, P0anual
        P0anual = np.sum(Nii * P0ii, axis=-1) / 365.25  # Eq. 3

        return P0anual.reshape(lat_d.shape)


class _ITU837_6():

    def __init__(self):
        self.__version__ = 6
        self.year = 2012
        self.month = 2
        self.link = 'https://www.p.int/rec/R-REC-P.837-6-201202-I/en'

        self._Pr6 = {}
        self._Mt = {}
        self._Beta = {}

    def Pr6(self, lat, lon):
        if not self._Pr6:
            vals = load_data(os.path.join(dataset_dir,
                                          'p837/ESARAIN_PR6_v5.txt'))
            lats = load_data(os.path.join(dataset_dir,
                                          'p837/ESARAIN_LAT_v5.txt'))
            lons = load_data(os.path.join(dataset_dir,
                                          'p837/ESARAIN_LON_v5.txt'))
            self._Pr6 = bilinear_2D_interpolator(lats, lons, vals)

        return self._Pr6(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def Mt(self, lat, lon):
        if not self._Mt:
            vals = load_data(os.path.join(dataset_dir,
                                          'p837/ESARAIN_MT_v5.txt'))
            lats = load_data(os.path.join(dataset_dir,
                                          'p837/ESARAIN_LAT_v5.txt'))
            lons = load_data(os.path.join(dataset_dir,
                                          'p837/ESARAIN_LON_v5.txt'))
            self._Mt = bilinear_2D_interpolator(lats, lons, vals)

        return self._Mt(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def Beta(self, lat, lon):
        if not self._Beta:
            vals = load_data(os.path.join(dataset_dir,
                                          'p837/ESARAIN_BETA_v5.txt'))
            lats = load_data(os.path.join(dataset_dir,
                                          'p837/ESARAIN_LAT_v5.txt'))
            lons = load_data(os.path.join(dataset_dir,
                                          'p837/ESARAIN_LON_v5.txt'))
            self._Beta = bilinear_2D_interpolator(lats, lons, vals)

        return self._Beta(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def rainfall_probability(self, lat_d, lon_d):
        """

        """
        Pr6 = self.Pr6(lat_d, lon_d)
        Mt = self.Mt(lat_d, lon_d)
        Beta = self.Beta(lat_d, lon_d)

        # Step 3: Convert MT and β to Mc and Ms as follows:
        Ms = (1 - Beta) * Mt

        # Step 4: Derive the percentage propability of rain in an average year,
        # P0:
        P0 = Pr6 * (1 - np.exp(-0.0079 * (Ms / Pr6)))  # Eq. 1

        return P0


__model = __ITU837()


def change_version(new_version):
    """
    Change the version of the ITU-R P.837 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.837-7 (02/12) (Current)
            * P.837-6
    """
    global __model
    __model = __ITU837(new_version)
    memory.clear()


def get_version():
    """
    Obtain the version of the ITU-R P.837 recommendation currently being used.
    """
    global __model
    return __model.__version__


@memory.cache
def rainfall_probability(lat, lon):
    """
    A method to compute the percentage probability of rain in an average
    year, P0


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points


    Returns
    -------
    - P0: numpy.ndarray
            Percentage probability of rain in an average year (%)


    References
    ----------
    [1] Characteristics of precipitation for propagation modelling
    https://www.p.int/rec/R-REC-P.837/en
    """
    global __model
    type_output = type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.rainfall_probability(lat, lon)
    return prepare_output_array(val, type_output) * u.pct

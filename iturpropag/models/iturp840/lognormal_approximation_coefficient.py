# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u

from iturpropag.models.iturp1144.bilinear_2D_interpolator import bilinear_2D_interpolator
from iturpropag.utils import load_data, dataset_dir, prepare_input_array, \
    prepare_output_array, prepare_quantity, memory


class __ITU840():
    """Attenuation due to clouds and fog: This Recommendation provides methods
    to predict the attenuation due to clouds and fog on Earth-space paths.

    Available versions include:
    * P.840-4 (10/09) (Superseded)
    * P.840-5 (02/12) (Superseded)
    * P.840-6 (09/13) (Superseded)
    * P.840-7 (12/17) (Current version)

    Non-available versions include:
    * P.840-1 (08/94) (Superseded) - Tentative similar to P.840-4
    * P.840-2 (08/97) (Superseded) - Tentative similar to P.840-4
    * P.840-3 (10/99) (Superseded) - Tentative similar to P.840-4

    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.840 recommendation.

    def __init__(self, version=7):
        if version == 7:
            self.instance = _ITU840_7()
        elif version == 6:
            self.instance = _ITU840_6()
        elif version == 5:
            self.instance = _ITU840_5()
        elif version == 4:
            self.instance = _ITU840_4()
        else:
            raise ValueError(
                'Version {0}  is not implemented for the ITU-R P.840 model.'
                .format(version))

        self._M = {}
        self._sigma = {}
        self._Pclw = {}

    @property
    def __version__(self):
        return self.instance.__version__

    def lognormal_approximation_coefficient(self, lat, lon):
        # Abstract method to compute the lognormal approximation coefficients
        return self.instance.lognormal_approximation_coefficient(lat, lon)


class _ITU840_7():

    def __init__(self):
        self.__version__ = 7
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.840-7-201712-I/en'

        self._M = {}
        self._sigma = {}
        self._Pclw = {}

    def M(self, lat, lon):
        if not self._M:
            vals = load_data(os.path.join(dataset_dir, 'p840/v7_M.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v7_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v7_Lon.txt'))
            self._M = bilinear_2D_interpolator(lats, lons, vals)

        return self._M(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def sigma(self, lat, lon):
        if not self._sigma:
            vals = load_data(os.path.join(dataset_dir, 'p840/v7_sigma.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v7_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v7_Lon.txt'))
            self._sigma = bilinear_2D_interpolator(lats, lons, vals)

        return self._sigma(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def Pclw(self, lat, lon):
        if not self._Pclw:
            vals = load_data(os.path.join(dataset_dir, 'p840/v7_Pclw.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v7_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v7_Lon.txt'))
            self._Pclw = bilinear_2D_interpolator(lats, lons, vals)

        return self._Pclw(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def lognormal_approximation_coefficient(self, lat, lon):
        m = self.M(lat, lon)
        sigma = self.sigma(lat, lon)
        Pclw = self.Pclw(lat, lon)

        return m, sigma, Pclw


class _ITU840_6():

    def __init__(self):
        self.__version__ = 6
        self.year = 2013
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.840-6-201202-I/en'

        self._M = {}
        self._sigma = {}
        self._Pclw = {}

    def M(self, lat, lon):
        if not self._M:
            vals = load_data(os.path.join(dataset_dir, 'p840/v6_M.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v6_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v6_Lon.txt'))
            self._M = bilinear_2D_interpolator(lats, lons, vals)

        return self._M(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def sigma(self, lat, lon):
        if not self._sigma:
            vals = load_data(os.path.join(dataset_dir, 'p840/v6_sigma.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v6_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v6_Lon.txt'))
            self._sigma = bilinear_2D_interpolator(lats, lons, vals)

        return self._sigma(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def Pclw(self, lat, lon):
        if not self._Pclw:
            vals = load_data(os.path.join(dataset_dir, 'p840/v6_Pclw.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v6_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v6_Lon.txt'))
            self._Pclw = bilinear_2D_interpolator(lats, lons, vals)

        return self._Pclw(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def lognormal_approximation_coefficient(self, lat, lon):
        m = self.M(lat, lon)
        sigma = self.sigma(lat, lon)
        Pclw = self.Pclw(lat, lon)

        return m, sigma, Pclw


class _ITU840_5():

    def __init__(self):
        self.__version__ = 5
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.840-5-201202-S/en'

        self._M = {}
        self._sigma = {}
        self._Pclw = {}

    def M(self, lat, lon):
        if not self._M:
            vals = load_data(os.path.join(dataset_dir,
                                          'p840/v4_WRED_LOGNORMAL_MEAN.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v4_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v4_Lon.txt'))
            self._M = bilinear_2D_interpolator(lats, lons, vals)

        return self._M(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def sigma(self, lat, lon):
        if not self._sigma:
            vals = load_data(os.path.join(dataset_dir,
                                          'p840/v4_WRED_LOGNORMAL_STDEV.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v4_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v4_Lon.txt'))
            self._sigma = bilinear_2D_interpolator(lats, lons, vals)

        return self._sigma(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def Pclw(self, lat, lon):
        if not self._Pclw:
            vals = load_data(os.path.join(dataset_dir,
                                          'p840/v4_WRED_LOGNORMAL_PCLW.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v4_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v4_Lon.txt'))
            self._Pclw = bilinear_2D_interpolator(lats, lons, vals)

        return self._Pclw(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def lognormal_approximation_coefficient(self, lat, lon):
        m = self.M(lat, lon)
        sigma = self.sigma(lat, lon)
        Pclw = self.Pclw(lat, lon)

        return m, sigma, Pclw


class _ITU840_4():

    def __init__(self):
        self.__version__ = 4
        self.year = 2013
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.840-6-201202-I/en'

        self._M = {}
        self._sigma = {}
        self._Pclw = {}

    def M(self, lat, lon):
        if not self._M:
            vals = load_data(os.path.join(dataset_dir,
                                          'p840/v4_WRED_LOGNORMAL_MEAN.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v4_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v4_Lon.txt'))
            self._M = bilinear_2D_interpolator(lats, lons, vals)

        return self._M(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def sigma(self, lat, lon):
        if not self._sigma:
            vals = load_data(os.path.join(dataset_dir,
                                          'p840/v4_WRED_LOGNORMAL_STDEV.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v4_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v4_Lon.txt'))
            self._sigma = bilinear_2D_interpolator(lats, lons, vals)

        return self._sigma(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def Pclw(self, lat, lon):
        if not self._Pclw:
            vals = load_data(os.path.join(dataset_dir,
                                          'p840/v4_WRED_LOGNORMAL_PCLW.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p840/v4_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v4_Lon.txt'))
            self._Pclw = bilinear_2D_interpolator(lats, lons, vals)

        return self._Pclw(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def lognormal_approximation_coefficient(self, lat, lon):
        m = self.M(lat, lon)
        sigma = self.sigma(lat, lon)
        Pclw = self.Pclw(lat, lon)

        return m, sigma, Pclw


__model = __ITU840()


def change_version(new_version):
    """
    Change the version of the ITU-R P.840 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * 7: P.840-7 (09/13) (Current version)
            * 6: P.840-6 
            * 5: P.840-5
            * 4: P.840-4
            
    """
    global __model
    __model = __ITU840(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.840 recommendation currently being used.
    """
    global __model
    return __model.__version__

@memory.cache
def lognormal_approximation_coefficient(lat, lon):
    """
    A method to estimate the paramerts of the lognormla distribution used to
    approximate the total columnar content of cloud liquid water


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points


    Returns
    -------
    - m: numpy.ndarray
            Mean of the lognormal distribution
    - sigma: numpy.ndarray
            Standard deviation of the lognormal distribution
    - Pclw: numpy.ndarray
            Probability of liquid water of the lognormal distribution



    References
    ----------
    [1] Attenuation due to clouds and fog:
    https://www.itu.int/rec/R-REC-P.840/en
    """
    type_output = type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.lognormal_approximation_coefficient(lat, lon)
    u_adim = u.dimensionless_unscaled
    return (prepare_output_array(val[0], type_output) * u_adim,
            prepare_output_array(val[1], type_output) * u_adim,
            prepare_output_array(val[2], type_output) * u_adim)

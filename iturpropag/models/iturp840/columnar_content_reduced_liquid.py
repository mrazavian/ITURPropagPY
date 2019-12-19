# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import warnings
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

        self._Lred = {}

    @property
    def __version__(self):
        return self.instance.__version__

    def columnar_content_reduced_liquid(self, lat, lon, p):
        # Abstract method to compute the columnar content of reduced liquid
        fcn = np.vectorize(self.instance.columnar_content_reduced_liquid,
                           excluded=[0, 1], otypes=[np.ndarray])
        return np.array(fcn(lat, lon, p).tolist())


class _ITU840_7():

    def __init__(self):
        self.__version__ = 7
        self.year = 2017
        self.month = 12
        self.link = 'https://www.p.int/rec/R-REC-P.840-7-201712-I/en'

        self._Lred = {}

    def Lred(self, lat, lon, p):
        if not self._Lred:
            ps = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30,
                  50, 60, 70, 80, 90, 95, 99]
            d_dir = os.path.join(dataset_dir, 'p840/v7_Lred_%s.txt')
            lats = load_data(os.path.join(dataset_dir, 'p840/v7_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v7_Lon.txt'))
            for p_load in ps:
                vals = load_data(d_dir % (str(p_load).replace('.', '')))
                self._Lred[float(p_load)] = bilinear_2D_interpolator(
                    lats, lons, vals)

        return self._Lred[float(p)](
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def columnar_content_reduced_liquid(self, lat, lon, p):
        """
        """
        if p > 99:
            warning_msg = 'maximum probability value for columnar content '+\
                    'reduced liquid water is 99%. Here the probability '+\
                    'is {:.2f}%. '.format(p)+\
                    'By default the probability is reset to 99%'
            p = 99
            warnings.warn(RuntimeWarning(warning_msg))

        available_p = np.array(
            [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0,
             60.0, 70.0, 80.0, 90.0, 95.0, 99.0])

        if p in available_p:
            p_below = p_above = p
            pExact = True
        else:
            pExact = False
            idx = available_p.searchsorted(p, side='right') - 1
            idx = np.clip(idx, 0, len(available_p))

            p_below = available_p[idx]
            p_above = available_p[idx + 1]

        # Compute the values of Lred_a
        Lred_a = self.Lred(lat, lon, p_above)
        if not pExact:
            Lred_b = self.Lred(lat, lon, p_below)
            Lred = Lred_b + (Lred_a - Lred_b) * (np.log(p) - np.log(p_below)) \
                / (np.log(p_above) - np.log(p_below))
            return Lred
        else:
            return Lred_a


class _ITU840_6():

    def __init__(self):
        self.__version__ = 6
        self.year = 2013
        self.month = 9
        self.link = 'https://www.p.int/rec/R-REC-P.840-6-201202-I/en'

        self._Lred = {}

    def Lred(self, lat, lon, p):
        if not self._Lred:
            ps = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30,
                  50, 60, 70, 80, 90, 95, 99]
            d_dir = os.path.join(dataset_dir, 'p840/v6_Lred_%s.txt')
            lats = load_data(os.path.join(dataset_dir, 'p840/v6_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v6_Lon.txt'))
            for p_load in ps:
                vals = load_data(d_dir % (str(p_load).replace('.', '')))
                self._Lred[float(p_load)] = bilinear_2D_interpolator(
                    lats, lons, vals)

        return self._Lred[float(p)](
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def columnar_content_reduced_liquid(self, lat, lon, p):
        """
        """
        if p > 99:
            warning_msg = 'maximum probability value for columnar content '+\
                    'reduced liquid water is 99%. Here the probability '+\
                    'is {:.2f}%. '.format(p)+\
                    'By default the probability is reset to 99%'
            p = 99
            warnings.warn(RuntimeWarning(warning_msg))

        available_p = np.array(
            [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0,
             60.0, 70.0, 80.0, 90.0, 95.0, 99.0])

        if p in available_p:
            p_below = p_above = p
            pExact = True
        else:
            pExact = False
            idx = available_p.searchsorted(p, side='right') - 1
            idx = np.clip(idx, 0, len(available_p))

            p_below = available_p[idx]
            p_above = available_p[idx + 1]

        # Compute the values of Lred_a
        Lred_a = self.Lred(lat, lon, p_above)
        if not pExact:
            Lred_b = self.Lred(lat, lon, p_below)
            Lred = Lred_b + (Lred_a - Lred_b) * (np.log(p) - np.log(p_below)) \
                / (np.log(p_above) - np.log(p_below))
            return Lred
        else:
            return Lred_a


class _ITU840_5():

    def __init__(self):
        self.__version__ = 5
        self.year = 2012
        self.month = 2
        self.link = 'https://www.p.int/rec/R-REC-P.840-5-201202-S/en'

        self._Lred = {}

    def Lred(self, lat, lon, p):
        if not self._Lred:
            ps = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30,
                  50, 60, 70, 80, 90, 95, 99]
            d_dir = os.path.join(dataset_dir, 'p840/v4_ESAWRED_%s.txt')
            lats = load_data(os.path.join(dataset_dir, 'p840/v4_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v4_Lon.txt'))
            for p_load in ps:
                vals = load_data(d_dir % (str(p_load).replace('.', '')))
                self._Lred[float(p_load)] = bilinear_2D_interpolator(
                    lats, lons, vals)

        return self._Lred[float(p)](
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def columnar_content_reduced_liquid(self, lat, lon, p):
        """
        """
        if p > 99:
            warning_msg = 'maximum probability value for columnar content '+\
                    'reduced liquid water is 99%. Here the probability '+\
                    'is {:.2f}%. '.format(p)+\
                    'By default the probability is reset to 99%'
            p = 99
            warnings.warn(RuntimeWarning(warning_msg))

        available_p = np.array(
            [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0,
             60.0, 70.0, 80.0, 90.0, 95.0, 99.0])

        if p in available_p:
            p_below = p_above = p
            pExact = True
        else:
            pExact = False
            idx = available_p.searchsorted(p, side='right') - 1
            idx = np.clip(idx, 0, len(available_p))

            p_below = available_p[idx]
            p_above = available_p[idx + 1]

        Lred_a = self.Lred(lat, lon, p_above)
        if not pExact:
            Lred_b = self.Lred(lat, lon, p_below)
            Lred = Lred_b + (Lred_a - Lred_b) * (np.log(p) - np.log(p_below)) \
                / (np.log(p_above) - np.log(p_below))
            return Lred
        else:
            return Lred_a


class _ITU840_4():

    def __init__(self):
        self.__version__ = 4
        self.year = 2013
        self.month = 9
        self.link = 'https://www.p.int/rec/R-REC-P.840-6-201202-I/en'

        self._Lred = {}
        
    def Lred(self, lat, lon, p):
        if not self._Lred:
            ps = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30,
                  50, 60, 70, 80, 90, 95, 99]
            d_dir = os.path.join(dataset_dir, 'p840/v4_ESAWRED_%s.txt')
            lats = load_data(os.path.join(dataset_dir, 'p840/v4_Lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p840/v4_Lon.txt'))
            for p_load in ps:
                vals = load_data(d_dir % (str(p_load).replace('.', '')))
                self._Lred[float(p_load)] = bilinear_2D_interpolator(
                    lats, lons, vals)

        return self._Lred[float(p)](
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def columnar_content_reduced_liquid(self, lat, lon, p):
        """
        """
        if p > 99:
            warning_msg = 'maximum probability value for columnar content '+\
                    'reduced liquid water is 99%. Here the probability '+\
                    'is {:.2f}%. '.format(p)+\
                    'By default the probability is reset to 99%'
            p = 99
            warnings.warn(RuntimeWarning(warning_msg))
            
        available_p = np.array(
            [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0,
             60.0, 70.0, 80.0, 90.0, 95.0, 99.0])

        if p in available_p:
            p_below = p_above = p
            pExact = True
        else:
            pExact = False
            idx = available_p.searchsorted(p, side='right') - 1
            idx = np.clip(idx, 0, len(available_p))

            p_below = available_p[idx]
            p_above = available_p[idx + 1]

        Lred_a = self.Lred(lat, lon, p_above)
        if not pExact:
            Lred_b = Lred_a = self.Lred(lat, lon, p_below)
            Lred = Lred_b + (Lred_a - Lred_b) * (np.log(p) - np.log(p_below)) \
                / (np.log(p_above) - np.log(p_below))
            return Lred
        else:
            return Lred_a


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
def columnar_content_reduced_liquid(lat, lon, p):
    """
    A method to compute the total columnar content of reduced cloud liquid
    water, Lred (kg/m2), exceeded for p% of the average year


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
    - Lred: numpy.ndarray
            Total columnar content of reduced cloud liquid water, Lred (kg/m2),
            exceeded for p% of the average year



    References
    ----------
    [1] Attenuation due to clouds and fog:
    https://www.p.int/rec/R-REC-P.840/en
    """
    global __model
    type_output = type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.columnar_content_reduced_liquid(lat, lon, p)
    return prepare_output_array(val, type_output) * u.kg / u.m**2

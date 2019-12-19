# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u
import scipy.stats as stats
from scipy.signal import lfilter

from iturpropag.models.iturp836.total_water_vapour_content import total_water_vapour_content
from iturpropag.models.iturp676.zenith_water_vapour_attenuation import zenith_water_vapour_attenuation
from iturpropag.utils import prepare_quantity



class __ITU1853():
    """Tropospheric attenuation time series synthesis

    Available versions include:
    * P.1853-0 (10/09) (Superseded)
    * P.1853-1 (02/12) (Superseded)
    * P.1853-2 (08/2019) (Current version)
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.1853 recommendation.

    def __init__(self, version=2):
        if version == 2:
            self.instance = _ITU1853_2()
        elif version == 1:
            self.instance = _ITU1853_1()
        elif version == 0:
            self.instance = _ITU1853_0()
        else:
            raise ValueError(
                'Version {0} is not implemented for the ITU-R P.1853 model.'
                .format(version))


    @property
    def __version__(self):
        return self.instance.__version__

    def integrated_water_vapour_coefficients(self, lat, lon, f):
        fcn = np.vectorize(self.instance.integrated_water_vapour_coefficients)
        return fcn(lat, lon, f)


class _ITU1853_2():

    def __init__(self):
        self.__version__ = 2
        self.year = 2019
        self.month = 8
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en'

    def integrated_water_vapour_coefficients(self, lat, lon, f):
        # Step A: Estimation of κ and λ (page=5)
        # SS_WV_1: Construct the pairs of [Pi, Awvi]
        ps = np.array([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50])
        Awvi = np.array([zenith_water_vapour_attenuation(lat, lon, p_i, \
                        f, V_t=None, h=None).value for p_i in ps])
        # SS_WV_2: Transform [Pi,Awvi]-->[ln(-ln(Pi/100)), ln(Awvi)]
        ln_lnPi = np.log(- np.log(ps / 100))
        ln_Awvi = np.log(Awvi)
        # SS_WV_3: determine the variables a, b
        a, b = np.linalg.lstsq(np.vstack([ln_lnPi, np.ones(len(ln_lnPi))]).T,
                               ln_Awvi, rcond=-1)[0]
        # SS_WV_4: parameters k_wv, lambda_wv
        kappa = 1/a
        lambd = np.exp(b)
        return kappa, lambd

class _ITU1853_1():

    def __init__(self):
        self.__version__ = 1
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-1-201202-I/en'

    
    def integrated_water_vapour_coefficients(self, lat, lon, f=None):
        # A Estimation of κ and λ
        ps = np.array([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50])
        Vi = np.array([total_water_vapour_content(lat, lon, p_i).value
                       for p_i in ps])

        ln_lnPi = np.log(- np.log(ps / 100))
        ln_Vi = np.log(Vi)

        a, b = np.linalg.lstsq(np.vstack([ln_Vi, np.ones(len(ln_Vi))]).T,
                               ln_lnPi, rcond=-1)[0]
        kappa = a
        lambd = np.exp(-b / a)
        return kappa, lambd


class _ITU1853_0():

    def __init__(self):
        self.__version__ = 0
        self.year = 2009
        self.month = 10
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-0-200910-I/en'

    def integrated_water_vapour_coefficients(self, *args, **kwargs):
        raise NotImplementedError(
            "Recommendation ITU-R P.1853-0 does not specify a method to compute "
            "time series for the water vapour content.")


__model = __ITU1853()


def change_version(new_version):
    """
    Change the version of the ITU-R P.1853 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.1853-1 (02/12)
            * P.1853-2 (08/2019) (Current version)
    """
    global __model
    __model = __ITU1853(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.1853 recommendation currently being used.
    """
    global __model
    return __model.__version__


def integrated_water_vapour_coefficients(lat, lon, f):
    """ The time series synthesis method generates a time series that
    reproduces the spectral characteristics and the distribution of water
    vapour content.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - f : number or Quantity
            Frequency (GHz).

    Returns
    -------
    - kappa:   numpy.ndarray
    - lambda:  numpy.ndarray
            parameters of the Weibull IWVC distribution


    References
    ----------
    [1] Characteristics of precipitation for propagation modelling
    https://www.itu.int/rec/R-REC-P.1853/en
    """
    global __model

    lon = np.mod(lon, 360)
    return __model.integrated_water_vapour_coefficients(lat, lon, f)

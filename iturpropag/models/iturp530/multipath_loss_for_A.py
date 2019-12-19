# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u
from scipy.optimize import bisect

from iturpropag.models.iturp453.DN65 import DN65
from iturpropag.models.iturp1144.bilinear_2D_interpolator import bilinear_2D_interpolator
from iturpropag.utils import prepare_input_array, prepare_quantity, load_data,\
    dataset_dir, prepare_output_array


class __ITU530():
    """Propagation data and prediction methods required for the design of
    terrestrial line-of-sight systems

    Available versions:
       * P.530-17 (07/15) (Current version)
       * P.530-16

    This recommendation includes prediction methods for the propagation effects
    that should be taken into account in the design of digital fixed
    line-of-sight links, both in clear-air and rainfall conditions. It also
    provides link design guidance in clear step-by-step procedures including
    the use of mitigation techniques to minimize propagation impairments. The
    final outage predicted is the base for other Recommendations addressing
    error performance and availability.
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.530 recommendation.

    def __init__(self, version=17):
        if version == 17:
            self.instance = _ITU530_17()
        elif version == 16:
            self.instance = _ITU530_16()
        else:
            raise ValueError(
                'Version ' +
                str(version) +
                ' is not implemented' +
                ' for the ITU-R P.530 model.')

    @property
    def __version__(self):
        return self.instance.__version__

    def multipath_loss_for_A(self, lat, lon, h_e, h_r, d, f, A):
        return self.instance.multipath_loss_for_A(lat, lon, h_e, h_r, d, f, A)


class _ITU530_17():

    def __init__(self):
        self.__version__ = 17
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.530-17-201712-S/en'

        self._s_a = {}

    def s_a(self, lat, lon):
        """ Standard deviation of terrain heights (m) within a 110 km × 110 km
        area with a 30 s resolution (e.g. the Globe “gtopo30” data).
        The value for the mid-path may be obtained from an area roughness
        with 0.5 × 0.5 degree resolution of geographical coordinates
        using bi-linear interpolation.
        """
        if not self._s_a:
            vals = load_data(os.path.join(dataset_dir, 'p530/v16_gtopo_30.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p530/v16_lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p530/v16_lon.txt'))
            self._Pr6 = bilinear_2D_interpolator(lats, lons, vals)

        return self._Pr6(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    ###########################################################################
    #                               Section 2.3                               #
    ###########################################################################
    
    def multipath_loss_for_A(self, lat, lon, h_e, h_r, d, f, A):
        """ Implementation of 'multipath_loss_for_A' method for recommendation
        ITU-P R.530-16. See documentation for function
        'ITUR530.multipath_loss_for_A'
        """

        # Step 1: Estimate the geoclimatic factor K
        # DN1 point refractivity gradient in the lowest 65 m of the atmospher
        # not exceeded for 1% of an average year
        # s_a is the area terrain roughness
        s_a = self.s_a(lat, lon)
        dN1 = DN65(lat, lon, 1).value
        K = 10**(-4.4 - 0.0027 * dN1) * (10 + s_a)**(-0.46)       # Eq. 4 [-]

        # Step 2: Claculate the magnitude of the path inclination
        # Eq. 5 [mrad]
        e_p = np.abs(h_r - h_e) / d

        # Step 3: For detailed link design applications calculate the
        # percentage of time (p_W) that fade depth A (dB) is exceeded in the
        # average worst month
        h_L = np.minimum(h_e, h_r)
        p_W = K * (d**3.4) * ((1 + e_p)**-1.03) * (f**0.8) * \
            10**(-0.00076 * h_L - A / 10)
        # Eq. 7 [%]
        return p_W


class _ITU530_16():

    def __init__(self):
        self.__version__ = 16
        self.year = 2015
        self.month = 7
        self.link = 'https://www.itu.int/rec/R-REC-P.530-16-201507-S/en'

        self._s_a = {}

    def s_a(self, lat, lon):
        """ 
        Standard deviation of terrain heights (m) within a 110 km × 110 km
        area with a 30 s resolution (e.g. the Globe “gtopo30” data).
        The value for the mid-path may be obtained from an area roughness
        with 0.5 × 0.5 degree resolution of geographical coordinates
        using bi-linear interpolation.
        """
        if not self._s_a:
            vals = load_data(os.path.join(dataset_dir, 'p530/v16_gtopo_30.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p530/v16_lat.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p530/v16_lon.txt'))
            self._Pr6 = bilinear_2D_interpolator(lats, lons, vals)

        return self._Pr6(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    ###########################################################################
    #                               Section 2.3                               #
    ###########################################################################
    def multipath_loss_for_A(self, lat, lon, h_e, h_r, d, f, A):
        """ 
        Implementation of 'multipath_loss_for_A' method for recommendation
        ITU-P R.530-16. See documentation for function
        'ITUR530.multipath_loss_for_A'
        """

        # Step 1: Estimate the geoclimatic factor K
        # DN1 point refractivity gradient in the lowest 65 m of the atmospher
        # not exceeded for 1% of an average year
        # s_a is the area terrain roughness
        s_a = self.s_a(lat, lon)
        dN1 = DN65(lat, lon, 1).value
        K = 10**(-4.4 - 0.0027 * dN1) * (10 + s_a)**(-0.46)       # Eq. 4 [-]

        # Step 2: Claculate the magnitude of the path inclination
        # Eq. 5 [mrad]
        e_p = np.abs(h_r - h_e) / d

        # Step 3: For detailed link design applications calculate the
        # percentage of time (p_W) that fade depth A (dB) is exceeded in the
        # average worst month
        h_L = np.minimum(h_e, h_r)
        p_W = K * (d**3.4) * ((1 + e_p)**-1.03) * (f**0.8) * \
            10**(-0.00076 * h_L - A / 10)
        # Eq. 7 [%]
        return p_W

__model = __ITU530()


def change_version(new_version):
    """
    Change the version of the ITU-R P.530 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.530-17 (07/15) (Current version)
            * P.530-16
    """
    global __model
    __model = __ITU530(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.530 recommendation currently being used.
    """
    global __model
    return __model.__version__


def multipath_loss_for_A(lat, lon, h_e, h_r, d, f, A):
    """ Method for predicting the single-frequency (or narrow-band) fading
    distribution at large fade depths in the average worst month in any part
    of the world. Given a fade depth value 'A', determines the amount of time
    it will be exceeded during a year

    This method does not make use of the path profile and can be used for
    initial planning, licensing, or design purposes.

    This method is only valid for small percentages of time.

    Multipath fading and enhancement only need to be calculated for path
    lengths longer than 5 km, and can be set to zero for shorter paths.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - h_e : number
            Emitter antenna height (above the sea level) [m]
    - h_r : number
            Receiver antenna height (above the sea level) [m]
    - d : number, sequence, or numpy.ndarray
            Distances between antennas [km]
    - f : number
            Frequency of the link [GHz]
    - A : number
            Fade depth [dB]


    Returns
    -------
    - p_w: Quantity
            percentage of time that fade depth A is exceeded in the average
            worst month  [%]


    References
    ----------
    [1] Propagation data and prediction methods required for the design of
    terrestrial line-of-sight systems: https://www.itu.int/rec/R-REC-P.530/en
    """
    global __model
    type_output = type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    h_e = prepare_quantity(
        h_e, u.m, 'Emitter antenna height (above sea level)')
    h_r = prepare_quantity(
        h_r, u.m, 'Receiver antenna height (above sea level)')
    d = prepare_quantity(d, u.km, 'Distance between antennas')
    f = prepare_quantity(f, u.GHz, 'Frequency')
    A = prepare_quantity(A, u.dB, 'Fade depth')

    val = __model.multipath_loss_for_A(lat, lon, h_e, h_r, d, f, A)
    return prepare_output_array(val, type_output) * u.percent


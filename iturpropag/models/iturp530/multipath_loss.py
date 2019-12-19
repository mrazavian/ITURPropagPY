# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u
from scipy.optimize import bisect

from iturpropag.models.iturp530.multipath_loss_for_A import multipath_loss_for_A
from iturpropag.utils import prepare_input_array, prepare_quantity, load_data,\
    dataset_dir, prepare_output_array


class __ITU530():
    """Propagation data and prediction methods required for the design of
    terrestrial line-of-sight systems

    Available versions:
       * P.530-17 (07/15) (Current version)
       * P.530-16

    Not available versions:

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

    def multipath_loss(self, lat, lon, h_e, h_r, d, f, A):
        return self.instance.multipath_loss(lat, lon, h_e, h_r, d, f, A)


class _ITU530_17():

    def __init__(self):
        self.__version__ = 17
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.530-17-201712-S/en'

    ###########################################################################
    #                               Section 2.3                               #
    ###########################################################################
    def multipath_loss(self, lat, lon, h_e, h_r, d, f, A):
        """ Implementation of 'multipath_loss' method for recommendation
        ITU-P R.530-16. See documentation for function
        'ITUR530.multipath_loss'
        """
        # Step 1: Using the method multipath_loss_for_A calculate the
        # multipath occurrence factor, p0
        p0 = multipath_loss_for_A(
            lat, lon, h_e, h_r, d, f, 0).value   # Eq. 10 [%]

        # Step 2: Calculate the value of fade depth, At, at which the
        # transition occurs between the deep-fading distribution and the
        # shallow-fading distribution
        At = 25 + 1.2 * np.log10(p0)                        # Eq. 12 [dB]

        # Step 3: Calculate the percentage of time that A is exceeded in the
        # average worst month:
        def step_3b(p_0, At, A):
            p_t = p_0 * 10 ** (-At / 10)
            qa_p = -20 * np.log10(-np.log((100 - p_t) / 100)) / At
            q_t = ((qa_p - 2) /
                   (1 + 0.3 * 10 ** (-At / 20) * 10 ** (-0.016 * At)) -
                   4.3 * (10**(-At / 20) + At / 800))
            q_a = 2 + (1 + 0.3 * 10**(-A / 20)) * (10**(-0.016 * A)) *\
                (q_t + 4.3 * (10**(-A / 20 + A / 800)))
            p_W = 100 * (1 - np.exp(-10 ** (-q_a * A / 20)))
            return p_W

        p_W = np.where(A >= At, p0 * 10 ** (-A / 10), step_3b(p0, At, A))
        # Eq. 13 and Eq. 18 [%]
        return p_W


class _ITU530_16():

    def __init__(self):
        self.__version__ = 16
        self.year = 2015
        self.month = 7
        self.link = 'https://www.itu.int/rec/R-REC-P.530-16-201507-S/en'

    ###########################################################################
    #                               Section 2.3                               #
    ###########################################################################
    def multipath_loss(self, *args, **kwargs):
        return _ITU530_17.multipath_loss(*args, **kwargs)


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


def multipath_loss(lat, lon, h_e, h_r, d, f, A):
    """ Method for predicting the percentage of time that any fade depth is
    exceeded. This method combines the deep fading distribution given in the
    multipath_loss_for_A' and an empirical interpolation procedure for shallow
    fading down to 0 dB.

    This method does not make use of the path profile and can be used for
    initial planning, licensing, or design purposes.

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

    val = __model.multipath_loss(lat, lon, h_e, h_r, d, f, A)
    return prepare_output_array(val, type_output) * u.percent


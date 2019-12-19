# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u
from scipy.optimize import bisect

from iturpropag.models.iturp530.multipath_loss_for_A import  multipath_loss_for_A
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

    def XPD_outage_clear_air(self, lat, lon, h_e, h_r,
                             d, f, XPD_g, C0_I, XPIF):
        return self.instance.XPD_outage_clear_air(lat, lon, h_e, h_r, d, f,
                                                  XPD_g, C0_I, XPIF)


class _ITU530_17():

    def __init__(self):
        self.__version__ = 17
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.530-17-201712-S/en'

    ###########################################################################
    #                                Section 4                                #
    ###########################################################################
    def XPD_outage_clear_air(self, lat, lon, h_e, h_r,
                             d, f, XPD_g, C0_I, XPIF):
        """ Implementation of 'XPD_outage_clear_air' method for recommendation
        ITU-P R.530-16. See documentation for function
        'ITUR530.XPD_outage_clear_air'
        """
        # Step 1
        XPD_0 = np.where(XPD_g <= 35, XPD_g + 5, 40)            # Eq. 101

        # Step 2: Evaluate the multipath activity parameter:
        P0 = multipath_loss_for_A(lat, lon, h_e, h_r, d, f, 0).value
        eta = 1 - np.exp(-0.2 * P0**0.75)                      # Eq. 102

        # Step 3:
        kXP = 0.7                                               # Eq. 104
        Q = - 10 * np.log10(kXP * eta / P0)                     # Eq. 103

        # Step 4: Derive the parameter C:
        C = XPD_0 + Q                                           # Eq. 105

        # Step 5: Calculate the probability of outage PXP due to clear-air
        # cross-polarization:
        M_XPD = C - C0_I + XPIF
        P_XP = P0 * 10 ** (- M_XPD / 10)                       # Eq. 106 [%]
        return P_XP


class _ITU530_16():

    def __init__(self):
        self.__version__ = 16
        self.year = 2015
        self.month = 7
        self.link = 'https://www.itu.int/rec/R-REC-P.530-16-201507-S/en'

    ###########################################################################
    #                                Section 4                                #
    ###########################################################################
    def XPD_outage_clear_air(self, *args, **kwargs):
        return _ITU530_17.XPD_outage_clear_air(*args, **kwargs)

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


def XPD_outage_clear_air(lat, lon, h_e, h_r, d, f, XPD_g, C0_I, XPIF):
    """ Estimate the probability of outage due to cross-polar discrimnation
    reduction due to clear-air effects, assuming that a targe C0_I is
    required.


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
    - XPD_g : number
            Manufacturer's guaranteed minimum XPD at boresight for both the
            transmitting and receiving antennas [dB]
    - C0_I : number
            Carrier-to-interference ratio for a reference BER [dB]
    - XPIF : number, optional
            Laboratory-measured cross-polarization improvement factor that gives
            the difference in cross-polar isolation (XPI) at sufficiently large
            carrier-to-noise ratio (typically 35 dB) and at a specific BER for
            systems with and without cross polar interference canceller (XPIC).
            A typical value of XPIF is about 20 dB. value 0 dB (no XPIC)
            [dB]


    Returns
    -------
    - p_XP: Quantity
            Probability of outage due to clear-air cross-polarization


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
    XPD_g = prepare_quantity(XPD_g, u.dB, 'Manufacturers minimum XPD')
    C0_I = prepare_quantity(C0_I, u.dB, 'Carrier-to-interference ratio')
    XPIF = prepare_quantity(
        XPIF, u.dB, 'Cross-polarization improvement factor')

    val = __model.XPD_outage_clear_air(
        lat, lon, h_e, h_r, d, f, XPD_g, C0_I, XPIF)
    return prepare_output_array(val, type_output) * u.percent

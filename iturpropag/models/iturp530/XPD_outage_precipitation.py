# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u
from scipy.optimize import bisect

from iturpropag.models.iturp530.rain_attenuation import rain_attenuation
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

    def XPD_outage_precipitation(self, lat, lon, d, f, el, C0_I, tau,
                                 U0, XPIF):
        return self.instance.XPD_outage_precipitation(lat, lon, d, f, el, C0_I,
                                                      tau, U0, XPIF)


class _ITU530_17():

    def __init__(self):
        self.__version__ = 17
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.530-17-201712-S/en'

    ###########################################################################
    #                                Section 4                                #
    ###########################################################################
    def XPD_outage_precipitation(self, lat, lon, d, f, el, C0_I, tau,
                                 U0, XPIF):
        """ Implementation of 'XPD_outage_precipitation' method for recommendation
        ITU-P R.530-16. See documentation for function
        'ITUR530.XPD_outage_precipitation'
        """
        # Step 1: Determine the path attenuation, A0.01 (dB), exceeded
        # for 0.01% of the time
        A001 = rain_attenuation(lat, lon, d, f, el, 0.01, tau).value

        # Step 2: Determine the equivalent path attenuation, Ap
        U = U0 + 30 * np.log10(f)
        V = np.where(np.logical_and(8 <= f, f <= 20), 12.8 * f**0.19,
            np.where(np.logical_and(20 < f, f <= 35), 22.6, np.nan))
        
        Ap = 10 ** ((U - C0_I + XPIF) / V)                      # Eq. 112

        # Step 3: Determine parameters m and n
        m = min(23.26 * np.log10(Ap / (0.12 * A001)), 40)      # Eq. 113
        n = (-12.7 + np.sqrt(161.23 - 4 * m)) / 2              # Eq. 114

        # Step 4 : Determine the outage probability
        P_XPR = 10**(n - 2)                                     # Eq. 115 [%]
        return P_XPR


class _ITU530_16():

    def __init__(self):
        self.__version__ = 16
        self.year = 2015
        self.month = 7
        self.link = 'https://www.itu.int/rec/R-REC-P.530-16-201507-S/en'

    def XPD_outage_precipitation(self, *args, **kwargs):
        return _ITU530_17.XPD_outage_precipitation(*args, **kwargs)


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


def XPD_outage_precipitation(lat, lon, d, f, el, C0_I, tau,
                             U0=15, XPIF=0):
    """ Estimate the probability of outage due to cross-polar discrimnation
    reduction due to clear-air effects, assuming that a targe C0_I is
    required.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - d : number, sequence, or numpy.ndarray
            Distances between antennas [km]
    - f : number
            Frequency of the link [GHz] (frequency should be 8<= f<=35 GHz)
    - el : sequence, or number
            Elevation angle (degrees)
    - C0_I : number
            Carrier-to-interference ratio for a reference BER [dB]
    - tau : number, optional
            Polarization tilt angle relative to the horizontal (degrees)
            (tau = 45 deg for circular polarization). Default value is 45
    - U0 : number, optional
            Coefficient for the cumulative distribution of the co-polar attenuation
            (CPA) for rain. Default 15 dB.
    - XPIF : number, optional
            Laboratory-measured cross-polarization improvement factor that gives
            the difference in cross-polar isolation (XPI) at sufficiently large
            carrier-to-noise ratio (typically 35 dB) and at a specific BER for
            systems with and without cross polar interference canceller (XPIC).
            A typical value of XPIF is about 20 dB. Default value 0 dB (no XPIC)
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
    d = prepare_quantity(d, u.km, 'Distance between antennas')
    f = prepare_quantity(f, u.GHz, 'Frequency')
    C0_I = prepare_quantity(C0_I, u.dB, 'Carrier-to-interference ratio')
    U0 = prepare_quantity(U0, u.dB, 'Coefficient for the CPA')
    XPIF = prepare_quantity(
        XPIF, u.dB, 'Cross-polarization improvement factor')

    val = __model.XPD_outage_precipitation(lat, lon, d, f, el, C0_I, tau, U0, XPIF)
    return prepare_output_array(val, type_output) * u.percent

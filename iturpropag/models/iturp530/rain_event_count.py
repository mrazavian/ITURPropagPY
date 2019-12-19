# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u
from scipy.optimize import bisect

from iturpropag.models.iturp530.inverse_rain_attenuation import inverse_rain_attenuation
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

    def rain_event_count(self, lat, lon, d, f, el, A, tau, R001=None):
        return self.instance.rain_event_count(lat, lon, d, f, el, A, tau, R001)


class _ITU530_17():

    def __init__(self):
        self.__version__ = 17
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.530-17-201712-S/en'
    ###########################################################################
    #                               Section 2.4                               #
    ###########################################################################
    def rain_event_count(self, lat, lon, d, f, el, A, tau, R001=None):
        """ Implementation of 'rain_event_count' method for recommendation
        ITU-P R.530-16. See documentation for function
        'ITUR530.rain_event_count'
        """
        # Compute the the percentage of time that the rain attenuation A(dB)
        # exceeded in the average year.
        p_A = inverse_rain_attenuation(lat, lon, d, f, el, A, tau).value

        # The number of fade events exceeding attenuation A for 10 s or longer
        N10s = 1 + 1313 * p_A**0.945                               # Eq. 78 [-]

        return N10s


class _ITU530_16():

    def __init__(self):
        self.__version__ = 16
        self.year = 2015
        self.month = 7
        self.link = 'https://www.itu.int/rec/R-REC-P.530-16-201507-S/en'

    ###########################################################################
    #                               Section 2.4                               #
    ###########################################################################
    def rain_event_count(self, *args, **kwargs):
        return _ITU530_17.rain_event_count(*args, **kwargs)


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


def rain_event_count(lat, lon, d, f, el, Ap, tau, R001=None):
    """ Estimate the number of fade events exceeding attenuation 'A'
    for 10 seconds or longer.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - d : number, sequence, or numpy.ndarray
            Path length [km]
    - f : number
            Frequency of the link [GHz]
    - el : sequence, or number
            Elevation angle (degrees)
    - Ap : number
            Fade depth
    - R001: number, optional
            Point rainfall rate for the location for 0.01% of an average year
            (mm/h). If not provided, an estimate is obtained from Recommendation
            Recommendation ITU-R P.837. Some useful values:
                * 0.25 mm/h : Drizzle
                *  2.5 mm/h : Light rain
                * 12.5 mm/h : Medium rain
                * 25.0 mm/h : Heavy rain
                * 50.0 mm/h : Downpour
                * 100  mm/h : Tropical
                * 150  mm/h : Monsoon
    - tau : number, optional
            Polarization tilt angle relative to the horizontal (degrees)
            (tau = 45 deg for circular polarization). Default value is 45


    Returns
    -------
    - p: Quantity
            Percentage of time that the attenuation A is exceeded.


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
    el = prepare_quantity(el, u.deg, 'Elevation Angle')
    Ap = prepare_quantity(Ap, u.dB, 'Fade depth')
    R001 = prepare_quantity(R001, u.mm / u.hr, 'Rainfall Rate')

    val = __model.rain_event_count(lat, lon, d, f, el, Ap, tau, R001)
    return prepare_output_array(val, type_output) * u.percent

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u
from scipy.optimize import bisect

from iturpropag.models.iturp837.rainfall_rate import rainfall_rate
from iturpropag.models.iturp838.rain_specific_attenuation import rain_specific_attenuation
from iturpropag.models.iturp838.rain_specific_attenuation_coefficients import rain_specific_attenuation_coefficients
from iturpropag.utils import prepare_input_array, prepare_quantity, load_data,\
    dataset_dir, prepare_output_array


class __ITU530():
    """Propagation data and prediction methods required for the design of
    terrestrial line-of-sight systems

    Available versions:
       * P.530-17 (07/15) (Current version)
       * P.530-17=6

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

    def inverse_rain_attenuation(self, lat, lon, d, f, el,
                                 Ap, tau, R001=None):
        return self.instance.inverse_rain_attenuation(lat, lon, d, f, el,
                                                      Ap, tau, R001)


class _ITU530_17():

    def __init__(self):
        self.__version__ = 17
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.530-17-201712-S/en'

    ###########################################################################
    #                               Section 2.4                               #
    ###########################################################################
    def inverse_rain_attenuation(
            self, lat, lon, d, f, el, Ap, tau, R001=None):
        """ Implementation of 'inverse_rain_attenuation' method for
        recommendation ITU-P R.530-16. See documentation for function
        'ITUR530.inverse_rain_attenuation'
        """
        # Step 1: Obtain the rain rate R0.01 exceeded for 0.01% of the time
        # (with an integration time of 1 min).
        if R001 is None:
            R001 = rainfall_rate(lat, lon, 0.01)

        # Step 2: Compute the specific attenuation, gammar (dB/km) for the
        # frequency, polarization and rain rate of interest using
        # Recommendation ITU-R P.838
        gammar = rain_specific_attenuation(R001, f, el, tau).value
        _, alpha = rain_specific_attenuation_coefficients(f, el, tau).value

        # Step 3: Compute the effective path length, deff, of the link by
        # multiplying the actual path length d by a distance factor r
        r = 1 / (0.477 * (d ** 0.633) * (R001 ** (0.073 * alpha)) *
                 f**(0.123) - 10.579 * (1 - np.exp(-0.024 * d)))
        deff = np.minimum(r, 2.5)

        # Step 4: An estimate of the path attenuation exceeded for 0.01% of
        # the time is given by:
        A001 = gammar * deff

        # Step 5: The attenuation exceeded for other percentages of time p in
        # the range 0.001% to 1% may be deduced from the following power law
        C0 = np.where(f >= 10, 0.12 + 0.4 * (np.log10(f / 10)**0.8), 0.12)
        C1 = (0.07**C0) * (0.12**(1 - C0))
        C2 = 0.855 * C0 + 0.546 * (1 - C0)
        C3 = 0.139 * C0 + 0.043 * (1 - C0)

        def func_bisect(p):
            return A001 * C1 * p ** (- (C2 + C3 * np.log10(p))) - Ap

        return bisect(func_bisect, 0, 100)


class _ITU530_16():

    def __init__(self):
        self.__version__ = 16
        self.year = 2015
        self.month = 7
        self.link = 'https://www.itu.int/rec/R-REC-P.530-16-201507-S/en'

    ###########################################################################
    #                               Section 2.4                               #
    ###########################################################################
    def inverse_rain_attenuation(self, *args, **kwargs):
        return _ITU530_17.inverse_rain_attenuation(*args, **kwargs)


__model = __ITU530()


def change_version(new_version):
    """
    Change the version of the ITU-R P.530 recommendation currently being used.


    Parameters
    ----------
    new_version : int
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


def inverse_rain_attenuation(lat, lon, d, f, el, Ap, tau, R001=None):
    """ Estimate the percentage of time a given attenuation is exceeded due to
    rain events.


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
                * 0.25 mm/h : Drizle
                *  2.5 mm/h : Light rain
                * 12.5 mm/h : Medium rain
                * 25.0 mm/h : Heavy rain
                * 50.0 mm/h : Dwonpour
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

    val = __model.inverse_rain_attenuation(lat, lon, d, f, el, Ap, tau, R001)
    return prepare_output_array(val, type_output) * u.percent

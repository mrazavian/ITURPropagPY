# -*- coding: utf-8 -*-
# checked by <TP>: 2019-06-07
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as stats
import scipy.special
import scipy.integrate
from astropy import units as u
import warnings

from iturpropag.utils import prepare_input_array, prepare_output_array,\
    prepare_quantity, compute_distance_earth_to_earth, memory


class _ITU618():
    """
    Propagation data and prediction methods required for the design of
    Earth-space telecommunication systems.

    Available versions include:
       * P.618-13 (12/17) (Current version)
       * P.618-12 (07/15) (Superseded)

    Versions that need to be implemented
       * P.618-11
       * P.618-10
       * P.618-09
       * P.618-08
       * P.618-07
       * P.618-06
       * P.618-05
       * P.618-04
       * P.618-03
       * P.618-02
       * P.618-01

    Recommendation ITU-R P.618 provides methods to estimate the propagation
    loss on an Earth-space path, relative to the free-space loss. This value
    is the sum of different contributions as follows:
    * attenuation by atmospheric gases;
    * attenuation by rain, other precipitation and clouds;
    * focusing and defocusing;
    * decrease in antenna gain due to wave-front incoherence;
    * scintillation and multipath effects;
    * attenuation by sand and dust storms.
    Each of these contributions has its own characteristics as a function of
    frequency, geographic location and elevation angle. As a rule, at elevation
    angles above 10°, only gaseous attenuation, rain and cloud attenuation and
    possibly scintillation will be significant, depending on propagation
    conditions.
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.618 recommendation.

    def __init__(self, version=13):
        if version == 13:
            self.instance = _ITU618_13()
        elif version == 12:
            self.instance = _ITU618_12()
        else:
            err_msg = 'Version {0} is not implemented for the ITU-R P.618 model.'
            raise ValueError(err_msg.format(version))

    @property
    def __version__(self):
        return self.instance.__version__

    def rain_cross_polarization_discrimination(self, Ap, f, el, p, tau):
        fcn = np.vectorize(
            self.instance.rain_cross_polarization_discrimination)
        return fcn(Ap, f, el, p, tau)


class _ITU618_13():

    def __init__(self):
        self.__version__ = 13
    
    def rain_cross_polarization_discrimination(self, Ap, f, el, p, tau):
        # Frequency reuse by means of orthogonal polarizations is often used to
        # increase the capacity of space telecommunication systems. This
        # technique is restricted, however, by depolarization on atmospheric
        # propagation paths. Various depolarization mechanisms, especially
        # hydrometeor effects, are important in the troposphere

        # The method described below to calculate cross-polarization
        # discrimination (XPD) statistics from rain attenuation statistics for
        # the same path is valid for 6 < f < 55 GHz and el < 60°.
        if f < 4 or f > 55:
            warning_msg = '\nThe method to compute the cross '+\
                          'polarization discrimination in re    commendation '+\
                          'ITU-P 618-13 is only valid for frequency values between'+\
                          ' 4 and 55 GHz. here the frequency is ' + str(f) + 'GHz.'
            warnings.warn(RuntimeWarning(warning_msg))

        if el >= 60:
            warning_msg = '\nThe method to compute the cross '+\
                          'polarization discrimination in recommendation ITU-P '+\
                          '618-13 is only valid for elevation angle values below '+\
                          '60 degrees. here the elevation is '+ str(el)+ ' degree.'
            warnings.warn(RuntimeWarning(warning_msg))
        
        # In case that the frequency is comprised between 4 and 6 GHz, scaling
        # is necessary
        scale_to_orig_f = False
        if 4 <= f < 6:
            f_orig = f
            f = 6.
            scale_to_orig_f = True

        # Step 1: Calculate the frequency-dependent term: Eq. 65
        if 6 <= f < 9:
            C_f = 60. * np.log10(f) - 28.3
        elif 9 <= f < 36:
            C_f = 26. * np.log10(f) + 4.1
        elif 36 <= f <= 55:
            C_f = 35.9 * np.log10(f) - 11.3

        # Step 2: Calculate the rain attenuation dependent term:
        if 6 <= f < 9:
            V = 30.8 * f**-0.21
        elif 9 <= f < 20:
            V = 12.8 * f**0.19
        elif 20 <= f < 40:
            V = 22.6
        elif 40 <= f <= 55:
            V = 13.0 * f**0.15

        C_a = V * np.log10(Ap)                          # Eq. 66

        # Step 3: Calculate the polarization improvement factor: Eq. 67
        C_tau = -10. * np.log10(1. - 0.484 * (1. + np.cos(np.deg2rad(4. * tau))))

        # Step 4: Calculate the elevation angle-dependent term: Eq. 68
        C_theta = -40. * np.log10(np.cos(np.deg2rad(el)))

        # Step 5: Calculate the canting angle dependent term: Eq. 79
        if p <= 0.001:
            C_sigma = 0.0053 * 15.**2
        elif p <= 0.01:
            C_sigma = 0.0053 * 10.**2
        elif p <= 0.1:
            C_sigma = 0.0053 * 5.**2
        else:
            C_sigma = 0.

        # Step 6: Calculate rain XPD not exceeded for p% of the time: Eq. 70
        XPD_rain = C_f - C_a + C_tau + C_theta + C_sigma

        # Step 7: Calculate the ice crystal dependent term: Eq. 71
        C_ice = XPD_rain * (0.3 + 0.1 * np.log10(p)) / 2.

        # Step 8: Calculate the XPD not exceeded for p% of the time,
        # including the effects of ice:                     Eq. 72
        XPD_p = XPD_rain - C_ice

        if scale_to_orig_f:
            # Long-term XPD statistics obtained at one frequency and
            # polarization tilt angle can be scaled to another frequency and
            # polarization tilt angle using the semi-empirical formula:
            # Eq. 73, with tilt angles tau1 == tau2
            XPD_p = XPD_p - 20. * np.log10(
              f_orig * np.sqrt(1. - 0.484 * (1. + np.cos(np.deg2rad(4. * tau)))) /
              (f * np.sqrt(1. - 0.484 * (1. + np.cos(np.deg2rad(4. * tau))))))
        return XPD_p


class _ITU618_12():

    def __init__(self):
        self.__version__ = 12

    def rain_cross_polarization_discrimination(self, *args, **kwargs):
        return _ITU618_13.rain_cross_polarization_discrimination(*args,
                                                                 **kwargs)


__model = _ITU618()


def change_version(new_version):
    global __model
    __model = _ITU618(new_version)
    memory.clear()


def get_version():
    global __model
    return __model.__version__


def rain_cross_polarization_discrimination(Ap, f, el, p, tau):
    """
    Calculation of the cross-polarization discrimination (XPD) statistics from
    rain attenuation statistics. The following procedure provides estimates of
    the long-term statistics of the cross-polarization discrimination (XPD)
    statistics for frequencies up to 55 GHz and elevation angles lower than 60
    deg.


    Parameters
    ----------
    - Ap : number, sequence, or numpy.ndarray
            Rain attenuation (dB) exceeded for the required percentage of time, p,
            for the path in question, commonly called co-polar attenuation (CPA)
    - f : number
            Frequency (GHz)
    - el : number, sequence, or numpy.ndarray
            Elevation angle (degrees)
    - p : number
            Percetage of the time the XPD is exceeded.
    - tau : number
            Polarization tilt angle relative to the horizontal (degrees)
            (tau = 45 deg for circular polarization).


    Returns
    -------
    - XPD: Quantity
            Cross-polarization discrimination (dB)


    References
    ----------
    [1] Propagation data and prediction methods required for the design of
    Earth-space telecommunication systems:
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-12-201507-I!!PDF-E.pdf
    """
    global __model
    type_output = type(Ap)
    Ap = prepare_input_array(Ap)
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(el, u.deg, 'Elevation angle')
    tau = prepare_quantity(tau, u.one, 'Polarization tilt angle')
    val = __model.rain_cross_polarization_discrimination(Ap, f, el, p, tau)
    return prepare_output_array(val, type_output) * u.dB


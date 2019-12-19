# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
import os
from astropy import units as u

from iturpropag.models.iturp676.slant_inclined_path_equivalent_height import slant_inclined_path_equivalent_height
from iturpropag.models.iturp676.gamma0_exact import gamma0_exact
from iturpropag.models.iturp676.gammaw_exact import gammaw_exact
from iturpropag.models.iturp676.gamma0_approx import gamma0_approx
from iturpropag.models.iturp676.gammaw_approx import gammaw_approx
from iturpropag.utils import prepare_quantity, prepare_output_array,\
    prepare_input_array, load_data, dataset_dir, memory


class __ITU676():
    """Attenuation by atmospheric gases.

    Available versions include:
       * P.676-9 (02/12) (Superseded)
       * P.676-10 (09/13) (Superseded)
       * P.676-11 (09/16) (Current version)
    Not available versions:
       * P.676-1 (03/92) (Superseded)
       * P.676-2 (10/95) (Superseded)
       * P.676-3 (08/97) (Superseded)
       * P.676-4 (10/99) (Superseded)
       * P.676-5 (02/01) (Superseded)
       * P.676-6 (03/05) (Superseded)
       * P.676-7 (02/07) (Superseded)
       * P.676-8 (10/09) (Superseded)
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.676 recommendation.

    def __init__(self, version=11):
        if version == 11:
            self.instance = _ITU676_11()
        elif version == 10:
            self.instance = _ITU676_10()
        elif version == 9:
            self.instance = _ITU676_9()
        else:
            raise ValueError(
                'Version {0} is not implemented for the ITU-R P.676 model.'
                .format(version))

    @property
    def __version__(self):
        return self.instance.__version__

    def gaseous_attenuation_inclined_path(self, f, el, rho, P, T, h1, h2, mode):
        # Abstract method to compute the gaseous attenuation over a slant path
        fcn = np.vectorize(self.instance.gaseous_attenuation_inclined_path)
        return fcn(f, el, rho, P, T, h1, h2, mode)


class _ITU676_11():

    def __init__(self):
        self.__version__ = 11
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.676-11-201712-S/en'

    def gamma_exact(self, f, P, rho, T):
        gamma0 = gamma0_exact(f, P, rho, T).value
        gammaw = gammaw_exact(f, P, rho, T).value
        return gamma0, gammaw

    def gaseous_attenuation_approximation(self, f, el, rho, P, T):
        """
        T goes in Kelvin
        """
        if np.any(f > 350):
            warning_msg = 'The approximated method to computes '+\
                          'the gaseous attenuation in recommendation ITU-P 676-11 '+\
                          'is only recommended for frequencies below 350GHz. '\
                            'Current frequency is '+ str(f) + 'GHz'
            warnings.warn(RuntimeWarning(warning_msg))

        if np.any(5 > el) or np.any(np.mod(el, 90) < 5):
            warning_msg = 'The approximated method to compute '+\
                          'the gaseous attenuation in recommendation ITU-P 676-11 '+\
                          'is only recommended for elevation angles between '+\
                          '5 and 90 degrees. Current elevation is '+ str(el)
            warnings.warn(RuntimeWarning(warning_msg))

        # Water vapour attenuation (gammaw) computation as in Section 1 of
        # Annex 2 of [1]
        gamma0 = gamma0_approx(f, P, rho, T).value
        gammaw = gammaw_approx(f, P, rho, T).value

        return gamma0, gammaw

    def gaseous_attenuation_inclined_path(
            self, f, el, rho, P, T, h1, h2, mode):
        """
        section 2.2.1 and 2.2.2 @ ITU-R P.676-11
        """
        if h1 > 10 or h2 > 10:
            raise ValueError(
                'Both the transmitter and the receiver must be at'
                'altitude of less than 10 km above the sea level.'
                'Current altitude Tx: %.2f km, Rx: %.2f km' % (h1, h2))

        if mode == 'approx':
            rho = rho * np.exp(h1 / 2)
            gamma0, gammaw = self.gaseous_attenuation_approximation(
                f, el, rho, P, T)
        else:
            gamma0, gammaw = self.gamma_exact(f, P, rho, T)

        h0, hw = slant_inclined_path_equivalent_height(f, P).to(u.km).value

        if 5 <= el and el <= 90:
            h0_p = h0 * (np.exp(-h1 / h0) - np.exp(-h2 / h0))
            hw_p = hw * (np.exp(-h1 / hw) - np.exp(-h2 / hw))
            return (gamma0 * h0_p + gammaw * hw_p) / np.sin(np.deg2rad(el))
        else:
            def F(x):
                return 1 / (0.661 * x + 0.339 * np.sqrt(x**2 + 5.51))

            Re = 8500   # TODO should be calculated by ITU-R P.834
            el1 = np.deg2rad(el)
            el2 = np.arccos((Re + h1) / (Re + h2) * np.cos(el1))
             

            def xi(eli, hi):
                return np.tan(eli) * np.sqrt((Re + hi) / h0)

            def xi_p(eli, hi):
                return np.tan(eli) * np.sqrt((Re + hi) / hw)

            def eq_33(h_num, h_den, el, x):
                return np.sqrt(Re + h_num) * F(x) * \
                    np.exp(-h_num / h_den) / np.cos(el)

            A = gamma0 * np.sqrt(h0) * (eq_33(h1, h0, el1, xi(el1, h1)) -
                                        eq_33(h2, h0, el2, xi(el2, h2))) +\
                gammaw * np.sqrt(hw) * (eq_33(h1, hw, el1, xi_p(el1, h1)) -
                                        eq_33(h2, hw, el2, xi_p(el2, h2)))
            return A


class _ITU676_10():

    def __init__(self):
        self.__version__ = 10
        self.year = 2013
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.676-10-201309-S/en'

    def gamma_exact(self, f, P, rho, T):
        gamma0 = gamma0_exact(f, P, rho, T).value
        gammaw = gammaw_exact(f, P, rho, T).value
        return gamma0, gammaw

    def gaseous_attenuation_approximation(self, f, el, rho, P, T):
        """
        T goes in Kelvin
        """
        if np.any(f > 350):
            warning_msg = 'The approximated method to computes '+\
                          'the gaseous attenuation in recommendation ITU-P 676-10 '+\
                          'is only recommended for frequencies below 350GHz. Current '+\
                          'frequency is '+ str(f)
            warnings.warn(RuntimeWarning(warning_msg))

        if np.any(5 > el) or np.any(np.mod(el, 90) < 5):
            warning_msg = 'The approximated method to compute '+\
                          'the gaseous attenuation in recommendation ITU-P 676-11 '+\
                          'is only recommended for elevation angles between'+\
                          '5 and 90 degrees. Current elevation is '+ str(el)
            warnings.warn(RuntimeWarning(warning_msg))

        # Water vapour attenuation (gammaw) computation as in Section 1 of
        # Annex 2 of [1]
        gamma0 = gamma0_approx(f, P, rho, T).value
        gammaw = gammaw_approx(f, P, rho, T).value

        return gamma0, gammaw

    def gaseous_attenuation_inclined_path(
            self, f, el, rho, P, T, h1, h2, mode):
        """ Sections 2.2.1.2 and 2.2.2.2 @ ITU-R P.676-10
        """
        if h1 > 10 or h2 > 10:
            raise ValueError(
                'Both the transmitter and the receiver must be at'
                'altitude of less than 10 km above the sea level.'
                'Current altitude Tx: %.2f km, Rx: %.2f km' % (h1, h2))

        if mode == 'approx':
            rho = rho * np.exp(h1 / 2)
            gamma0, gammaw = self.gaseous_attenuation_approximation(
                f, el, rho, P, T)
        else:
            gamma0, gammaw = self.gamma_exact(f, P, rho, T)
  
        h0, hw = slant_inclined_path_equivalent_height(f, P).to(u.km).value

        if 5 <= el and el <= 90:
            h0_p = h0 * (np.exp(-h1 / h0) - np.exp(-h2 / h0))
            hw_p = hw * (np.exp(-h1 / hw) - np.exp(-h2 / hw))
            return (gamma0 * h0_p + gammaw * hw_p) / np.sin(np.deg2rad(el))
        else:
            def F(x):
                return 1 / (0.661 * x + 0.339 * np.sqrt(x**2 + 5.51))

            Re = 8500   # TODO should be calculated by ITU-R P.834
            el1 = np.deg2rad(el)
            el2 = np.arccos((Re + h1) / (Re + h2) * np.cos(el1))

            def xi(eli, hi):
                return np.tan(eli) * np.sqrt((Re + hi) / h0)

            def xi_p(eli, hi):
                return np.tan(eli) * np.sqrt((Re + hi) / hw)

            def eq_33(h_num, h_den, el, x):
                return np.sqrt(Re + h_num) * F(x) * \
                    np.exp(-h_num / h_den) / np.cos(el)

            A = gamma0 * np.sqrt(h0) * (eq_33(h1, h0, el1, xi(el1, h1)) -
                                        eq_33(h2, h0, el2, xi(el2, h2))) +\
                gammaw * np.sqrt(hw) * (eq_33(h1, hw, el1, xi_p(el1, h1)) -
                                        eq_33(h2, hw, el2, xi_p(el2, h2)))
            return A


class _ITU676_9():

    def __init__(self):
        self.__version__ = 9
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.676-9-201202-S/en'

    # Recommendation ITU-P R.676-9 has most of the methods similar to those
    # in Recommendation ITU-P R.676-10.
    def gaseous_attenuation_inclined_path(self, *args, **kwargs):
        return _ITU676_10.gaseous_attenuation_inclined_path(*args, **kwargs)


__model = __ITU676()


def change_version(new_version):
    """
    Change the version of the ITU-R P.676 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.676-11 (02/12) (Current version)
            * P.676-10 
            * P.676-9 
    """
    global __model
    __model = __ITU676(new_version)
    memory.clear()


def get_version():
    """
    Obtain the version of the ITU-R P.676 recommendation currently being used.
    """
    global __model
    return __model.__version__


def gaseous_attenuation_inclined_path(f, el, rho, P, T, h1, h2, mode='approx'):
    """
    Estimate the attenuation of atmospheric gases on inclined paths between two
    ground stations at heights h1 and h2. This function operates in two modes,
    'approx', and 'exact':

    * 'approx': a simplified approximate method to estimate gaseous attenuation
    that is applicable in the frequency range 1-350 GHz.
    * 'exact': an estimate of gaseous attenuation computed by summation of
    individual absorption lines that is valid for the frequency
    range 1-1,000 GHz


    Parameters
    ----------
    - f : number or Quantity
            Frequency (GHz)
    - el : sequence, number or Quantity
            Elevation angle at altitude h1 (degrees)
    - rho : number or Quantity
            Water vapor density (g/m3)
    - P : number or Quantity
            Total atmospheric pressure (hPa) (Ptot = Pdry + e)
    - T : number or Quantity
            Absolute temperature (K)
    - h1 : number or Quantity
            Height of ground station 1 (km)
    - h2 : number or Quantity
            Height of ground station 2 (km)
    - mode : string, optional
            Mode for the calculation. Valid values are 'approx', 'exact'. If
            'approx' Uses the method in Annex 2 of the recommendation (if any),
            else uses the method described in Section 1. Default, 'approx'


    Returns
    -------
    - attenuation: Quantity
            Inclined path attenuation (dB)

    References
    --------
    [1] Attenuation by atmospheric gases:
    https://www.itu.int/rec/R-REC-P.676/en
    """
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(el, u.deg, 'Elevation angle')
    type_output = type(el)
    rho = prepare_quantity(rho, u.g / u.m**3, 'Water vapor density')
    P = prepare_quantity(P, u.hPa, 'Total Atmospheric pressure')
    T = prepare_quantity(T, u.K, 'Temperature')
    h1 = prepare_quantity(h1, u.km, 'Height of Ground Station 1')
    h2 = prepare_quantity(h2, u.km, 'Height of Ground Station 2')
    val = __model.gaseous_attenuation_inclined_path(
        f, el, rho, P, T, h1, h2, mode)
    return prepare_output_array(val, type_output) * u.dB

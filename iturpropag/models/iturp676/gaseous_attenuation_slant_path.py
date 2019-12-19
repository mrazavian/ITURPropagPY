# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
import os
from astropy import units as u

from iturpropag.models.iturp453.radio_refractive_index import radio_refractive_index
from iturpropag.models.iturp835.standard_pressure import standard_pressure
from iturpropag.models.iturp835.standard_temperature import standard_temperature
from iturpropag.models.iturp835.standard_water_vapour_density import standard_water_vapour_density
from iturpropag.models.iturp676.slant_inclined_path_equivalent_height import slant_inclined_path_equivalent_height
from iturpropag.models.iturp676.zenith_water_vapour_attenuation import zenith_water_vapour_attenuation
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

    def gaseous_attenuation_slant_path(self, f, el, rho, P, T, V_t, h, mode):
        # Abstract method to compute the gaseous attenuation over a slant path
        fcn = np.vectorize(self.instance.gaseous_attenuation_slant_path)
        return fcn(f, el, rho, P, T, V_t, h, mode)


class _ITU676_11():

    def __init__(self):
        self.__version__ = 11
        self.year = 2017
        self.month = 12
        self.link = 'https://www.p.int/rec/R-REC-P.676-11-201712-S/en'

    def gamma_exact(self, f, P, rho, T):
        return (gamma0_exact(f, P, rho, T).value + \
                gammaw_exact(f, P, rho, T).value)
                

    def gaseous_attenuation_approximation(self, f, el, rho, P, T):
        """
        T goes in Kelvin
        """
        if np.any(f > 350):
            warning_text = 'The approximated method to computes '+\
                    'the gaseous attenuation in recommendation ITU-P 676-11 '+\
                    'is only recommended for frequencies below 350GHz; '+\
                    'here the frequency is '+ str(f) + ' GHz.'

            warnings.warn(RuntimeWarning(warning_text))

        # <TP> replaced: if np.any(5 > el) or np.any(np.mod(el, 90) < 5):
        if np.any(el < 5) or np.any(el > 90):
            warning_text = 'The approximated method to compute '+\
                    'the gaseous attenuation in recommendation ITU-P 676-11 '+\
                    'is only recommended for elevation angles between'+\
                    '5 and 90 degrees;' + 'here the elevation is ' + str(el)

            warnings.warn(RuntimeWarning(warning_text))

        # Water vapour attenuation (gammaw) computation as in Section 1 of
        # Annex 2 of [1]
        gamma0 = gamma0_approx(f, P, rho, T).value
        gammaw = gammaw_approx(f, P, rho, T).value
        return gamma0, gammaw

    
    def gaseous_attenuation_slant_path(self, f, el, rho, P, T, V_t=None,
                                       h=None, mode='approx'):
        if mode == 'approx':
            gamma0, gammaw = self.gaseous_attenuation_approximation(
                f, el, rho, P, T)

            h0, hw = slant_inclined_path_equivalent_height(f, P).to(u.km).value

            # Use the zenit water-vapour method if the values of V_t
            # and h are provided
            if V_t is not None and h is not None:
                Aw = zenith_water_vapour_attenuation(0, 0, 0, f, V_t=V_t, h=h).value
            else:
                Aw = gammaw * hw
            
            A0 = gamma0 * h0
            return (A0 + Aw) / np.sin(np.deg2rad(el))

        else:
            if h is None:
                warning_msg = 'The exact method to compute '+\
                    'the slant path gaseous attenuation in recommendation ITU-R P.676-11 '+\
                    'needs the height of ground station (h (km)) '+\
                    'which currently the h is equal None.'
                raise NameError(warning_msg)

            delta_h = 0.0001 * np.exp((np.arange(0, 922)) / 100)
            h_n = np.cumsum(delta_h)
            
            ind = np.searchsorted(h_n, h, side='right')
            h_n = h_n[ind:]
            delta_h = delta_h[ind:]
            delta_h[0] = h_n[0] - h
            
            T_n = standard_temperature(h_n).to(u.K).value
            press_n = standard_pressure(h_n).value
            rho_n = standard_water_vapour_density(h_n, rho_0=rho).value

            e_n = rho_n * T_n / 216.7
            n_n = radio_refractive_index(press_n, e_n, T_n).value
            n_ratio = np.pad(n_n[1:], (0, 1), mode='edge') / n_n
            r_n = 6371 + h_n

            b = np.pi / 2 - np.deg2rad(el)
            Agas = 0
            for t, press, rho, r, delta, n_r in zip(
                    T_n, press_n, rho_n, r_n, delta_h, n_ratio):
                a = - r * np.cos(b) + 0.5 * np.sqrt(
                    4 * r**2 * np.cos(b)**2 + 8 * r * delta + 4 * delta**2)
                a_cos_arg = np.clip((-a**2 - 2 * r * delta - delta**2) /
                                    (2 * a * r + 2 * a * delta), -1, 1)
                alpha = np.pi - np.arccos(a_cos_arg)
                gamma = self.gamma_exact(f, press, rho, t)
                Agas += a * gamma
                b = np.arcsin(np.sin(alpha) / n_r)

            return Agas


class _ITU676_10():

    def __init__(self):
        self.__version__ = 10
        self.year = 2013
        self.month = 9
        self.link = 'https://www.p.int/rec/R-REC-P.676-10-201309-S/en'

    
    def gamma_exact(self, f, P, rho, T):
        return (gamma0_exact(f, P, rho, T).value +
                gammaw_exact(f, P, rho, T).value)

    
    def gaseous_attenuation_approximation(self, f, el, rho, P, T):
        """
        T goes in Kelvin
        """
        if np.any(f > 350):
            warning_text = 'The approximated method to computes '+\
                    'the gaseous attenuation in recommendation ITU-P 676-10 '+\
                    'is only recommended for frequencies below 350GHz. '+\
                    'here the frequency is '+ str(f) + ' GHz.'

            warnings.warn(RuntimeWarning(warning_text))

        if np.any(5 > el) or np.any(np.mod(el, 90) < 5):
            warning_text = 'The approximated method to compute '+\
                    'the gaseous attenuation in recommendation ITU-P 676-10 '+\
                    'is only recommended for elevation angles between'+\
                    '5 and 90 degrees. '+'here the elevation is ' + str(el)
            warnings.warn(RuntimeWarning(warning_text))

        # Water vapour attenuation (gammaw) computation as in Section 1 of
        # Annex 2 of [1]
        gamma0 = gamma0_approx(f, P, rho, T).value
        gammaw = gammaw_approx(f, P, rho, T).value

        return gamma0, gammaw

    
    def gaseous_attenuation_slant_path(self, f, el, rho, P, T, V_t=None,
                                       h=None, mode='approx'):
        """
        """
        if mode == 'approx':
            gamma0, gammaw = self.gaseous_attenuation_approximation(
                f, el, rho, P, T)

            h0, hw = slant_inclined_path_equivalent_height(f, P).value

            # Use the zenit water-vapour method if the values of V_t
            # and h are provided
            if V_t is not None and h is not None:
                Aw = zenith_water_vapour_attenuation(0, 0, 0,
                                                         f, V_t, h).value
            else:
                Aw = gammaw * hw

            A0 = gamma0 * h0
            return (A0 + Aw) / np.sin(np.deg2rad(el))

        else:
            if h is None:
                warning_msg = 'The exact method to compute '+\
                    'the slant path gaseous attenuation in recommendation ITU-R P.676-10 '+\
                    'needs the height of ground station (h (km))'+\
                    'which currently the h is equal None.'
                raise NameError(warning_msg)

            delta_h = 0.0001 * np.exp((np.arange(0, 922)) / 100)
            h_n = np.cumsum(delta_h)

            ind = np.searchsorted(h_n, h, side='right')
            h_n = h_n[ind:]
            delta_h = delta_h[ind:]
            delta_h[0] = h_n[0] - h

            T_n = standard_temperature(h_n).to(u.K).value
            press_n = standard_pressure(h_n).value
            rho_n = standard_water_vapour_density(h_n, rho_0=rho).value

            e_n = rho_n * T_n / 216.7
            n_n = radio_refractive_index(press_n, e_n, T_n).value
            n_ratio = np.pad(n_n[1:], (0, 1), mode='edge') / n_n
            r_n = 6371 + h_n

            b = np.pi / 2 - np.deg2rad(el)
            Agas = 0
            for t, press, rho, r, delta, n_r in zip(
                    T_n, press_n, rho_n, r_n, delta_h, n_ratio):
                a = - r * np.cos(b) + 0.5 * np.sqrt(
                    4 * r**2 * np.cos(b)**2 + 8 * r * delta + 4 * delta**2)
                a_cos_arg = np.clip((-a**2 - 2 * r * delta - delta**2) /
                                    (2 * a * r + 2 * a * delta), -1, 1)
                alpha = np.pi - np.arccos(a_cos_arg)
                gamma = self.gamma_exact(f, press, rho, t)
                Agas += a * gamma
                b = np.arcsin(np.sin(alpha) / n_r)

            return Agas


class _ITU676_9():

    def __init__(self):
        self.__version__ = 9
        self.year = 2012
        self.month = 2
        self.link = 'https://www.p.int/rec/R-REC-P.676-9-201202-S/en'

    # Recommendation ITU-P R.676-9 has most of the methods similar to those
    # in Recommendation ITU-P R.676-10.

    def gaseous_attenuation_slant_path(self, *args, **kwargs):
        return _ITU676_10.gaseous_attenuation_slant_path(*args, **kwargs)


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


def gaseous_attenuation_slant_path(f, el, rho, P, T, V_t=None, h=None,
                                   mode='approx'):
    """
    Estimate the attenuation of atmospheric gases on slant paths. This function
    operates in two modes, 'approx', and 'exact':

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
            Elevation angle (degrees)
    - rho : number or Quantity
            Water vapor density (g/m3)
    - P : number or Quantity
            Total atmospheric pressure (hPa)(Ptot = Pdry + e)
    - T : number or Quantity
            Absolute temperature (K)
    - V_t: number or Quantity (kg/m2)
            Integrated water vapour content from: a) local radiosonde or
            radiometric data or b) at the required percentage of time (kg/m2)
            obtained from the digital maps in Recommendation ITU-R P.836 (kg/m2).
            If None, use general method to compute the wet-component of the
            gaseous attenuation. If provided, 'h' must be also provided. Default
            is None.
    - h : number, sequence, or numpy.ndarray, optional
            Altitude of the receivers (km). If None, use the topographical altitude as
            described in recommendation ITU-R P.1511. If provided, 'V_t' needs to
            be also provided. Default is None.
    - mode : string, optional
            Mode for the calculation. Valid values are 'approx', 'exact'. If
            'approx' Uses the method in Annex 2 of the recommendation (if any),
            else uses the method described in Section 1. Default, 'approx'


    Returns
    -------
    - attenuation: Quantity
            Slant path attenuation (dB)

    References
    --------
    [1] Attenuation by atmospheric gases:
    https://www.p.int/rec/R-REC-P.676/en
    """
    type_output = type(el)
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(prepare_input_array(el), u.deg, 'Elevation angle')
    rho = prepare_quantity(rho, u.g / u.m**3, 'Water vapor density')
    P = prepare_quantity(P, u.hPa, 'Total Atmospheric pressure')
    T = prepare_quantity(T, u.K, 'Temperature')
    V_t = prepare_quantity(V_t, u.kg / u.m**2,
                           'Integrated water vapour content')
    h = prepare_quantity(h, u.km, 'Altitude')
    val = __model.gaseous_attenuation_slant_path(
            f, el, rho, P, T, V_t, h, mode)
    return prepare_output_array(val, type_output) * u.dB

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
import os
from astropy import units as u

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

    def gamma0_approx(self, f, P, rho, t):
        # Abstract method to compute the specific attenuation due to dry
        # atmoshere
        fcn = np.vectorize(self.instance.gamma0_approx)
        return fcn(f, P, rho, t)


class _ITU676_11():

    tmp = load_data(os.path.join(dataset_dir, 'p676/v11_lines_oxygen.txt'),
                    skip_header=1)
    f_ox = tmp[:, 0]
    a1 = tmp[:, 1]
    a2 = tmp[:, 2]
    a3 = tmp[:, 3]
    a4 = tmp[:, 4]
    a5 = tmp[:, 5]
    a6 = tmp[:, 6]


    def __init__(self):
        self.__version__ = 11
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.676-11-201712-S/en'

    def gamma0_approx(self, f, P, rho, T):
        # T in Kelvin
        # e : water vapour partial pressure in hPa (total barometric pressure
        # ptot = p + e)
        theta = 300 / T
        e = rho * T / 216.7   # water vapour partial pressure
        p = P - e             # Dry air pressure

        f_ox = self.f_ox

        D_f_ox = self.a3 * 1e-4 * (p * (theta ** (0.8 - self.a4)) +
                                   1.1 * e * theta)

        delta_ox = (self.a5 + self.a6 * theta) * 1e-4 * (p + e) * theta**0.8

        F_i_ox = f / f_ox * ((D_f_ox - delta_ox * (f_ox - f)) /
                             ((f_ox - f) ** 2 + D_f_ox ** 2) +
                             (D_f_ox - delta_ox * (f_ox + f)) /
                             ((f_ox + f) ** 2 + D_f_ox ** 2))

        Si_ox = self.a1 * 1e-7 * p * theta**3 * np.exp(self.a2 * (1 - theta))

        N_pp_ox = Si_ox * F_i_ox

        d = 5.6e-4 * (p + e) * theta**0.8

        N_d_pp = f * p * theta**2 * \
            (d * 6.14e-5 / (d**2 + f**2) +
             1.4e-12 * p * theta**1.5 / (1 + 1.9e-5 * f**1.5))

        N_pp = N_pp_ox.sum() + N_d_pp

        gamma = 0.1820 * f * N_pp           # Eq. 1 [dB/km]
        return gamma

    
class _ITU676_10():

    def __init__(self):
        self.__version__ = 10
        self.year = 2013
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.676-10-201309-S/en'


    def gamma0_approx(self, f, P, rho, T):
        rp = P / 1013.0
        rt = 288.0 / (T)

        def phi(rp, rt, a, b, c, d): 
            return np.power(rp, a) * np.power(rt, b) * \
                   np.exp(c * (1 - rp) + d * (1 - rt))

        # Dry air attenuation (gamma0) computation as in Section 1 of Annex 2
        # of [1]
        delta = -0.00306 * phi(rp, rt, 3.211, -14.94, 1.583, -16.37)

        xi1 = phi(rp, rt, 0.0717, -1.8132, 0.0156, -1.6515)
        xi2 = phi(rp, rt, 0.5146, -4.6368, -0.1921, -5.7416)
        xi3 = phi(rp, rt, 0.3414, -6.5851, 0.2130, -8.5854)
        xi4 = phi(rp, rt, -0.0112, 0.0092, -0.1033, -0.0009)
        xi5 = phi(rp, rt, 0.2705, -2.7192, -0.3016, -4.1033)
        xi6 = phi(rp, rt, 0.2445, -5.9191, 0.0422, -8.0719)
        xi7 = phi(rp, rt, -0.1833, 6.5589, -0.2402, 6.131)

        gamma54 = 2.192 * phi(rp, rt, 1.8286, -1.9487, 0.4051, -2.8509)
        gamma58 = 12.59 * phi(rp, rt, 1.0045, 3.5610, 0.1588, 1.2834)
        gamma60 = 15.00 * phi(rp, rt, 0.9003, 4.1335, 0.0427, 1.6088)
        gamma62 = 14.28 * phi(rp, rt, 0.9886, 3.4176, 0.1827, 1.3429)
        gamma64 = 6.819 * phi(rp, rt, 1.4320, 0.6258, 0.3177, -0.5914)
        gamma66 = 1.908 * phi(rp, rt, 2.0717, -4.1404, 0.4910, -4.8718)

        def fcn_le_54():
            return (((7.2 * rt**2.8) / (f**2 + 0.34 * rp**2 * rt**1.6) +
                     (0.62 * xi3) / ((54 - f)**(1.16 * xi1) + 0.83 * xi2)) *
                    f**2 * rp**2 * 1e-3)

        def fcn_le_60():
            return (np.exp(np.log(gamma54) / 24.0 * (f - 58) * (f - 60) -
                           np.log(gamma58) / 8.0 * (f - 54) * (f - 60) +
                           np.log(gamma60) / 12.0 * (f - 54) * (f - 58)))

        def fcn_le_62():
            return (gamma60 + (gamma62 - gamma60) * (f - 60) / 2.0)

        def fcn_le_66():
            return (np.exp(np.log(gamma62) / 8.0 * (f - 64) * (f - 66) -
                           np.log(gamma64) / 4.0 * (f - 62) * (f - 66) +
                           np.log(gamma66) / 8.0 * (f - 62) * (f - 64)))

        def fcn_le_120():
            return ((3.02e-4 * rt**3.5 + (0.283 * rt**3.8) /
                     ((f - 118.75)**2 + 2.91 * rp**2 * rt**1.6) +
                     (0.502 * xi6 * (1 - 0.0163 * xi7 * (f - 66))) /
                     ((f - 66)**(1.4346 * xi4) + 1.15 * xi5)) *
                    f**2 * rp**2 * 1e-3)

        def fcn_rest():
            return (((3.02e-4) / (1 + 1.9e-5 * f**1.5) +
                     (0.283 * rt**0.3) / ((f - 118.75)**2 +
                                          2.91 * rp**2 * rt**1.6)) *
                    f**2 * rp**2 * rt**3.5 * 1e-3 + delta)

        gamma0 = \
            np.where(
                f <= 54, fcn_le_54(),
                np.where(
                    np.logical_and(54 < f, f <= 60), fcn_le_60(),
                    np.where(
                        np.logical_and(60 < f, f <= 62), fcn_le_62(),
                        np.where(
                            np.logical_and(62 < f, f <= 66), fcn_le_66(),
                            np.where(
                                np.logical_and(66 < f, f <= 120),
                                fcn_le_120(),
                                fcn_rest())))))

        return gamma0


class _ITU676_9():

    def __init__(self):
        self.__version__ = 9
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.676-9-201202-S/en'

    # Recommendation ITU-P R.676-9 has most of the methods similar to those
    # in Recommendation ITU-P R.676-10.
    def gamma0_approx(self, *args, **kwargs):
        return _ITU676_10.gamma0_approx(*args, **kwargs)


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


def gamma0_approx(f, P, rho, T):
    """
    Method to estimate the specific attenuation due to dry atmosphere using the
    approximate method descibed in Annex 2.

    Parameters
    ----------
    - f : number or Quantity
            Frequency (GHz)
    - P : number or Quantity
            Total atmospheric pressure (hPa) (Ptot = Pdry + e)
    - rho : number or Quantity
            Water vapor density (g/m3)
    - T : number or Quantity
            Absolute temperature (K)


    Returns
    -------
    - gamma_0 : Quantity
            Dry atmosphere specific attenuation (dB/km)

    References
    --------
    [1] Attenuation by atmospheric gases:
    https://www.itu.int/rec/R-REC-P.676/en
    """
    global __model
    type_output = type(f)
    f = prepare_quantity(f, u.GHz, 'Frequency')
    P = prepare_quantity(P, u.hPa, 'Total atmospheric pressure')
    rho = prepare_quantity(rho, u.g / u.m**3, 'Water vapour density')
    T = prepare_quantity(T, u.K, 'Temperature')
    val = __model.gamma0_approx(f, P, rho, T)
    return prepare_output_array(val, type_output) * u.dB / u.km

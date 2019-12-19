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

    def gammaw_approx(self, f, P, rho, t):
        # Abstract method to compute the specific attenuation due to water
        # vapour
        fcn = np.vectorize(self.instance.gammaw_approx)
        return fcn(f, P, rho, t)


class _ITU676_11():

    tmp = load_data(os.path.join(dataset_dir,
                                 'p676//v11_lines_water_vapour.txt'),
                    skip_header=1)
    f_wv = tmp[:, 0]
    b1 = tmp[:, 1]
    b2 = tmp[:, 2]
    b3 = tmp[:, 3]
    b4 = tmp[:, 4]
    b5 = tmp[:, 5]
    b6 = tmp[:, 6]

    idx_approx = np.zeros_like(b1, dtype=bool).squeeze()
    asterisk_rows = [0, 3, 4, 5, 7, 12, 20, 24, 34]
    idx_approx[np.array(asterisk_rows)] = True

    def __init__(self):
        self.__version__ = 11
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.676-11-201712-S/en'

    def gammaw_approx(self, f, P, rho, T):
        # T in Kelvin
        # e : water vapour partial pressure in hPa (total barometric pressure
        # ptot = p + e)
        theta = 300 / T
        e = rho * T / 216.7  # water vapour partial pressure
        p = P - e           # Dry air pressure

        f_wv = self.f_wv[self.idx_approx]
        b1 = self.b1[self.idx_approx]
        b2 = self.b2[self.idx_approx]
        b3 = self.b3[self.idx_approx]
        b4 = self.b4[self.idx_approx]
        b5 = self.b5[self.idx_approx]
        b6 = self.b6[self.idx_approx]

        D_f_wv = b3 * 1e-4 * (p * theta ** b4 +
                              b5 * e * theta ** b6)

        F_i_wv = f / f_wv * ((D_f_wv) / ((f_wv - f)**2 + D_f_wv**2) +
                             (D_f_wv) / ((f_wv + f)**2 + D_f_wv**2))

        Si_wv = b1 * 1e-1 * e * theta**3.5 * np.exp(b2 * (1 - theta))

        N_pp_wv = Si_wv * F_i_wv

        N_pp = N_pp_wv.sum()

        gamma = 0.1820 * f * N_pp           # Eq. 1 [dB/km]
        return gamma

    
class _ITU676_10():

    def __init__(self):
        self.__version__ = 10
        self.year = 2013
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.676-10-201309-S/en'

    def gammaw_approx(self, f, P, rho, T): 
        rp = P / 1013
        rt = 288 / (T)
        eta1 = 0.955 * rp * rt**0.68 + 0.006 * rho
        eta2 = 0.735 * rp * rt**0.50 + 0.0353 * rt**4 * rho

        def g(f, fi): return 1 + ((f - fi) / (f + fi))**2
        gammaw = (
            (3.98 * eta1 * np.exp(2.23 * (1 - rt))) /
            ((f - 22.235) ** 2 + 9.42 * eta1 ** 2) * g(f, 22.0) +
            (11.96 * eta1 * np.exp(0.70 * (1 - rt))) /
            ((f - 183.310) ** 2 + 11.14 * eta1 ** 2) +
            (0.081 * eta1 * np.exp(6.44 * (1 - rt))) /
            ((f - 321.226) ** 2 + 6.29 * eta1 ** 2) +
            (3.660 * eta1 * np.exp(1.60 * (1 - rt))) /
            ((f - 325.153) ** 2 + 9.22 * eta1 ** 2) +
            (25.37 * eta1 * np.exp(1.09 * (1 - rt))) / ((f - 380.000) ** 2) +
            (17.40 * eta1 * np.exp(1.46 * (1 - rt))) / ((f - 448.000) ** 2) +
            (844.6 * eta1 * np.exp(0.17 * (1 - rt))) / ((f - 557.000) ** 2) *
            g(f, 557.0) + (290.0 * eta1 * np.exp(0.41 * (1 - rt))) /
            ((f - 752.000) ** 2) * g(f, 752.0) +
            (8.3328e4 * eta2 * np.exp(0.99 * (1 - rt))) /
            ((f - 1780.00) ** 2) *
            g(f, 1780.0)) * f ** 2 * rt ** 2.5 * rho * 1e-4
        return gammaw


class _ITU676_9():

    def __init__(self):
        self.__version__ = 9
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.676-9-201202-S/en'

    # Recommendation ITU-P R.676-9 has most of the methods similar to those
    # in Recommendation ITU-P R.676-10.
    def gammaw_approx(self, *args, **kwargs):
        return _ITU676_10.gammaw_approx(*args, **kwargs)


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


def gammaw_approx(f, P, rho, T):
    """
    Method to estimate the specific attenuation due to water vapour using the
    approximate method descibed in Annex 2.


    Parameters
    ----------
    - f : number or Quantity
            Frequency (GHz)
    - P : number or Quantity
            Total atmospheric pressure (hPa)(Ptot = Pdry + e)
    - rho : number or Quantity
            Water vapor density (g/m3)
    - T : number or Quantity
            Absolute temperature (K)


    Returns
    -------
    - gamma_w : Quantity
            Water vapour specific attenuation (dB/km)

    References
    --------
    [1] Attenuation by atmospheric gases:
    https://www.itu.int/rec/R-REC-P.676/en
    """
    global __model
    type_output = type(f)
    f = prepare_quantity(f, u.GHz, 'Frequency')
    P = prepare_quantity(P, u.hPa, 'Atmospheric pressure ')
    rho = prepare_quantity(rho, u.g / u.m**3, 'Water vapour density')
    T = prepare_quantity(T, u.K, 'Temperature')
    val = __model.gammaw_approx(f, P, rho, T)
    return prepare_output_array(val, type_output) * u.dB / u.km

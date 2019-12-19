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
from iturpropag.models.iturp836.total_water_vapour_content import total_water_vapour_content
from iturpropag.models.iturp1511.topographic_altitude import topographic_altitude
from iturpropag.utils import prepare_quantity, prepare_output_array,\
    prepare_input_array, load_data, dataset_dir, memory


def __gamma0_exact__676_9_11__(self, f, P, rho, T):
    # T in Kelvin
    # p : dry air pressure (total barometric pressure
    # ptot = p + e)
    theta = 300 / T
    e = rho * T / 216.7   # water vapour partial pressure
    p = P - e              #  Dry air pressure

    f_ox = self.f_ox

    D_f_ox = self.a3 * 1e-4 * (p * (theta ** (0.8 - self.a4)) +
                               1.1 * e * theta)
    D_f_ox = np.sqrt(D_f_ox**2 + 2.25 * 1e-6)

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

    def gamma0_exact(self, f, P, rho, t):
        # Abstract method to compute the specific attenuation due to dry
        # atmoshere
        fcn = np.vectorize(self.instance.gamma0_exact)
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

    def gamma0_exact(self, f, P, rho, T):
        return __gamma0_exact__676_9_11__(self, f, P, rho, T)


class _ITU676_10():

    tmp = load_data(os.path.join(dataset_dir, 'p676/v10_lines_oxygen.txt'),
                    skip_header=1)
    f_ox = tmp[:, 0]
    a1 = tmp[:, 1]
    a2 = tmp[:, 2]
    a3 = tmp[:, 3]
    a4 = tmp[:, 4]
    a5 = tmp[:, 5]
    a6 = tmp[:, 6]

    def __init__(self):
        self.__version__ = 10
        self.year = 2013
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.676-10-201309-S/en'

    def gamma0_exact(self, f, P, rho, T):
        return __gamma0_exact__676_9_11__(self, f, P, rho, T)


class _ITU676_9():

    tmp = load_data(os.path.join(dataset_dir, 'p676//v9_lines_oxygen.txt'),
                    skip_header=1)
    f_ox = tmp[:, 0]
    a1 = tmp[:, 1]
    a2 = tmp[:, 2]
    a3 = tmp[:, 3]
    a4 = tmp[:, 4]
    a5 = tmp[:, 5]
    a6 = tmp[:, 6]

    def __init__(self):
        self.__version__ = 9
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.676-9-201202-S/en'

    # Recommendation ITU-P R.676-9 has most of the methods similar to those
    # in Recommendation ITU-P R.676-10.
    def gamma0_exact(self, f, P, rho, T):
        return __gamma0_exact__676_9_11__(self, f, P, rho, T)


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

@memory.cache
def gamma0_exact(f, P, rho, T):
    """
    Method to estimate the specific attenuation due to dry atmosphere using
    the line-by-line method described in Annex 1 of the recommendation.

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
    P = prepare_quantity(P, u.hPa, 'Total Atmospheric pressure')
    rho = prepare_quantity(rho, u.g / u.m**3, 'Water vapour density')
    T = prepare_quantity(T, u.K, 'Temperature')
    val = __model.gamma0_exact(f, P, rho, T)
    return prepare_output_array(val, type_output) * u.dB / u.km

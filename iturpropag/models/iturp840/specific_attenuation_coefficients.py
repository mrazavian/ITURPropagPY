# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u

from iturpropag.utils import load_data, dataset_dir, prepare_input_array, \
    prepare_output_array, prepare_quantity, memory


class __ITU840():
    """Attenuation due to clouds and fog: This Recommendation provides methods
    to predict the attenuation due to clouds and fog on Earth-space paths.

    Available versions include:
    * P.840-4 (10/09) (Superseded)
    * P.840-5 (02/12) (Superseded)
    * P.840-6 (09/13) (Superseded)
    * P.840-7 (12/17) (Current version)

    Non-available versions include:
    * P.840-1 (08/94) (Superseded) - Tentative similar to P.840-4
    * P.840-2 (08/97) (Superseded) - Tentative similar to P.840-4
    * P.840-3 (10/99) (Superseded) - Tentative similar to P.840-4

    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.840 recommendation.

    def __init__(self, version=7):
        if version == 7:
            self.instance = _ITU840_7()
        elif version == 6:
            self.instance = _ITU840_6()
        elif version == 5:
            self.instance = _ITU840_5()
        elif version == 4:
            self.instance = _ITU840_4()
        else:
            raise ValueError(
                'Version {0}  is not implemented for the ITU-R P.840 model.'
                .format(version))

        self._Lred = {}
        self._M = {}
        self._sigma = {}
        self._Pclw = {}

    @property
    def __version__(self):
        return self.instance.__version__

    def specific_attenuation_coefficients(self, f, T):
        # Abstract method to compute the specific attenuation coefficients
        fcn = np.vectorize(self.instance.specific_attenuation_coefficients, excluded=[1])
        return fcn(f, T)


class _ITU840_7():

    def __init__(self):
        self.__version__ = 7
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.840-7-201712-I/en'

    def specific_attenuation_coefficients(self, f, T):
        """
        """
        if np.any(f > 1000):
            raise ValueError('Frequency must be introduced in GHz and the '
                             'maximum range is 1000 GHz')

        T_kelvin = T + 273.15
        theta = 300.0 / T_kelvin                # Eq. 9

        # Compute the values of the epsilons
        epsilon0 = 77.66 + 103.3 * (theta - 1)  # Eq. 6
        epsilon1 = 0.0671 * epsilon0            # Eq. 7
        epsilon2 = 3.52                         # Eq. 8

        # Compute the principal and secondary relacation frequencies
        fp = 20.20 - 146 * (theta - 1) + 316.0 * (theta - 1)**2     # Eq. 10
        fs = 39.8 * fp                                              # Eq. 11

        # Compute the dielectric permitivity of water
        epsilonp = (epsilon0 - epsilon1) / (1 + (f / fp) ** 2) + \
            (epsilon1 - epsilon2) / (1 + (f / fs) ** 2) + epsilon2  # Eq. 5

        epsilonpp = f * (epsilon0 - epsilon1) / (fp * (1 + (f / fp)**2)) + \
            f * (epsilon1 - epsilon2) / (fs * (1 + (f / fs)**2))       # Eq. 4

        eta = (2 + epsilonp) / epsilonpp                    # Eq. 3
        Kl = (0.819 * f) / (epsilonpp * (1 + eta**2))       # Eq. 2

        return Kl       # Specific attenuation coefficient  (dB/km)/(g/m3)

    
class _ITU840_6():

    def __init__(self):
        self.__version__ = 6
        self.year = 2013
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.840-6-201202-I/en'

    def specific_attenuation_coefficients(self, f, T):
        """
        """
        if np.any(f > 1000):
            raise ValueError('Frequency must be introduced in GHz and the '
                             'maximum range is 1000 GHz')

        T_kelvin = T + 273.15
        theta = 300.0 / T_kelvin                # Eq. 9

        # Compute the values of the epsilons
        epsilon0 = 77.66 + 103.3 * (theta - 1)  # Eq. 6
        epsilon1 = 0.0671 * epsilon0            # Eq. 7
        epsilon2 = 3.52                         # Eq. 8

        # Compute the principal and secondary relacation frequencies
        fp = 20.20 - 146 * (theta - 1) + 316.0 * (theta - 1)**2     # Eq. 10
        fs = 39.8 * fp                                              # Eq. 11

        # Compute the dielectric permitivity of water
        epsilonp = (epsilon0 - epsilon1) / (1 + (f / fp) ** 2) + \
            (epsilon1 - epsilon2) / (1 + (f / fs) ** 2) + epsilon2  # Eq. 5

        epsilonpp = f * (epsilon0 - epsilon1) / (fp * (1 + (f / fp)**2)) + \
            f * (epsilon1 - epsilon2) / (fs * (1 + (f / fs)**2))       # Eq. 4

        eta = (2 + epsilonp) / epsilonpp                    # Eq. 3
        Kl = (0.819 * f) / (epsilonpp * (1 + eta**2))       # Eq. 2

        return Kl       # Specific attenuation coefficient  (dB/km)/(g/m3)

   
class _ITU840_5():

    def __init__(self):
        self.__version__ = 5
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.840-5-201202-S/en'

    def specific_attenuation_coefficients(self, f, T):
        """
        """
        if np.any(f > 1000):
            raise ValueError(
                'Frequency must be introduced in GHz and the maximum range '
                'is 1000 GHz')

        T_kelvin = T + 273.15
        theta = 300.0 / T_kelvin                # Eq. 9

        # Compute the values of the epsilons
        epsilon0 = 77.66 + 103.3 * (theta - 1)  # Eq. 6
        epsilon1 = 5.48                         # Eq. 7
        epsilon2 = 3.51                         # Eq. 8

        # Compute the principal and secondary relacation frequencies
        fp = 20.09 - 142 * (theta - 1) + 294.0 * (theta - 1)**2     # Eq. 10
        fs = 590 - 1500 * (theta - 1)                               # Eq. 11

        # Compute the dielectric permitivity of water
        epsilonp = (epsilon0 - epsilon1) / (1 + (f / fp) ** 2) + \
            (epsilon1 - epsilon2) / (1 + (f / fs) ** 2) + epsilon2  # Eq. 5

        epsilonpp = f * (epsilon0 - epsilon1) / (fp * (1 + (f / fp)**2)) + \
            f * (epsilon1 - epsilon2) / (fs * (1 + (f / fs)**2))       # Eq. 4

        eta = (2 + epsilonp) / epsilonpp                    # Eq. 3
        Kl = (0.819 * f) / (epsilonpp * (1 + eta**2))       # Eq. 2

        return Kl       # Specific attenuation coefficient  (dB/km)/(g/m3)

    
class _ITU840_4():

    def __init__(self):
        self.__version__ = 4
        self.year = 2013
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.840-6-201202-I/en'

    def specific_attenuation_coefficients(self, f, T):
        """
        """
        if np.any(f > 1000):
            raise ValueError(
                'Frequency must be introduced in GHz and the maximum range'
                ' is 1000 GHz')

        T_kelvin = T + 273.15
        theta = 300.0 / T_kelvin                # Eq. 9

        # Compute the values of the epsilons
        epsilon0 = 77.66 + 103.3 * (theta - 1)  # Eq. 6
        epsilon1 = 5.48                         # Eq. 7
        epsilon2 = 3.51                         # Eq. 8

        # Compute the principal and secondary relacation frequencies
        fp = 20.09 - 142 * (theta - 1) + 294.0 * (theta - 1)**2     # Eq. 10
        fs = 590 - 1500 * (theta - 1)                               # Eq. 11

        # Compute the dielectric permitivity of water
        epsilonp = (epsilon0 - epsilon1) / (1 + (f / fp) ** 2) + \
            (epsilon1 - epsilon2) / (1 + (f / fs) ** 2) + epsilon2  # Eq. 5

        epsilonpp = f * (epsilon0 - epsilon1) / (fp * (1 + (f / fp)**2)) + \
            f * (epsilon1 - epsilon2) / (fs * (1 + (f / fs)**2))       # Eq. 4

        eta = (2 + epsilonp) / epsilonpp                    # Eq. 3
        Kl = (0.819 * f) / (epsilonpp * (1 + eta**2))       # Eq. 2

        return Kl       # Specific attenuation coefficient  (dB/km)/(g/m3)


__model = __ITU840()


def change_version(new_version):
    """
    Change the version of the ITU-R P.840 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * 7: P.840-7 (09/13) (Current version)
            * 6: P.840-6 
            * 5: P.840-5 
            * 4: P.840-4 
    """
    global __model
    __model = __ITU840(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.840 recommendation currently being used.
    """
    global __model
    return __model.__version__


def specific_attenuation_coefficients(f, T):
    """
    A method to compute the specific attenuation coefficient. The method is
    based on Rayleigh scattering, which uses a double-Debye model for the
    dielectric permittivity of water.

    This model can be used to calculate the value of the specific attenuation
    coefficient for frequencies up to 1000 GHz:


    Parameters
    ----------
    - f : number
            Frequency (GHz)
    - T : sequence, array
            Temperature (degrees C)


    Returns
    -------
    - Kl: numpy.ndarray
            Specific attenuation coefficient (dB/km)/(g/m3)


    References
    ----------
    [1] Attenuation due to clouds and fog:
    https://www.itu.int/rec/R-REC-P.840/en
    """
    global __model
    f = prepare_quantity(f, u.GHz, 'Frequency')
    T = prepare_quantity(T, u.deg_C, 'Temperature')
    return __model.specific_attenuation_coefficients(f, T)

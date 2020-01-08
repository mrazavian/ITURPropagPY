# -*- coding: utf-8 -*-
# checked by <TP>: 2019-06-06
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u

from iturpropag.utils import prepare_input_array, prepare_output_array,\
    prepare_quantity

class __ITU835():
    """Reference Standard Atmospheres

    Available versions:
       * P.835-6 (12/17) (Current version)
       * P.835-5 (02/12) (Superseded)

    The procedures to compute the reference standard atmosphere parameters
    pressented in these versions are identical to those included in version
    ITU_T P.835-5. 
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.835 recommendation.

    def __init__(self, version=6):
        if version == 6:
            self.instance = _ITU835_6()
        elif version == 5:
            self.instance = _ITU835_5()
        else:
            raise ValueError(
                'Version ' +
                str(version) +
                ' is not implemented' +
                ' for the ITU-R P.835 model.')

    @property
    def __version__(self):
        return self.instance.__version__

    def standard_pressure(self, h, T_0, P_0):
        return self.instance.standard_pressure(h, T_0, P_0)


class _ITU835_6():

    def __init__(self):
        self.__version__ = 6
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.835-6-201712-I/en'

    #  Mean annual global reference atmosphere (Section 1 of ITU-R P.835-6)  #
    def standard_pressure(self, h, T_0, P_0):
        """Section 1.1 of Recommendation ITU-R P.835-6
        """
       
        h_p = 6356.766 * h / (6356.766 + h)                         # Eq. 1a
        arg1 = 288.15 / (288.15 - 6.5 * h_p) # avoid error having negative power
        # First height regime
        P = np.where(np.logical_and(0 <= h_p, h_p <= 11),             # Eq. 3a
                1013.25 * np.sign(arg1) * np.power(np.abs(arg1), -34.1632 / 6.5),
            np.where(np.logical_and(11 < h_p, h_p <= 20),           # Eq. 3b
                226.3226 * np.exp(-34.1632 * (h_p - 11.) / 216.65),
            np.where(np.logical_and(20 < h_p, h_p <= 32),           # Eq. 3c
                54.74980 * np.power(216.65 / (216.65 + (h_p - 20.)), 34.1632),
            np.where(np.logical_and(32 < h_p, h_p <= 47),           # Eq. 3d
                8.680422 * np.power(228.65 / (228.65 + 2.8 * (h_p - 32.)),
                     34.1632 / 2.8),
            np.where(np.logical_and(47 < h_p, h_p <= 51),           # Eq. 3e
                1.109106 * np.exp(-34.1632 * (h_p - 47.) / 270.65),
            np.where(np.logical_and(51 < h_p, h_p <= 71),           # Eq. 3f
                0.6694167 * np.power(270.65 / (270.65 - 2.8 * (h_p - 51.)),
                    -34.1632 / 2.8),
            np.where(np.logical_and(71 < h_p, h_p <= 84.852),       # Eq. 3g
                0.03956649 * np.power(214.65 / (214.65 - 2.0 * (h_p - 71.)),
                    -34.1632 / 2.0),
            # Second height regime
            np.where(np.logical_and(86 <= h, h <= 100),             # Eq. 5
                np.exp(95.571899 -4.011801 * h + 
                        6.424731e-2 * h**2 -
                        4.789660e-4 * h**3 + 
                        1.1340543e-6 * h**4),
                np.nan)))))))).astype(float)

        return P


class _ITU835_5():

    def __init__(self):
        self.__version__ = 5
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.835-5-201202-I/en'

    #  Mean annual global reference atmosphere (Section 1 of ITU-R P.835-5)  #
    def standard_pressure(self, h, T_0 =288.15, P_0=1013.25): # Eq. 5
        """
        section 1.1 of ITU-R P.835-5
        """
        # Table 1
        H = [0., 11., 20., 32., 47., 51., 71., 85.]
        L = [-6.5, 0., 1., 2.8, 0., -2.8, -2.]
        # Figure 1
        T = np.array([0., -71.5, -71.5, -59.5, -17.5, -17.5, -73.5, -101.5]) + T_0
        
        num_splits = np.minimum(np.searchsorted(H, h), 7.)
        if not hasattr(num_splits, '__iter__'):
            num_splits = list([num_splits])

        ret = np.ones_like(h) * P_0
        for ret_i, n in enumerate(num_splits):
            n = int(n.squeeze())
            P = np.zeros((n + 1))
            P[0] = P_0
            for i in range(n):
                h_p = h[ret_i] if i == (n - 1) else H[i + 1]
                if L[i] != 0:                                       # Eq. 3
                    P[i + 1] = P[i] * \
                        (T[i] / (T[i] + L[i] * (h_p - H[i])))**(34.163 / L[i])
                else:                                               # Eq. 4
                    P[i + 1] = P[i] * np.exp(-34.163 * (h_p - H[i]) / T[i])

            ret[ret_i] = P[-1]

        return ret


__model = __ITU835()


def change_version(new_version):
    """
    Change the version of the ITU-R P.835 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.835-6 (12/17) (Current version)
            * P.835-5 (02/12) (Superseded)
           
    """
    global __model
    __model = __ITU835(new_version)



def get_version():
    """
    Obtain the version of the ITU-R P.835 recommendation currently being used.
    """
    global __model
    return __model.__version__


def standard_pressure(h, T_0=288.15, P_0=1013.25):
    """
    Method to compute the total atmopsheric pressure of an standard atmosphere
    at a given height.

    The reference standard atmosphere is based on the United States Standard
    Atmosphere, 1976, in which the atmosphere is divided into seven successive
    layers showing linear variation with temperature.


    Parameters
    ----------
    - h : number or Quantity
            Height (km)
    - T_0 : number or Quantity
            Surface absolute temperature (K)
    - P_0 : number or Quantity
            Surface pressure (hPa)


    Returns
    -------
    - P: Quantity
            Pressure (hPa)


    References
    ----------
    [1] Reference Standard Atmospheres
    https://www.itu.int/rec/R-REC-P.835/en
    """
    global __model

    type_output = type(h)
    h = prepare_quantity(h, u.km, 'Height')
    T_0 = prepare_quantity(T_0, u.Kelvin, 'Surface temperature')
    P_0 = prepare_quantity(P_0, u.hPa, 'Surface pressure')
    val = __model.standard_pressure(h, T_0, P_0)
    return prepare_output_array(val, type_output) * u.hPa

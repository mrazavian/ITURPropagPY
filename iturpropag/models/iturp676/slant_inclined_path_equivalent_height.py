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

    def slant_inclined_path_equivalent_height(self, f, P):
        
        return self.instance.slant_inclined_path_equivalent_height(f, P)


class _ITU676_11():

    def __init__(self):
        self.__version__ = 11
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.676-11-201712-S/en'

    
    def slant_inclined_path_equivalent_height(self, f, P):
        # page 18
        rp = P / 1013.25  
        t1 = 4.64 / (1 + 0.066 * rp**-2.3) * \
            np.exp(- ((f - 59.7) / (2.87 + 12.4 * np.exp(-7.9 * rp)))**2)
        t2 = (0.14 * np.exp(2.12 * rp)) / \
            ((f - 118.75)**2 + 0.031 * np.exp(2.2 * rp))
        t3 = 0.0114 / (1 + 0.14 * rp**-2.6) * f * \
            (-0.0247 + 0.0001 * f + 1.61e-6 * f**2) / \
            (1 - 0.0169 * f + 4.1e-5 * f**2 + 3.2e-7 * f**3)

        h0 = 6.1 / (1 + 0.17 * rp**-1.1) * (1 + t1 + t2 + t3)

        h0 = np.where(f < 70,
                      np.minimum(h0, 10.7 * rp**0.3),
                      h0)

        sigmaw = 1.013 / (1 + np.exp(-8.6 * (rp - 0.57)))
        hw = 1.66 * (1 + (1.39 * sigmaw) / ((f - 22.235)**2 + 2.56 * sigmaw) +
                     (3.37 * sigmaw) / ((f - 183.31)**2 + 4.69 * sigmaw) +
                     (1.58 * sigmaw) / ((f - 325.1)**2 + 2.89 * sigmaw))

        return h0, hw


class _ITU676_10():

    def __init__(self):
        self.__version__ = 10
        self.year = 2013
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.676-10-201309-S/en'

    
    def slant_inclined_path_equivalent_height(self, f, P):
        
        rp = P / 1013.0
        t1 = (4.64) / (1 + 0.066 * rp**-2.3) * \
            np.exp(- ((f - 59.7) / (2.87 + 12.4 * np.exp(-7.9 * rp)))**2)
        t2 = (0.14 * np.exp(2.21 * rp)) / \
            ((f - 118.75)**2 + 0.031 * np.exp(2.2 * rp))
        t3 = (0.0114) / (1 + 0.14 * rp**-2.6) * f * \
             (-0.0247 + 0.0001 * f + 1.61e-6 * f**2) / \
             (1 - 0.0169 * f + 4.1e-5 * f**2 + 3.2e-7 * f**3)

        h0 = (6.1) / (1 + 0.17 * rp**-1.1) * (1 + t1 + t2 + t3)

        h0 = np.where(f < 70,
                      np.minimum(h0, 10.7 * rp**0.3),
                      h0)

        sigmaw = (1.013) / (1 + np.exp(-8.6 * (rp - 0.57)))
        hw = 1.66 * (1 + (1.39 * sigmaw) / ((f - 22.235)**2 + 2.56 * sigmaw) +
                     (3.37 * sigmaw) / ((f - 183.31)**2 + 4.69 * sigmaw) +
                     (1.58 * sigmaw) / ((f - 325.1)**2 + 2.89 * sigmaw))

        return h0, hw


class _ITU676_9():

    def __init__(self):
        self.__version__ = 9
        self.year = 2012
        self.month = 2
        self.link = 'https://www.itu.int/rec/R-REC-P.676-9-201202-S/en'

    # Recommendation ITU-P R.676-9 has most of the methods similar to those
    # in Recommendation ITU-P R.676-10.
    def slant_inclined_path_equivalent_height(self, *args, **kwargs):
        return _ITU676_10.slant_inclined_path_equivalent_height(*args,
                                                                **kwargs)


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


def slant_inclined_path_equivalent_height(f, P):
    """ Computes the equivalent height to be used for oxygen and water vapour
    gaseous attenuation computations.

    Parameters
    ----------
    - f : number or Quantity
            Frequency (GHz)
    - P : number or Quantity
            Total Atmospheric Pressure (hPa)(Ptot = Pdry + e)

    Returns
    -------
    - ho, hw : Quantity
            Equivalent height for oxygen and water vapour (km)

    References
    --------
    [1] Attenuation by atmospheric gases:
    https://www.itu.int/rec/R-REC-P.676/en

    """
    type_output = type(f)
    f = prepare_quantity(f, u.GHz, 'Frequency')
    P = prepare_quantity(P, u.hPa, 'Total atmospheric pressure')
    val = __model.slant_inclined_path_equivalent_height(f, P)
    return prepare_output_array(val, type_output) * u.km

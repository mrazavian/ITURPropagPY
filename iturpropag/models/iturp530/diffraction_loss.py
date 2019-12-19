# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u
from scipy.optimize import bisect

from iturpropag.models.iturp530.fresnel_ellipse_radius import fresnel_ellipse_radius
from iturpropag.utils import prepare_input_array, prepare_quantity, load_data,\
    dataset_dir, prepare_output_array


class __ITU530():
    """Propagation data and prediction methods required for the design of
    terrestrial line-of-sight systems

    Available versions:
       * P.530-16 (07/15) (Current version)

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

    def diffraction_loss(self, d1, d2, h, f):
        return self.instance.diffraction_loss(d1, d2, h, f)


class _ITU530_17():

    def __init__(self):
        self.__version__ = 17
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.530-17-201712-S/en'

    ###########################################################################
    #                               Section 2.2                               #
    ###########################################################################
    
    def diffraction_loss(self, d1, d2, h, f):
        """ Implementation of 'diffraction_loss' method for recommendation
        ITU-P R.530-16. See documentation for function
        'ITUR530.diffraction_loss'
        """
        F1 = fresnel_ellipse_radius(d1, d2, f).value      # Eq. 2 [m]
        Ad = -20 * h / F1 + 10                            # Eq. 3 [dB]
        return Ad

    
class _ITU530_16():

    def __init__(self):
        self.__version__ = 16
        self.year = 2015
        self.month = 7
        self.link = 'https://www.itu.int/rec/R-REC-P.530-16-201507-S/en'

    def diffraction_loss(self, *args, **kwargs):
        return _ITU530_17.diffraction_loss(*args, **kwargs)


__model = __ITU530()


def change_version(new_version):
    """
    Change the version of the ITU-R P.530 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
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


def diffraction_loss(d1, d2, h, f):
    """
    Diffraction loss over average terrain. This value is valid for losses
    greater than 15 dB.


    Parameters
    ----------
    - d1 : number, sequence, or numpy.ndarray
            Distances from the first terminal to the path obstruction. [km]
    - d2 : number, sequence, or numpy.ndarray
            Distances from the second terminal to the path obstruction. [km]
    - h : number, sequence, or numpy.ndarray
            Height difference between most significant path blockages
            and the path trajectory. h is negative if the top of the obstruction
            of interest is above the virtual line-of-sight. [m]
    - f : number
            Frequency of the link [GHz]


    Returns
    -------
    - A_d: Quantity
            Diffraction loss over average terrain  [dB]


    References
    ----------
    [1] Propagation data and prediction methods required for the design of
    terrestrial line-of-sight systems: https://www.itu.int/rec/R-REC-P.530/en
    """
    global __model
    type_output = type(d1)
    d1 = prepare_quantity(d1, u.km, 'Distance to the first terminal')
    d2 = prepare_quantity(d2, u.km, 'Distance to the second terminal')
    h = prepare_quantity(h, u.m, 'Height difference')
    f = prepare_quantity(f, u.GHz, 'Frequency')

    val = __model.diffraction_loss(d1, d2, h, f)
    return prepare_output_array(val, type_output) * u.m


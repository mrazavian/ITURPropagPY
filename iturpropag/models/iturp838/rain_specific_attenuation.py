# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u

from iturpropag.models.iturp838.rain_specific_attenuation_coefficients import rain_specific_attenuation_coefficients
from iturpropag.utils import prepare_quantity

def rain_specific_attenuation(R, f, el, tau):
    """
    Specific attenuation model for rain for use in prediction methods
    A method to compute the specific attenuation γ_R (dB/km) from rain. The
    value is obtained from the rain rate R (mm/h) using a power law
    relationship.

    ..math:
        \\gamma_R = k R^\\alpha


    Parameters
    ----------
    R : number, sequence, numpy.ndarray or Quantity
        Rain rate (mm/h)
    f : number or Quantity
        Frequency (GHz)
    el : number, sequence, or numpy.ndarray
        Elevation angle of the receiver points
    tau : number, sequence, or numpy.ndarray
        Polarization tilt angle relative to the horizontal (degrees). Tau = 45
        deg for circular polarization)


    Returns
    -------
    gamma_R: numpy.ndarray
        Specific attenuation from rain (dB/km)


    References
    ----------
    [1] Rain height model for prediction methods:
    https://www.itu.int/rec/R-REC-P.838/en
    """
    R = prepare_quantity(R, u.mm / u.hr, 'Rain rate')
    f = prepare_quantity(f, u.GHz, 'Frequency')
    
    k, alpha = rain_specific_attenuation_coefficients(f, el, tau)
    val = k * (R**alpha)
    return val * u.dB / u.km
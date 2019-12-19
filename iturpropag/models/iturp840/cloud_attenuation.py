# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u

from iturpropag.models.iturp840.specific_attenuation_coefficients import specific_attenuation_coefficients
from iturpropag.models.iturp840.columnar_content_reduced_liquid import columnar_content_reduced_liquid
from iturpropag.utils import prepare_input_array, prepare_output_array, prepare_quantity



def cloud_attenuation(lat, lon, el, f, p):
    """
    Attenuation due to clouds and fog: This Recommendation provides methods
    to predict the attenuation due to clouds and fog on Earth-space paths.

    A method to estimate the attenuation due to clouds along slant paths for
    a given probability.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - el : number, sequence, or numpy.ndarray
            Elevation angle of the receiver points (deg)
    - f : number
            Frequency (GHz)
    - p : number
            Percentage of time exceeded for p% of the average year


    Returns
    -------
    - p: numpy.ndarray
            Cloud attenuation (dB)



    References
    ----------
    [1] Attenuation due to clouds and fog:
    https://www.p.int/rec/R-REC-P.840/en
    """

    type_output = type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    el = prepare_quantity(el, u.deg, 'Elevation angle')
    f = prepare_quantity(f, u.GHz, 'Frequency')

    Kl = specific_attenuation_coefficients(f, T=0)
    Lred = columnar_content_reduced_liquid(lat, lon, p).value
    A = Lred * Kl / np.sin(np.deg2rad(el))

    return prepare_output_array(A, type_output) * u.dB

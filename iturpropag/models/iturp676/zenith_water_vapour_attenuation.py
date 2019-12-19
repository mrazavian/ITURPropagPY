# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
import os
from astropy import units as u

from iturpropag.models.iturp836.total_water_vapour_content import total_water_vapour_content
from iturpropag.models.iturp1511.topographic_altitude import topographic_altitude
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

    def zenith_water_vapour_attenuation(
            self, lat, lon, p, f, V_t=None, h=None):
        # Abstract method to compute the water vapour attenuation over the
        # slant path
        fcn = np.vectorize(self.instance.zenith_water_vapour_attenuation,
                           excluded=[0, 1, 4, 5], otypes=[np.ndarray])
        return np.array(fcn(lat, lon, p, f, V_t, h).tolist())


class _ITU676_11():

    def __init__(self):
        self.__version__ = 11
        self.year = 2017
        self.month = 12
        self.link = 'https://www.p.int/rec/R-REC-P.676-11-201712-S/en'

    
    def zenith_water_vapour_attenuation(
            self, lat, lon, p, f, V_t=None, h=None):
        f_ref = 20.6        # [GHz]
        p_ref = 815         # [hPa] Dry reference pressure
         
        if h is None:
            h = topographic_altitude(lat, lon).value

        if V_t is None:
            V_t = total_water_vapour_content(lat, lon, p, h).value

        rho_ref = V_t / 3.67
        t_ref = 14 * np.log(0.22 * V_t / 3.67) + 3    # [Celsius]

        a = (0.2048 * np.exp(- ((f - 22.43)/3.097)**2) +
             0.2326 * np.exp(- ((f-183.5)/4.096)**2) +
             0.2073 * np.exp(- ((f-325)/3.651)**2) - 0.113)

        b = 8.741e4 * np.exp(-0.587 * f) + 312.2 * f**(-2.38) + 0.723
        h = np.minimum(h, 4)

        gammaw_approx_vect = gammaw_approx

        P_ref = p_ref + rho_ref * (t_ref + 273.15) / 216.7  # Total reference pressure
        Aw_term1 = (0.0176 * V_t *
                    gammaw_approx_vect(f, P_ref, rho_ref, t_ref + 273.15).value /
                    gammaw_approx_vect(f_ref, P_ref, rho_ref, t_ref + 273.15).value)

        return np.where(np.logical_and(1 <= f, f <= 20), Aw_term1, Aw_term1 * (a * h ** b + 1))


class _ITU676_10():

    def __init__(self):
        self.__version__ = 10
        self.year = 2013
        self.month = 9
        self.link = 'https://www.p.int/rec/R-REC-P.676-10-201309-S/en'

    
    def zenith_water_vapour_attenuation(
            self, lat, lon, p, f, V_t=None, h=None):
        f_ref = 20.6        # [GHz]
        p_ref = 780         # [hPa]  Dry reference pressure

        if h is None:
            h = topographic_altitude(lat, lon).value
            
        if V_t is None:
            V_t = total_water_vapour_content(lat, lon, p, h).value

        rho_ref = V_t / 4     # [g/m3]
        t_ref = 14 * np.log(0.22 * V_t / 4) + 3    # [Celsius]

        gammaw_approx_vect = np.vectorize(gammaw_approx)

        P_ref = p_ref + rho_ref * (t_ref + 273.15) / 216.7  # Total reference pressure
        Aw = (0.0173 * V_t *
                gammaw_approx_vect(f, P_ref, rho_ref, t_ref + 273.15).value /
                gammaw_approx_vect(f_ref, P_ref, rho_ref, t_ref + 273.15).value)
        return Aw


class _ITU676_9():

    def __init__(self):
        self.__version__ = 9
        self.year = 2012
        self.month = 2
        self.link = 'https://www.p.int/rec/R-REC-P.676-9-201202-S/en'

    # Recommendation ITU-P R.676-9 has most of the methods similar to those
    # in Recommendation ITU-P R.676-10.
    def zenith_water_vapour_attenuation(self, *args, **kwargs):
        return _ITU676_10.zenith_water_vapour_attenuation(*args, **kwargs)


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
def zenith_water_vapour_attenuation(lat, lon, p, f, V_t=None, h=None):
    """
    An alternative method may be used to compute the slant path attenuation by
    water vapour, in cases where the integrated water vapour content along the
    path, ``V_t``, is known.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - p : number
            Percentage of time exceeded for p% of the average year for calculation of 
            total water vapour content(Vt)
    - f : number or Quantity
            Frequency (GHz)
    - V_t : number or Quantity, optional
            Integrated water vapour content along the path (kg/m2 or mm).
            If not provided this value is estimated using Recommendation
            ITU-R P.836. Default value None
    - h : number, sequence, or numpy.ndarray
            Altitude of the receivers. If None, use the topographical altitude as
            described in recommendation ITU-R P.1511


    Returns
    -------
    - A_w : Quantity
            Zenith Water vapour attenuation along the slant path (dB)

    References
    --------
    [1] Attenuation by atmospheric gases:
    https://www.p.int/rec/R-REC-P.676/en
    """
    type_output = type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    f = prepare_quantity(f, u.GHz, 'Frequency')
    p = prepare_quantity(p, u.pct, 'Percentage of the time')
    V_t = prepare_quantity(V_t, u.kg / u.m**2,
                'Integrated water vapour content along the path')
    h = prepare_quantity(h, u.km, 'Altitude')
    val = __model.zenith_water_vapour_attenuation(lat, lon, p, f, V_t=V_t, h=h)
    return prepare_output_array(val, type_output) * u.dB

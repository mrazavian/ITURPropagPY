# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from astropy import units as u

from iturpropag.models.iturp1144.bilinear_2D_interpolator import bilinear_2D_interpolator
from iturpropag.utils import prepare_input_array, prepare_quantity, load_data,\
    prepare_output_array, dataset_dir


class __ITU453():
    """ Implementation of the methods in Recommendation ITU-R P.453
    "The radio refractive index: its formula and refractivity data"

    Available versions:
       * P.453-13 (12/17)
       * P.453-12 (07/15)

    Recommendation ITU-R P.453 provides methods to estimate the radio
    refractive index and its behaviour for locations worldwide; describes both
    surface and vertical profile characteristics; and provides global maps for
    the distribution of refractivity parameters and their statistical
    variation.
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.453 recommendation.

    def __init__(self, version=13):
        if version == 13:
            self.instance = _ITU453_13()
        elif version == 12:
            self.instance = _ITU453_12()
        else:
            raise ValueError(
                'Version {0} is not implemented for the ITU-R P.453 model.'
                .format(version))

    @property
    def __version__(self):
        return self.instance.__version__

    def map_wet_term_radio_refractivity(self, lat, lon, p):
        fcn = np.vectorize(self.instance.map_wet_term_radio_refractivity,
                           excluded=[0, 1], otypes=[np.ndarray])
        return np.array(fcn(lat, lon, p).tolist())

class _ITU453_13():

    def __init__(self):
        self.__version__ = 13
        self.year = 2017
        self.month = 12
        self.link = 'https://www.itu.int/rec/R-REC-P.453-13-201712-I/en'

        self._N_wet = {}

    def N_wet(self, lat, lon, p):
        if not self._N_wet:
            ps = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 60, 70, 80,
                  90, 95, 99]
            d_dir = os.path.join(dataset_dir, 'p453/v13_NWET_Annual_%s.txt')
            lats = load_data(os.path.join(dataset_dir, 'p453/v13_LAT_N.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p453/v13_LON_N.txt'))
            for p_loads in ps:
                vals = load_data(d_dir % (str(p_loads).replace('.', '')))
                self._N_wet[float(p_loads)] = bilinear_2D_interpolator(
                    np.flipud(lats), lons, np.flipud(vals))

        lon[lon > 180] = lon[lon > 180] - 360
        return self._N_wet[float(p)](
                np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def map_wet_term_radio_refractivity(self, lat, lon, p):
       
        # Fix lon because the data-set is now indexed -180 to 180 instead
        # of 0 to 360
        lon[lon > 180] = lon[lon > 180] - 360

        lat_f = lat.flatten()
        lon_f = lon.flatten()

        available_p = np.array([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10,
                                20, 30, 50, 60, 70, 80, 90, 95, 99])

        if p in available_p:
            p_below = p_above = p
            pExact = True
        else:
            pExact = False
            idx = available_p.searchsorted(p, side='right') - 1
            idx = np.clip(idx, 0, len(available_p) - 1)

            p_below = available_p[idx]
            idx = np.clip(idx + 1, 0, len(available_p) - 1)
            p_above = available_p[idx]

        R = -(lat_f - 90) // 0.75
        C = (lon_f + 180) // 0.75

        lats = np.array([90 - R * 0.75, 90 - (R + 1) * 0.75,
                         90 - R * 0.75, 90 - (R + 1) * 0.75])

        lons = np.array([C * 0.75, C * 0.75,
                         (C + 1) * 0.75, (C + 1) * 0.75]) - 180

        r = - (lat_f - 90) / 0.75
        c = (lon_f + 180) / 0.75

        N_wet_a = self.N_wet(lats, lons, p_above)
        N_wet_a = (N_wet_a[0, :] * ((R + 1 - r) * (C + 1 - c)) +
                   N_wet_a[1, :] * ((r - R) * (C + 1 - c)) +
                   N_wet_a[2, :] * ((R + 1 - r) * (c - C)) +
                   N_wet_a[3, :] * ((r - R) * (c - C)))

        if not pExact:
            N_wet_b = self.N_wet(lats, lons, p_below)
            N_wet_b = (N_wet_b[0, :] * ((R + 1 - r) * (C + 1 - c)) +
                       N_wet_b[1, :] * ((r - R) * (C + 1 - c)) +
                       N_wet_b[2, :] * ((R + 1 - r) * (c - C)) +
                       N_wet_b[3, :] * ((r - R) * (c - C)))

        # Compute the values of Lred_a
        if not pExact:
            rho = N_wet_b + (N_wet_a - N_wet_b) * \
                (np.log(p) - np.log(p_below)) / \
                (np.log(p_above) - np.log(p_below))
            return rho.reshape(lat.shape)
        else:
            return N_wet_a.reshape(lat.shape)


class _ITU453_12():

    def __init__(self):
        self.__version__ = 12
        self.year = 2016
        self.month = 9
        self.link = 'https://www.itu.int/rec/R-REC-P.453-12-201609-I/en'

        self._N_wet = {}

    def N_wet(self, lat, lon):
        if not self._N_wet:
            vals = load_data(os.path.join(dataset_dir, 'p453/v12_ESANWET.txt'))
            lats = load_data(os.path.join(dataset_dir, 'p453/v12_ESALAT.txt'))
            lons = load_data(os.path.join(dataset_dir, 'p453/v12_ESALON.txt'))
            self._N_wet = bilinear_2D_interpolator(lats, lons, vals)

        return self._N_wet(
            np.array([lat.ravel(), lon.ravel()]).T).reshape(lat.shape)

    def map_wet_term_radio_refractivity(self, lat, lon, p):
        return self.N_wet(lat, lon)


__model = __ITU453()


def change_version(new_version):
    """
    Change the version of the ITU-R P.453 recommendation currently being used.


    Parameters
    ----------
    new_version : int
        Number of the version to use.
        Valid values are:
           * P.453-13  (Current version)
           * p.453-12
    """
    global __model
    __model = __ITU453(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.453 recommendation currently being used.
    """
    global __model
    return __model.__version__


def map_wet_term_radio_refractivity(lat, lon, p):
    """
    Method to determine the wet term of the radio refractivity


    Parameters
    ------------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points


    Returns
    -----------
    - N_wet : Quantity
            Wet term of the radio refractivity (-)



    References
    ----------
    [1] The radio refractive index: its formula and refractivity data
    https://www.itu.int/rec/R-REC-P.453/en
    """
    global __model
    type_output = type(lat)
    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)
    lon = np.mod(lon, 360)
    val = __model.map_wet_term_radio_refractivity(lat, lon, p)
    return prepare_output_array(val, type_output) * u.dimensionless_unscaled


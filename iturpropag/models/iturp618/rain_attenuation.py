# -*- coding: utf-8 -*-
# checked by <TP>: 2019-06-06
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as stats
import scipy.special
import scipy.integrate
from astropy import units as u
import warnings


from iturpropag.models.iturp837.rainfall_rate import rainfall_rate
from iturpropag.models.iturp838.rain_specific_attenuation import rain_specific_attenuation
from iturpropag.models.iturp839.rain_height import rain_height
from iturpropag.models.iturp1511.topographic_altitude import topographic_altitude
from iturpropag.utils import prepare_input_array, prepare_output_array,\
    prepare_quantity, compute_distance_earth_to_earth, memory



class _ITU618():
    """
    Propagation data and prediction methods required for the design of
    Earth-space telecommunication systems.

    Available versions include:
       * P.618-13 (12/17) (Current version)
       * P.618-12 (07/15) (Superseded)

    Versions that need to be implemented
       * P.618-11
       * P.618-10
       * P.618-09
       * P.618-08
       * P.618-07
       * P.618-06
       * P.618-05
       * P.618-04
       * P.618-03
       * P.618-02
       * P.618-01

    Recommendation ITU-R P.618 provides methods to estimate the propagation
    loss on an Earth-space path, relative to the free-space loss. This value
    is the sum of different contributions as follows:
    * attenuation by atmospheric gases;
    * attenuation by rain, other precipitation and clouds;
    * focusing and defocusing;
    * decrease in antenna gain due to wave-front incoherence;
    * scintillation and multipath effects;
    * attenuation by sand and dust storms.
    Each of these contributions has its own characteristics as a function of
    frequency, geographic location and elevation angle. As a rule, at elevation
    angles above 10°, only gaseous attenuation, rain and cloud attenuation and
    possibly scintillation will be significant, depending on propagation
    conditions.
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.618 recommendation.

    def __init__(self, version=13):
        if version == 13:
            self.instance = _ITU618_13()
        elif version == 12:
            self.instance = _ITU618_12() # Rev.12 is equal Rev. 13
        else:
            err_msg = 'Version {0} is not implemented for the ITU-R P.618 model.'
            raise ValueError(err_msg.format(version))

    @property
    def __version__(self):
        return self.instance.__version__

    def rain_attenuation(self, lat, lon, f, el, p, tau, hs=None, R001=None):
        fcn = np.vectorize(self.instance.rain_attenuation,
                           excluded=[0, 1, 3, 4, 6], otypes=[np.ndarray])
        return np.array(fcn(lat, lon, f, el, hs, p, R001, tau).tolist())


class _ITU618_13():

    def __init__(self):
        self.__version__ = 13

    def rain_attenuation(self, lat, lon, f, el, hs=None, p=0.01, R001=None,
                         tau=45):
        if np.logical_or(p < 0.001, p > 5).any():
            warning_msg = '\n\"The method to compute the rain attenuation in '+\
                          'recommendation ITU-P 618-13 is only valid for '+\
                          'unavailability values of p between 0.001 and 5. '+\
                          'here the p is '+ str(p) +'\"'      
            warnings.warn(RuntimeWarning(warning_msg))


        Re = 8500.   # Efective radius of the Earth (8500 km)

        if hs is None:
            hs = topographic_altitude(lat, lon).to(u.km).value # P.1511
        
        # Step 1: Compute the rain height (hr) based on ITU - R P.839
        hr = rain_height(lat, lon).value
        
        # Step 2: Compute the slant path length
        
        Ls = np.where(el >= 5,
                (hr - hs) / (np.sin(np.deg2rad(el))),                   # Eq. 1
                2* (hr - hs) / (((np.sin(np.deg2rad(el)))**2 +          # Eq. 2
                2* (hr - hs) / Re)**0.5 + (np.sin(np.deg2rad(el)))))
        
        # Step 3: Calculate the horizontal projection, LG, of the
        # slant-path length
        Lg = np.abs(Ls * np.cos(np.deg2rad(el)))                        # Eq. 3
        
        # Obtain the rainfall rate, exceeded for 0.01% of an average year,
        # with ab ubtegratuib tune if 1 minute,
        # if not provided, as described in ITU-R P.837.
        if R001 is None:
            R001 = rainfall_rate(lat, lon, 0.01).to(u.mm / u.hr).value
        
        # Step 5: Obtain the specific attenuation gammar using the frequency
        # dependent coefficients as given in ITU-R P.838
        # https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.838-3-200503-I!!PDF-E.pdf
        # Eq. 4
        gammar = rain_specific_attenuation(R001, f, el, tau).to(u.dB / u.km).value
        
        # Step 6: Calculate the horizontal reduction factor, r0.01,
        # for 0.01% of the time:
        r001 = 1. / (1. + 0.78 * np.sqrt(Lg * gammar / f) - # Eq. 5
                     0.38 * (1. - np.exp(-2. * Lg)))

        # Step 7: Calculate the vertical adjustment factor, v0.01,
        # for 0.01% of the time:
        eta = np.rad2deg(np.arctan2(hr - hs, Lg * r001)) # <TP> TBC
        
        Lr = np.where(eta > el,
                        Lg * r001 / np.cos(np.deg2rad(el)),
                        (hr - hs) / np.sin(np.deg2rad(el)))

        xi = np.where(np.abs(lat) < 36, 36. - np.abs(lat), 0.)

        v001 = 1. / (1 + np.sqrt(np.sin(np.deg2rad(el))) *
                     (31 * (1 - np.exp(-(el / (1 + xi)))) *
                      np.sqrt(Lr * gammar) / f**2 - 0.45))
        
        # Step 8: calculate the effective path length:
        Le = Lr * v001   # (km)                             # Eq. 6

        # Step 9: The predicted attenuation exceeded for 0.01% of an
        # average year
        A001 = gammar * Le   # (dB)                         # Eq. 7
        
        # Step 10: The estimated attenuation to be exceeded for other
        # percentages of an average year
        if p >= 1:
            beta = np.zeros_like(A001)
        else:
            beta = np.where(np.abs(lat) > 36,
                            np.zeros_like(A001),
                            np.where((np.abs(lat) < 36) & (el > 25),
                                     -0.005 * (np.abs(lat) - 36),
                                     -0.005 * (np.abs(lat) - 36) + 1.8 -
                                     4.25 * np.sin(np.deg2rad(el))))
        
        # <TP> following to avoid division by zero error when calculating log(A001==0)
        loc = np.where(A001 > 0)
        log_res = np.zeros_like(A001, dtype=float)
        log_res[loc] = np.log(A001[loc])
        A = A001 * np.power(p / 0.01, (-1)*(0.655 + 0.033 * np.log(p) - 0.045 * 
                log_res - beta * (1 - p) * np.sin(np.deg2rad(el))))  # Eq. 8   
        return A


class _ITU618_12():

    def __init__(self):
        self.__version__ = 12

    def rain_attenuation(self, *args, **kwargs):
        return _ITU618_13().rain_attenuation(*args, **kwargs)

__model = _ITU618()


def change_version(new_version):
    global __model
    __model = _ITU618(new_version)
    memory.clear()


def get_version():
    global __model
    return __model.__version__


def rain_attenuation(lat, lon, f, el, p, tau, hs=None, R001=None):
    """
    Calculation of long-term rain attenuation statistics from point rainfall
    rate.
    The following procedure provides estimates of the long-term statistics of
    the slant-path rain attenuation at a given location for frequencies up
    to 55 GHz.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - f : number
            Frequency (GHz)
    - el : sequence, or number
            Elevation angle (degrees)
    - hs : number, sequence, or numpy.ndarray, optional
            Heigh above mean sea level of the earth station (km). If local data for
            the earth station height above mean sea level is not available, an
            estimate is obtained from the maps of topographic altitude
            given in Recommendation ITU-R P.1511.
    - p : number, optional
            Percetage of the time the rain attenuation value is exceeded.
    - R001: number, optional
            Point rainfall rate for the location for 0.01% of an average year
            (mm/h).
            If not provided, an estimate is obtained from Recommendation
            Recommendation ITU-R P.837. Some useful values:
                * 0.25 mm/h : Drizzle
                * 2.5  mm/h : Light rain
                * 12.5 mm/h : Medium rain
                * 25.0 mm/h : Heavy rain
                * 50.0 mm/h : Downpour
                * 100  mm/h : Tropical
                * 150  mm/h : Monsoon
    - tau : number, optional
            Polarization tilt angle relative to the horizontal (degrees)
            (tau = 45 deg for circular polarization).


    Returns
    -------
    - attenuation: Quantity
            Attenuation due to rain (dB)

    References
    --------
    [1] Propagation data and prediction methods required for the design of
    Earth-space telecommunication systems:
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-12-201507-I!!PDF-E.pdf
    """
    global __model
    type_output = type(lat)

    lat = prepare_input_array(lat)
    lon = prepare_input_array(lon)

    lon = np.mod(lon, 360)
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(prepare_input_array(el), u.deg, 'Elevation angle')
    hs = prepare_quantity(
        hs, u.km, 'Heigh above mean sea level of the earth station')
    R001 = prepare_quantity(R001, u.mm / u.hr, 'Point rainfall rate')
    tau = prepare_quantity(tau, u.one, 'Polarization tilt angle')
    p = prepare_quantity(p, u.pct, 'Percetage of the time')

    val = __model.rain_attenuation(lat, lon, f, el, p, tau, hs=hs, R001=R001)
    return prepare_output_array(val, type_output) * u.dB

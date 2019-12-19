# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u
import scipy.stats as stats
from scipy.signal import lfilter

from iturpropag.models.iturp836.total_water_vapour_content import total_water_vapour_content
from iturpropag.models.iturp1853.integrated_water_vapour_coefficients import integrated_water_vapour_coefficients
from iturpropag.models.iturp1853.integrated_water_vapour_synthesis import integrated_water_vapour_synthesis
from iturpropag.models.iturp676.zenith_water_vapour_attenuation import zenith_water_vapour_attenuation
from iturpropag.utils import prepare_quantity, prepare_input_array,\
                            compute_distance_earth_to_earth


class __ITU1853():
    """Tropospheric attenuation time series synthesis

    Available versions include:
    * P.1853-0 (10/09) (Superseded)
    * P.1853-1 (02/12) (Superseded)
    * P.1853-2 (08/2019) (Current Version)
    """
    # This is an abstract class that contains an instance to a version of the
    # ITU-R P.1853 recommendation.

    def __init__(self, version=2):
        if version == 2:
            self.instance = _ITU1853_2()
        elif version == 1:
            self.instance = _ITU1853_1()
        elif version == 0:
            self.instance = _ITU1853_0()
        else:
            raise ValueError(
                'Version {0} is not implemented for the ITU-R P.1853 model.'
                .format(version))

    @property
    def __version__(self):
        return self.instance.__version__

    def water_vapour_attenuation_synthesis(self, lat, lon, f, Ns, Ts=1, n=None):
        return self.instance.water_vapour_attenuation_synthesis(
                lat, lon, f, Ns, Ts=Ts, n=n)


class _ITU1853_2():

    def __init__(self):
        self.__version__ = 2
        self.year = 2019
        self.month = 8
        self.link = 'https://www.itu.int/rec/R-REC-P.1853-2-201908-I/en'

        # Step B:
        # SS_WV_5:  Low-pass filter parameter
        self.beta_wv = 3.65e-6

    def single_site_water_vapour_attenuation_synthesis(
                            self, lat, lon, f, n, Ts=1):
        # Step A: Estimation of κ and λ
        kappa, lambd = integrated_water_vapour_coefficients(lat, lon, f)

        # Step C: 
        # SS_WV_8: Filter the noise time series, with two recursive low-pass
        # filters
        rho = np.exp(-self.beta_wv * Ts)
        Gwv = lfilter([np.sqrt(1 - rho**2)], [1, -rho], n, 0)
        # SS_WV_9: Compute Awv(kTs),
        Awv = lambd * (- np.log10(stats.norm.sf(Gwv)))**(1 / kappa)

        return Awv.flatten()

    def water_vapour_attenuation_synthesis(self, lat, lon, f, Ns, Ts=1, n=None):
        # Step C: 
        # MS_WV_3: Synthesize a white Gaussian noise time series page=7
        if n is None:
            n = np.random.normal(0, 1, \
                (np.size(lat), int(Ns * Ts + 5e6)))[::Ts]
            discard_samples = True
        else:
            discard_samples = False

        # MS_WV_4: calculate the matrix Rn=[rnij]
        rho_wv = np.exp(-self.beta_wv * Ts)

        D = np.zeros( (np.size(lat), np.size(lat)) )
        for ii,_ in enumerate(lat):
            D[ii,:] = compute_distance_earth_to_earth(lat[ii], lon[ii], lat, lon)

        rG = 0.29 * np.exp(-D/38) + 0.71 * np.exp(-D/900)

        Rn = rG 

        # MS_WV_5~6: calculate cholesky factorization
        CR = np.linalg.cholesky(Rn)

        n =  np.matmul(CR, n)

        # MS_WV_7: single site water vapour attenuation
        A_wv = np.zeros_like(n)
        for ii,_ in enumerate(lat):
            A_wv[ii,:] = self.single_site_water_vapour_attenuation_synthesis(
                                    lat[ii], lon[ii], f[ii], n[ii,:], Ts=1)

        # D6: Discard the first 5 000 000 samples from the synthesized
        if discard_samples:
            A_wv = A_wv[:, np.ceil(5e6/Ts).astype(int):]

        return A_wv

class _ITU1853_1():

    def __init__(self):
        self.__version__ = 1
        self.year = 2012
        self.month = 2
        self.link = 'https://www.p.int/rec/R-REC-P.1853-1-201202-I/en'

    def water_vapour_attenuation_synthesis(self, lat, lon, f, Ns, Ts=1, n=None):
        
        V = integrated_water_vapour_synthesis(lat, lon, Ns, Ts=Ts, n=n)

        Awv = zenith_water_vapour_attenuation(lat, lon, None, f, V_t=V).value

        return Awv.flatten()


class _ITU1853_0():

    def __init__(self):
        self.__version__ = 0
        self.year = 2009
        self.month = 10
        self.link = 'https://www.p.int/rec/R-REC-P.1853-0-200910-I/en'

    def integrated_water_vapour_synthesis(self, *args, **kwargs):
        raise NotImplementedError(
            "Recommendation ITU-R P.1853-0 does not specify a method to compute "
            "time series for the water vapour content.")


__model = __ITU1853()


def change_version(new_version):
    """
    Change the version of the ITU-R P.1853 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.1853-2 (08/2019) (Current version)
            * P.1853-1 (02/12)
    """
    global __model
    __model = __ITU1853(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.1853 recommendation currently being used.
    """
    global __model
    return __model.__version__


def water_vapour_attenuation_synthesis(lat, lon, f, Ns, Ts=1, n=None):
    """ The time series synthesis method generates a time series that
    reproduces the spectral characteristics and the distribution of water
    vapour content.


    Parameters
    ----------
    - lat : number, sequence, or numpy.ndarray
            Latitudes of the receiver points
    - lon : number, sequence, or numpy.ndarray
            Longitudes of the receiver points
    - f : number or Quantity
            Frequency (GHz)
    - Ns : int
            Number of samples
    - Ts : int
            Time step between consecutive samples (seconds)
    - n : list, np.array, optional
            Additive White Gaussian Noise used as input for the

    Returns
    -------
    - Awv: numpy.ndarray
            Synthesized water vapour attenuation time series (dB)



    References
    ----------
    [1] P.1853 : Time series synthesis of tropospheric impairments
    https://www.itu.int/rec/R-REC-P.1853/en
    """
    global __model

    # prepare the input array
    lon = np.mod(lon, 360)
    lat = prepare_input_array(lat).flatten()
    lon = prepare_input_array(lon).flatten()
    f = prepare_input_array(f).flatten()

    # prepare quantity
    f = prepare_quantity(f, u.GHz, 'Frequency')

    # calculate the output
    val_1 = __model.water_vapour_attenuation_synthesis(lat, lon, f, Ns, Ts=Ts, n=n)
    return val_1 * u.dB

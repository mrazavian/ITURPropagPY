# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy import units as u
import scipy.stats as stats
from scipy.signal import lfilter

from iturpropag.models.iturp618.scintillation_attenuation_sigma import scintillation_attenuation_sigma
from iturpropag.models.iturp676.gamma0_exact import gamma0_exact
from iturpropag.models.iturp676.slant_inclined_path_equivalent_height import slant_inclined_path_equivalent_height
from iturpropag.models.iturp676.zenith_water_vapour_attenuation import zenith_water_vapour_attenuation
from iturpropag.models.iturp840.specific_attenuation_coefficients import specific_attenuation_coefficients
from iturpropag.models.iturp835.standard_pressure import standard_pressure
from iturpropag.models.iturp835.standard_water_vapour_density import standard_water_vapour_density
from iturpropag.models.iturp1510.surface_mean_temperature import surface_mean_temperature
from iturpropag.models.iturp1511.topographic_altitude import topographic_altitude
from iturpropag.models.iturp1853.rain_attenuation_synthesis import rain_attenuation_synthesis
from iturpropag.models.iturp1853.scintillation_attenuation_synthesis import scintillation_attenuation_synthesis
from iturpropag.models.iturp1853.cloud_attenuation_synthesis import cloud_attenuation_synthesis
from iturpropag.models.iturp1853.cloud_liquid_water_synthesis import cloud_liquid_water_synthesis
from iturpropag.models.iturp1853.integrated_water_vapour_coefficients import integrated_water_vapour_coefficients
from iturpropag.models.iturp1853.water_vapour_attenuation_synthesis import water_vapour_attenuation_synthesis
from iturpropag.models.iturp1853.integrated_water_vapour_synthesis import integrated_water_vapour_synthesis
from iturpropag.models.iturp1853.surface_mean_pressure import surface_mean_pressure
from iturpropag.models.iturp1853.surface_mean_water_vapour_density import surface_mean_water_vapour_density
from iturpropag.utils import prepare_quantity, prepare_input_array



class __ITU1853():
    """Tropospheric attenuation time series synthesis

    Available versions include:
    * P.1853-0 (10/09) (Superseded)
    * P.1853-1 (02/12) (Superseded)
    * P.1853-2 (08/2019) (Current version)
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

    def total_attenuation_synthesis(self, lat, lon, f, el, p, D, Ns, tau, eta, 
                                    Ts=1, hs=None, rho=None,
                                    H=None, P=None, hL=1000,
                                    return_contributions=False):
        return self.instance.total_attenuation_synthesis(
            lat, lon, f, el, p, D, Ns, tau, eta, Ts=Ts, hs=hs, rho=rho,
            H=H, P=P, hL=hL, return_contributions=return_contributions)


class _ITU1853_2():

    def __init__(self):
        self.__version__ = 2
        self.year = 2019
        self.month = 8
        self.link = 'https://www.p.int/rec/R-REC-P.1853-2-201905-I/en'

    
    def total_attenuation_synthesis(self, lat, lon, f, el, p, D, Ns, tau, eta, 
                                    Ts=1, hs=None, rho=None,
                                    H=None, P=None, hL=1000,
                                   return_contributions=False):
        t_disc = int(5e6)
 
        # Step MS_TOT_1: 
        # Synthesize a white Gaussian noise time series page=18
        n = np.random.normal(0, 1, \
                        (np.size(lat), int(Ns * Ts + t_disc)))[::Ts]
        # Step MS_TOT_2:
        # Compute the annual mean oxygen gaseous attenuation
        Tm = surface_mean_temperature(lat, lon).value

        if P is None:
            P = surface_mean_pressure(lat, lon).value
        
        if rho is None:
            rho = surface_mean_water_vapour_density(lat, lon).value
        # Convert the mean annual temperature, pressure, water vapour to
        # oxygen attenuation AO following the method recommended in
        # Recommendation ITU-R P.676.
        e = Tm * rho / 216.7
        go = gamma0_exact(f, P, rho, Tm).value
        
        ho, hw = slant_inclined_path_equivalent_height(f, P).value

        Ao = np.atleast_2d(ho * go / np.sin(np.deg2rad(el))).T @ np.ones((1, n.shape[-1]))
        Ao = Ao[:, np.ceil(t_disc/Ts).astype(int):]

        # Step MS_TOT_3:
        # calculate the water vapour attenuation time-series
        Awv = water_vapour_attenuation_synthesis(lat, lon, f, Ns, Ts=Ts, n=n).value
        Awv = Awv[:, np.ceil(t_disc/Ts).astype(int):]

        # Step MS_TOT_4:
        # Calculate the cloud attenuation time-series
        Ac = cloud_attenuation_synthesis(lat, lon, f, el, Ns, Ts=1, n=n,\
                    rain_contribution=True).value
        Ac = Ac[:, np.ceil(t_disc/Ts).astype(int):]

        # Step MS_TOT_5:
        # Calculate the rain attenuation time series
        Ar = rain_attenuation_synthesis(lat, lon, f, el, tau, 
                            Ns, hs=hs, Ts=1, n=n).value
        Ar = Ar[:, np.ceil(t_disc/Ts).astype(int):]

        # Step MS_TOT_6:
        # limit the cloud attenuation time-series
        Ac_thresh = specific_attenuation_coefficients(f, T=0) / np.sin(np.deg2rad(el))
        Ac_thresh = np.atleast_2d(Ac_thresh).T @ np.ones((1, Ac.shape[-1])) 
        Ac = np.where(np.logical_and(0 < Ar, Ac_thresh < Ac),\
                        Ac_thresh, Ac)

        # Step MS_TOT_7: Scintillation fading and enhancement polynomials
        def a_Fade(p):
            return -0.061 * np.log10(p)**3 + 0.072 * \
                np.log10(p)**2 - 1.71 * np.log10(p) + 3

        def a_Enhanc(p):
            return -0.0597 * np.log10(p)**3 - 0.0835 * \
                np.log10(p)**2 - 1.258 * np.log10(p) + 2.672

        # Step MS_TOT_8: Synthesize unit variance scintillation time series
        sci_0 = np.zeros_like(Awv)
        for ii,_ in enumerate(lat):
            sci_0[ii,:] = scintillation_attenuation_synthesis(Ns, Ts=Ts).value

        # Step MS_TOT_9: Compute the correction coefficient time series Cx(kTs) in
        # order to distinguish between scintillation fades and enhancements:
        Q_sci = 100 * stats.norm.sf(sci_0)
        C_x = np.where(sci_0 > 0, a_Fade(Q_sci) / a_Enhanc(Q_sci), 1)

        #Step MS_TOT_10:
        # limitation on Cx
        C_x = np.where(np.logical_or(C_x < 1, Q_sci > 45), 1, C_x)
        
        # Step MS_TOT_11: Compute the scintillation standard deviation σ following
        # the method recommended in Recommendation ITU-R P.618.
        sigma = np.atleast_1d(
                scintillation_attenuation_sigma(lat, lon, f, el, D, \
                                            eta, Tm, H, P, hL).value)
        
        # Step MS_TOT_12: 
        # Transform the intermediate underlying Gaussian 
        # process Gwv(kTs) into Gamma distribution time series Z(kTs) as follows:
        kappa, lambd = integrated_water_vapour_coefficients(lat, lon, f)

        Z = np.zeros_like(Awv)
        for ii,_ in enumerate(lat):
            Q_Gwv = np.exp(-(Awv[ii,:] / lambd[ii])**kappa[ii])
            Z[ii,:] = stats.gamma.ppf(1 - Q_Gwv, 10, scale=sigma[ii]/10)

        # Step MS_TOT_13: 
        # Compute the scintillation time series sci:
        As = np.where(Ar > 1, sci_0 * C_x * Z * Ar ** (5 / 12),
                      sci_0 * C_x * Z)

        # Step MS_TOT_14: Compute total tropospheric attenuation time series A(kTs)
        # as follows:
        A = Ar + Ac + Awv + Ao + As

        if return_contributions:
            return (Ao + Awv)[::Ts], Ac[::Ts], Ar[::Ts], As[::Ts], A[::Ts]
        else:
            return A[::Ts]



class _ITU1853_1():

    def __init__(self):
        self.__version__ = 1
        self.year = 2012
        self.month = 2
        self.link = 'https://www.p.int/rec/R-REC-P.1853-1-201202-I/en'

    
    def total_attenuation_synthesis(self, lat, lon, f, el, p, D, Ns, tau, eta, 
                                    Ts=1, hs=None, rho=None,
                                    H=None, P=None, hL=1000,
                                    return_contributions=False):
        t_disc = int(5e6)
        # Step A Correlation coefficients:
        C_RC = 1
        C_CV = 0.8

        # Step B Scintillation polynomials
        def a_Fade(p):
            return -0.061 * np.log10(p)**3 + 0.072 * \
                np.log10(p)**2 - 1.71 * np.log10(p) + 3

        def a_Enhanc(p):
            return -0.0597 * np.log10(p)**3 - 0.0835 * \
                np.log10(p)**2 - 1.258 * np.log10(p) + 2.672

        # Step C1-C3:
        n_R = np.random.normal(0, 1, int((Ns * Ts + t_disc)))
        n_L0 = np.random.normal(0, 1, int((Ns * Ts + t_disc)))
        n_V0 = np.random.normal(0, 1, int((Ns * Ts + t_disc)))

        # Step C4-C5:
        n_L = C_RC * n_R + np.sqrt(1 - C_RC**2) * n_L0
        n_V = C_CV * n_L + np.sqrt(1 - C_CV**2) * n_V0

        # Step C6: Compute the rain attenuation time series
        if hs is None:
            hs = topographic_altitude(lat, lon)
        Ar = rain_attenuation_synthesis(
            lat, lon, f, el, tau, Ns, hs=hs, Ts=1, n=n_R).value
        Ar = Ar[t_disc:]

        # Step C7: Compute the cloud integrated liquid water content time
        # series
        L = cloud_liquid_water_synthesis(lat, lon, Ns, Ts=1, n=n_L).value
        L = L[t_disc:]
        Ac = L * \
            specific_attenuation_coefficients(f, T=0) / np.sin(np.deg2rad(el))
        Ac = Ac.flatten()

        # Step C9: Identify time stamps where A_R > 0 L > 1
        idx = np.where(np.logical_and(Ar > 0, L > 1))[0]
        idx_no = np.where(np.logical_not(
                    np.logical_and(Ar > 0, L > 1)))[0]

        # Step C10: Discard the previous values of Ac and re-compute them by
        # linear interpolation vs. time starting from the non-discarded cloud
        # attenuations values
        Ac[idx] = np.interp(idx, idx_no, Ac[idx_no])

        # Step C11: Compute the integrated water vapour content time series
        V = integrated_water_vapour_synthesis(lat, lon, Ns, Ts=1, n=n_V).value
        V = V[t_disc:]

        # Step C12: Convert the integrated water vapour content time series
        # V into water vapour attenuation time series AV(kTs)
        Av = zenith_water_vapour_attenuation(lat, lon, p, f, V_t=V).value

        # Step C13: Compute the mean annual temperature Tm for the location of
        # interest using experimental values if available.
        Tm = surface_mean_temperature(lat, lon).value

        # Step C14: Convert the mean annual temperature Tm into mean annual
        # oxygen attenuation AO following the method recommended in
        # Recommendation ITU-R P.676.
        if P is None:
            P = standard_pressure(hs).value

        if rho is None:
            rho = standard_water_vapour_density(hs).value

        e = Tm * rho / 216.7
        go = gamma0_exact(f, P, rho, Tm).value
        ho, hw = slant_inclined_path_equivalent_height(f, P).value
        Ao = ho * go * np.ones_like(Ar)

        # Step C15: Synthesize unit variance scintillation time series
        sci_0 = scintillation_attenuation_synthesis(Ns, Ts=1).value

        # Step C16: Compute the correction coefficient time series Cx(kTs) in
        # order to distinguish between scintillation fades and enhancements:
        Q_sci = 100 * stats.norm.sf(sci_0)
        C_x = np.where(sci_0 > 0, a_Fade(Q_sci) / a_Enhanc(Q_sci), 1)

        # Step C17: Transform the integrated water vapour content time series
        # V(kTs) into the Gamma distributed time series Z(kTs) as follows:
        kappa, lambd = integrated_water_vapour_coefficients(lat, lon, None)
        Z = stats.gamma.ppf(np.exp(-(V / lambd)**kappa), 10, scale=0.1) 

        # Step C18: Compute the scintillation standard deviation σ following
        # the method recommended in Recommendation ITU-R P.618.
        sigma = scintillation_attenuation_sigma(lat, lon, f, el, D, eta, Tm,
                                                H, P, hL).value

        # Step C19: Compute the scintillation time series sci:
        As = np.where(Ar > 1, sigma * sci_0 * C_x * Z * Ar ** (5 / 12),
                      sigma * sci_0 * C_x * Z)

        # Step C20: Compute total tropospheric attenuation time series A(kTs)
        # as follows:
        A = Ar + Ac + Av + Ao + As

        if return_contributions:
            return (Ao + Av)[::Ts], Ac[::Ts], Ar[::Ts], As[::Ts], A[::Ts]
        else:
            return A[::Ts]


class _ITU1853_0():

    def __init__(self):
        self.__version__ = 0
        self.year = 2009
        self.month = 10
        self.link = 'https://www.p.int/rec/R-REC-P.1853-0-200910-I/en'

    def total_attenuation_synthesis(self, *args, **kwargs):
        raise NotImplementedError(
            "Recommendation ITU-R P.1853 does not specify a method to compute "
            "time series for the total atmospheric attenuation.")


__model = __ITU1853()


def change_version(new_version):
    """
    Change the version of the ITU-R P.1853 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * 2 : P.1853-2 (08/2019) (Current version)
            * 1 : P.1853-1 (02/2012) (Superseded)
    """
    global __model
    __model = __ITU1853(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.1853 recommendation currently being used.
    """
    global __model
    return __model.__version__


def total_attenuation_synthesis(lat, lon, f, el, p, D, Ns, tau, eta, Ts=1, 
                                hs=None, rho=None, H=None, P=None,
                                hL=1000, return_contributions=False):
    """ The time series synthesis method generates a time series that
    reproduces the spectral characteristics, rate of change and duration
    statistics of the total atmospheric attenuation events.

    The time series is obtained considering the contributions of gaseous,
    cloud, rain, and scintillation attenuation.

    Parameters
    ----------
    - lat : number
            Latitudes of the receiver points
    - lon : number
            Longitudes of the receiver points
    - f : number or Quantity
            Frequency (GHz)
    - el : number
            Elevation angle (degrees)
    - p : number
            Percetage of the time the rain attenuation value is exceeded.
    - D: number or Quantity
            Physical diameter of the earth-station antenna (m)
    - Ns : int
            Number of samples
    - Ts : int
            Time step between consecutive samples (seconds)
    - tau : number
            Polarization tilt angle relative to the horizontal (degrees)
            (tau = 45 deg for circular polarization).
    - eta: number
            Antenna efficiency.
    - hs : number, sequence, or numpy.ndarray, optional
            Heigh above mean sea level of the earth station (km). If local data for
            the earth station height above mean sea level is not available, an
            estimate is obtained from the maps of topographic altitude
            given in Recommendation ITU-R P.1511. Deafult value is None.
    - rho : number or Quantity, optional
            Water vapor density (g/m3). If not provided, an estimate is obtained
            from Recommendation Recommendation ITU-R P.836.Default value is None.
    - H: number, sequence, or numpy.ndarray, optional
            Average surface relative humidity (%) at the site. If None, uses the
            ITU-R P.453 to estimate the wet term of the radio refractivity.Default value is None.
    - P: number, sequence, or numpy.ndarray, optional
            Average surface pressure (hPa) at the site. If None, uses the
            ITU-R P.453 to estimate the wet term of the radio refractivity. Deafult value is None
    - hL : number, optional
            Height of the turbulent layer (m). Default value 1000 m
    - return_contributions: bool, optional
            Determines whether individual contributions from gases, rain, clouds
            and scintillation are returned in addition ot the total attenuation
            (True), or just the total atmospheric attenuation (False).
            Default is False

    Returns
    ---------
    - A : Quantity
            Synthesized total atmospheric attenuation time series (dB)

    - Ag, Ac, Ar, As, A : tuple
            Synthesized Gaseous, Cloud, Rain, Scintillation contributions to total
            attenuation time series, and synthesized total attenuation time seires
            (dB).

    References
    ----------
    [1] Characteristics of precipitation for propagation modelling
    https://www.itu.int/rec/R-REC-P.1853/en
    """
    global __model

    # prepare the input array
    lon = np.mod(lon, 360)
    lat = prepare_input_array(lat).flatten()
    lon = prepare_input_array(lon).flatten()
    f = prepare_input_array(f).flatten()
    el = prepare_input_array(el).flatten()
    tau = prepare_input_array(tau).flatten()
    eta = prepare_input_array(eta).flatten()
    D = prepare_input_array(D).flatten()

    # prepare quantity
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(el, u.deg, 'Elevation angle')
    Ts = prepare_quantity(Ts, u.second, 'Time step between samples')
    D = prepare_quantity(D, u.m, 'Antenna diameter')
    hs = prepare_quantity(
        hs, u.km, 'Heigh above mean sea level of the earth station')
    eta = prepare_quantity(eta, u.one, 'Antenna efficiency')
    rho = prepare_quantity(rho, u.g / u.m**3, 'Water vapor density')
    H = prepare_quantity(H, u.percent, 'Average surface relative humidity')
    P = prepare_quantity(P, u.hPa, 'Average surface pressure')
    hL = prepare_quantity(hL, u.m, 'Height of the turbulent layer')
    # calculate the output
    val = __model.total_attenuation_synthesis(
            lat, lon, f, el, p, D, Ns, tau, eta, Ts=Ts, hs=hs, rho=rho,
            H=H, P=P, hL=hL, return_contributions=return_contributions)
    if return_contributions:
        return tuple([v * u.dB for v in val])
    else:
        return val * u.dB

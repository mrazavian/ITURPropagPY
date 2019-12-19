# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt

import unittest as test
import numpy as np
import scipy
import sys
from astropy import units as u

import iturpropag
# ------import from ITU-R P.618 functions------#
from iturpropag.models.iturp618.rain_attenuation_probability import rain_attenuation_probability
from iturpropag.models.iturp618.rain_attenuation import rain_attenuation
from iturpropag.models.iturp618.scintillation_attenuation import scintillation_attenuation
# ------import from ITU-R P.1510-1511 functions------#
from iturpropag.models.iturp1510.surface_mean_temperature import surface_mean_temperature
from iturpropag.models.iturp1511.topographic_altitude import topographic_altitude
# ------import from ITU-R P.676 functions------#
from iturpropag.models.iturp676.gamma0_exact import gamma0_exact
from iturpropag.models.iturp676.gammaw_exact import gammaw_exact
from iturpropag.models.iturp676.gaseous_attenuation_slant_path import gaseous_attenuation_slant_path
from iturpropag.models.iturp676.zenith_water_vapour_attenuation import zenith_water_vapour_attenuation
# ------import from ITU-R P.835 functions------#
from iturpropag.models.iturp835.standard_water_vapour_density import standard_water_vapour_density
from iturpropag.models.iturp835.water_vapour_density import water_vapour_density
from iturpropag.models.iturp835.standard_pressure import standard_pressure
from iturpropag.models.iturp835.pressure import pressure
from iturpropag.models.iturp835.standard_temperature import standard_temperature
from iturpropag.models.iturp835.temperature import temperature
# ------import from ITU-R P.836 functions------#
from iturpropag.models.iturp836.surface_water_vapour_density import surface_water_vapour_density
from iturpropag.models.iturp836.total_water_vapour_content import total_water_vapour_content
# ------import from ITU-R P.837 functions------#
from iturpropag.models.iturp837.rainfall_probability import rainfall_probability
from iturpropag.models.iturp837.rainfall_rate import rainfall_rate
# ------import from ITU-R P.839 functions------#
from iturpropag.models.iturp839.isotherm_0 import isotherm_0
from iturpropag.models.iturp839.rain_height import rain_height
# ------import from ITU-R P.840 functions------#
from iturpropag.models.iturp840.columnar_content_reduced_liquid import columnar_content_reduced_liquid
from iturpropag.models.iturp840.specific_attenuation_coefficients import specific_attenuation_coefficients
from iturpropag.models.iturp840.cloud_attenuation import cloud_attenuation
# ------import from ITU-R P.1853 functions------#
from iturpropag.models.iturp1853.rain_attenuation_synthesis import rain_attenuation_synthesis
from iturpropag.models.iturp1853.cloud_liquid_water_synthesis import cloud_liquid_water_synthesis
#------------------------------------------------------------------------

def suite():
    """ A test suite for the ITU-P Recommendations. Recommendations tested:
    """
    suite = test.TestSuite()

    # Test valid versions
    suite.addTest(TestMapAfrica('test_map_africa'))
    suite.addTest(TestGaseousAttenuation('test_gaseous_attenuation'))
    suite.addTest(TestMultipleLocations('test_multiple_locations'))
    suite.addTest(TestSingleLocation('test_single_location'))
    suite.addTest(TestSingleLocationVsFrequency('test_single_location_vs_f'))
    suite.addTest(TestSingleLocationVsUnavailability(
             'test_single_location_vs_p'))
    suite.addTest(TestTimeSeries('rain_att_time_series'))
    suite.addTest(TestTimeSeries('cloud_att_time_series'))

    return suite


class TestMapAfrica(test.TestCase):

    def test_map_africa(self):
        # Generate a regular grid of latitude and longitudes with 0.1
        # degree resolution for the region of interest.
        lat, lon = iturpropag.utils.regular_lat_lon_grid(lat_max=60,
                                                   lat_min=-60,
                                                   lon_max=65,
                                                   lon_min=-35,
                                                   resolution_lon=1,
                                                   resolution_lat=1)

        # Satellite coordinates (GEO, 4 E)
        lat_sat = 0
        lon_sat = 4
        h_sat = 35786 * u.km

        # Compute the elevation angle between satellite and ground stations
        el = iturpropag.utils.elevation_angle(h_sat, lat_sat, lon_sat, lat, lon)

        # Set the link parameters
        f = 22.5 * u.GHz    # Link frequency
        D = 1.2 * u.m       # Antenna diameters
        p = 0.1                  # Unavailability (Vals exceeded 0.1% of time)
        tau = 0                  # polarization tilt angle
        eta = 0.5                # antenna efficiency

        # Compute the atmospheric attenuation
        Att = iturpropag.atmospheric_attenuation_slant_path(lat, lon, f, el, p, D, tau, eta)

        # Now we show the surface mean temperature distribution
        T = surface_mean_temperature(lat, lon)\
            .to(u.Celsius, equivalencies=iturpropag.u.temperature())

        # Plot the results
        try:
            m = iturpropag.utils.plot_in_map(Att.value, lat, lon,
                                       cbar_text='Atmospheric attenuation [dB]',
                                       cmap='magma',ax=plt.subplot(121))

            # Plot the satellite location
            m.scatter(lon_sat, lat_sat, c='white', s=20)
            # plt.show()
            m = iturpropag.utils.plot_in_map(
                T.value, lat, lon, cbar_text='Surface mean temperature [C]',
                cmap='RdBu_r',ax=plt.subplot(122))
        except RuntimeError as e:
            print(e)


class TestMultipleLocations(test.TestCase):

    def test_multiple_locations(self):
        # Obtain the coordinates of the different cities
        # cities = {'Boston': (42.36, -71.06),
        #           'New York': (40.71, -74.01),
        #           'Los Angeles': (34.05, -118.24),
        #           'Denver': (39.74, -104.99),
        #           'Las Vegas': (36.20, -115.14),
        #           'Seattle': (47.61, -122.33),
        #           'Washington DC': (38.91, -77.04)}
        cities = {#'Geneva': (46.20, 6.15),
                  'New York': (40.71, -74.01),
                  'Nairobi': (-1.28, 36.82),
                  'Beijing': (39.93, 116.38),
                  'Moskwa': (55.75, 37.62),
                  'Buenos Aires': (-34.60, -58.38),
                  'Sidney': (-33.85, 151.20),
                  'Cherrapunji': (25.30, 91.70)}

        lat = [coords[0] for coords in cities.values()]
        lon = [coords[1] for coords in cities.values()]

        # Satellite coordinates (GEO, 4 E)
        lat_sat = 0
        lon_sat = -77
        h_sat = 35786 * u.km

        # Compute the elevation angle between satellite and ground stations
        el = iturpropag.utils.elevation_angle(h_sat, lat_sat, lon_sat, lat, lon)

        # Set the link parameters
        f = 22.5 * u.GHz    # Link frequency
        D = 1.2 * u.m       # Antenna diameters
        p = 0.1                  # Unavailability (Vals exceeded 0.1% of time)
        tau = 0                  # polarization tilt angle
        eta = 0.5                # antenna efficiency

        # Compute the atmospheric attenuation
        Ag, Ac, Ar, As, Att = iturpropag.atmospheric_attenuation_slant_path(
            lat, lon, f, el, p, D, tau, eta, return_contributions=True)

        # Plot the results
        city_idx = np.arange(len(cities))
        width = 0.15

        fig, ax = plt.subplots(1, 1)
        ax.bar(city_idx, Att.value, width=0.6, label='Total atmospheric Attenuation')
        ax.bar(city_idx - 1.5 * width, Ar.value, width=width,
               label='Rain attenuation')
        ax.bar(city_idx - 0.5 * width, Ag.value, width=width,
               label='Gaseous attenuation')
        ax.bar(city_idx + 0.5 * width, Ac.value, width=width,
               label='Clouds attenuation')
        ax.bar(city_idx + 1.5 * width, As.value, width=width,
               label='Scintillation attenuation')

        # Set the labels
        ticks = ax.set_xticklabels([''] + list(cities.keys()))
        for t in ticks:
            t.set_rotation(45)
        ax.set_ylabel('Atmospheric attenuation exceeded for 0.1% [dB]')

        # Format image
        ax.yaxis.grid(which='both', linestyle=':')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)
        plt.tight_layout(rect=(0, 0, 1, 0.85))



class TestSingleLocation(test.TestCase):

    def test_single_location(self):

        # Location of the receiver ground stations (Boston, MA)
        lat = 42.31
        lon = -70.91
        print('\nThe ITU recommendations predict the following values\nfor the Rx ground station coordinates (Boston, MA)')
        print('Lat = 42.31, Lon = -70.91')

        # Link parameters
        el = 60                # Elevation angle equal to 60 degrees
        f = 22.5 * u.GHz       # Frequency equal to 22.5 GHz
        D = 1 * u.m            # Receiver antenna diameter of 1 m
        p = 0.1                # We compute values exceeded during 0.1 % of
                               # the average year
        eta = 0.6              # Antenna efficiency 
        tau = 45               # Polarization tilt
        print('Elevation angle:\t\t',el,'°')
        print('Frequency:\t\t\t',f)
        print('Rx antenna diameter:\t\t',D)
        print('Values exceeded during 0.1 percent of the year')
        print('Antenna efficiency:\t\t',eta)
        print('Polarization tilt:\t\t',tau,'°')

        # Compute atmospheric parameters
        hs = topographic_altitude(lat, lon)
        print('Topographic altitude                  [ITU-R P.1511]:\t',hs)
        T = surface_mean_temperature(lat, lon)
        print('Surface mean temperature              [ITU-R P.1510]:\t',T,'=',T.value-273.15,'°C')
        P = pressure(lat, hs)
        print('Surface pressure                      [ITU-R P.835]:\t',P)
        T_sa = temperature(lat, hs)
        print('Standard surface temperature          [ITU-R P.835]:\t',T_sa,'=',T_sa.value-273.15,'°C')
        rho_sa = water_vapour_density(lat, hs)
        print('Standard water vapour density         [ITU-R P.835]:\t',rho_sa)
        rho_p = surface_water_vapour_density(lat, lon, p, hs)
        print('Surface water vapour density (p=0.1%) [ITU-R P.836]:\t',rho_p)
        V = total_water_vapour_content(lat, lon, p, hs)
        print('Total water vapour content   (p=0.1%) [ITU-R P.836]:\t',V)

        # Compute rain and cloud-related parameters
        R_prob = rain_attenuation_probability(lat, lon, el, hs)
        print('Rain attenuation probability          [ITU-R P.618]:\t',R_prob)
        R_pct_prob = rainfall_probability(lat, lon)
        print('Rainfall probability                  [ITU-R P.837]:\t',R_pct_prob)
        R001 = rainfall_rate(lat, lon, p)
        print('Rainfall rate exceeded for p=0.1%     [ITU-R P.836]:\tt',R001)
        h_0 = isotherm_0(lat, lon)
        print('0°C Isotherm height                   [ITU-R P.839]:\t',h_0)
        h_rain = rain_height(lat, lon)
        print('Rain height                           [ITU-R P.839]:\t',h_rain)
        L_red = columnar_content_reduced_liquid(lat, lon, p)
        print('Reduced liquid columnar content (p=0.1%)    [P.836]:\t',L_red)
        A_w = zenith_water_vapour_attenuation(lat, lon, p, f, h=hs)
        print('Zenith water vapour attenuation:      [ITU-R P.676]:\t',A_w)

        # Compute attenuation values
        A_g = gaseous_attenuation_slant_path(f, el, rho_p, P, T)
        print('Gaseous attenuation slant path        [ITU-R P.676]:\t',A_g)
        A_r = rain_attenuation(lat, lon, f, el, p, tau, hs=hs)
        print('Rain attenuation                      [ITU-R P.618]:\t',A_r)
        A_c = cloud_attenuation(lat, lon, el, f, p)
        print('Cloud attenuation                     [ITU-R P.840]:\t',A_c)
        A_s = scintillation_attenuation(lat, lon, f, el, p, D, eta)
        print('Scintillation attenuation             [ITU-R P.618]:\t',A_s)
        A_t = iturpropag.atmospheric_attenuation_slant_path(lat, lon, f, el, p, D, tau, eta)
        print('Atmospheric attenuation slant path    [ITU-R P.618]:\t',A_t)


class TestSingleLocationVsFrequency(test.TestCase):

    def test_single_location_vs_f(self):
        # Ground station coordinates (Boston)
        lat_GS = 42.3601
        lon_GS = -71.0942

        ################################################
        # First case: Attenuation vs. frequency        #
        ################################################

        # Satellite coordinates (GEO, 77 W)
        lat_sat = 0
        lon_sat = -77
        h_sat = 35786 * u.km

        # Compute the elevation angle between satellite and ground station
        el = iturpropag.utils.elevation_angle(h_sat, lat_sat, lon_sat,
                                        lat_GS, lon_GS)

        f = 22.5 * u.GHz    # Link frequency
        D = 1.2 * u.m       # Antenna diameters
        p = 1
        tau = 0                  # polarization tilt angle
        eta = 0.5                # antenna efficiency

        f = np.logspace(-0.2, 2, 100) * u.GHz

        Ag, Ac, Ar, As, A =\
            iturpropag.atmospheric_attenuation_slant_path(lat_GS, lon_GS, f,
                                                    el, p, D, tau, eta,
                                                    return_contributions=True)

        # Plot the results
        fig, ax = plt.subplots(1, 1)
        ax.loglog(f, Ag, label='Gaseous attenuation')
        ax.loglog(f, Ac, label='Cloud attenuation')
        ax.loglog(f, Ar, label='Rain attenuation')
        ax.loglog(f, As, label='Scintillation attenuation')
        ax.loglog(f, A, label='Total atmospheric attenuation')

        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel('Atmospheric attenuation [dB]')
        ax.grid(which='both', linestyle=':')
        plt.legend(loc='upper left')

        ################################################
        # Second case: Attenuation vs. elevation angle #
        ################################################

        f = 22.5 * u.GHz
        el = np.linspace(5, 90, 100)

        Ag, Ac, Ar, As, A =\
            iturpropag.atmospheric_attenuation_slant_path(lat_GS, lon_GS,
                                                    f, el, p, D, tau, eta,
                                                    return_contributions=True)

        # Plot the results
        fig, ax = plt.subplots(1, 1)
        ax.plot(el, Ag, label='Gaseous attenuation')
        ax.plot(el, Ac, label='Cloud attenuation')
        ax.plot(el, Ar, label='Rain attenuation')
        ax.plot(el, As, label='Scintillation attenuation')
        ax.plot(el, A, label='Total atmospheric attenuation')

        ax.set_xlabel('Elevation angle [deg]')
        ax.set_ylabel('Atmospheric attenuation [dB]')
        ax.grid(which='both', linestyle=':')
        plt.legend()
        


class TestSingleLocationVsUnavailability(test.TestCase):

    def test_single_location_vs_p(self):
        # Ground station coordinates (Boston)
        lat_GS = 42.3601
        lon_GS = -71.0942

        # Satellite coordinates (GEO, 77 W)
        lat_sat = 0
        lon_sat = -77
        h_sat = 35786 * u.km

        # Compute the elevation angle between satellite and ground station
        el = iturpropag.utils.elevation_angle(h_sat, lat_sat, lon_sat,
                                        lat_GS, lon_GS)

        f = 22.5 * u.GHz    # Link frequency
        D = 1.2 * u.m       # Antenna diameters
        tau = 0             # polarization tilt angle
        eta = 0.5           # antenna efficiency
        
        # Define unavailabilities vector in logarithmic scale
        # p = np.logspace(-1.5, 1.5, 100)
        p = np.logspace(0.05, 0.695, 100)

        A_g, A_c, A_r, A_s, A_t = \
            iturpropag.atmospheric_attenuation_slant_path(
                    lat_GS, lon_GS, f, el, p, D, tau, eta, return_contributions=True)

        # Plot the results using matplotlib
        f, ax = plt.subplots(1, 1)
        ax.semilogx(p, A_g.value, label='Gaseous attenuation')
        ax.semilogx(p, A_c.value, label='Cloud attenuation')
        ax.semilogx(p, A_r.value, label='Rain attenuation')
        ax.semilogx(p, A_s.value, label='Scintillation attenuation')
        ax.semilogx(p, A_t.value, label='Total atmospheric attenuation')

        ax.set_xlabel('Percentage of time attenuation value is exceeded [%]')
        ax.set_ylabel('Attenuation [dB]')
        ax.grid(which='both', linestyle=':')
        plt.legend()


class TestGaseousAttenuation(test.TestCase):

    def test_gaseous_attenuation(self):
        # Define atmospheric parameters
        rho_wet = 7.5 * u.g / u.m**3
        rho_dry = 0 * u.g / u.m**3
        P = 1013.25 * u.hPa
        T = 15 * u.deg_C

        # Define frequency logspace parameters
        N_freq = 1000
        fs = np.linspace(0, 1000, N_freq)

        # Compute the attenuation values
        att_wet = gammaw_exact(fs, P, rho_wet, T) 

        att_dry = gamma0_exact(fs, P, rho_dry, T)

        # Plot the results
        plt.figure()
        plt.plot(fs, att_wet.value, 'b--', label='Wet atmosphere')
        plt.plot(fs, att_dry.value, 'r', label='Dry atmosphere')
        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Specific attenuation [dB/km]')
        plt.yscale('log')
        plt.xscale('linear')
        plt.xlim(0, 1000)
        plt.ylim(1e-3, 1e5)
        plt.legend()
        plt.grid(which='both', linestyle=':', color='gray',
                 linewidth=0.3, alpha=0.5)
        plt.grid(which='major', linestyle=':', color='black')
        plt.title('FIGURE 1. - Specific attenuation due to atmospheric gases,'
                  '\ncalculated at 1 GHz intervals, including line centres')
        plt.tight_layout()

        #######################################################################
        #               Specific attenuation at different altitudes           #
        #######################################################################

        # Define atmospheric parameters
        hs = np.array([0, 5, 10, 15, 20]) * u.km

        # Define frequency logspace parameters
        N_freq = 2001
        fs = np.linspace(50, 70, N_freq)

        # Plot the results
        plt.figure()

        # Loop over heights and compute values
        for h in hs:
            rho = standard_water_vapour_density(h)
            P = standard_pressure(h)
            T = standard_temperature(h)
            atts_dry = gamma0_exact(fs * u.GHz, P, rho, T).value
            atts_wet = gammaw_exact(fs * u.GHz, P, rho, T).value
            atts = atts_dry + atts_wet
            plt.plot(fs, atts, label='Altitude {0} km'.format(h.value))

        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Specific attenuation [dB/km]')
        plt.yscale('log')
        plt.xscale('linear')
        plt.xlim(50, 70)
        plt.ylim(1e-3, 1e2)
        plt.legend()
        plt.grid(which='both', linestyle=':', color='gray',
                 linewidth=0.3, alpha=0.5)
        plt.grid(which='major', linestyle=':', color='black')
        plt.title('FIGURE 2. - Specific attenuation in the range 50-70 GHz'
                  ' at the\n altitudes indicated, calculated at intervals of'
                  ' 10 MHz\nincluding line centers (0, 5, 10 15, 20) km')
        plt.tight_layout()

        #######################################################################
        #           Comparison of line-by-line and approximate method         #
        #######################################################################
        # Define atmospheric parameters
        el = 90
        rho = 7.5 * u.g / u.m**3
        P = 1013.25 * u.hPa
        T = 15 * u.deg_C
        e = rho.value * T.value / 216.7   # Water vapour partial pressure
        p = P.value - e     # Dry Air Pressure

        # Define frequency logspace parameters
        N_freq = 350
        fs = np.linspace(1, 350, N_freq)   # GHz

        # Initialize result vectors
        atts_approx = []
        atts_exact = []

        # Loop over frequencies and compute values
        atts_approx = gaseous_attenuation_slant_path(
            fs, el, rho, p, T, mode='approx')

        atts_exact = gaseous_attenuation_slant_path(
            fs, el, rho, p, T, h=0, mode='exact')

        # Plot the results
        plt.figure()
        plt.plot(fs, atts_approx.value, 'b--',
                 label='Approximate method Annex 2')
        plt.plot(fs, atts_exact.value, 'r', label='Exact line-by-line method')
        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Attenuation [dB]')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.grid(which='both', linestyle=':', color='gray',
                 linewidth=0.3, alpha=0.5)
        plt.grid(which='major', linestyle=':', color='black')
        plt.title('Comparison of line-by-line method to approximate method')
        plt.tight_layout()


class TestTimeSeries(test.TestCase):

    def rain_att_time_series(self):
        iturpropag.models.iturp1853.__version.change_version(1)
        # Location of the receiver ground stations (Louvain-La-Neuve, Belgium)
        lat = 50.66
        lon = 4.62
        print('\nThe ITU-R P.1853 recommendation predict rain attenuation time-series\n'+\
              'the following values for the Rx ground station coordinates (Louvain-La-Neuve, Belgium)')
        print('Lat = 50.66, Lon = 4.62')

        # Link parameters
        el = 30                # Elevation angle equal to 60 degrees
        f = 39.4 * u.GHz       # Frequency equal to 22.5 GHz
        tau = 45               # Polarization tilt
        D = 900*24*3600        #  duration (second) (900 days)
        Ts = 1                 # Ts : sampling
        Ns = int(D / (Ts**2))  # number of samples

        print('Elevation angle:\t\t',el,'°')
        print('Frequency:\t\t\t',f)
        print('Polarization tilt:\t\t',tau,'°')
        print('Sampling Duration:\t\t',900,'days')

        #--------------------------------------------------------
        #  rain attenuation time series synthesis by ITU-R P.1853
        #--------------------------------------------------------
        ts_rain = rain_attenuation_synthesis(lat, lon, f, el,\
                                            tau, Ns, Ts=Ts).value
                                            
        stat = iturpropag.utils.ccdf(ts_rain, bins=300)
        #--------------------------------------------------------
        #  calculating the m_lna and sigma_lna for log-normal distribution
        #--------------------------------------------------------
        P_rain = rainfall_probability(lat, lon).to(u.dimensionless_unscaled).value

        p_i = np.array([0.01, 0.02, 0.03, 0.05, 0.1,\
                    0.2, 0.3, 0.5, 1, 2, 3, 5, 10])
        Pi = np.array([p for p in p_i if p < P_rain * 100], dtype=np.float)

        Ai = rain_attenuation(lat, lon, f, el, Pi, tau).value

        Q = scipy.stats.norm.ppf(1-(Pi / 100))
        lnA = np.log(Ai)

        m_lna, sigma_lna = np.linalg.lstsq(np.vstack([np.ones(len(Q)), Q]).T,
                                    lnA, rcond=None)[0]
        #--------------------------------------------------------
        # rain attenuation by ITU-R P.618 and log_normal distribution
        #--------------------------------------------------------
        P = np.array([0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1,\
                       0.2, 0.3, 0.5, 1, 2, 3, 5])
        
        A_rain = rain_attenuation(lat, lon, f, el, P, tau).value

        A_rain_log = np.exp(m_lna + sigma_lna * scipy.stats.norm.ppf(1-(P / 100)))

        #--------------------------------------------------------
        # ploting the results
        #--------------------------------------------------------
        plt.figure()
        plt.plot(P, A_rain, '-b', lw=2, label='ITU-R P.618')
        plt.plot(P, A_rain_log, '-y', lw=2, label='Log Normal Approx.')
        plt.plot(stat.get('ccdf'), stat.get('bin_edges')[1:], '-r', lw=2, label='ITU-R P.1853 (900 days)')
        plt.xlim((10**-3.5, 5))
        plt.xscale('log')
        plt.xlabel('Time percentage (%)')
        plt.ylabel('Rain attenuation CCDF (dB)')
        plt.legend()
        plt.grid(which='both', linestyle=':', color='gray',
                 linewidth=0.3, alpha=0.5)
        plt.tight_layout()


    def cloud_att_time_series(self):
        # Location of the receiver ground stations (Louvain-La-Neuve, Belgium)
        lat = 50.66
        lon = 4.62
        print('\nThe ITU-R P.1853 recommendation predict cloud attenuation time-series\n'+\
              'the following values for the Rx ground station coordinates (Louvain-La-Neuve, Belgium)')
        print('Lat = 50.66, Lon = 4.62')

        # Link parameters
        el = 30                # Elevation angle equal to 60 degrees
        f = 39.4 * u.GHz       # Frequency equal to 22.5 GHz
        tau = 45               # Polarization tilt
        D = 900*24*3600        #  duration (second) (900 days)
        Ts = 1                 # Ts : sampling
        Ns = int(D / (Ts**2))  # number of samples

        print('Elevation angle:\t\t',el,'°')
        print('Frequency:\t\t\t',f)
        print('Polarization tilt:\t\t',tau,'°')
        print('Sampling Duration:\t\t',900,'days')

        #--------------------------------------------------------
        #  cloud attenuation time series synthesis by ITU-R P.1853
        #--------------------------------------------------------
        iturpropag.models.iturp1853.__version.change_version(1)
        L = cloud_liquid_water_synthesis(lat, lon, Ns, Ts=1).value
        Ac_timeseries = L * \
            specific_attenuation_coefficients(f, T=0) / np.sin(np.deg2rad(el))
        Ac_timeseries = Ac_timeseries.flatten()

        stat = iturpropag.utils.ccdf(Ac_timeseries, bins=300)
        #--------------------------------------------------------
        # cloud attenuation by ITU-R P.840
        #--------------------------------------------------------
        P = np.array([0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1,\
                       0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50,\
                        60, 70, 80, 90, 100])
        
        Ac = cloud_attenuation(lat, lon, el, f, P).value
        #--------------------------------------------------------
        # ploting the results
        #--------------------------------------------------------
        plt.figure()
        plt.plot(P, Ac, '-b', lw=2, label='ITU-R P.840')
        plt.plot(stat.get('ccdf'), stat.get('bin_edges')[1:], '-r', lw=2, label='ITU-R P.1853 (900 days)')
        plt.xlim((10**-3.5, 100))
        plt.xscale('log')
        plt.xlabel('Time percentage (%)')
        plt.ylabel('Cloud attenuation CCDF (dB)')
        plt.legend()
        plt.grid(which='both', linestyle=':', color='gray',
                 linewidth=0.3, alpha=0.5)
        plt.tight_layout()




if __name__ == '__main__':
    pass
    suite = suite()
    print('Test examples of the code')
    print('------------------------')
    print(
        'A total of %d test-cases are going to be tested' %
        suite.countTestCases())
    sys.stdout.flush()
    test.TextTestRunner(verbosity=2).run(suite)
    plt.show()

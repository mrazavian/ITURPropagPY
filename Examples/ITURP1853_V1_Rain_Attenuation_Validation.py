import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import scipy

import iturpropag
from iturpropag.models.iturp618.rain_attenuation import rain_attenuation
from iturpropag.models.iturp837.rainfall_probability import rainfall_probability
from iturpropag.models.iturp1853.rain_attenuation_synthesis import rain_attenuation_synthesis
from iturpropag.utils import ccdf
       
##################################################################################
##               validate the ITU-R P.1853.1--section 2
##            Rain Attenuation time series synthesis method
##################################################################################
# Location of the receiver ground stations
cities = {'Louvain-La-Neuve': (50.66, 4.62),
          #'Geneva': (46.20, 6.15)
          }

lat = [coords[0] for coords in cities.values()]
lon = [coords[1] for coords in cities.values()]
print('\nThe ITU-R P.1853-1 recommendation predict rain attenuation time-series\n'+\
              'the following values for the Rx ground station coordinates')

# Link parameters
el = [35]             # Elevation angle equal to 60 degrees
f = [39.4] * u.GHz       # Frequency equal to 22.5 GHz
tau = [45]               # Polarization tilt
D = 3*365*24*3600        #  duration (second) (3 years)
Ts = 1                 # Ts : sampling
Ns = int(D / (Ts**2))  # number of samples

print('Elevation angle:\t\t',el,'°')
print('Frequency:\t\t\t',f)
print('Polarization tilt:\t\t',tau,'°')
print('Sampling Duration:\t\t',D/(24*3600),'days')
#--------------------------------------------------------
#  rain attenuation time series synthesis by ITU-R P.1853
#--------------------------------------------------------
iturpropag.models.iturp1853.__version.change_version(1)
ts_rain = rain_attenuation_synthesis(lat, lon, f, el,\
                                    tau, Ns,Ts=Ts).value
#--------------------------------------------------------
#  calculating the m and sigma
#--------------------------------------------------------
P_rain = rainfall_probability(lat, lon).to(u.dimensionless_unscaled).value

p_i = np.array([0.01, 0.02, 0.03, 0.05, 0.1,\
               0.2, 0.3, 0.5, 1, 2, 3, 5, 10])
Pi = np.array([p for p in p_i if p < P_rain * 100], dtype=np.float)
Ai = np.zeros_like(Pi)

for i, p in enumerate(Pi):
    Ai[i] = rain_attenuation(lat, lon, f, el, p, tau).value

Q = scipy.stats.norm.ppf(1-(Pi / 100))
lnA = np.log(Ai)

m, sigma = np.linalg.lstsq(np.vstack([np.ones(len(Q)), Q]).T,
                            lnA, rcond=None)[0]
#--------------------------------------------------------
# rain attenuation by ITU-R P.618 and log_normal distribution
#--------------------------------------------------------
P = np.array([0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1,\
               0.2, 0.3, 0.5, 1, 2, 3, 5])
A_rain = np.zeros_like(P)
for ii, p in enumerate(P):
	A_rain[ii] = rain_attenuation(lat, lon, f, el, p, tau).value

A_rain_log = np.exp(m + sigma * scipy.stats.norm.ppf(1-(P / 100)))

#--------------------------------------------------------
# calculating the ccdf of rain attenuation time_series
#--------------------------------------------------------
stat = ccdf(ts_rain, bins=300)

#--------------------------------------------------------
# ploting the results
#--------------------------------------------------------
plt.figure()
plt.plot(P, A_rain, '-b', lw=2, label='ITU-R P.618')
plt.plot(P, A_rain_log, '-y', lw=2, label='Log Normal')
plt.plot(stat.get('ccdf'), stat.get('bin_edges')[1:], '-r', lw=2, label='ITU-R P.1853')
plt.xlim((10**-3.5, 5))
plt.xscale('log')
plt.xlabel('Time percentage (%)')
plt.ylabel('Rain attenuation CCDF (dB)')
plt.title('Rain Attenuation Statistics at Ground station Louvain-La-Neuve')
plt.legend()
plt.grid(which='both', linestyle=':', color='gray',
                 linewidth=0.3, alpha=0.5)
plt.tight_layout()
plt.show()


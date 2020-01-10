import numpy as np 
import matplotlib.pyplot as plt
import scipy
import astropy.units as u 

from iturpropag.models.iturp1853.total_attenuation_synthesis import total_attenuation_synthesis
from iturpropag.utils import ccdf


"""
This script will calculate the total attenuation time series for 3 years duration
using the procedure in ITU-R P.1853-2 using multi-site configuration
"""

# Location of the receiver ground stations
cities = {'Louvain-La-Neuve': (50.66, 4.62),
          'Geneva': (46.20, 6.15)
          }

lat = [coords[0] for coords in cities.values()]
lon = [coords[1] for coords in cities.values()]
print('\nThe ITU-R P.1853-2 recommendation predict total attenuation time-series\n'+\
              'the following values for the Rx ground station coordinates')

# Link parameters
el = [35, 35]             # Elevation angle equal to 60 degrees
f = [39.4, 39.4] * u.GHz       # Frequency equal to 22.5 GHz
tau = [45, 45]               # Polarization tilt
D = 3*365*24*3600        #  duration (second) (3 years)
Ts = 1                 # Ts : sampling time (second)
Ns = int(D / (Ts**2))  # number of samples
eta = [0.6, 0.6]    # Antenna efficiency
Dia = [1, 1]       # Antenna Diameter [m]

print('Ground station locations:\t\t', cities)
print('Elevation angle:\t\t',el,'°')
print('Frequency:\t\t\t',f)
print('Polarization tilt:\t\t',tau,'°')
print('Sampling Duration:\t\t',D/(24*3600),'days')

#--------------------------------------------------------
#  total attenuation time series synthesis by ITU-R P.1853-2
#--------------------------------------------------------

time_series = total_attenuation_synthesis(lat, lon, f, el,\
                                    0.1, Dia, Ns, tau, eta, Ts=Ts).value
stat_lln = ccdf(time_series[0,:], bins=300)
stat_gnv = ccdf(time_series[1,:], bins=300)

plt.plot(stat_lln.get('ccdf'), stat_lln.get('bin_edges')[1:],\
                      lw=2, label='Louvain-La-Neuve')
plt.plot(stat_gnv.get('ccdf'), stat_gnv.get('bin_edges')[1:],\
                      lw=2, label='Geneva')


plt.xlim((10**-3.5, 100))
plt.xscale('log')
plt.xlabel('Time percentage (%)')
plt.ylabel('Total attenuation CCDF (dB)')
plt.title('ITU-R P.1853-2 Total Attenuation - Multi-Site Configuration')
plt.legend()
plt.grid(which='both', linestyle=':', color='gray',
            linewidth=0.3, alpha=0.5)
plt.tight_layout()
plt.show()
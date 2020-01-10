import numpy as np 
import matplotlib.pyplot as plt
import scipy
import astropy.units as u 

import iturpropag
from iturpropag.models.iturp1853.rain_attenuation_synthesis import rain_attenuation_synthesis
from iturpropag.utils import ccdf

"""
This script will calculate the rain attenuation time series for 3 years duration
using the procedure in ITU-R P.1853-1 and 2 at Louvain-La-Neuve.
"""

# Location of the receiver ground stations
cities = {'Louvain-La-Neuve': (50.66, 4.62),
          #'Geneva': (46.20, 6.15)
          }

lat = [coords[0] for coords in cities.values()]
lon = [coords[1] for coords in cities.values()]
print('\nThe ITU-R P.1853-2 recommendation predict rain attenuation time-series\n'+\
              'the following values for the Rx ground station coordinates')

# Link parameters
el = [35]             # Elevation angle equal to 60 degrees
f = [39.4] * u.GHz       # Frequency equal to 22.5 GHz
tau = [45]               # Polarization tilt
D = 3*365*24*3600        #  duration (second) (3 years)
Ts = 1                 # Ts : sampling
Ns = int(D / (Ts**2))  # number of samples

print('Ground station locations:\t\t', cities)
print('Elevation angle:\t\t',el,'°')
print('Frequency:\t\t\t',f)
print('Polarization tilt:\t\t',tau,'°')
print('Sampling Duration:\t\t',D/(24*3600),'days')

#--------------------------------------------------------
#  rain attenuation time series synthesis by ITU-R P.1853-2
#--------------------------------------------------------

ts_rain_v2 = rain_attenuation_synthesis(lat, lon, f, el,\
                                    tau, Ns, Ts=Ts).value

iturpropag.models.iturp1853.__version.change_version(1)  # change the vesrion of ITU-R P.1853-2 to ITU-R P.1853-1

ts_rain_v1 = rain_attenuation_synthesis(lat, lon, f, el,\
                                    tau, Ns, Ts=Ts).value

stat_v1 = ccdf(ts_rain_v1, bins=300)   # calculate the statistical of rain attenuation at Louvain-La-Neuve
stat_v2 = ccdf(ts_rain_v2, bins=300)   

plt.figure()
plt.plot(stat_v1.get('ccdf'), stat_v1.get('bin_edges')[1:],\
                      lw=2, label='ITU-R P.1853-1')
plt.plot(stat_v2.get('ccdf'), stat_v2.get('bin_edges')[1:],\
                      lw=2, label='ITU-R P.1853-2')

plt.xlim((10**-3.5, 15))
plt.xscale('log')
plt.xlabel('Time percentage (%)')
plt.ylabel('Rain attenuation CCDF (dB)')
plt.title('Rain Attenuation statistics at Louvain-La-Neuve')
plt.legend()
plt.grid(which='both', linestyle=':', color='gray',
            linewidth=0.3, alpha=0.5)
plt.tight_layout()
plt.show()
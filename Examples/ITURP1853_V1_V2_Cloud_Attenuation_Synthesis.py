import numpy as np 
import matplotlib.pyplot as plt
import scipy
import astropy.units as u 

import iturpropag
from iturpropag.models.iturp1853.cloud_attenuation_synthesis import cloud_attenuation_synthesis
from iturpropag.utils import ccdf

"""
This script will calculate the cloud attenuation time series for 3 years duration
using the procedure in ITU-R P.1853-1 and 2.

first the cloud attenuation time series using ITU-R P.1853-2 in two location 
    (a) Louvain-La-Neuve
    (b) Geneva
with Multi-Site configuration is calculated.
second, the cloud attenuation time series at Louvain-La-Neuve using
the method at ITU-R P.1853-1 is calculated and the result are plotted.
"""

# Location of the receiver ground stations
cities = {'Louvain-La-Neuve': (50.66, 4.62),
          'Geneva': (46.20, 6.15)
          }

lat = [coords[0] for coords in cities.values()]
lon = [coords[1] for coords in cities.values()]
print('\nThe ITU-R P.1853-2 recommendation predict cloud attenuation time-series\n'+\
              'the following values for the Rx ground station coordinates')

# Link parameters
el = [35, 35]             # Elevation angle equal to 60 degrees
f = [39.4, 39.4] * u.GHz       # Frequency equal to 22.5 GHz
tau = [45, 45]               # Polarization tilt
D = 3*365*24*3600        #  duration (second) (3 years)
Ts = 1                 # Ts : sampling
Ns = int(D / (Ts**2))  # number of samples

print('Ground station locations:\t\t', cities)
print('Elevation angle:\t\t',el,'°')
print('Frequency:\t\t\t',f)
print('Polarization tilt:\t\t',tau,'°')
print('Sampling Duration:\t\t',D/(24*3600),'days')

#--------------------------------------------------------
#  cloud attenuation time series synthesis by ITU-R P.1853-2
#--------------------------------------------------------

ts_cloud_v2 = cloud_attenuation_synthesis(lat, lon, f, el,\
                                    Ns, Ts=Ts).value

iturpropag.models.iturp1853.__version.change_version(1)  # change the version of ITU-R P.1853-2 to ITU-R P.1853-1

ts_cloud_v1 = cloud_attenuation_synthesis(lat[0], lon[0], f[0], el[0],\
                                    Ns, Ts=Ts).value     

stat_v1 = ccdf(ts_cloud_v1, bins=300)   
stat_v2_lln = ccdf(ts_cloud_v2[0,:], bins=300)
stat_v2_gnv = ccdf(ts_cloud_v2[1,:], bins=300) 

plt.figure()
plt.plot(stat_v1.get('ccdf'), stat_v1.get('bin_edges')[1:],\
                      lw=2, label='ITU-R P.1853-1 - LLN')
plt.plot(stat_v2_lln.get('ccdf'), stat_v2_lln.get('bin_edges')[1:],\
                      lw=2, label='ITU-R P.1853-2 - LLN')
plt.plot(stat_v2_gnv.get('ccdf'), stat_v2_gnv.get('bin_edges')[1:],\
                      lw=2, label='ITU-R P.1853-2 - Geneva')

plt.xlim((10**-3.5, 15))
plt.xscale('log')
plt.xlabel('Time percentage (%)')
plt.ylabel('Cloud attenuation CCDF (dB)')
plt.title('Cloud Attenuation statistics')
plt.legend()
plt.grid(which='both', linestyle=':', color='gray',
            linewidth=0.3, alpha=0.5)
plt.tight_layout()
plt.show()
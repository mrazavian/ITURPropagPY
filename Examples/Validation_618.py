import matplotlib.pyplot as plt
import pandas as pd
import unittest as test
import numpy as np
import sys
import os
from astropy import units as u

import iturpropag.utils as utils
from iturpropag.models.iturp837.rainfall_rate import rainfall_rate
from iturpropag.__init__ import atmospheric_attenuation_slant_path
from iturpropag.models.iturp618.rain_attenuation import rain_attenuation
from iturpropag.models.iturp618.scintillation_attenuation import scintillation_attenuation
from iturpropag.models.iturp618.site_diversity_rain_outage_probability import site_diversity_rain_outage_probability
from iturpropag.models.iturp618.rain_attenuation_probability import rain_attenuation_probability
from iturpropag.models.iturp618.rain_cross_polarization_discrimination import rain_cross_polarization_discrimination                  


file = os.path.dirname(os.path.realpath(__file__)) + "/SG3.xlsx"
xl = pd.ExcelFile(file)
##################################################################################
##             validate the ITU-R P.618.13--section 2.2.1.1
##                          Rain Attenuation
##################################################################################
df1 = xl.parse('P618-13 A_Rain')
lat = df1.values[2:,0]
lon = df1.values[2:,1]
hs = df1.values[2:,3] * u.km
f = df1.values[2:,4] * u.GHz
el = df1.values[2:,5]
tau = df1.values[2:,6]
P = df1.values[2:,7]
R001 = df1.values[2:,9]                              
A_rain = df1.values[2:,18]

A_r=np.zeros(np.size(f))

#R001=A_r
for ii in np.arange(np.size(f)):
    #R001[ii] = rainfall_rate(lat[ii], lon[ii], 0.01).value
    A_r[ii] = rain_attenuation(lat[ii], lon[ii], f[ii], el[ii], P[ii], tau[ii],\
                                hs=hs[ii], R001=R001[ii]).value

err_rain = (np.abs(A_r-A_rain)/A_rain) * 100
plt.figure(1)
plt.plot(err_rain)
plt.title('PYTHON : Rain Attenuation Validation')
plt.ylabel('(A-A_rain)/A_rain x 100')
plt.xlabel('item number refer to test values')
#plt.show()

##################################################################################
##             validate the ITU-R P.618.13--section 2.2.1.2
##            Probability of rain attenuation on a slant path
##################################################################################
df1 = xl.parse('P618-13 PofA')
lat = df1.values[2:,0]
lon = df1.values[2:,1]
hs = df1.values[2:,2] * u.km
el = df1.values[2:,4]
P0 = df1.values[2:,5]
Ls = df1.values[2:,6] * u.km                            
PA_rain = df1.values[2:,11]

PA_r = np.zeros(np.size(lat))
for ii in np.arange(np.size(PA_r)):
    PA_r[ii] = rain_attenuation_probability(lat[ii], lon[ii], el[ii], hs=hs[ii]).value
       
err_PA = (np.abs(PA_r-PA_rain)/PA_rain) * 100
plt.figure(2)
plt.semilogy(err_PA,'-s',lw=2)
plt.title('PYTHON : probability of Rain Attenuation Validation (modified)')
plt.ylabel('| (Y - Y_ref) / Y_ref x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)
#plt.show()

##################################################################################
##             validate the ITU-R P.618.13--section 2.2.4.1
##       Outage probability due to rain attenuation with site diversity
##################################################################################
df1 = xl.parse('P618-13 SD-JP')
site_A = {'lat': df1.values[2,1], 'lon': df1.values[2,2], 'hs': df1.values[2,3]*u.km,
          'P0': df1.values[2,4], 'el': df1.values[2,5], 'tau': df1.values[2,6]}
site_B = {'lat': df1.values[3,1], 'lon': df1.values[3,2], 'hs': df1.values[3,3]*u.km,
          'P0': df1.values[3,4], 'el': df1.values[3,5], 'tau': df1.values[3,6]}
site_C = {'lat': df1.values[4,1], 'lon': df1.values[4,2], 'hs': df1.values[4,3]*u.km,
          'P0': df1.values[4,4], 'el': df1.values[4,5], 'tau': df1.values[4,6]}
site_D = {'lat': df1.values[5,1], 'lon': df1.values[5,2], 'hs': df1.values[5,3]*u.km,
          'P0': df1.values[5,4], 'el': df1.values[5,5], 'tau': df1.values[5,6]}     
d_AB = df1.values[2,7] * u.km
d_CD = df1.values[4,7] * u.km
f = np.array([14.5, 14.5, 14.5, 18.0, 18.0, 18.0 , 29.0, 29.0, 29.0]) * u.GHz
a1 = df1.values[32:41,1]
a2 = df1.values[32:41,2]
tau = 0
Pjoint = df1.values[32:41,13]


Pj = np.zeros(np.size(a1))
for ii in np.arange(np.size(Pj)):
    Pj[ii] = site_diversity_rain_outage_probability(site_A['lat'], site_A['lon'], a1[ii],\
                                            site_A['el'], site_B['lat'],site_B['lon'],\
                                            a2[ii], site_B['el'], f[ii], tau, hs1=site_A['hs'],\
                                            hs2=site_B['hs']).value
    
err_Pj = (np.abs(Pj - Pjoint)/Pjoint) * 100
plt.figure(3)
plt.plot(err_Pj,'-s',lw=2)
plt.title('PYTHON : Site diversity joint probability')
plt.ylabel('| (Y - Y_ref) / Y_ref | x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)
#plt.show()

##################################################################################
##               validate the ITU-R P.618.13--section 2.4.1
##                   Scintillation Attenuation for el>5
##################################################################################
df1 = xl.parse('P618-13 A_Scint')
lat = df1.values[2:,0]
lon = df1.values[2:,1]
hs = df1.values[2:,2] * u.km
f = df1.values[2:,4] * u.GHz
el = df1.values[2:,5]
D = df1.values[2:,6] * u.m
eta = df1.values[2:,7]
P = df1.values[2:,8]
Nwet = df1.values[2:,9]                             
A_scin = df1.values[2:,15]

A_s = np.zeros(np.size(f))
#Nwet = np.zeros(np.size(f))
for ii in np.arange(np.size(f)):
    #Nwet[ii] = map_wet_term_radio_refractivity(lat[ii], lon[ii], 50).value
    A_s[ii] = scintillation_attenuation(lat[ii], lon[ii], f[ii], el[ii], P[ii], D[ii], eta[ii]).value
       
err_scin=(np.abs(A_s-A_scin)/A_scin)*100
plt.figure(4)
plt.plot(err_scin)
plt.title('PYTHON : Scintillation Attenuation error')
plt.ylabel('(As-A_scin)/A_scin x 100')
plt.xlabel('item number refer to test values')
#plt.show()

##################################################################################
##               validate the ITU-R P.618.13--section 2.5
##                         Total Attenuation
##################################################################################
df1 = xl.parse('P618-13 Att_Tot')
lat = df1.values[2:,0]
lon = df1.values[2:,1]
hs = df1.values[2:,2] * u.km
f = df1.values[2:,4] * u.GHz
el = df1.values[2:,5]
D = df1.values[2:,6] * u.m
eta = df1.values[2:,7]
tau = df1.values[2:,8]
P = df1.values[2:,9]
A_total = df1.values[2:,16]

A_t = np.zeros(np.size(A_total))
for ii in np.arange(np.size(A_t)):
    A_t[ii] = atmospheric_attenuation_slant_path(lat[ii], lon[ii], f[ii], el[ii], P[ii], D[ii],\
                                                 tau[ii], eta[ii], hs=hs[ii], mode='approx').value
       
err_total=(np.abs(A_t-A_total)/A_total)*100
plt.figure(5)
plt.plot(err_total)
plt.title('PYTHON : Total Attenuation error')
plt.ylabel('(At-A_total)/A_total x 100')
plt.xlabel('item number refer to test values')
#plt.show()

##################################################################################
##               validate the ITU-R P.618.13--section 4.1
##                      Cross Polarization Effects
##################################################################################
df1 = xl.parse('P618-13 XPD')
P = df1.values[2:,4]
f = df1.values[2:,5] * u.GHz
el = df1.values[2:,6]
tau = df1.values[2:,7]
Ap = df1.values[2:,8]
XPD_total = df1.values[2:,17]

XPD_t = np.zeros(np.size(XPD_total))
for ii in np.arange(np.size(XPD_t)):
    XPD_t[ii] = rain_cross_polarization_discrimination(Ap[ii], f[ii], el[ii], P[ii], tau[ii]).value
       
err_xpd=(np.abs(XPD_t-XPD_total)/XPD_total)*100
plt.figure(6)
plt.plot(err_xpd,'-s',lw=2)
plt.title('PYTHON : XPD error')
plt.ylabel('|(Y - Y_ref) / Y_ref| x 100')
plt.xlabel('item number refer to test values')
plt.show()
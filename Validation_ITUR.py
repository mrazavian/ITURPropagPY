import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from astropy import units as u

from iturpropag.models.iturp453.map_wet_term_radio_refractivity import map_wet_term_radio_refractivity
from iturpropag.models.iturp836.surface_water_vapour_density import surface_water_vapour_density
from iturpropag.models.iturp836.total_water_vapour_content import total_water_vapour_content
from iturpropag.models.iturp837.rainfall_probability import rainfall_probability
from iturpropag.models.iturp837.rainfall_rate import rainfall_rate
from iturpropag.models.iturp838.rain_specific_attenuation_coefficients import rain_specific_attenuation_coefficients
from iturpropag.models.iturp839.rain_height import rain_height
from iturpropag.models.iturp840.columnar_content_reduced_liquid import columnar_content_reduced_liquid
from iturpropag.models.iturp840.cloud_attenuation import cloud_attenuation
from iturpropag.models.iturp676.gammaw_exact import gammaw_exact
from iturpropag.models.iturp676.gamma0_exact import gamma0_exact
from iturpropag.models.iturp676.gammaw_approx import gammaw_approx
from iturpropag.models.iturp676.gamma0_approx import gamma0_approx
from iturpropag.models.iturp676.slant_inclined_path_equivalent_height import slant_inclined_path_equivalent_height
from iturpropag.models.iturp676.gaseous_attenuation_slant_path import gaseous_attenuation_slant_path

file = os.path.dirname(os.path.realpath(__file__)) + "/SG3.xlsx"
xl = pd.ExcelFile(file)
##################################################################################
##             validate the ITU-R P.453.13
##                     Nwet
##################################################################################
df1 = xl.parse('P453-13 Nwet')
lat = df1.values[2:,0]
lon = df1.values[2:,1]
Nwet_ref = df1.values[2:,2]
p = 50
Nwet = np.zeros(np.size(Nwet_ref))
for ii in np.arange(np.size(Nwet)):
    Nwet[ii] = map_wet_term_radio_refractivity(lat[ii], lon[ii], p).value

err_Nwet = np.abs(Nwet - Nwet_ref) / Nwet_ref * 100
plt.figure(1)
plt.semilogy(err_Nwet,lw=2)
plt.title('PYTHON : Nwet Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)
#plt.show()

##################################################################################
##                       validate the ITU-R P.836.6
##           Water Vapour: surface density and total columnar content
##################################################################################
df1 = xl.parse('P836-6 WV')
lat = df1.values[3:,0]
lon = df1.values[3:,1]
hs = df1.values[3:,2]
P = df1.values[3:,3]
SWVD = df1.values[3:,4]
TWVC = df1.values[3:,17]

swvd_i = np.zeros(np.size(SWVD))
twvc_i = np.zeros(np.size(TWVC))
for ii in np.arange(np.size(swvd_i)):
   swvd_i[ii] = surface_water_vapour_density(lat[ii], lon[ii], P[ii], alt=hs[ii]).value
   twvc_i[ii] = total_water_vapour_content(lat[ii], lon[ii], P[ii], alt=hs[ii]).value

err_swvd = np.abs(swvd_i - SWVD) / SWVD * 100
err_twvc = np.abs(twvc_i - TWVC) / TWVC * 100
plt.figure(2)
plt.subplot(211)
plt.semilogy(err_swvd,'-s',lw=2)
plt.title('PYTHON : Surface Water Vapour Density Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.grid(True)

plt.subplot(212)
plt.semilogy(err_twvc,'-s',lw=2)
plt.title('PYTHON : Total Water Vapour Content Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)
#plt.show()

##################################################################################
##                       validate the ITU-R P.837.7
##        Characteristics of precipitation for propagation modelling (P0,  Rp)
##################################################################################
df1 = xl.parse('P837-7 Rp')
lat = df1.values[2:,0]
lon = df1.values[2:,1]
P = df1.values[2:,2]
P0 = df1.values[2:,39] + 1e-13
Rp = df1.values[2:,40] + 1e-13

PP = np.zeros(np.size(P0))
RR = np.zeros(np.size(Rp))
for ii in np.arange(np.size(PP)):
    PP[ii] = rainfall_probability(lat[ii], lon[ii]).value
    RR[ii] = rainfall_rate(lat[ii], lon[ii], P[ii]).value

err_P0 = np.abs(PP - P0) / P0 * 100
err_Rp = np.abs(RR - Rp) / Rp * 100
plt.figure(3)
plt.subplot(211)
plt.semilogy(err_P0,'-s',lw=2)
plt.title('PYTHON : P0 anual Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.grid(True)

plt.subplot(212)
plt.semilogy(err_Rp,'-s',lw=2)
plt.title('PYTHON : Rainfall Rain (Rp) Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)
#plt.show()

##################################################################################
##                       validate the ITU-R P.838.3
##       Specific attenuation model for rain for use in prediction methods (a,k)
##################################################################################
df1 = xl.parse('P838-3 Sp.Att')
el = df1.values[2:,3]
f = df1.values[2:,4] * u.GHz
tau = df1.values[2:,6]
K = df1.values[2:,11]
alpha = df1.values[2:,12]

k_t = np.zeros(np.size(K))
alpha_t = np.zeros(np.size(alpha))
for ii in np.arange(np.size(k_t)):
    k_t[ii], alpha_t[ii] = rain_specific_attenuation_coefficients(f[ii], el[ii], tau[ii])

err_k = np.abs(k_t - K) / K * 100
err_alpha = np.abs(alpha_t - alpha) / alpha * 100
plt.figure(4)
plt.subplot(211)
plt.semilogy(err_k,'-s',lw=2)
plt.title('PYTHON : Specific attenuation (k) Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.grid(True)

plt.subplot(212)
plt.semilogy(err_alpha,'-s',lw=2)
plt.title('PYTHON : Specific attenuation (alpha) Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)
#plt.show()

##################################################################################
##                       validate the ITU-R P.839.4
##              Rain height model for prediction methods (hr)
##################################################################################
df1 = xl.parse('P839-4 Rain_Height')
lat = df1.values[2:,0]
lon = df1.values[2:,1]
HR = df1.values[2:,3]

hr = np.zeros(np.size(HR))
for ii in np.arange(np.size(hr)):
    hr[ii] = rain_height(lat[ii], lon[ii]).value

err_hr = np.abs(hr - HR) / HR * 100
plt.figure(5)
plt.semilogy(err_hr,'-s',lw=2)
plt.title('PYTHON : Rain Height(hr) Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)
#plt.show()

##################################################################################
##                       validate the ITU-R P.840-7
##      Total columnar content of reduced cloud liquid water, (Lred)
##################################################################################
df1 = xl.parse('P840-7 Lred')
lat = df1.values[2:,0]
lon = df1.values[2:,1]
P = df1.values[2:,2]
Lred = df1.values[2:,3]

l_red = np.zeros(np.size(Lred))
for ii in np.arange(np.size(Lred)):
    l_red[ii] = columnar_content_reduced_liquid(lat[ii], lon[ii], P[ii]).value

err_Lred = np.abs(l_red - Lred) / Lred * 100
plt.figure(6)
plt.semilogy(err_Lred,'-s',lw=2)
plt.title('PYTHON : Total columnar content of reduced cloud liquid water,(Lred) Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)
#plt.show()

##################################################################################
##                       validate the ITU-R P.840-7
##                Attenuation due to clouds and fog. (Ac)
##################################################################################
df1 = xl.parse('P840-7 A_Clouds')
lat = df1.values[2:,2]
lon = df1.values[2:,3]
f = df1.values[2:,4] * u.GHz
el =df1.values[2:,5]
P = df1.values[2:,6]
Ac = df1.values[2:,12]

Acc = np.zeros(np.size(Ac))
for ii in np.arange(np.size(Acc)):
    Acc[ii] = cloud_attenuation(lat[ii], lon[ii], el[ii], f[ii], P[ii]).value

err_Ac = np.abs(Acc - Ac) / Ac * 100
plt.figure(7)
plt.semilogy(err_Ac,'-s',lw=2)
plt.title('PYTHON : Attenuation due to clouds and fog (Ac) Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)
#plt.show()

##################################################################################
##                       validate the ITU-R P.676.11
##                Specific attenuation (gamma_oxy  /  gamma_wv)
##################################################################################
df1 = xl.parse('P676-11 SpAtt')

f = np.array([12,20,60,90,130]) * u.GHz
p = 1013.25 * u.hPa
T = 288.15 * u.K
rho = 7.5 * u.g / u.m**3
P = p.value + rho.value * T.value/ 216.7 
Gox_exact = df1.values[9,2:7]
Gwv_exact = df1.values[9,10:15]
Gox_approx = df1.values[9,18:23]
Gwv_approx = df1.values[9,26:31]

g0_exact = gamma0_exact(f, P, rho, T).value
gw_exact = gammaw_exact(f, P, rho, T).value

g0_approx = gamma0_approx(f, P, rho, T).value
gw_approx = gammaw_approx(f, P, rho, T).value

err_g0_exact = np.abs(g0_exact - Gox_exact) / Gox_exact * 100
err_gw_exact = np.abs(gw_exact - Gwv_exact) / Gwv_exact * 100

err_g0_approx = np.abs(g0_approx - Gox_approx) / Gox_approx * 100
err_gw_approx = np.abs(gw_approx - Gwv_approx) / Gwv_approx * 100

plt.figure(8)
plt.subplot(221)
plt.semilogy(err_g0_exact,lw=2)
plt.title('PYTHON : gamma_oxy_exact Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.grid(True)

plt.subplot(222)
plt.semilogy(err_gw_exact,lw=2)
plt.title('PYTHON : gamma_wv_exact Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)

plt.subplot(223)
plt.semilogy(err_g0_approx,lw=2)
plt.title('PYTHON : gamma_oxy_approx Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.grid(True)

plt.subplot(224)
plt.semilogy(err_gw_approx,lw=2)
plt.title('PYTHON : gamma_wv_approx Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.xlabel('item number refer to test values')
plt.grid(True)
#plt.show()

##################################################################################
##                       validate the ITU-R P.676.11
##                      Slant Paths (approx) attenuation
##################################################################################
df1 = xl.parse('P676-11 A_Gas ')

hs = df1.values[2:,2]
rho = df1.values[2:,5]
el = df1.values[2:,6]
f = df1.values[2:,7] * u.GHz
T = df1.values[2:,8]
p = df1.values[2:,10]
e = df1.values[2:,9]
V = df1.values[2:,18]
Agas = df1.values[2:,31]

Ag = np.zeros(np.size(Agas))
for ii in np.arange(np.size(Ag)):
    Ag[ii] = gaseous_attenuation_slant_path(f[ii], el[ii], rho[ii], p[ii] + e[ii], T[ii], V_t=V[ii],\
                                             h=hs[ii],mode='approx').value

err_Ag = np.abs(Ag - Agas) / Agas * 100

plt.figure(9)
plt.semilogy(err_Ag,lw=2)
plt.title('PYTHON : Slant Paths (approx.) Validation')
plt.ylabel('| (Y-Y_ref) / Y_ref | x 100')
plt.grid(True)
plt.show()
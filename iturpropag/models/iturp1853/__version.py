# -*- coding: utf-8 -*-

from iturpropag.models.iturp1853 import cloud_attenuation_synthesis
from iturpropag.models.iturp1853 import cloud_liquid_water_synthesis
from iturpropag.models.iturp1853 import integrated_water_vapour_coefficients
from iturpropag.models.iturp1853 import integrated_water_vapour_synthesis
from iturpropag.models.iturp1853 import water_vapour_attenuation_synthesis
from iturpropag.models.iturp1853 import rain_attenuation_synthesis
from iturpropag.models.iturp1853 import scintillation_attenuation_synthesis
from iturpropag.models.iturp1853 import total_attenuation_synthesis
from iturpropag.models.iturp1853 import surface_mean_pressure
from iturpropag.models.iturp1853 import surface_mean_water_vapour_density


def change_version(new_version):
    """
    Change the version of the ITU-R P.1853 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.1853-2 (Current version)
            * P.1853-1
            * P.1853-0
    """
    cloud_attenuation_synthesis.change_version(new_version)
    cloud_liquid_water_synthesis.change_version(new_version)
    integrated_water_vapour_coefficients.change_version(new_version)
    integrated_water_vapour_synthesis.change_version(new_version)
    water_vapour_attenuation_synthesis.change_version(new_version)
    rain_attenuation_synthesis.change_version(new_version)
    scintillation_attenuation_synthesis.change_version(new_version)
    total_attenuation_synthesis.change_version(new_version)
    surface_mean_pressure.change_version(new_version)
    surface_mean_water_vapour_density.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.1853 recommendation currently being used.
    """
    val = {"cloud_attenuation_synthesis" : cloud_attenuation_synthesis.get_version(),
            "cloud_liquid_water_synthesis" : cloud_liquid_water_synthesis.get_version(),
            "integrated_water_vapour_coefficients" : integrated_water_vapour_coefficients.get_version(),
            "integrated_water_vapour_synthesis" : integrated_water_vapour_synthesis.get_version(),
            "water_vapour_attenuation_synthesis" : water_vapour_attenuation_synthesis.get_version(),
            "rain_attenuation_synthesis" : rain_attenuation_synthesis.get_version(),
            "scintillation_attenuation_synthseis" : scintillation_attenuation_synthesis.get_version(),
            "total_attenuation_synthesis" : total_attenuation_synthesis.get_version(),
            "surface_mean_pressure" : surface_mean_pressure.get_version(),
            "surface_mean_water_vapour_density" : surface_mean_water_vapour_density.get_version()}

    return val


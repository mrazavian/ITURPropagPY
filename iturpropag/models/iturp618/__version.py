# -*- coding: utf-8 -*-

from iturpropag.models.iturp618 import fit_rain_attenuation_to_lognormal
from iturpropag.models.iturp618 import rain_attenuation
from iturpropag.models.iturp618 import rain_attenuation_probability
from iturpropag.models.iturp618 import rain_cross_polarization_discrimination
from iturpropag.models.iturp618 import scintillation_attenuation
from iturpropag.models.iturp618 import scintillation_attenuation_sigma
from iturpropag.models.iturp618 import site_diversity_rain_outage_probability



def change_version(new_version):
    """
    Change the version of the ITU-R P.618 recommendation currently being used.


    Parameters
    ----------
    new_version : int
        Number of the version to use.
        Valid values are:
           * P.618-13 (02/12) (Current version)
             P.618-12
    """
    fit_rain_attenuation_to_lognormal.change_version(new_version)
    rain_attenuation.change_version(new_version)
    rain_attenuation_probability.change_version(new_version)
    rain_cross_polarization_discrimination.change_version(new_version)
    scintillation_attenuation.change_version(new_version)
    scintillation_attenuation_sigma.change_version(new_version)
    site_diversity_rain_outage_probability.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.618 recommendation currently being used.
    """
    val = {"fit_rain_attenuation_to_lognormal" : fit_rain_attenuation_to_lognormal.get_version(),
            "rain_attenuation" : rain_attenuation.get_version(),
            "rain_attenuation_probability" : rain_attenuation_probability.get_version(),
            "rain_cross_polarization_discrimination" : rain_cross_polarization_discrimination.get_version(),
            "scintillation_attenuation" : scintillation_attenuation.get_version(),
            "scintillation_attenuation_sigma" : scintillation_attenuation_sigma.get_version(),
            "site_diversity_rain_outage_probability" : site_diversity_rain_outage_probability.get_version()}

    return val


# -*- coding: utf-8 -*-

from iturpropag.models.iturp453 import DN1
from iturpropag.models.iturp453 import DN65
from iturpropag.models.iturp453 import dry_term_radio_refractivity
from iturpropag.models.iturp453 import map_wet_term_radio_refractivity
from iturpropag.models.iturp453 import radio_refractive_index
from iturpropag.models.iturp453 import saturation_vapour_pressure
from iturpropag.models.iturp453 import water_vapour_pressure
from iturpropag.models.iturp453 import wet_term_radio_refractivity




def change_version(new_version):
    """
    Change the version of the ITU-R P.453 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
           	* P.453-13 (02/12) (Current version)
           	* P.453-12
    """
    DN1.change_version(new_version)
    DN65.change_version(new_version)
    dry_term_radio_refractivity.change_version(new_version)
    map_wet_term_radio_refractivity.change_version(new_version)
    radio_refractive_index.change_version(new_version)
    saturation_vapour_pressure.change_version(new_version)
    water_vapour_pressure.change_version(new_version)
    wet_term_radio_refractivity.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.453 recommendation currently being used.
    """
    val = {"DN1" : DN1.get_version(),
            "DN65" : DN65.get_version(),
            "dry_term_radio_refractivity" : dry_term_radio_refractivity.get_version(),
            "map_wet_term_radio_refractivity" : map_wet_term_radio_refractivity.get_version(),
            "radio_refractive_index" : radio_refractive_index.get_version(),
            "saturation_vapour_pressure" : saturation_vapour_pressure.get_version(),
            "water_vapour_pressure" : water_vapour_pressure.get_version(),
            "wet_term_radio_refractivity" : wet_term_radio_refractivity.get_version()}

    return val


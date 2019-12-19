# -*- coding: utf-8 -*-

from iturpropag.models.iturp836 import surface_water_vapour_density
from iturpropag.models.iturp836 import total_water_vapour_content



def change_version(new_version):
    """
    Change the version of the ITU-R P.836 recommendation currently being used.


    Parameters
    ----------
    new_version : int
        Number of the version to use.
        Valid values are:
           * P.836-6  (Current version)
             P.836-5
             P.836-4
    """
    surface_water_vapour_density.change_version(new_version)
    total_water_vapour_content.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.836 recommendation currently being used.
    """
    val = {"surface_water_vapour_density" : surface_water_vapour_density.get_version(),
            "total_water_vapour_content" : total_water_vapour_content.get_version()}

    return val


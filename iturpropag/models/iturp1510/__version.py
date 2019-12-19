# -*- coding: utf-8 -*-

from iturpropag.models.iturp1510 import surface_mean_temperature
from iturpropag.models.iturp1510 import surface_month_mean_temperature


def change_version(new_version):
    """
    Change the version of the ITU-R P.1510 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
                Number of the version to use.
                Valid values are:
                * P.1510-1  (Current version)
                * P.1510-0
    """
    surface_mean_temperature.change_version(new_version)
    surface_month_mean_temperature.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.1510 recommendation currently being used.
    """
    val = {"surface_mean_temperature" : surface_mean_temperature.get_version(),
            "surface_month_mean_temperature" : surface_month_mean_temperature.get_version()}

    return val


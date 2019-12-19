# -*- coding: utf-8 -*-

from iturpropag.models.iturp839 import isotherm_0
from iturpropag.models.iturp839 import rain_height



def change_version(new_version):
    """
    Change the version of the ITU-R P.839 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.839-4  (Current version)
            * P.839-3
            * P.839-2

    """
    isotherm_0.change_version(new_version)
    rain_height.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.839 recommendation currently being used.
    """
    val = {"isotherm_0" : isotherm_0.get_version(),
            "rain_height" : rain_height.get_version()}

    return val


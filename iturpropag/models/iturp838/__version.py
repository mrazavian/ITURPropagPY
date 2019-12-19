# -*- coding: utf-8 -*-
from iturpropag.models.iturp838 import rain_specific_attenuation_coefficients



def change_version(new_version):
    """
    Change the version of the ITU-R P.838 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
          Number of the version to use.
          Valid values are:
            * P.838-3 (02/12) (Current version)
            * P.838-2
            * P.838-1
            * P.838-0
    """
    rain_specific_attenuation_coefficients.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.838 recommendation currently being used.
    """
    val = {"rain_specific_attenuation_coefficients" : rain_specific_attenuation_coefficients.get_version()}

    return val


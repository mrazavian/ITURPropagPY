# -*- coding: utf-8 -*-

from iturpropag.models.iturp837 import rainfall_probability
from iturpropag.models.iturp837 import rainfall_rate



def change_version(new_version):
    """
    Change the version of the ITU-R P.837 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
            Number of the version to use.
            Valid values are:
            * P.837-8 (Current version - only editorial changes)
            * P.837-7
            * P.837-6
    """
    rainfall_probability.change_version(new_version)
    rainfall_rate.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.837 recommendation currently being used.
    """
    val = {"rainfall_probability" : rainfall_probability.get_version(),
            "rainfall_rate" : rainfall_rate.get_version()}

    return val


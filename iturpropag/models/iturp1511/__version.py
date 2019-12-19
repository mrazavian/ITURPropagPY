# -*- coding: utf-8 -*-

from iturpropag.models.iturp1511 import topographic_altitude



def change_version(new_version):
    """
    Change the version of the ITU-R P.1511 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
        	Number of the version to use.
          	Valid values are:
            * P.1511-2  (Current version)
        	* P.1511-1 
            * P.1511-0
    """
    topographic_altitude.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.1511 recommendation currently being used.
    """
    val = {"topographic_altitude" : topographic_altitude.get_version()}

    return val


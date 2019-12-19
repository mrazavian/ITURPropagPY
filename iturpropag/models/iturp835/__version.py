# -*- coding: utf-8 -*-

from iturpropag.models.iturp835 import pressure
from iturpropag.models.iturp835 import standard_pressure
from iturpropag.models.iturp835 import standard_temperature
from iturpropag.models.iturp835 import standard_water_vapour_density
from iturpropag.models.iturp835 import standard_water_vapour_pressure
from iturpropag.models.iturp835 import temperature
from iturpropag.models.iturp835 import water_vapour_density



def change_version(new_version):
    """
    Change the version of the ITU-R P.835 recommendation currently being used.


    Parameters
    ----------
    new_version : int
        Number of the version to use.
        Valid values are:
           * P.835-6  (Current version)
             P.835-5
    """
    pressure.change_version(new_version)
    standard_pressure.change_version(new_version)
    standard_temperature.change_version(new_version)
    standard_water_vapour_density.change_version(new_version)
    standard_water_vapour_pressure.change_version(new_version)
    temperature.change_version(new_version)
    water_vapour_density.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.835 recommendation currently being used.
    """
    val = {"pressure" : pressure.get_version(),
           "standard_pressure" : standard_pressure.get_version(),
           "standard_temperature" : standard_temperature.get_version(),
           "standard_water_vapour_density" : standard_water_vapour_density.get_version(),
           "standard_water_vapour_pressure" : standard_water_vapour_pressure.get_version(),
           "temperature" : temperature.get_version(),
           "water_vapour_density" : water_vapour_density.get_version()}

    return val


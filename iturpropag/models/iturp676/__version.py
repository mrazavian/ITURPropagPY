# -*- coding: utf-8 -*-

from iturpropag.models.iturp676 import gamma0_approx
from iturpropag.models.iturp676 import gamma0_exact
from iturpropag.models.iturp676 import gammaw_approx
from iturpropag.models.iturp676 import gammaw_exact
from iturpropag.models.iturp676 import gaseous_attenuation_inclined_path
from iturpropag.models.iturp676 import gaseous_attenuation_slant_path
from iturpropag.models.iturp676 import gaseous_attenuation_terrestrial_path
from iturpropag.models.iturp676 import slant_inclined_path_equivalent_height
from iturpropag.models.iturp676 import zenith_water_vapour_attenuation



def change_version(new_version):
    """
    Change the version of the ITU-R P.676 recommendation currently being used.


    Parameters
    ----------
    new_version : int
        Number of the version to use.
        Valid values are:
           * P.676-11 (02/12) (Current version)
             P.676-10
             P.676-9
    """
    gamma0_approx.change_version(new_version)
    gamma0_exact.change_version(new_version)
    gammaw_approx.change_version(new_version)
    gammaw_exact.change_version(new_version)
    gaseous_attenuation_inclined_path.change_version(new_version)
    gaseous_attenuation_slant_path.change_version(new_version)
    gaseous_attenuation_terrestrial_path.change_version(new_version)
    slant_inclined_path_equivalent_height.change_version(new_version)
    zenith_water_vapour_attenuation.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.676 recommendation currently being used.
    """
    val = {"gamma0_approx" : gamma0_approx.get_version(),
            "gamma0_exact" : gamma0_exact.get_version(),
            "gammaw_approx" : gammaw_approx.get_version(),
            "gammaw_exact" : gammaw_exact.get_version(),
            "gaseous_attenuation_inclined_path" : gaseous_attenuation_inclined_path.get_version(),
            "gaseous_attenuation_slant_path" : gaseous_attenuation_slant_path.get_version(),
            "gaseous_attenuation_terrestrial_path" : gaseous_attenuation_terrestrial_path.get_version(),
            "slant_inclined_path_equivalent_height" : slant_inclined_path_equivalent_height.get_version(),
            "zenith_water_vapour_attenuation" : zenith_water_vapour_attenuation.get_version()}

    return val


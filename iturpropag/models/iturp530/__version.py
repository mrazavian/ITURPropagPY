# -*- coding: utf-8 -*-

from iturpropag.models.iturp530 import diffraction_loss
from iturpropag.models.iturp530 import fresnel_ellipse_radius
from iturpropag.models.iturp530 import inverse_rain_attenuation
from iturpropag.models.iturp530 import multipath_loss
from iturpropag.models.iturp530 import multipath_loss_for_A
from iturpropag.models.iturp530 import rain_attenuation
from iturpropag.models.iturp530 import rain_event_count
from iturpropag.models.iturp530 import XPD_outage_clear_air
from iturpropag.models.iturp530 import XPD_outage_precipitation



def change_version(new_version):
    """
    Change the version of the ITU-R P.530 recommendation currently being used.


    Parameters
    ----------
    new_version : int
        Number of the version to use.
        Valid values are:
           * P.530-17 (02/12) (Current version)
             P.530-16
    """
    diffraction_loss.change_version(new_version)
    fresnel_ellipse_radius.change_version(new_version)
    inverse_rain_attenuation.change_version(new_version)
    multipath_loss.change_version(new_version)
    multipath_loss_for_A.change_version(new_version)
    rain_attenuation.change_version(new_version)
    rain_event_count.change_version(new_version)
    XPD_outage_clear_air.change_version(new_version)
    XPD_outage_precipitation.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.530 recommendation currently being used.
    """
    val = {"diffraction_loss" : diffraction_loss.get_version(),
            "fresnel_ellipse_radius" : fresnel_ellipse_radius.get_version(),
            "inverse_rain_attenuation" : inverse_rain_attenuation.get_version(),
            "multipath_loss" : multipath_loss.get_version(),
            "multipath_loss_for_A" : multipath_loss_for_A.get_version(),
            "rain_attenuation" : rain_attenuation.get_version(),
            "rain_event_count" : rain_event_count.get_version(),
            "XPD_outage_clear_air" : XPD_outage_clear_air.get_version(),
            "XPD_outage_precipitation" : XPD_outage_precipitation.get_version()}

    return val


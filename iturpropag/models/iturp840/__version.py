# -*- coding: utf-8 -*-

from iturpropag.models.iturp840 import columnar_content_reduced_liquid
from iturpropag.models.iturp840 import lognormal_approximation_coefficient
from iturpropag.models.iturp840 import specific_attenuation_coefficients



def change_version(new_version):
    """
    Change the version of the ITU-R P.840 recommendation currently being used.


    Parameters
    ----------
    - new_version : int
                Number of the version to use.
                Valid values are:
                * P.840-7  (Current version)
                * P.840-6
                * P.840-5
                * P.840-4
    """
    columnar_content_reduced_liquid.change_version(new_version)
    lognormal_approximation_coefficient.change_version(new_version)
    specific_attenuation_coefficients.change_version(new_version)


def get_version():
    """
    Obtain the version of the ITU-R P.840 recommendation currently being used.
    """
    val = {"columnar_content_reduced_liquid" : columnar_content_reduced_liquid.get_version(),
            "lognormal_approximation_coefficient" : lognormal_approximation_coefficient.get_version(),
            "specific_attenuation_coefficients" : specific_attenuation_coefficients.get_version()}

    return val


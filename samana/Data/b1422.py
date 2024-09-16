import matplotlib.pyplot as plt
from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _B1422(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):

        z_lens = 0.34
        z_source = 3.62
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1]
        super(_B1422, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

class B1422_HST(_B1422):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = [-0.347, -0.734, -1.096, 0.207]
        y_image = [0.964, 0.649, -0.079, -0.148]
        magnifications = [0.88, 1., 0.474, 0.025]
        #flux_uncertainties = [0.01 / 0.88, 0.01, 0.006 / 0.47, 0.01]
        flux_uncertainties = None
        image_position_uncertainties = [0.005] * 4
        super(B1422_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=True)


import matplotlib.pyplot as plt
from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _MG0414(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):

        z_lens = 0.96
        z_source = 2.64

        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_MG0414, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

class MG0414(_MG0414):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        gx, gy = 0.482, -1.279
        x_image = np.array([-0.576, -0.739, 0.0, 1.345]) - gx
        y_image = np.array([-1.993, -1.520, 0.0, -1.648]) - gy
        magnifications = np.array([1.0, 0.86, 0.36, 0.16])
        flux_uncertainties = np.array([0.05 / 0.83, 0.04 / 0.36, 0.04 / 0.34])
        image_position_uncertainties = [0.005] * 4
        super(MG0414, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)


import matplotlib.pyplot as plt
from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _RXJ0911(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes):

        z_lens = 0.77
        z_source = 2.76
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_RXJ0911, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

    @property
    def coordinate_properties(self):
        window_size = 6
        deltaPix = 0.05
        ra_at_xy_0 = -3
        dec_at_xy_0 = -3
        transform_pix2angle = np.array([[0.05, 0.], [0., 0.05]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

class RXJ0911_HST(_RXJ0911):
    g2x = -0.767
    g2y = 0.657
    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([0.688, 0.946, 0.672, -2.283])
        y_image = np.array([-0.517, -0.112, 0.442, 0.274])
        magnifications = np.array([0.56, 1.0, 0.53, 0.24])
        flux_uncertainties = np.array([0.04 / 0.56, 0.05, 0.04 / 0.53, 0.04 / 0.24])
        image_position_uncertainties = [0.005] * 4
        super(RXJ0911_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=True)


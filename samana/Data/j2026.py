import matplotlib.pyplot as plt
from samana.Data.data_base import QuadNoImageDataBase
import numpy as np

class _J2026(QuadNoImageDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, z_lens):

        z_source = 2.23
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        super(_J2026, self).__init__(z_lens, z_source, x_image, y_image, magnifications, image_position_uncertainties,
                                       flux_uncertainties, uncertainty_in_fluxes, keep_flux_ratio_index)

class J2026(_J2026):

    def __init__(self, z_lens=0.5):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        reorder = [0, 1, 2, 3]
        x_image = np.array([0.0, 0.252, -0.164, -0.733])[reorder]
        y_image = np.array([0.0, 0.219, 1.431, 0.386])[reorder]
        x_image -= np.mean(x_image) - 0.1
        y_image -= np.mean(y_image)

        # mags HST: check image ordering
        # m = [1.0, 0.75, 0.31, 0.28]
        # flux_uncertainties = [0.02, 0.02/0.75, 0.02/0.31, 0.01/0.28]

        image_position_uncertainties = [0.005] * 4 # 5 marcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J2026, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False, z_lens=z_lens)

    def redshift_sampling(self):
        return True

    def sample_z_lens(self):
        z_lens = np.random.uniform(0.2, 1.0)
        return np.round(z_lens, 2)

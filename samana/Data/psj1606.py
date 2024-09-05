from samana.Data.data_base import ImagingDataBase
import numpy as np
from samana.Data.ImageData.psj1606_814w import image_data, psf_model, psf_error_map

class _PSJ1606(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor=1.0):

        z_lens = 0.92
        z_source = 1.7
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self._psf_estimate_init = psf_model
        self._psf_error_map_init = psf_error_map
        self._image_data = image_data
        self._supersample_factor = supersample_factor
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(_PSJ1606, self).__init__(z_lens, z_source,
                                        kwargs_data_joint, x_image, y_image,
                                        magnifications, image_position_uncertainties, flux_uncertainties,
                                        uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
                                        likelihood_mask_imaging_weights)

    def likelihood_masks(self, x_image, y_image):

        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        _x = np.linspace(-window_size / 2, window_size / 2, image_data.shape[0])
        _y = np.linspace(-window_size / 2, window_size / 2, image_data.shape[0])
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx)
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
        likelihood_mask[inds] = 0.0

        return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.006833,
                       'exposure_time': 1428.0,
                       'ra_at_xy_0': ra_at_xy_0,
                       'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': transform_pix2angle,
                       'image_data': self._image_data}
        return kwargs_data

    @property
    def kwargs_numerics(self):
        return {'supersampling_factor': int(self._supersample_factor),
                'supersampling_convolution': False}

    @property
    def coordinate_properties(self):

        deltaPix = 0.04
        window_size = 86 * deltaPix
        ra_at_xy_0 = 1.720723
        dec_at_xy_0 = -1.719579
        transform_pix2angle = np.array([[-4.00070427e-02, -9.77154997e-06],
                                        [-9.76233061e-06, 3.99999798e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_error_map': self._psf_error_map_init}
        return kwargs_psf

class PSJ1606_HST(_PSJ1606):

    def __init__(self, supersample_factor=1.0):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([0.8715, -0.7495, 0.0795, -0.2575])
        y_image = np.array([0.28925, -0.30275, -0.61575, 0.44125])
        x_offset = -0.016
        y_offset = -0.005
        x_image += x_offset
        y_image += y_offset
        image_position_uncertainties = [0.005] * 4
        flux_uncertainties = [0.02/1.01, 0.01/0.59, 0.02/0.75]
        magnifications = np.array([1.0, 1.01, 0.59, 0.75])
        super(PSJ1606_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False, supersample_factor=supersample_factor)

# class PSJ1606_HST(_PSJ1606):
#
#     def __init__(self, supersample_factor=1.0):
#         """
#
#         :param image_position_uncertainties: list of astrometric uncertainties for each image
#         i.e. [0.003, 0.003, 0.003, 0.003]
#         :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
#         post-processing
#         :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
#         :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
#         """
#         x_image = [0.838, -0.784, 0.048, -0.289]
#         y_image = [0.378, -0.211, -0.527, 0.528]
#
#         x_offset = 0.01
#         y_offset = -0.08
#         x_image = np.array(x_image) + x_offset
#         y_image = np.array(y_image) + y_offset
#         magnifications = np.array([1.0, 1.0, 0.59, 0.79])
#         image_position_uncertainties = [0.005] * 4
#         flux_uncertainties = [0.03, 0.03, 0.02 / 0.6, 0.02 / 0.78]
#         super(PSJ1606_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
#                                           uncertainty_in_fluxes=True, supersample_factor=supersample_factor)

# import matplotlib.pyplot as plt
#
# xc = np.array([1.6217, -0.0005, 0.8328, 0.4948]) - 0.5
# yc = np.array([0.5890, 0.0003, -0.3170, 0.7377]) - 0.2
# fc = np.array([0.867,1.0, 0.670, 0.694])
#
# lens = PSJ1606_HST()
# flux_ratios = np.round(lens.magnifications/lens.magnifications[0],2)
#
# colors = ['k', 'r','g','m']
# for i in range(0, 4):
#     plt.scatter(lens.x_image[i], lens.y_image[i],color=colors[i])
#     plt.scatter(xc[i], yc[i], color=colors[i],marker='+')
#     plt.annotate(str(flux_ratios[i]),
#                  xy=(lens.x_image[i], lens.y_image[i]),color=colors[i])
#
# plt.show()

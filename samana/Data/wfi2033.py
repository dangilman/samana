from samana.Data.data_base import ImagingDataBase
import numpy as np
from samana.Data.ImageData.wfi2033_814w import image_data, psf_error_map, psf_model


class _WFI2033(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor=1.0):

        z_lens = 0.66
        z_source = 1.66
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
        super(_WFI2033, self).__init__(z_lens, z_source,
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
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2.)
        likelihood_mask[inds] = 0.0
        return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.0058,
                       'exposure_time': 2085.0,
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
        deltaPix = 0.05
        window_size = 112 * deltaPix
        ra_at_xy_0 = 2.8044480233
        dec_at_xy_0 = -2.7982320
        transform_pix2angle = np.array([[-5.00479193e-02, -3.15096429e-05],
                                        [-3.15326310e-05,  4.99999618e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_error_map': self._psf_error_map_init}
        return kwargs_psf

class WFI2033_HST(_WFI2033):

    def __init__(self, supersample_factor=1.0):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """

        x_image = np.array([-0.751, -0.039,  1.445, -0.668])
        y_image = np.array([ 0.953,  1.068, -0.307, -0.585])
        # caluclated from image data
        x_shifts = -0.05
        y_shifts = -0.04 - 0.01
        x_image += x_shifts
        y_image += y_shifts

        # delta_x_image = np.array([-0.01314664, -0.00057129, 0.00157636, -0.00519599])
        # delta_y_image = np.array([0.01332557, 0.03394125, 0.01965914, -0.00873674])
        # x_image += delta_x_image
        # y_image += delta_y_image

        magnifications = [1.,   0.65, 0.5,  0.53]
        image_position_uncertainties = [0.005] * 4
        flux_uncertainties = [0.03, 0.03/0.64, 0.02/0.5, 0.02/0.53]
        uncertainty_in_fluxes = True
        super(WFI2033_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                          flux_uncertainties, uncertainty_in_fluxes=uncertainty_in_fluxes,
                                         supersample_factor=supersample_factor)


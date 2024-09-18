from samana.Data.data_base import ImagingDataBase
import numpy as np
from samana.Data.ImageData.j0405_814w import image_data, psf_error_map, psf_model

class _DESJ0405(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties,
                 flux_uncertainties, uncertainty_in_fluxes, supersample_factor,
                 mask_quasar_images_for_logL=True):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        z_lens = 0.85 #photo-z
        z_source = 1.70

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
        super(_DESJ0405, self).__init__(z_lens, z_source,
                                        kwargs_data_joint, x_image, y_image,
                                        magnifications, image_position_uncertainties, flux_uncertainties,
                                        uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
                                        likelihood_mask_imaging_weights)

    def likelihood_masks(self, x_image, y_image):

        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        _x = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
        _y = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[1])
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx)
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
        likelihood_mask[inds] = 0.0
        if self._mask_quasar_images_for_logL:
            likelihood_mask_imaging_weights = self.quasar_image_mask(
                likelihood_mask,
                x_image,
                y_image,
                self._image_data.shape
            )
            return likelihood_mask, likelihood_mask_imaging_weights
        else:
            return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.00541091,
                       'exposure_time': 1428.0,
                       'ra_at_xy_0': ra_at_xy_0,
                       'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': transform_pix2angle,
                       'image_data': self._image_data}
        return kwargs_data

    @property
    def coordinate_properties(self):

        deltaPix = 0.04
        window_size = 72 * deltaPix
        ra_at_xy_0 = 1.441213025
        dec_at_xy_0 = -1.43946221
        transform_pix2angle = np.array([[-4.00187709e-02, -1.49197263e-05],
       [-1.49162211e-05,  3.99999777e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_numerics(self):
        kwargs_numerics = {
            'supersampling_factor': int(self._supersample_factor),
            'supersampling_convolution': False,  # try with True
            'point_source_supersampling_factor': 1}
        return kwargs_numerics

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init / np.sum(self._psf_estimate_init),
                      'psf_error_map': self._psf_error_map_init,
                      'point_source_supersampling_factor': 1
                      }
        return kwargs_psf

class J0405_HST(_DESJ0405):

    def __init__(self, supersample_factor=1):

        x_image = np.array([ 0.693, -0.372,  0.349, -0.53 ])
        y_image = np.array([-0.29 , -0.615,  0.546,  0.407])
        horizontal_shift = 0.00
        vertical_shift = 0.007
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4  # m.a.s.
        magnifications = np.array([1.00, 0.70, 1.07, 1.28])
        flux_uncertainties = np.array([0.03] * 3)
        uncertainty_in_fluxes = False
        super(J0405_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                        flux_uncertainties, uncertainty_in_fluxes, supersample_factor)

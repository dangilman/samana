from samana.Data.data_base import ImagingDataBase
import numpy as np
from samana.Data.ImageData.j1251_f814W import psf_model, psf_error_map, image_data

class _J1251(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor=1,
                 mask_quasar_images_for_logL=True):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        z_lens = 0.4
        z_source = 0.8
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self._psf_estimate_init = psf_model
        self._psf_error_map_init = self.mask_psf_error_map(psf_error_map)
        self._image_data = image_data
        self._supersample_factor = supersample_factor
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(_J1251, self).__init__(z_lens, z_source,
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

        x_main = -0.0105
        y_main = -0.0094
        r_main = 0.3
        _xx, _yy = np.meshgrid(_x - x_main, _y - y_main)
        inds_main_deflector = np.where(np.sqrt(_xx ** 2 + _yy ** 2) < r_main / 2)
        likelihood_mask[inds_main_deflector] = 0.0

        if self._mask_quasar_images_for_logL:
            likelihood_mask_imaging_weights = self.quasar_image_mask(
                likelihood_mask,
                x_image,
                y_image,
                self._image_data.shape, radius_arcsec=0.15
            )
            return likelihood_mask, likelihood_mask_imaging_weights
        else:
            return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.0054732,
                       'exposure_time': 1428.0,
                       'ra_at_xy_0': ra_at_xy_0,
                       'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': transform_pix2angle,
                       'image_data': self._image_data}
        return kwargs_data

    @property
    def coordinate_properties(self):

        deltaPix = 0.04
        window_size = 3.6
        ra_at_xy_0 = 1.67922
        dec_at_xy_0 = -1.680470
        transform_pix2angle = np.array([[-3.99927081e-02,  1.12125733e-05],
       [ 1.12227424e-05,  3.99999831e-02]])
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
                      'psf_variance_map': self._psf_error_map_init,
                      'point_source_supersampling_factor': 1
                      }
        return kwargs_psf

class J1251_HST(_J1251):

    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([-1.0734222, 0.63920452, 0.70877884, 0.34943884])[[2,3,1,0]]
        y_image = np.array([0.32981853, 0.3459498, -0.24232032, -0.61744801])[[2,3,1,0]]
        horizontal_shift = 0.0
        vertical_shift = -0.015
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J1251_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False, supersample_factor=supersample_factor)


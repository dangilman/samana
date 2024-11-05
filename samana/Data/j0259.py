from samana.Data.data_base import ImagingDataBase
import numpy as np
from samana.Data.ImageData.j0259_f814W import psf_model, psf_error_map, image_data

class _J0259(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor, band,
                 mask_quasar_images_for_logL=True):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        z_lens = 0.91
        z_source = 2.16
        self.band = band
        if band == 'F814W':
            from samana.Data.ImageData.j0259_f814W import psf_model, psf_error_map, image_data
            self._background_rms = 0.0054476
            self._exp_time = 1428.0
        elif band == 'F475X':
            from samana.Data.ImageData.j0259_f475X import psf_model, psf_error_map, image_data
            self._background_rms = 0.0076711
            self._exp_time = 994.0
        else:
            raise Exception('only F814W and F475X available')

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
        super(_J0259, self).__init__(z_lens, z_source,
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
        kwargs_data = {'background_rms': self._background_rms,
                       'exposure_time': self._exp_time,
                       'ra_at_xy_0': ra_at_xy_0,
                       'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': transform_pix2angle,
                       'image_data': self._image_data}
        return kwargs_data

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

    @property
    def coordinate_properties(self):

        deltaPix = 0.04
        window_size = 2.88
        ra_at_xy_0 = 1.4405518
        dec_at_xy_0 = -1.4397599
        transform_pix2angle = np.array([[-4.00086717e-02, -6.65875035e-06],
       [-6.64879554e-06,  3.99999805e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

# class J0259_HST_F475X(_J0259):
#
#     def __init__(self, super_sample_factor=1):
#         """
#
#         :param image_position_uncertainties: list of astrometric uncertainties for each image
#         i.e. [0.003, 0.003, 0.003, 0.003]
#         :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
#         post-processing
#         :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
#         :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
#         """
#         x_image = np.array([ 0.67688171, -0.04485759, -0.79680975,  0.35393413])
#         y_image = np.array([-0.29873911, -0.68926224,  0.25911033,  0.57734639])
#         horizontal_shift = -0.01
#         vertical_shift = 0.005
#         x_image += horizontal_shift
#         y_image += vertical_shift
#         image_position_uncertainties = [0.005] * 4 # 5 arcsec
#         flux_uncertainties = None
#         magnifications = np.array([1.0] * 4)
#         uncertainty_in_fluxes= False
#         band = 'F475X'
#         super(J0259_HST_F475X, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
#                                               uncertainty_in_fluxes, super_sample_factor, band)

class J0259_HST_F814W(_J0259):

    def __init__(self, super_sample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        #x_image = np.array([ 0.67688171, -0.04485759, -0.79680975,  0.35393413])
        #y_image = np.array([-0.29873911, -0.68926224,  0.25911033,  0.57734639])
        x_image = np.array([ 0.67321505, -0.04478495, -0.80178495,  0.38521505])
        y_image = np.array([-0.29821417, -0.68721417,  0.26278583,  0.57578583])
        horizontal_shift = 0.0
        vertical_shift = 0.002
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        uncertainty_in_fluxes= False
        band = 'F814W'
        super(J0259_HST_F814W, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                              uncertainty_in_fluxes, super_sample_factor, band)


from samana.Data.data_base import ImagingDataBase
import numpy as np


class _J0248(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor, image_data_type,
                 mask_quasar_images_for_logL=True):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        #z_lens = 0.96
        z_lens = 0.5
        z_source = 2.44
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self._image_data_type = image_data_type
        if image_data_type == 'HST814W':
            from samana.Data.ImageData.j0248_f814W import psf_model, psf_error_map, image_data
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_error_map
            self._image_data = image_data
            self._psf_supersampling_factor = 1
            self._deltaPix = 0.04
            self._window_size = 2.96
            self._ra_at_xy_0 = 1.4794297
            self._dec_at_xy_0 = -1.48027077
            self._transform_pix2angle = np.array([[-3.99919120e-02,  7.32418926e-06],
                                            [ 7.33473592e-06,  3.99999834e-02]])
            self._background_rms = 0.005795
            self._exposure_time = 1428.0
            self._noise_map = None

        elif image_data_type == 'MIRI540W':
            from samana.Data.ImageData.j0248_MIRI540W import psf_model, image_data, noise_map
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = None
            self._image_data = image_data
            self._psf_supersampling_factor = 3
            self._deltaPix = 0.11086
            self._window_size = 3.8801
            self._ra_at_xy_0 = 1.47216860106189
            self._dec_at_xy_0 = 2.3172012476596215
            self._transform_pix2angle = np.array([[0.02414379, -0.10826771],
                                                  [-0.10826771, -0.02414379]])
            self._background_rms = None
            self._exposure_time = None
            self._noise_map = noise_map
        else:
            raise Exception('image data type must be either HST814W or MIRI540W')

        self._supersample_factor = supersample_factor

        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(_J0248, self).__init__(z_lens, z_source,
                                       kwargs_data_joint, x_image, y_image,
                                       magnifications, image_position_uncertainties, flux_uncertainties,
                                       uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
                                       likelihood_mask_imaging_weights)

    @property
    def coordinate_properties(self):

        deltaPix = self._deltaPix
        window_size = self._window_size
        ra_at_xy_0 = self._ra_at_xy_0
        dec_at_xy_0 = self._dec_at_xy_0
        transform_pix2angle = self._transform_pix2angle
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    def likelihood_masks(self, x_image, y_image):

        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        _x = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
        _y = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[1])
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx)
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
        likelihood_mask[inds] = 0.0

        if self._image_data_type == 'HST814W':
            x_main = -0.0239
            y_main = 0.060
            r_main = 0.4
            _xx, _yy = np.meshgrid(_x - x_main, _y - y_main)
            inds_main_deflector = np.where(np.sqrt(_xx ** 2 + _yy ** 2) < r_main / 2)
            likelihood_mask[inds_main_deflector] = 0.0

        if self._mask_quasar_images_for_logL:
            likelihood_mask_imaging_weights = self.quasar_image_mask(
                likelihood_mask,
                x_image,
                y_image,
                self._image_data.shape, radius_arcsec=0.3
            )
            inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2.5)
            likelihood_mask_imaging_weights[inds] = 0.0
            return likelihood_mask, likelihood_mask_imaging_weights
        else:
            return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': self._background_rms,
                       'exposure_time': self._exposure_time,
                       'ra_at_xy_0': ra_at_xy_0,
                       'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': transform_pix2angle,
                       'image_data': self._image_data,
                       'noise_map': self._noise_map}
        return kwargs_data

    @property
    def kwargs_numerics(self):
        kwargs_numerics = {
            'supersampling_factor': int(self._supersample_factor * max(1, self._psf_supersampling_factor)),
            'supersampling_convolution': False,  # try with True
            'point_source_supersampling_factor': self._psf_supersampling_factor}
        return kwargs_numerics

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init / np.sum(self._psf_estimate_init),
                      'psf_variance_map': self._psf_error_map_init,
                      'point_source_supersampling_factor': self._psf_supersampling_factor
                      }
        return kwargs_psf

class J0248_HST(_J0248):

    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        reorder = [3,2,1,0] # so that HST and MIRI image positions are in same order
        x_image = np.array([ 0.3827822 , -0.52452975, -0.66897894,  0.33092538])[reorder]
        y_image = np.array([ 0.62079534,  0.65874007, -0.17619568, -0.78839384])[reorder]
        horizontal_shift = 0.0
        vertical_shift = 0.0
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        uncertainty_in_fluxes = False
        magnifications = np.array([1.0] * 4)
        super(J0248_HST, self).__init__(x_image,
                                     y_image,
                                     magnifications,
                                     image_position_uncertainties,
                                     flux_uncertainties,
                                     uncertainty_in_fluxes,
                                     supersample_factor=supersample_factor,
                                     image_data_type='HST814W')

class J0248_MIRI(_J0248):

    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([0.34, -0.66019, -0.5158, 0.3901])
        y_image = np.array([-0.8, -0.18994, 0.646989, 0.60956])
        horizontal_shift = 0.005
        vertical_shift = -0.01
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        uncertainty_in_fluxes = False
        magnifications = np.array([1.0] * 4)
        image_data_type = 'MIRI540W'
        super(J0248_MIRI, self).__init__(x_image,
                                     y_image,
                                     magnifications,
                                     image_position_uncertainties,
                                     flux_uncertainties,
                                     uncertainty_in_fluxes,
                                     supersample_factor,
                                     image_data_type)


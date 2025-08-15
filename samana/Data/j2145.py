from samana.Data.data_base import ImagingDataBase
import numpy as np

class _J2145(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor, image_data_type,
                 mask_quasar_images_for_logL=True):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        z_lens = 0.50 # pm 0.14
        z_source = 1.56
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self._image_data_type = image_data_type
        if image_data_type == 'HST814W':
            from samana.Data.ImageData.j2145_F814W import image_data, psf_model, psf_error_map
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_error_map
            self._image_data = image_data
            self._psf_supersampling_factor = 1
            self._deltaPix = 0.03999
            self._window_size = 4.639
            self._ra_at_xy_0 = 2.3150230
            self._dec_at_xy_0 = -2.322383
            self._transform_pix2angle = np.array([[-3.99553555e-02,  4.11644862e-05],
                                            [ 4.11364255e-05,  3.99999640e-02]])
            self._background_rms = 0.006003
            self._exposure_time = 1428.0
            self._noise_map = None
        elif image_data_type == 'HST814W_WIDE':
            from samana.Data.ImageData.j2145_F814W import psf_model, psf_error_map
            from samana.Data.ImageData.j2145_F814W_wide import image_data
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_error_map
            self._image_data = image_data
            self._psf_supersampling_factor = 1
            self._deltaPix = 0.03999
            self._window_size = 13.999999
            self._ra_at_xy_0 = 6.98562714
            self._dec_at_xy_0 = -7.0068722
            self._transform_pix2angle = np.array([[-3.99571973e-02,  3.93279777e-05],
                                        [ 3.93029328e-05,  3.99999672e-02]])
            self._background_rms = 0.006003
            self._exposure_time = 1428.0
            self._noise_map = None
        elif image_data_type == 'MIRI560W':
            from samana.Data.ImageData.j2145_MIRI540W import psf_model, image_data, noise_map
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = None
            self._image_data = image_data
            self._psf_supersampling_factor = 3
            self._deltaPix = 0.11083407701179139
            self._window_size = 5.320035696565987
            self._ra_at_xy_0 = -1.6305016314724083
            self._dec_at_xy_0 = 3.3901112584633015
            self._transform_pix2angle = np.array([[0.1045961, -0.03665853],
                                                  [-0.03665853, -0.1045961]])
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
        super(_J2145, self).__init__(z_lens, z_source,
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
        if self._image_data_type == 'MIRI560W':
            maskpixes_vert = np.arange(0, 13)
            maskpixels_hor = np.arange(25, 35)
            xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
            xx, yy = xx.ravel(), yy.ravel()
            likelihood_mask[xx, yy] = 0.0
            maskpixes_vert = np.arange(33, 44)
            maskpixels_hor = np.arange(24, 34)
            xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
            xx, yy = xx.ravel(), yy.ravel()
            likelihood_mask[xx, yy] = 0.0
            maskpixes_vert = np.arange(22, 26)
            maskpixels_hor = np.arange(0, 21)
            xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
            xx, yy = xx.ravel(), yy.ravel()
            likelihood_mask[xx, yy] = 0.0
            maskpixes_vert = np.arange(16, 20)
            maskpixels_hor = np.arange(0, 11)
            xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
            xx, yy = xx.ravel(), yy.ravel()
            likelihood_mask[xx, yy] = 0.0

        else:
            pass

        if self._mask_quasar_images_for_logL:

            likelihood_mask_imaging_weights = self.quasar_image_mask(
                likelihood_mask,
                x_image,
                y_image,
                self._image_data.shape, radius_arcsec=0.45
            )
            inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2.3)
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

class J2145_HST(_J2145):
    # satellite positions relative to image 0
    sat1_x_wrt0 = 5.43
    sat1_y_wrt0 = 0.00
    sat2_x_wrt0 = -5.2
    sat2_y_wrt0 = 0.5
    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([0.60435097, 0.91743508, -0.46027406, -0.92551199])
        y_image = np.array([-0.40934611, 0.15995747, 0.90484046, -0.76745182])
        horizontal_shift = -0.18 - 0.005
        vertical_shift = -0.23 + 0.02
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        uncertainty_in_fluxes = False
        magnifications = np.array([1.0] * 4)
        image_data_type = 'HST814W'
        super(J2145_HST, self).__init__(x_image,
                                     y_image,
                                     magnifications,
                                     image_position_uncertainties,
                                     flux_uncertainties,
                                     uncertainty_in_fluxes,
                                     supersample_factor,
                                     image_data_type)

class J2145_HST_WIDEFIELD(_J2145):
    # satellite positions relative to image 0
    sat1_x_wrt0 = 5.43
    sat1_y_wrt0 = 0.00
    sat2_x_wrt0 = -5.2
    sat2_y_wrt0 = 0.5
    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([0.60435097, 0.91743508, -0.46027406, -0.92551199])
        y_image = np.array([-0.40934611, 0.15995747, 0.90484046, -0.76745182])
        horizontal_shift = -0.18 - 0.005
        vertical_shift = -0.23 + 0.02
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        uncertainty_in_fluxes = False
        magnifications = np.array([1.0] * 4)
        image_data_type = 'HST814W_WIDE'
        super(J2145_HST_WIDEFIELD, self).__init__(x_image,
                                     y_image,
                                     magnifications,
                                     image_position_uncertainties,
                                     flux_uncertainties,
                                     uncertainty_in_fluxes,
                                     supersample_factor,
                                     image_data_type)

class J2145_MIRI(_J2145):
    sat1_x_wrt0 = 5.43
    sat1_y_wrt0 = 0.00
    sat2_x_wrt0 = -5.2
    sat2_y_wrt0 = 0.5
    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """

        x_image = np.array([ 0.60435097,  0.91743508, -0.46027406, -0.92551199])
        y_image = np.array([-0.40934611,  0.15995747,  0.90484046, -0.76745182])
        horizontal_shift = -0.01
        vertical_shift = -0.01
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        uncertainty_in_fluxes = False
        magnifications = np.array([1.0] * 4)
        image_data_type = 'MIRI560W'
        super(J2145_MIRI, self).__init__(x_image,
                                     y_image,
                                     magnifications,
                                     image_position_uncertainties,
                                     flux_uncertainties,
                                     uncertainty_in_fluxes,
                                     supersample_factor,
                                     image_data_type)

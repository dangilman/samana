from samana.Data.data_base import ImagingDataBase
import numpy as np

class _J1042(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor=1,
                 mask_quasar_images_for_logL=True, band='814W'):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        z_lens = 0.59
        z_source = 2.5
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self._band = band
        if band == '814W':
            from samana.Data.ImageData.j1042_f814w import psf_model, psf_error_map, image_data
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_error_map
            self._image_data = image_data
            self._noise_map = None
            self._background_rms = 0.005579
            self._exposure_time = 1298.0
            self._deltaPix = 0.04
            self._window_size = 3.04
            self._ra_at_xy_0 = 1.519597
            self._dec_at_xy_0 = -1.52023
            self._transform_pix2angle = np.array([[-3.99956893e-02,  6.29091659e-06],
                        [ 6.30266385e-06,  3.99999824e-02]])
        elif band == '160W':
            from samana.Data.ImageData.j1042_F160W import image_data, psf_model, psf_error_map
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_error_map
            self._image_data = image_data
            self._background_rms = 0.0066673
            self._exposure_time = 1596.92626
            self._deltaPix = 0.06
            self._window_size = 3.83999
            self._ra_at_xy_0 = 2.460896
            self._dec_at_xy_0 = -1.147514
            self._transform_pix2angle = np.array([[-0.05638142, -0.02052159],
                                                  [-0.02052159, 0.05638142]])
        elif band == '125W':
            from samana.Data.ImageData.j1042_F125W import image_data, psf_model, psf_error_map
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_error_map
            self._image_data = image_data
            self._background_rms = 0.010539
            self._exposure_time = 896.935304
            self._deltaPix = 0.06
            self._window_size = 3.839999
            self._ra_at_xy_0 = 2.46089
            self._dec_at_xy_0 = -1.147514
            self._transform_pix2angle = np.array([[-0.05638142, -0.02052159],
                                                  [-0.02052159,  0.05638142]])
        else:
            raise Exception('band must be either 814W or 160W')
        self._supersample_factor = supersample_factor
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(_J1042, self).__init__(z_lens, z_source,
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

        if self._band == '814W':
            star_x = 0.05
            star_y = 0.87
            mask_radius = 0.175
            dr2 = (_xx - star_x) ** 2 + (_yy - star_y) ** 2
            inds = np.where(dr2 <= mask_radius ** 2)
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
    def coordinate_properties(self):

        deltaPix = self._deltaPix
        window_size = self._window_size
        ra_at_xy_0 = self._ra_at_xy_0
        dec_at_xy_0 = self._dec_at_xy_0
        transform_pix2angle = self._transform_pix2angle
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': self._background_rms,
                       'exposure_time': self._exposure_time,
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
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_error_map': self._psf_error_map_init}
        return kwargs_psf

class J1042_HST(_J1042):
    gx = 1.782
    gy = -0.317
    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([0.827, 0.68, 0.01, -0.757])
        y_image = np.array([0.10825, -0.45675, -0.80575, 0.65425])
        horizontal_shift = -0.005
        vertical_shift = 0.0
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J1042_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False,
                                        supersample_factor=supersample_factor,
                                        band='814W')

class J1042_HST_160W(_J1042):
    gx = 1.782
    gy = -0.317
    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([0.77927611, 0.63227611, -0.03772389, -0.80472389])
        y_image = np.array([0.12817213, -0.43682787, -0.78582787, 0.67417213])
        horizontal_shift = 0.02
        vertical_shift = -0.03
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J1042_HST_160W, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False,
                                             supersample_factor=supersample_factor,
                                             band='160W'
                                             )

class J1042_HST_125W(_J1042):
    gx = 1.782
    gy = -0.317
    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([ 0.78284624,  0.63584624, -0.03415376, -0.80115376])
        y_image = np.array([ 0.0939989, -0.4710011, -0.8200011,  0.6399989])
        horizontal_shift = 0.07
        vertical_shift = -0.045
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J1042_HST_125W, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False,
                                             supersample_factor=supersample_factor,
                                             band='125W'
                                             )


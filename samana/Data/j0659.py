from samana.Data.data_base import ImagingDataBase
import numpy as np

class _J0659MIRIImaging(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor=1):

        from samana.Data.ImageData.j0659_miri import image_data, psf_model, image_data_sigma
        z_lens = 0.77
        z_source = 3.1
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self._psf_estimate_init = psf_model
        scale_error_map = 0.1
        self._psf_error_map_init = self._psf_estimate_init * scale_error_map
        self._image_data = image_data
        self._image_data_noise_map = image_data_sigma
        self._supersample_factor = supersample_factor
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(_J0659MIRIImaging, self).__init__(z_lens, z_source,
                                     kwargs_data_joint, x_image, y_image,
                                     magnifications, image_position_uncertainties, flux_uncertainties,
                                     uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
                                     likelihood_mask_imaging_weights)

    def likelihood_masks(self, x_image, y_image):

        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        _x = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
        _y = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx)
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
        likelihood_mask[inds] = 0.0
        #
        # star_x = -0.45
        # star_y = 1.54
        # mask_radius = 0.275
        # dr2 = (_xx - star_x)**2 + (_yy - star_y)**2
        # inds = np.where(dr2 <= mask_radius**2)
        # likelihood_mask[inds] = 0.0
        return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'noise_map': self._image_data_noise_map,
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
        deltaPix = 0.11
        window_size = 3.96 * 2
        ra_at_xy_0 = 3.96
        dec_at_xy_0 = -3.96
        transform_pix2angle = np.array([[-deltaPix,  0.0],
       [0.0, deltaPix]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_psf(self):
        # npix = self._psf_estimate_init.shape[0]
        # psf_model_norm = np.sum(self._psf_estimate_init)/npix**2
        psf_model_norm = np.sum(self._psf_estimate_init)
        psf_model = self._psf_estimate_init / psf_model_norm
        psf_error_map = 0.01 * psf_model
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': psf_model,
                      # 'psf_error_map': psf_error_map * psf_model ** 2,
                      'point_source_supersampling_factor': 3}
        return kwargs_psf

class _J0659HSTImaging(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor=1):

        from samana.Data.ImageData.j0659_f814w import image_data, psf_model, psf_error_map
        z_lens = 0.77
        z_source = 3.1
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
        super(_J0659HSTImaging, self).__init__(z_lens, z_source,
                                     kwargs_data_joint, x_image, y_image,
                                     magnifications, image_position_uncertainties, flux_uncertainties,
                                     uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
                                     likelihood_mask_imaging_weights)

    def likelihood_masks(self, x_image, y_image):

        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        _x = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
        _y = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx)
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
        likelihood_mask[inds] = 0.0
        #
        # star_x = -0.45
        # star_y = 1.54
        # mask_radius = 0.275
        # dr2 = (_xx - star_x)**2 + (_yy - star_y)**2
        # inds = np.where(dr2 <= mask_radius**2)
        # likelihood_mask[inds] = 0.0
        return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.007518,
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
        window_size = 8.48
        ra_at_xy_0 = 4.238418
        dec_at_xy_0 = -4.240695
        transform_pix2angle = np.array([[-3.99916465e-02,  6.56928592e-06],
       [ 6.58148735e-06,  3.99999809e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_error_map': self._psf_error_map_init}
        return kwargs_psf

class J0659JWST_HSTImaging(_J0659HSTImaging):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([ 1.86868455, -2.79631545,  0.88968455,  1.95268455])
        y_image = np.array([-0.92780858, -1.26280858,  1.96419142,  0.97519142])
        # vertical_shift = 0.04
        # horizontal_shift = -0.022
        vertical_shift = 0.0
        horizontal_shift = 0.0
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J0659JWST_HSTImaging, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)

    @property
    def satellite_or_star_coords(self):
        return 0.48, 1.5

class J0659JWST_MIRIImaging(_J0659MIRIImaging):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([ 1.92868455, -2.63631545,  0.69968455,  1.84268455])
        y_image = np.array([-0.74780858, -1.50780858,  2.03419142,  1.13519142])
        # from fitting
        horizontal_shift = -0.02
        vertical_shift = 0.01
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J0659JWST_MIRIImaging, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)

    @property
    def satellite_or_star_coords(self):
        return 0.28, 1.55

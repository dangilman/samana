from samana.Data.data_base import ImagingDataBase
import numpy as np

class _PG1115(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, image_data, psf_model, psf_error_map,
                 image_likelihood_mask, supersample_factor=1.0,
                 mask_quasar_images_for_logL=True):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        z_lens = 0.31
        z_source = 1.72
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self._psf_estimate_init = psf_model
        self._psf_error_map_init = psf_error_map
        self._image_data = image_data
        self._supersample_factor = supersample_factor
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        if image_likelihood_mask is None:
            likelihood_mask, _ = self.likelihood_masks(None, None)
        else:
            likelihood_mask, _ = image_likelihood_mask, image_likelihood_mask
        if self._mask_quasar_images_for_logL:
            likelihood_mask_imaging_weights = self.quasar_image_mask(
                likelihood_mask,
                x_image,
                y_image,
                self._image_data.shape
            )
        else:
            likelihood_mask_imaging_weights = likelihood_mask
        super(_PG1115, self).__init__(z_lens, z_source,
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
        return likelihood_mask, likelihood_mask

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_error_map': self._psf_error_map_init}
        return kwargs_psf

class PG1115_JWST(_PG1115):

    def __init__(self, supersample_factor=1.0):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """

        from samana.Data.ImageData.pg1115_nircam import image_data
        from samana.Data.ImageData.pg1115_nircam import psf_model
        from samana.Data.ImageData.pg1115_nircam import psf_error_map

        theta = 0.14 * np.pi
        x_image = np.array([0.947, 1.096, - 0.722, - 0.381])
        y_image = np.array([-0.69, -0.232, -0.617, 1.344])
        y_image, x_image = self.rotate_to_image_data(x_image, y_image, theta)
        magnifications = [1.0] * 4
        image_position_uncertainties = [0.005]*4
        flux_uncertainties = [0.03] * 3  # percent uncertainty
        uncertainty_in_fluxes = False
        super(PG1115_JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                          flux_uncertainties,
                                          uncertainty_in_fluxes,
                                          image_data, psf_model, psf_error_map,
                                          image_likelihood_mask=None,
                                          supersample_factor=supersample_factor)

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.03075,
                       'exposure_time': 451.0,
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
        window_size = 150 * deltaPix
        ra_at_xy_0 = -2.17
        dec_at_xy_0 = -2.17
        transform_pix2angle = np.array([[0.031, 0.0],
                                        [0.0, 0.031]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @staticmethod
    def rotate_to_image_data(x, y, theta):
        return x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)

class PG1115_HST(_PG1115):

    def __init__(self, supersample_factor=1.0):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        from samana.Data.ImageData.pg1115_f160w import image_data, psf_error_map, psf_model, image_likelihood_mask

        x_image = np.array([0.947,  1.096, - 0.722, - 0.381])
        y_image = np.array([-0.69,  -0.232, -0.617,  1.344])
        magnifications = [1.0, 0.93, 0.16, 0.21]
        image_position_uncertainties = [0.005]*4
        flux_uncertainties = [0.06/0.93, 0.07/0.16, 0.04/0.21]  # percent uncertainty
        uncertainty_in_fluxes = False
        super(PG1115_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                         flux_uncertainties,
                                         uncertainty_in_fluxes,
                                         image_data, psf_model, psf_error_map,
                                          image_likelihood_mask,
                                          supersample_factor=supersample_factor)

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.0019075,
                       'exposure_time': 10878.9613,
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
        window_size = 120 * deltaPix
        ra_at_xy_0 = 2.999932662
        dec_at_xy_0 = -3.000034482
        transform_pix2angle = np.array([[-4.99994518e-02, 5.74056003e-07],
                                        [5.75227714e-07, 4.99999995e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size


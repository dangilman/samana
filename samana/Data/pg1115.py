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
        if self._mask_quasar_images_for_logL:
            likelihood_mask_imaging_weights = self.quasar_image_mask(
                image_likelihood_mask,
                x_image,
                y_image,
                self._image_data.shape,
                radius_arcsec=0.4
            )
        else:
            likelihood_mask_imaging_weights = image_likelihood_mask
        super(_PG1115, self).__init__(z_lens, z_source,
                                      kwargs_data_joint, x_image, y_image,
                                      magnifications, image_position_uncertainties, flux_uncertainties,
                                      uncertainty_in_fluxes, keep_flux_ratio_index,
                                      image_likelihood_mask,
                                      likelihood_mask_imaging_weights)

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
        self.band = 'HST160W'
        x_image = np.array([-0.75849943, -0.41470272, 1.06139943, 0.91180272])
        y_image = np.array([-0.61868201, 1.34351346, -0.23637735, -0.6884541])
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
        deltaPix = 0.05
        window_size = 120 * deltaPix
        ra_at_xy_0 = 2.999932662
        dec_at_xy_0 = -3.000034482
        transform_pix2angle = np.array([[-4.99994518e-02, 5.74056003e-07],
                                        [5.75227714e-07, 4.99999995e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

class PG1115_NIRCAM(_PG1115):

    def __init__(self, supersample_factor=1.0):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        from samana.Data.ImageData.pg1115_f115w import image_data, psf_error_map, psf_model
        self.band = 'NIRCAM115W'
        x_image = np.array([-0.74449943, -0.40070272, 1.07539943, 0.92580272])
        y_image = np.array([-0.60868201, 1.35351346, -0.22637735, -0.6784541])
        horizontal_shift = 0.00
        vertical_shift = -0.003
        x_image += horizontal_shift
        y_image += vertical_shift
        magnifications = [1.0, 0.93, 0.16, 0.21]
        image_position_uncertainties = [0.005]*4
        flux_uncertainties = [0.06/0.93, 0.07/0.16, 0.04/0.21]  # percent uncertainty
        uncertainty_in_fluxes = False
        self._image_data_nircam = image_data
        image_likelihood_mask = self.likelihood_masks(x_image, y_image)[0]

        inds_nan = (np.array([104, 105, 106, 106, 106, 107, 107]), np.array([78, 78, 62, 63, 64, 63, 64]))
        self._image_data_nircam[inds_nan] = 300.0 # bad pixels at the quasar image positions
        super(PG1115_NIRCAM, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                         flux_uncertainties,
                                         uncertainty_in_fluxes,
                                         self._image_data_nircam, psf_model, psf_error_map,
                                          image_likelihood_mask,
                                          supersample_factor=supersample_factor)

    def likelihood_masks(self, x_image, y_image):

        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        _x = np.linspace(-window_size / 2, window_size / 2, self._image_data_nircam.shape[0])
        _y = np.linspace(-window_size / 2, window_size / 2, self._image_data_nircam.shape[1])
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx)
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
        likelihood_mask[inds] = 0.0
        inds_nan = (np.array([104, 105, 106, 106, 106, 107, 107]), np.array([78, 78, 62, 63, 64, 63, 64]))
        likelihood_mask[inds_nan] = 0.0
        return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.0157,
                       'exposure_time': 1803.776,
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
        deltaPix = 0.03122
        window_size = 140 * deltaPix
        ra_at_xy_0 = -2.9037
        dec_at_xy_0 = -1.059812
        transform_pix2angle = np.array([[ 0.01317066,  0.02831084],
       [ 0.02831084, -0.01317066]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

class PG1115_HST_AstrometricOffsets(_PG1115):

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
        self.band = 'HST160W'
        x_image = np.array([-0.75849943, -0.41470272, 1.06139943, 0.91180272])
        y_image = np.array([-0.61868201, 1.34351346, -0.23637735, -0.6884541])
        magnifications = [1.0, 0.93, 0.16, 0.21]
        image_position_uncertainties = [0.005]*4
        flux_uncertainties = [0.06/0.93, 0.07/0.16, 0.04/0.21]  # percent uncertainty
        uncertainty_in_fluxes = False
        # delta_r =[-0.00215152  0.02729128 -0.00764127 -0.00760529]
        delta_x_image = np.array([-0.00192527, -0.01212327,  0.00747577,  0.00709375])
        delta_y_image = np.array([-0.00096041, -0.02445078, -0.00158174,  0.00274212])
        x_image += delta_x_image
        y_image += delta_y_image
        super(PG1115_HST_AstrometricOffsets, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                         flux_uncertainties,
                                         uncertainty_in_fluxes,
                                         image_data, psf_model, psf_error_map,
                                          image_likelihood_mask,
                                          supersample_factor=supersample_factor)

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
        deltaPix = 0.05
        window_size = 120 * deltaPix
        ra_at_xy_0 = 2.999932662
        dec_at_xy_0 = -3.000034482
        transform_pix2angle = np.array([[-4.99994518e-02, 5.74056003e-07],
                                        [5.75227714e-07, 4.99999995e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size


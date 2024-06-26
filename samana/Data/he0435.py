from samana.Data.data_base import ImagingDataBase
import numpy as np


class _HE0435(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor, image_data_filter):

        z_lens = 0.45
        z_source = 1.69
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self._filter = image_data_filter
        if self._filter == 'f814w':
            from samana.Data.ImageData.he0435_814w import image_data, psf_error_map, psf_model
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_error_map
            self._image_data = image_data
        elif self._filter == 'f555w':
            from samana.Data.ImageData.he0435_f555W import image_data as image_data_f555w
            from samana.Data.ImageData.he0435_f555W import psf_model as psf_model_f555w
            from samana.Data.ImageData.he0435_f555W import psf_error_map as psf_error_map_f555w
            self._psf_estimate_init = psf_model_f555w
            self._psf_error_map_init = psf_error_map_f555w
            self._image_data = image_data_f555w
        elif self._filter == 'jwst_nircam':
            from samana.Data.ImageData.he0435_nircam_dithered import image_data as image_data_nircam
            from samana.Data.ImageData.he0435_nircam_dithered import psf_model as psf_model_nircam
            from samana.Data.ImageData.he0435_nircam_dithered import psf_error_map as psf_error_map_nircam
            self._psf_estimate_init = psf_model_nircam
            self._psf_error_map_init = psf_error_map_nircam
            self._image_data = image_data_nircam
        else:
            raise Exception('filter '+str(image_data_filter)+' not recognized.')
        self._supersample_factor = supersample_factor
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(_HE0435, self).__init__(z_lens, z_source,
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
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        if self._filter == 'f814w':
            kwargs_data = {'background_rms': 0.01181,
                           'exposure_time': 1445.0,
                           'ra_at_xy_0': ra_at_xy_0,
                           'dec_at_xy_0': dec_at_xy_0,
                           'transform_pix2angle': transform_pix2angle,
                           'image_data': self._image_data}
        elif self._filter == 'f555w':
            kwargs_data = {'background_rms': 0.007946,
                           'exposure_time': 2030.0,
                           'ra_at_xy_0': ra_at_xy_0,
                           'dec_at_xy_0': dec_at_xy_0,
                           'transform_pix2angle': transform_pix2angle,
                           'image_data': self._image_data}
        elif self._filter == 'jwst_nircam':
            kwargs_data = {'background_rms': 0.01539,
                           'exposure_time': 1803.776,
                           'ra_at_xy_0': ra_at_xy_0,
                           'dec_at_xy_0': dec_at_xy_0,
                           'transform_pix2angle': transform_pix2angle,
                           'image_data': self._image_data}
        else:
            raise Exception('filter must be either f814w or f555w')
        return kwargs_data

    @property
    def kwargs_numerics(self):
        return {'supersampling_factor': int(self._supersample_factor),
                'supersampling_convolution': False}

    @property
    def coordinate_properties(self):
        if self._filter == 'f814w':
            deltaPix = 0.05
            window_size = 110 * deltaPix
            ra_at_xy_0 = 2.75069576
            dec_at_xy_0 = -2.74962
            transform_pix2angle = np.array([[-5.00058809e-02, -6.76934349e-06],
                                            [-6.75231528e-06,  4.99999709e-02]])
            return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size
        elif self._filter == 'f555w':
            deltaPix = 0.05
            window_size = 110 * deltaPix
            ra_at_xy_0 = 2.750695
            dec_at_xy_0 = -2.74962
            transform_pix2angle = np.array([[-5.00058808e-02, -6.76926675e-06],
                                            [-6.75236526e-06,  4.99999710e-02]])
            return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size
        elif self._filter == 'jwst_nircam':
            deltaPix = 0.031
            window_size = 160 * deltaPix
            ra_at_xy_0 = -2.48
            dec_at_xy_0 = -2.48
            transform_pix2angle = np.array([[0.031, 0.0],
                                            [0.0, 0.031]])
            return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size
        else:
            raise Exception('filter must be either f814w or f555w')

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_error_map': self._psf_error_map_init}
        return kwargs_psf

class HE0435_HST(_HE0435):

    def __init__(self, supersample_factor=1.0, image_data_filter='f814w'):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """

        # x_image = np.array([-1.272, -0.306,  1.152,  0.384])
        # y_image = np.array([-0.156,  1.092,  0.636, -1.026])
        x_image = np.array([-1.27134834, -0.30946454,  1.15665179,  0.32363394])
        y_image = np.array([-0.15831931,  1.09475682,  0.62941412, -1.06071974])
        # caluclated from image data
        # x_shifts = np.array([-0.01, 0., 0.025, -0.149])
        # y_shifts = np.array([0.12, 0.026, -0.08, -0.038])
        # x_image += x_shifts
        # y_image += y_shifts

        magnifications = [0.96, 0.976, 1.0, 0.65]
        image_position_uncertainties = [0.005] * 4
        flux_uncertainties = [0.05, 0.049, 0.048, 0.056]
        uncertainty_in_fluxes = True
        super(HE0435_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                          flux_uncertainties, uncertainty_in_fluxes=uncertainty_in_fluxes,
                                         supersample_factor=supersample_factor, image_data_filter=image_data_filter)

class HE0435_NIRCAM(_HE0435):

    def __init__(self, supersample_factor=1.0):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """

        image_data_filter = 'jwst_nircam'

        x_image = np.array([-0.41975347,  0.97504454,  0.88339521, -0.92230729])
        y_image = np.array([ 1.21060859,  0.63044535, -0.94073114, -0.53980543])

        magnifications = [0.96, 0.976, 1.0, 0.65]
        image_position_uncertainties = [0.005] * 4
        flux_uncertainties = [0.05, 0.049, 0.048, 0.056]
        uncertainty_in_fluxes = True
        super(HE0435_NIRCAM, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                                       flux_uncertainties, uncertainty_in_fluxes=uncertainty_in_fluxes,
                                                       supersample_factor=supersample_factor, image_data_filter=image_data_filter)

    def likelihood_masks(self, x_image=None, y_image=None):
        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        _x = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
        _y = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[1])
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx)
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2.1)
        likelihood_mask[inds] = 0.0

        rad = 1.8
        s = 1.7
        xp = 1.7 * s
        yp = 1.3 * s
        inds = np.where(np.sqrt((_xx - xp) ** 2 + (_yy - yp) ** 2) <= rad)
        likelihood_mask[inds] = 0.0

        rad = 2.0
        s = 1.
        xp = 3.9 * s
        yp = 0.0 * s
        inds = np.where(np.sqrt((_xx - xp) ** 2 + (_yy - yp) ** 2) <= rad)
        likelihood_mask[inds] = 0.0

        return likelihood_mask, likelihood_mask

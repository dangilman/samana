from samana.Data.data_base import ImagingDataBase
import numpy as np

class _J0607(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor, image_data_type,
                 mask_quasar_images_for_logL=True, use_noise_map=True):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        z_lens = 0.56
        z_source = 1.3
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        if image_data_type == 'HSTF160W':
            self._image_band = 'HSTF160W'
            from samana.Data.ImageData.j0607_f160w import (psf_model, psf_variance_map, image_data,
                                                           noise_map)
            background_rms = 0.0110187046
            exposure_time = 2396.928712
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = self.mask_psf_error_map(psf_variance_map)
            self._image_data = image_data
            self._psf_supersampling_factor = 1
            self._deltaPix = 0.0975059926
            self._window_size = 5.0703116152
            self._supersampling_convolution = True
            ra_shift = 0.0
            dec_shift = 0.0
            self._ra_at_xy_0 = -2.931770012795 + ra_shift
            self._dec_at_xy_0 = -1.941422655878 + dec_shift
            self._transform_pix2angle = np.array([[0.01942566, 0.09554572],
                                                  [0.09555749, -0.01942327]])
            if use_noise_map:
                # built from the exposure map: sqrt(rms**2 + max(data,0)/exposure_map)
                self._background_rms = None
                self._exposure_time = None
                self._noise_map = noise_map
            else:
                # let lenstronomy build the Poisson term from scalars instead
                self._background_rms = background_rms
                self._exposure_time = exposure_time
                self._noise_map = None

        elif image_data_type == 'MIRI560W':
            self._image_band = 'MIRI560W'
            from samana.Data.ImageData.j0607_MIRI560W import psf_model, image_data, noise_map
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = None
            self._image_data = image_data
            self._psf_supersampling_factor = 3
            self._supersampling_convolution = False
            self._deltaPix = 0.11090924511658345
            self._window_size = 4.658188294896505
            self._ra_at_xy_0 = -1.3761772259088505
            self._dec_at_xy_0 = -2.4651762507314614
            self._transform_pix2angle = np.array([[-0.03024997, 0.10670426],
                                                  [0.10670426, 0.03024997]])
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
        super(_J0607, self).__init__(z_lens, z_source,
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
        if self._image_band == 'HSTF160W':
            inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2.5)
            likelihood_mask[inds] = 0.0
            galx, galy = 0.0, 0.0
            mask_radius = 0.3
            dr2 = (_xx - galx) ** 2 + (_yy - galy) ** 2
            inds = np.where(dr2 <= mask_radius ** 2)
            likelihood_mask[inds] = 0.0
        else:
            inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
            likelihood_mask[inds] = 0.0
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
            'supersampling_convolution': self._supersampling_convolution,  # try with True
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

class J0607_HSTF160W(_J0607):
    satx = 1.186191
    saty = 0.206641
    def __init__(self, supersample_factor=1, use_noise_map=True):
        """
        Positions are centroids from the second F160W reduction, re-referenced to
        the lens galaxy and kept in the original image ordering.  They sit ~0.035"
        in x from the values used with the older reduction, so horizontal_shift is
        now 0.

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        :param use_noise_map: use the exposure-map-derived noise map (default) instead of
        scalar background_rms / exposure_time
        """
        x_image = np.array([0.54255323, 0.68011541, 0.21289095, -0.75544418])
        y_image = np.array([-0.75708658, 0.40902659, 0.80252704, -0.04544046])
        horizontal_shift = 0.0
        vertical_shift = 0.0
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        uncertainty_in_fluxes = False
        magnifications = np.array([1.0] * 4)
        image_data_type = 'HSTF160W'
        super(J0607_HSTF160W, self).__init__(x_image,
                                     y_image,
                                     magnifications,
                                     image_position_uncertainties,
                                     flux_uncertainties,
                                     uncertainty_in_fluxes,
                                     supersample_factor,
                                     image_data_type,
                                     use_noise_map=use_noise_map)

class J0607_MIRI(_J0607):
    satx = 1.22
    saty = 0.24
    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([0.50291994, 0.64119422, 0.1809345, -0.78504867])
        y_image = np.array([-0.7428513, 0.38855852, 0.78825482, -0.01396204])
        horizontal_shift = 0.005
        vertical_shift = 0.0
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        uncertainty_in_fluxes = False
        magnifications = np.array([1.0] * 4)
        image_data_type = 'MIRI560W'
        super(J0607_MIRI, self).__init__(x_image,
                                     y_image,
                                     magnifications,
                                     image_position_uncertainties,
                                     flux_uncertainties,
                                     uncertainty_in_fluxes,
                                     supersample_factor,
                                     image_data_type)

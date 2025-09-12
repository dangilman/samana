from samana.Data.data_base import ImagingDataBase
import numpy as np

class _RXJ1131(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor,
                 mask_quasar_images_for_logL=True,band=None):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        z_lens = 0.3
        z_source = 0.66
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self.band = band
        if band == 'f560w':
            from samana.Data.ImageData.rxj1131_MIRI560W import psf_model, image_data, noise_map
            self._deltaPix = 0.11081954147285053
            self._window_size = 8.533104693409491
            self._ra_at_xy_0 = -5.8277140575119954
            self._dec_at_xy_0 = -1.5635493951661728
            self._transform_pix2angle = np.array([[0.05537876, 0.09599043],
                                                  [0.09599043, -0.05537876]])
            self._background_rms = None
            self._exposure_time = None
            self._noise_map = noise_map
            self._psf_supersampling_factor = 3
            self._psf_error_map_init = None
            self._psf_estimate_init = psf_model
            self._psf_estimate_init /= np.sum(self._psf_estimate_init)

        elif band == 'f1280w':
            from samana.Data.ImageData.rxj1131_MIRI1280W import psf_model, noise_map, image_data
            self._deltaPix = 0.11081954153620512
            self._window_size = 8.533104698287795
            self._ra_at_xy_0 = -5.827706197355736
            self._dec_at_xy_0 = -1.5635787082007524
            self._transform_pix2angle = np.array([[0.05537828, 0.09599071],
                                                  [0.09599071, -0.05537828]])
            self._background_rms = None
            self._exposure_time = None
            self._noise_map = noise_map
            self._psf_supersampling_factor = 3
            self._psf_error_map_init = None
            self._psf_estimate_init = psf_model
            self._psf_estimate_init /= np.sum(self._psf_estimate_init)

        else:
            from samana.Data.ImageData.rxj1131_f814W import psf_model, psf_error_map, image_data
            self._deltaPix = 0.04999
            self._window_size = 8.399346
            self._ra_at_xy_0 = -4.89782
            self._dec_at_xy_0 = -1.840399
            self._transform_pix2angle = np.array([[ 0.02065827,  0.04552853],
                                                [ 0.04552853, -0.02065827]])
            self._background_rms = 0.0075095
            self._exposure_time = 1980.0
            self._noise_map = None
            self._psf_supersampling_factor = 1
            self._psf_estimate_init = psf_model
            self._psf_estimate_init /= np.sum(self._psf_estimate_init)
            self._psf_error_map_init = self.mask_psf_error_map(psf_error_map)

        self._image_data = image_data
        self._supersample_factor = supersample_factor
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(_RXJ1131, self).__init__(z_lens, z_source,
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

        x_main = 0.02
        y_main = -0.44
        r_main = 0.5
        _xx, _yy = np.meshgrid(_x - x_main, _y - y_main)
        inds_main_deflector = np.where(np.sqrt(_xx ** 2 + _yy ** 2) < r_main / 2)
        likelihood_mask[inds_main_deflector] = 0.0

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
                       'exposure_time': self._exposure_time,
                       'ra_at_xy_0': ra_at_xy_0,
                       'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': transform_pix2angle,
                       'image_data': self._image_data,
                       'noise_map': self._noise_map}
        return kwargs_data

    @property
    def coordinate_properties(self):

        deltaPix = self._deltaPix
        window_size = self._window_size
        ra_at_xy_0 = self._ra_at_xy_0
        dec_at_xy_0 = self._dec_at_xy_0
        transform_pix2angle = self._transform_pix2angle
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

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
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_variance_map': self._psf_error_map_init,
                      'point_source_supersampling_factor': 1
                      }
        return kwargs_psf

class RXJ1131_HST(_RXJ1131):
    g2x = -0.328
    g2y = 0.700
    def __init__(self, super_sample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([1.622, 1.6567, 1.03, -1.4934])
        y_image = np.array([-0.45, 0.737, -1.5646, 0.4301])
        horizontal_shift = -0.02
        vertical_shift = 0.0
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        uncertainty_in_fluxes= False
        super(RXJ1131_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                        uncertainty_in_fluxes, super_sample_factor,band='hst814w')

class RXJ1131_MIRI_f560w(_RXJ1131):
    g2x = 0.062
    g2y = 0.549
    def __init__(self, super_sample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([2.012, 2.0467, 1.42, -1.1034])
        y_image = np.array([-0.6, 0.587, -1.7146, 0.2801])
        horizontal_shift = -0.0
        vertical_shift = 0.02
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        uncertainty_in_fluxes= False
        super(RXJ1131_MIRI_f560w, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                        uncertainty_in_fluxes, super_sample_factor, band='f560w')


class RXJ1131_MIRI_f1280w(_RXJ1131):
    g2x = 0.062
    g2y = 0.55
    def __init__(self, super_sample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([2.012, 2.0467, 1.42, -1.1034])
        y_image = np.array([-0.6, 0.587, -1.7146, 0.2801])
        horizontal_shift = -0.0
        vertical_shift = 0.02
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4  # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        uncertainty_in_fluxes = False
        super(RXJ1131_MIRI_f1280w, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                                 flux_uncertainties,
                                                 uncertainty_in_fluxes, super_sample_factor,band='f1280w')


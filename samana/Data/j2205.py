from samana.Data.data_base import ImagingDataBase
import numpy as np
from samana.Data.ImageData.j2205_f814W import psf_model, psf_error_map, image_data

class _J2205(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, image_data_band, supersample_factor=1,
                 mask_quasar_images_for_logL=True):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        z_lens = 0.63 # measured by Ken Wong unpublished
        z_source = 1.85
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        self._image_data_band = image_data_band
        if image_data_band == '814W':
            from samana.Data.ImageData.j2205_f814W import psf_model, psf_error_map, image_data
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_error_map
            self._image_data = image_data
            self._psf_supersampling_factor = 1
            self._deltaPix = 0.04
            self._window_size = 2.8
            self._ra_at_xy_0 = 1.40133
            self._dec_at_xy_0 = -1.39939
            self._transform_pix2angle = np.array([[-4.00207636e-02, -1.73841613e-05],
                                                  [-1.73835284e-05,  3.99999772e-02]])
            self._background_rms = 0.006125
            self._exposure_time = 1428.0
            self._noise_map = None

        elif image_data_band == 'MIRI540W':
            from samana.Data.ImageData.j2205_MIRI540W import psf_model, image_data, noise_map
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = None
            self._image_data = image_data
            self._psf_supersampling_factor = 3
            self._deltaPix = 0.11092269738466404
            self._window_size = 3.549526316309249
            self._ra_at_xy_0 = 1.977689863103727
            self._dec_at_xy_0 = 1.545416233428675
            self._transform_pix2angle = np.array([[-0.01350855, -0.11009707],
                                                  [-0.11009707, 0.01350855]])
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
        super(_J2205, self).__init__(z_lens, z_source,
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
        if self._image_data_band == '814W':
            blobx, bloby = -1.122, 0.194
            likelihood_mask = self.quasar_image_mask(
                likelihood_mask,
                [blobx],
                [bloby],
                self._image_data.shape, radius_arcsec=0.1
            )
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
                      'kernel_point_source': self._psf_estimate_init / np.sum(self._psf_estimate_init),
                      'psf_variance_map': self._psf_error_map_init,
                      'point_source_supersampling_factor': self._psf_supersampling_factor
                      }
        return kwargs_psf

class J2205_MIRI(_J2205):
    band = 'MIRI'
    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([0.98748892, -0.21109039, -0.65488568, -0.36151285])
        y_image = np.array([0.14718422, -0.57865477, 0.02586794, 0.58560261])
        horizontal_shift = -0.005
        vertical_shift = 0.012
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J2205_MIRI, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                        image_data_band='MIRI540W',
                                         uncertainty_in_fluxes=False,
                                         supersample_factor=supersample_factor,
                                         )


class J2205_HST(_J2205):
    band = 'HST'
    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([0.97748892, -0.22109039, -0.66488568, -0.37151285])
        y_image = np.array([0.15218422, -0.57365477, 0.03086794, 0.59060261])
        horizontal_shift = -0.066
        vertical_shift = -0.036
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J2205_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties,
                                        flux_uncertainties, image_data_band='814W',
                                        uncertainty_in_fluxes=False, supersample_factor=supersample_factor)


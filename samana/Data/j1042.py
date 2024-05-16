from samana.Data.data_base import ImagingDataBase
import numpy as np
from samana.Data.ImageData.j1042_f814w import psf_model, psf_error_map, image_data

class _J1042(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor=1):

        z_lens = 0.59 #0.59 in schmidt
        z_source = 2.5
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
        super(_J1042, self).__init__(z_lens, z_source,
                                       kwargs_data_joint, x_image, y_image,
                                       magnifications, image_position_uncertainties, flux_uncertainties,
                                       uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
                                       likelihood_mask_imaging_weights)

    def likelihood_masks(self, x_image, y_image):

        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        _x = np.linspace(-window_size / 2, window_size / 2, image_data.shape[0])
        _y = np.linspace(-window_size / 2, window_size / 2, image_data.shape[0])
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx)
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
        likelihood_mask[inds] = 0.0

        star_x = 0.05
        star_y = 0.87
        mask_radius = 0.175
        dr2 = (_xx - star_x) ** 2 + (_yy - star_y) ** 2
        inds = np.where(dr2 <= mask_radius ** 2)
        likelihood_mask[inds] = 0.0

        return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.005579,
                       'exposure_time': 1298.0,
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
        window_size = 3.04
        ra_at_xy_0 = 1.519597
        dec_at_xy_0 = -1.52023
        transform_pix2angle = np.array([[-3.99956893e-02,  6.29091659e-06],
       [ 6.30266385e-06,  3.99999824e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_error_map': self._psf_error_map_init}
        return kwargs_psf

class J1042JWST(_J1042):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([-0.76930023,  0.81439063, 0.66778075, -0.00159419])
        y_image = np.array([ 0.65240278,  0.10628356, -0.45875266, -0.80610109])
        horizontal_shift = -0.01
        vertical_shift = 0.01
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J1042JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)

# data = J1042JWST()
# import matplotlib.pyplot as plt
# mask = data.likelihood_masks(0.0, 0.0)[0]
# satellite_x = 1.6553
# satellite_y = -0.3
# plt.imshow(np.log10(image_data*mask), origin='lower', extent=[-1.52, 1.52, -1.52, 1.52])
# plt.scatter(-satellite_x, satellite_y)
# plt.scatter(-data.x_image, data.y_image, marker='+')
# plt.show()

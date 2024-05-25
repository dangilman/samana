from samana.Data.data_base import ImagingDataBase
import numpy as np
from samana.Data.ImageData.j0248_f814W import psf_model, psf_error_map, image_data
from samana.data_util import mask_quasar_images
from lenstronomy.Data.coord_transforms import Coordinates
from copy import deepcopy


class _J0248(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor=1):

        z_lens = 0.5
        z_source = 2.44
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
        super(_J0248, self).__init__(z_lens, z_source,
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
        # coords = Coordinates(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
        # ra_grid, dec_grid = coords.coordinate_grid(*_xx.shape)
        # mask_radius_arcsec = 0.3
        # likelihood_mask_imaging_weights = mask_quasar_images(deepcopy(likelihood_mask),
        #                                                      x_image, y_image,
        #                                                      ra_grid,
        #                                                      dec_grid,
        #                                                      mask_radius_arcsec)
        return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.005795,
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
        window_size = 2.96
        ra_at_xy_0 = 1.4794297
        dec_at_xy_0 = -1.48027077
        transform_pix2angle = np.array([[-3.99919120e-02,  7.32418926e-06],
       [ 7.33473592e-06,  3.99999834e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_error_map': self._psf_error_map_init}
        return kwargs_psf

class J0248_JWST(_J0248):

    def __init__(self):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([ 0.3827822 , -0.52452975, -0.66897894,  0.33092538])
        y_image = np.array([ 0.62079534,  0.65874007, -0.17619568, -0.78839384])
        horizontal_shift = 0.0
        vertical_shift = 0.0
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(J0248_JWST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False)


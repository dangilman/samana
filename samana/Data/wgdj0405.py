import matplotlib.pyplot as plt
from samana.Data.data_base import ImagingDataBase
from samana.data_util import mask_quasar_images
from copy import deepcopy
import numpy as np
from samana.Data.ImageData.wgdj0405_814w import image_data, psf_error_map, psf_model
from lenstronomy.Data.coord_transforms import Coordinates


class _WGDJ0405(ImagingDataBase):

    def __init__(self, image_position_uncertainties, flux_uncertainties, magnifications,
                 uncertainty_in_fluxes, supersample_factor):

        z_lens = 0.3
        z_source = 1.7

        x_image_offset = -0.01
        y_image_offset = -0.05
        # astrometric data
        x_image = np.array([0.708, -0.358, 0.363, -0.515]) + x_image_offset
        y_image = np.array([-0.244, -0.567, 0.592, 0.454]) + y_image_offset
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
        super(_WGDJ0405, self).__init__(z_lens, z_source,
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
        coords = Coordinates(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
        ra_grid, dec_grid = coords.coordinate_grid(*_xx.shape)
        mask_radius_arcsec = 0.25
        likelihood_mask_imaging_weights = mask_quasar_images(deepcopy(likelihood_mask), x_image, y_image, ra_grid, dec_grid,
                                                             mask_radius_arcsec)
        return likelihood_mask, likelihood_mask_imaging_weights

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': 0.00541091,
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
        window_size = 72 * deltaPix
        ra_at_xy_0 = 1.441213
        dec_at_xy_0 = -1.4394622
        transform_pix2angle = np.array([[-4.00187502e-02, -1.48942746e-05],
                                        [-1.48908009e-05, 3.99999777e-02]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_psf(self):
        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': self._psf_estimate_init,
                      'psf_error_map': self._psf_error_map_init}
        return kwargs_psf

class WGDJ0405_HST(_WGDJ0405):

    def __init__(self, supersample_factor=1):

        image_position_uncertainty = [0.005] * 4 # m.a.s.
        normalized_magnifications = np.array([0.8, 0.52, 1.0, 0.94])
        flux_uncertainties = np.array([0.04, 0.04/0.65, 0.03/1.25, 0.04/1.17])
        uncertainty_in_fluxes = True
        super(WGDJ0405_HST, self).__init__(image_position_uncertainty, flux_uncertainties, normalized_magnifications,
                                                       uncertainty_in_fluxes, supersample_factor)

class WGDJ0405_JWST(_WGDJ0405):

    def __init__(self, supersample_factor=1):

        image_position_uncertainty = [0.005] * 4  # m.a.s.
        normalized_magnifications = np.array([1.00, 0.70, 1.07, 1.28])
        flux_uncertainties = np.array([0.03] * 3)
        uncertainty_in_fluxes = False
        super(WGDJ0405_JWST, self).__init__(image_position_uncertainty, flux_uncertainties,
                                                       normalized_magnifications,
                                                       uncertainty_in_fluxes, supersample_factor)

# xc = np.array([1.0656, 0.0026, 0.7222, -0.1562]) - 0.35
# yc = np.array([0.3204, -0.0017, 1.1589, 1.0206]) - 0.6
# fc = np.array([1.0, 0.508, 0.920, 0.658])
#
# lens = WGDJ0405_HST()
# flux_ratios = np.round(lens.magnifications/lens.magnifications[0],2)
#
# colors = ['k', 'r','g','m']
# for i in range(0, 4):
#     plt.scatter(lens.x_image[i], lens.y_image[i],color=colors[i])
#     plt.scatter(xc[i], yc[i], color=colors[i],marker='+')
#     plt.annotate(str(flux_ratios[i]),
#                  xy=(lens.x_image[i], lens.y_image[i]),color=colors[i])
#
# plt.show()

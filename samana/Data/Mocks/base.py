from samana.Data.data_base import ImagingDataBase
import numpy as np
from lenstronomy.Data.coord_transforms import Coordinates
from samana.data_util import mask_quasar_images
from copy import deepcopy

class MockBase(ImagingDataBase):

    def __init__(self, z_lens, z_source, x_image, y_image, magnifications,
                 astrometric_uncertainties, flux_ratio_uncertainties, image_data, super_sample_factor):
        self._super_sample_factor = super_sample_factor
        # here we specify whether measurement uncertainties are quoted for flux ratios or fluxes
        uncertainty_in_fluxes = False
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        background_rms = 0.006
        exposure_time = 1428.0
        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        kwargs_data = {'background_rms': background_rms,
                       'exposure_time': exposure_time,
                       'ra_at_xy_0': ra_at_xy_0,
                       'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': transform_pix2angle,
                       'image_data': image_data}
        self._kwargs_data = kwargs_data
        image_band = [kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image, image_data)
        super(MockBase, self).__init__(z_lens, z_source,
                                        kwargs_data_joint, np.array(x_image), np.array(y_image),
                                        np.array(magnifications), astrometric_uncertainties, flux_ratio_uncertainties,
                                        uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
                                        likelihood_mask_imaging_weights)

    def perturb_flux_measurements(self, measurement_uncertainties):
        """

        :param measurement_uncertainties:
        :param uncertainty_in_flux_ratios:
        :return:
        """
        magnifications = []
        for i in range(0, len(self.magnifications)):
            dm = self.magnifications[i] * measurement_uncertainties[i]
            magnifications.append(self.magnifications[i] + np.random.normal(0.0, dm))
        self.magnifications = np.round(magnifications, 4)

    def likelihood_masks(self, x_image, y_image, image_data):

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
    def kwargs_numerics(self):
        return {'supersampling_factor': int(self._super_sample_factor),
                'supersampling_convolution': False}

    @property
    def coordinate_properties(self):

        window_size = 3.5
        deltaPix = 0.05
        ra_at_xy_0 = -1.725
        dec_at_xy_0 = -1.725
        transform_pix2angle = np.array([[0.05, 0.], [0., 0.05]])
        return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size

    @property
    def kwargs_psf(self):
        fwhm = 0.1
        deltaPix = self.coordinate_properties[0]
        kwargs_psf = {'psf_type': 'GAUSSIAN',
                      'fwhm': fwhm,
                      'pixel_size': deltaPix,
                      'truncation': 5}
        return kwargs_psf

    @property
    def kwargs_data(self):
        return self._kwargs_data

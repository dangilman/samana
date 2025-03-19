import numpy as np
from lenstronomy.Data.coord_transforms import Coordinates
from samana.data_util import mask_quasar_images
from copy import deepcopy

class ImagingDataBase(object):

    def __init__(self, zlens, zsource, kwargs_data_joint, x_image, y_image,
                 magnifications, image_position_uncertainties, flux_uncertainty,
                 uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask, likelihood_mask_imaging_weights):
        """

        :param zlens: the main deflector redshift
        :param zsource: source redshift
        :param kwargs_data_joint: keyword argument relevant for the data; see docs in lenstronomy
        :param x_image: quasar image positions (x)
        :param y_image: quasar image positions (y
        :param magnifications: image magnifications (can be normalized to 1 being the brightnest)
        :param image_position_uncertainties: astrometric uncertainties; must be same size as x_image
        :param flux_uncertainty: uncertainty in the flux (array of length x_image) or flux ratio (array of length x_image-1)
        :param uncertainty_in_fluxes: bool; whether the uncertainties are quoted for measured fluxes or flux ratios
        :param keep_flux_ratio_index: indexes of which flux ratios to store; default is all of them [0, 1, 2]
        :param likelihood_mask: mask used to compute the likelihood of the imaging data during the lens modeling
        :param likelihood_mask_imaging_weights: mask used to compute the log-likelihood of the imaging data after lens
        modeling
        """
        self._z_lens = zlens
        self.z_source = zsource
        self._kwargs_data_joint = kwargs_data_joint
        self._x_image_init = np.array(x_image)
        self._y_image_init = np.array(y_image)
        self._x = np.array(x_image)
        self._y = np.array(y_image)
        self.magnifications = np.array(magnifications)
        self.image_position_uncertainty = image_position_uncertainties
        if len(self.image_position_uncertainty) != len(self._x_image_init):
            raise Exception('image position uncertainties must have the same shape as point source arrays')
        self.flux_uncertainty = flux_uncertainty
        self.uncertainty_in_fluxes = uncertainty_in_fluxes
        self.keep_flux_ratio_index = keep_flux_ratio_index
        self._likelihood_mask = likelihood_mask
        self._likelihood_mask_imaging_weights = likelihood_mask_imaging_weights
        self.mask_quasar_image_for_reconstruction(False)

    def mask_psf_error_map(self, psf_error_map):
        """
        Set the psf error map to 0 far from the center
        :param psf_error_map:
        :return:
        """
        size = psf_error_map.shape[0]
        r = np.linspace(-size/2, size/2, size)
        xx, yy = np.meshgrid(r, r)
        rr = np.sqrt(xx ** 2 + yy ** 2)
        inds = np.where(rr > size/3)
        psf_error_map[inds] = 0
        return psf_error_map

    @property
    def redshift_sampling(self):
        return False

    @property
    def z_lens(self):
        return self._z_lens

    def perturb_image_positions(self, delta_x_image=None, delta_y_image=None):
        if delta_x_image is None:
            delta_x_image = np.random.normal(0.0, self.image_position_uncertainty)
        if delta_y_image is None:
            delta_y_image = np.random.normal(0.0, self.image_position_uncertainty)
        self._x = self._x_image_init + delta_x_image
        self._y = self._y_image_init + delta_y_image
        return delta_x_image, delta_y_image

    def quasar_image_mask(self, likelihood_mask,
                          x_image,
                          y_image,
                          image_data_shape,
                          radius_arcsec=0.2):
        """

        :param likelihood_mask:
        :param x_image:
        :param y_image:
        :param image_data_shape:
        :param radius_arcsec:
        :return:
        """
        coords = self.coordinate_system
        ra_grid, dec_grid = coords.coordinate_grid(*image_data_shape)
        likelihood_mask_imaging_weights = mask_quasar_images(deepcopy(likelihood_mask),
                                                             x_image,
                                                             y_image,
                                                             ra_grid,
                                                             dec_grid,
                                                             radius_arcsec)
        return likelihood_mask_imaging_weights

    def mask_quasar_image_for_reconstruction(self, value):
        self._mask_quasar_images_for_reconstruction = value

    @property
    def likelihood_mask(self):
        if self._mask_quasar_images_for_reconstruction is True:
            return self.likelihood_mask_imaging_weights
        else:
            return self._likelihood_mask

    @property
    def likelihood_mask_imaging_weights(self):
        return self._likelihood_mask_imaging_weights

    @property
    def coordinate_system(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        coords = Coordinates(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
        return coords

    @property
    def x_image(self):
        return self._x

    @property
    def y_image(self):
        return self._y

    @property
    def coordinate_properties(self):
        raise Exception('must define a coordinate_properties property in the data class')

    @property
    def kwargs_data_joint(self):
        return self._kwargs_data_joint

    @property
    def kwargs_psf(self):
        raise Exception('must define a kwargs_psf property in the data class')


class QuadNoImageDataBase(ImagingDataBase):
    """
    Base class for a lens system with no available imaging data
    """

    def __init__(self, zlens, zsource, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainty,
                 uncertainty_in_fluxes, keep_flux_ratio_index):
        """

        :param zlens:
        :param zsource:
        :param x_image:
        :param y_image:
        :param magnifications:
        :param image_position_uncertainties:
        :param flux_uncertainty:
        :param uncertainty_in_fluxes:
        :param keep_flux_ratio_index:
        """
        # this will be a placeholder until imaging data is available
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(QuadNoImageDataBase, self).__init__(zlens, zsource, kwargs_data_joint, x_image, y_image,
                 magnifications, image_position_uncertainties, flux_uncertainty,
                 uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask, likelihood_mask_imaging_weights)

    @property
    def kwargs_data(self):
        deltapix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        npixels = int(window_size / deltapix)
        image_data = np.zeros(shape=(npixels, npixels))
        kwargs_data = {'background_rms': 0.00541091,
                       'exposure_time': 1428.0,
                       'ra_at_xy_0': ra_at_xy_0,
                       'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': transform_pix2angle,
                       'image_data': image_data}
        return kwargs_data

    @property
    def kwargs_numerics(self):
        return {'supersampling_factor': 1,
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

    def likelihood_masks(self, x_image, y_image):

        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        npixels = int(window_size / deltaPix)
        _x = np.linspace(-window_size / 2, window_size / 2, npixels)
        _y = np.linspace(-window_size / 2, window_size / 2, npixels)
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx)
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
        likelihood_mask[inds] = 0.0
        return likelihood_mask, likelihood_mask

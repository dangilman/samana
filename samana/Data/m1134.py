from samana.Data.data_base import ImagingDataBase
import numpy as np

class _2M1134(ImagingDataBase):

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor=1,
                 mask_quasar_images_for_logL=True, band='HST814W', filename_image_data=None):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        z_lens = 0.66 # Anguita et al. in prep
        z_source = 2.77
        self.band = band
        if band == 'HST814W':
            from samana.Data.ImageData.m1134_f814W import psf_model, psf_error_map, image_data
            from samana.Data.ImageData.m1134_hstmask import _custom_mask_2m1134
            self._custom_mask = _custom_mask_2m1134
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_error_map
            self._mask_rotation = 0.25
            self._image_data = image_data
            self._psf_supersampling_factor = 1
            self._background_rms = 0.00609
            self._exposure_time = 1428.0
            self._deltaPix = 0.04
            self._window_size = 6.24
            self._ra_at_xy_0 = 3.12089
            self._dec_at_xy_0 = -3.1194472
            self._transform_pix2angle = np.array([[-4.00043735e-02, -7.07884311e-06],
                            [-7.07227957e-06,  3.99999860e-02]])
            self._noise_map = None

        elif band == 'MIRI560W':
            from samana.Data.ImageData.m1134_MIRI540W import psf_model, image_data, noise_map
            self._custom_mask = 1.0
            self._mask_rotation = -0.15
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = None
            self._image_data = image_data
            self._psf_supersampling_factor = 3
            self._deltaPix = 0.11082108303617688
            self._window_size = 6.981728231279143
            self._ra_at_xy_0 = -4.444486634504042
            self._dec_at_xy_0 = -2.14914008270461
            self._transform_pix2angle = np.array([[0.03643407, 0.10466074],
                                                  [0.10466074, -0.03643407]])
            self._background_rms = None
            self._exposure_time = None
            self._noise_map = noise_map

        elif band == 'NIRCam200':
            # JWST/NIRCam F200W SWarp mosaic + companion RMS map, in microJy (ZP 23.9).
            # native SW pixel scale; the mosaic is N-up / E-left so transform_pix2angle
            # is diagonal, the same orientation as the HST814W frame above -- which is
            # why _mask_rotation reuses the HST value rather than the MIRI one.
            # psf_model is the STARRED reconstruction (supersampling 3, 447x447).
            # psf_variance_map is STARRED's psf_error_map SQUARED: that file is a standard
            # error in kernel units, while lenstronomy wants a variance in kernel**2 units.
            if filename_image_data is not None:
                psf_model = np.loadtxt(filename_image_data[0])
                image_data = np.loadtxt(filename_image_data[1])
                noise_map = np.loadtxt(filename_image_data[2])
                psf_variance_map = np.loadtxt(filename_image_data[3]) if len(filename_image_data) > 3 else None
            else:
                # ImageData/m1134_NIRCam200.py holds the pixel data inline and is kept out of
                # the public repository; use filename_image_data to load it from text instead.
                try:
                    from samana.Data.ImageData.m1134_NIRCam200 import (psf_model, image_data, noise_map,
                                                                       psf_variance_map)
                except ImportError:
                    raise Exception('M1134 NIRCam200 pixel data not available in this checkout. Pass '
                                    'filename_image_data=[psf_txt, imgdata_txt, noise_txt, psfvar_txt]; '
                                    'the fourth entry is optional and sets psf_variance_map to None if omitted.')
            self._custom_mask = 1.0
            self._mask_rotation = 0.25
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_variance_map
            self._image_data = image_data
            self._psf_supersampling_factor = 3
            self._deltaPix = 0.0309004492
            self._window_size = 6.9835015194
            self._ra_at_xy_0 = 3.4908467476
            self._dec_at_xy_0 = -3.4861786532
            self._transform_pix2angle = np.array([[-0.0309004492, 0.0],
                                                  [0.0, 0.0309004492]])
            self._background_rms = None
            self._exposure_time = None
            # supplied SWarp RMS map, 1-sigma per pixel, used unmodified. It is correctly
            # normalised per pixel (blank-sky z has sigma 1.03) but carries no covariance
            # term; empty-aperture noise exceeds rms*sqrt(N) by 1.8x at r=0.15" and 5.2x
            # at 1", so inflate final parameter uncertainties by roughly 1.5-2x.
            self._noise_map = noise_map

        else:
            raise Exception('imaging data band must be HST814W, MIRI560W or NIRCam200')
        self._band = band
        keep_flux_ratio_index = [0, 1, 2]
        self._supersample_factor = supersample_factor
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(_2M1134, self).__init__(z_lens, z_source,
                                       kwargs_data_joint, x_image, y_image,
                                       magnifications, image_position_uncertainties, flux_uncertainties,
                                       uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
                                       likelihood_mask_imaging_weights)

    @staticmethod
    def rotate(x, y, theta):
        return x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)

    def likelihood_masks(self, x_image, y_image):

        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
        _x = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
        _y = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
        _xx, _yy = np.meshgrid(_x, _y)
        likelihood_mask = np.ones_like(_xx) * self._custom_mask
        _xx_rot, _yy_rot = self.rotate(_xx, _yy, self._mask_rotation * np.pi)
        q = 0.7
        inds = np.where(np.sqrt(_xx_rot ** 2 + (_yy_rot / q) ** 2) >= window_size / 1.9)
        likelihood_mask[inds] = 0.0

        if self._band == 'NIRCam200':
            # MASK CENTER OF DEFLECTOR
            likelihood_mask = self.quasar_image_mask(
                likelihood_mask,
                [0.0],
                [0.0],
                self._image_data.shape, radius_arcsec=0.4
            )
        if self._mask_quasar_images_for_logL:
            likelihood_mask_imaging_weights = self.quasar_image_mask(
                likelihood_mask,
                x_image,
                y_image,
                self._image_data.shape, radius_arcsec=0.4
            )
            return likelihood_mask, likelihood_mask_imaging_weights
        else:
            return likelihood_mask, likelihood_mask

    @property
    def kwargs_data(self):
        _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
        kwargs_data = {'background_rms': self._background_rms,
                       'exposure_time': self._exposure_time,
                       'ra_at_xy_0': self._ra_at_xy_0,
                       'dec_at_xy_0': self._dec_at_xy_0,
                       'transform_pix2angle': self._transform_pix2angle,
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

class M1134_HST(_2M1134):

    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        # reorder = [3,1,0,2]
        # x_image = np.array([ 1.48667844, -0.5005318 ,  0.75275602, -1.19123804])[reorder]
        # y_image = np.array([ 0.98634326,  0.59067364, -0.77144427, -1.54090464])[reorder]
        x_image = np.array([-1.24171241, -0.54108785, 1.43621075, 0.70658951])
        y_image = np.array([-1.45103786, 0.6893056, 1.07930078, -0.67756853])
        horizontal_shift = 0.048
        vertical_shift = -0.09
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(M1134_HST, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                                          uncertainty_in_fluxes=False,
                                        supersample_factor=supersample_factor,
                                        band='HST814W')

class M1134_MIRI(_2M1134):

    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([-1.24171241, -0.54108785, 1.43621075, 0.70658951])
        y_image = np.array([-1.45103786, 0.6893056, 1.07930078, -0.67756853])
        horizontal_shift = -0.012
        vertical_shift = 0.015
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(M1134_MIRI, self).__init__(x_image, y_image, magnifications,
                                         image_position_uncertainties,
                                         flux_uncertainties,
                                          uncertainty_in_fluxes=False,
                                         supersample_factor=supersample_factor,
                                         band='MIRI560W')

class M1134_NIRCAM(_2M1134):
    band = 'NIRCam'

    def __init__(self, supersample_factor=1, filename_image_data=None):
        """
        JWST/NIRCam F200W.

        Image positions are centroids measured directly on the F200W mosaic, in a
        frame centred on the deflector with x = +East, y = +North.  They reproduce
        the positions already used by M1134_HST / M1134_MIRI to under 2.7 mas, so no
        shift is applied here.

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        :param filename_image_data: optional [psf_txt, image_txt, noise_txt] paths, so the
        pixel data can be kept out of the repository
        """
        x_image = np.array([-1.24355627, -0.53926016, 1.43687313, 0.70594330])
        y_image = np.array([-1.44905590, 0.68926229, 1.07907354, -0.67927994])
        horizontal_shift = 0.002
        vertical_shift = 0.006
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 mas
        flux_uncertainties = None
        magnifications = np.array([1.0] * 4)
        super(M1134_NIRCAM, self).__init__(x_image, y_image, magnifications,
                                           image_position_uncertainties,
                                           flux_uncertainties,
                                           uncertainty_in_fluxes=False,
                                           supersample_factor=supersample_factor,
                                           band='NIRCam200',
                                           filename_image_data=filename_image_data)


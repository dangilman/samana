from samana.Data.data_base import ImagingDataBase
import numpy as np

class _J0659(ImagingDataBase):

    # --- NIRCam200 western mask ---------------------------------------------------
    # The outer mask radius is reduced from window_size/2 (4.004") to
    # west_mask_radius inside a wedge of half-width west_mask_pa_width/2 about
    # west_mask_pa_center, so the region removed is a partial annulus.
    #
    # Position angle is measured in DEGREES WEST OF NORTH, i.e.
    # PA = degrees(arctan2(-ra, dec)) with ra = +East.  For reference in this frame:
    #   counter image (img 1)            PA = +112.6 deg,  r = 3.11"
    #   images 0 / 2 / 3                 PA = -115.5 / -22.0 / -61.0 deg (east side)
    #   compact bright feature NNE       PA =  -14 deg
    # The default wedge 60-110 deg therefore lies north of the counter image and
    # west of that feature, and excludes all four images.
    #
    # west_mask_radius = 2.8" is set so the wedge clears the lensed arc: inside
    # r = 2.2" the high-pass signal in this sector is flat at 0.8-2.3 sigma, and the
    # discrete features that motivate the mask all sit at r >= 3.08" (the four
    # brightest are at r = 3.08, 3.18, 3.61, 3.83", PA 102-123 deg).
    # Set west_mask_radius = 4.004 (= window_size/2) to switch the wedge off.
    west_mask_radius = 2.8
    west_mask_pa_center = 85.0
    west_mask_pa_width = 50.0
    keep_images_in_west_mask = True
    west_mask_keep_radius = 0.45

    def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
                 uncertainty_in_fluxes, supersample_factor, image_data_type,
                 mask_quasar_images_for_logL=True, filename_image_data=None):

        self._mask_quasar_images_for_logL = mask_quasar_images_for_logL
        self._image_data_type = image_data_type
        z_lens = 0.77
        z_source = 3.10
        # we use all three flux ratios to constrain the model
        keep_flux_ratio_index = [0, 1, 2]
        if image_data_type == 'HST814W':
            raise Exception('not HST imaging avaialble for this system')
        elif image_data_type == 'MIRI540W':
            from samana.Data.ImageData.j0659_MIRI540W import psf_model, image_data, noise_map
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = None
            self._image_data = image_data
            self._psf_supersampling_factor = 3
            self._deltaPix = 0.11090762486874971
            self._window_size = 7.985348990549979
            self._ra_at_xy_0 = -4.3480206889223165
            self._dec_at_xy_0 = -3.602445745739124
            self._transform_pix2angle = np.array([[0.01035521, 0.11042314],
                                                  [0.11042314, -0.01035521]])
            self._background_rms = None
            self._exposure_time = None
            self._noise_map = noise_map
            self._supersampling_convolution = False

        elif image_data_type == 'NIRCam200':
            # JWST/NIRCam F200W, SWarp mosaic + companion RMS map.
            # Units are microJy (mosaic ZP = 23.9). Pixel scale is the native
            # NIRCam SW scale; the mosaic is N-up / E-left, so transform_pix2angle
            # is diagonal. Window is matched to the MIRI540W window above.
            # psf_model is the STARRED reconstruction (supersampling 3, 447x447).
            # psf_variance_map is STARRED's psf_error_map SQUARED: that file is a standard
            # error in kernel units, while lenstronomy wants a variance in kernel**2 units.
            if filename_image_data is not None:
                psf_model = np.loadtxt(filename_image_data[0])
                image_data = np.loadtxt(filename_image_data[1])
                noise_map = np.loadtxt(filename_image_data[2])
                psf_variance_map = np.loadtxt(filename_image_data[3]) if len(filename_image_data) > 3 else None
            else:
                # ImageData/j0659_NIRCam200.py holds the pixel data inline and is kept out of
                # the public repository; use filename_image_data to load it from text instead.
                try:
                    from samana.Data.ImageData.j0659_NIRCam200 import (psf_model, image_data, noise_map,
                                                                       psf_variance_map)
                except ImportError:
                    raise Exception('J0659 NIRCam200 pixel data not available in this checkout. Pass '
                                    'filename_image_data=[psf_txt, imgdata_txt, noise_txt, psfvar_txt]; '
                                    'the fourth entry is optional and sets psf_variance_map to None if omitted.')
            self._psf_estimate_init = psf_model
            self._psf_error_map_init = psf_variance_map
            self._image_data = image_data
            self._psf_supersampling_factor = 3
            self._deltaPix = 0.0309195442
            self._window_size = 8.0081619541
            self._ra_at_xy_0 = 3.9744066384
            self._dec_at_xy_0 = -3.9884616346
            self._transform_pix2angle = np.array([[-0.0309195442, 0.0],
                                                  [0.0, 0.0309195442]])
            self._background_rms = None
            self._exposure_time = None
            # supplied SWarp RMS map, 1-sigma per pixel, used unmodified.
            # Pixel noise is strongly correlated: expect chi2/dof ~ 0.54 from a
            # good fit and inflate parameter uncertainties by ~1.5-2x.
            self._noise_map = noise_map
            self._supersampling_convolution = False

        else:
            raise Exception('image data type must be HST814W, MIRI540W or NIRCam200')

        self._supersample_factor = supersample_factor
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
        likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
        super(_J0659, self).__init__(z_lens, z_source,
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
        inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
        likelihood_mask[inds] = 0.0
        if self._image_data_type == 'NIRCam200':
            # MASK CENTER OF DEFLECTOR
            likelihood_mask = self.quasar_image_mask(
                likelihood_mask,
                [0.0],
                [0.0],
                self._image_data.shape, radius_arcsec=1.4
            )
            # Shrink the mask radius inside a wedge to the west, so what gets removed
            # is a partial annulus: west_mask_radius < r < window_size/2 within
            # west_mask_pa_width degrees about west_mask_pa_center.  There are faint
            # curved features out there, probably lensed background galaxies, which the
            # lens model does not describe and which show up as a coherent positive
            # residual.
            #
            # This uses the TRUE coordinate grid rather than the _xx/_yy linspace proxy
            # above.  That matters: this mosaic is N-up/E-left, so _xx increases toward
            # WEST and its sign is the opposite of the sky RA offset.  ra_grid is +East.
            ra_grid, dec_grid = self.coordinate_system.coordinate_grid(*self._image_data.shape)
            rr = np.sqrt(ra_grid ** 2 + dec_grid ** 2)
            pa = np.degrees(np.arctan2(-ra_grid, dec_grid))   # degrees west of north
            dpa = (pa - self.west_mask_pa_center + 180.0) % 360.0 - 180.0
            inds = np.where((rr >= self.west_mask_radius) &
                            (np.abs(dpa) <= self.west_mask_pa_width / 2.0))
            likelihood_mask[inds] = 0.0
            # the default wedge excludes all four images, but restore a disc around
            # each one anyway so no image can be silently dropped if the wedge is
            # widened; set keep_images_in_west_mask = False to disable.
            if self.keep_images_in_west_mask:
                for _xq, _yq in zip(x_image, y_image):
                    _d = np.sqrt((ra_grid - _xq) ** 2 + (dec_grid - _yq) ** 2)
                    likelihood_mask[np.where(_d < self.west_mask_keep_radius)] = 1.0
        if self._mask_quasar_images_for_logL:
            if self._image_data_type == 'NIRCam200':
                # the NIRCam PSF is much sharper than MIRI, so a smaller mask ok
                mask_radius = 0.2
            else:
                mask_radius = 0.35
            likelihood_mask_imaging_weights = self.quasar_image_mask(
                likelihood_mask,
                x_image,
                y_image,
                self._image_data.shape, radius_arcsec=mask_radius
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
    def kwargs_numerics(self):
        kwargs_numerics = {
            'supersampling_factor': int(self._supersample_factor * max(1, self._psf_supersampling_factor)),
            'supersampling_convolution': self._supersampling_convolution,
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

class J0659_MIRI(_J0659):

    def __init__(self, supersample_factor=1):
        """

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        """
        x_image = np.array([1.74995434, -2.91498935, 0.77093583, 1.83409918])
        y_image = np.array([-0.91482953, -1.25060444, 1.97732951, 0.98810446])
        horizontal_shift = -0.008
        vertical_shift = 0.0
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 arcsec
        flux_uncertainties = None
        uncertainty_in_fluxes = False
        magnifications = np.array([1.0] * 4)
        image_data_type = 'MIRI540W'
        super(J0659_MIRI, self).__init__(x_image,
                                     y_image,
                                     magnifications,
                                     image_position_uncertainties,
                                     flux_uncertainties,
                                     uncertainty_in_fluxes,
                                     supersample_factor,
                                     image_data_type)

    @property
    def satellite_or_star_coords(self):
        return 0.28, 1.55

class J0659_NIRCAM(_J0659):
    band = 'NIRCam'

    def __init__(self, supersample_factor=1, filename_image_data=None):
        """
        JWST/NIRCam F200W.

        Image positions are centroids measured directly on the F200W mosaic,
        in a frame centred on the lens galaxy with x = +East, y = +North.
        They agree with the MIRI540W positions to 2-3 mas rms after removing a
        constant offset of (+0.049, +0.055) arcsec; set the shifts below to
        that offset if you want the two bands on an identical frame.

        :param image_position_uncertainties: list of astrometric uncertainties for each image
        i.e. [0.003, 0.003, 0.003, 0.003]
        :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
        post-processing
        :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
        :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
        :param filename_image_data: optional [psf_txt, image_txt, noise_txt] paths, so the
        pixel data can be kept out of the repository
        """
        x_image = np.array([1.80108812, -2.86662765, 0.82118058, 1.88327995])
        y_image = np.array([-0.85803028, -1.19468746, 2.02849772, 1.04438631])
        horizontal_shift = 0.0
        vertical_shift = 0.0
        x_image += horizontal_shift
        y_image += vertical_shift
        image_position_uncertainties = [0.005] * 4 # 5 mas
        flux_uncertainties = None
        uncertainty_in_fluxes = False
        magnifications = np.array([1.0] * 4)
        image_data_type = 'NIRCam200'
        super(J0659_NIRCAM, self).__init__(x_image,
                                     y_image,
                                     magnifications,
                                     image_position_uncertainties,
                                     flux_uncertainties,
                                     uncertainty_in_fluxes,
                                     supersample_factor,
                                     image_data_type,
                                     filename_image_data=filename_image_data)

    @property
    def satellite_or_star_coords(self):
        # centroided directly on the F200W data (MIRI quotes 0.28, 1.55)
        return 0.4124, 1.5907

# class _J0659HSTImaging(ImagingDataBase):
#
#     def __init__(self, x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
#                  uncertainty_in_fluxes, supersample_factor=1):
#
#         from samana.Data.ImageData.j0659_f814w import image_data, psf_model, psf_error_map
#         z_lens = 0.77
#         z_source = 3.1
#         # we use all three flux ratios to constrain the model
#         keep_flux_ratio_index = [0, 1, 2]
#         self._psf_estimate_init = psf_model
#         self._psf_error_map_init = psf_error_map
#         self._image_data = image_data
#         self._supersample_factor = supersample_factor
#         image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
#         multi_band_list = [image_band]
#         kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
#         likelihood_mask, likelihood_mask_imaging_weights = self.likelihood_masks(x_image, y_image)
#         super(_J0659HSTImaging, self).__init__(z_lens, z_source,
#                                      kwargs_data_joint, x_image, y_image,
#                                      magnifications, image_position_uncertainties, flux_uncertainties,
#                                      uncertainty_in_fluxes, keep_flux_ratio_index, likelihood_mask,
#                                      likelihood_mask_imaging_weights)
#
#     def likelihood_masks(self, x_image, y_image):
#
#         deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = self.coordinate_properties
#         _x = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
#         _y = np.linspace(-window_size / 2, window_size / 2, self._image_data.shape[0])
#         _xx, _yy = np.meshgrid(_x, _y)
#         likelihood_mask = np.ones_like(_xx)
#         inds = np.where(np.sqrt(_xx ** 2 + _yy ** 2) >= window_size / 2)
#         likelihood_mask[inds] = 0.0
#         #
#         # star_x = -0.45
#         # star_y = 1.54
#         # mask_radius = 0.275
#         # dr2 = (_xx - star_x)**2 + (_yy - star_y)**2
#         # inds = np.where(dr2 <= mask_radius**2)
#         # likelihood_mask[inds] = 0.0
#         return likelihood_mask, likelihood_mask
#
#     @property
#     def kwargs_data(self):
#         _, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, _ = self.coordinate_properties
#         kwargs_data = {'background_rms': 0.007518,
#                        'exposure_time': 1428.0,
#                        'ra_at_xy_0': ra_at_xy_0,
#                        'dec_at_xy_0': dec_at_xy_0,
#                        'transform_pix2angle': transform_pix2angle,
#                        'image_data': self._image_data}
#         return kwargs_data
#
#     @property
#     def kwargs_numerics(self):
#         return {'supersampling_factor': int(self._supersample_factor),
#                 'supersampling_convolution': False}
#
#     @property
#     def coordinate_properties(self):
#         deltaPix = 0.04
#         window_size = 8.48
#         ra_at_xy_0 = 4.238418
#         dec_at_xy_0 = -4.240695
#         transform_pix2angle = np.array([[-3.99916465e-02,  6.56928592e-06],
#        [ 6.58148735e-06,  3.99999809e-02]])
#         return deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size
#
#     @property
#     def kwargs_psf(self):
#         kwargs_psf = {'psf_type': 'PIXEL',
#                       'kernel_point_source': self._psf_estimate_init,
#                       'psf_error_map': self._psf_error_map_init}
#         return kwargs_psf
# class J0659JWST_HSTImaging(_J0659HSTImaging):
#
#     def __init__(self):
#         """
#
#         :param image_position_uncertainties: list of astrometric uncertainties for each image
#         i.e. [0.003, 0.003, 0.003, 0.003]
#         :param flux_uncertainties: list of flux ratio uncertainties in percentage, or None if these are handled
#         post-processing
#         :param magnifications: image magnifications; can also be a vector of 1s if tolerance is set to infintiy
#         :param uncertainty_in_fluxes: bool; the uncertainties quoted are for fluxes or flux ratios
#         """
#         x_image = np.array([ 1.86868455, -2.79631545,  0.88968455,  1.95268455])
#         y_image = np.array([-0.92780858, -1.26280858,  1.96419142,  0.97519142])
#         # vertical_shift = 0.04
#         # horizontal_shift = -0.022
#         vertical_shift = 0.0
#         horizontal_shift = 0.0
#         x_image += horizontal_shift
#         y_image += vertical_shift
#         image_position_uncertainties = [0.005] * 4 # 5 arcsec
#         flux_uncertainties = None
#         magnifications = np.array([1.0] * 4)
#         super(J0659JWST_HSTImaging, self).__init__(x_image, y_image, magnifications, image_position_uncertainties, flux_uncertainties,
#                                           uncertainty_in_fluxes=False)
#
#     @property
#     def satellite_or_star_coords(self):
#         return 0.48, 1.5

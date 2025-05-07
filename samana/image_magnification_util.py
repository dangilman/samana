from lenstronomy.LensModel.Util.decouple_multi_plane_util import setup_grids, coordinates_and_deflections, setup_lens_model
import numpy as np
from lenstronomy.LightModel.light_model import LightModel
from copy import deepcopy
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Util import util
from lenstronomy.Util.util import make_grid_with_coordtransform
from lenstronomy.Data.coord_transforms import Coordinates


def perturbed_flux_ratios_from_flux_ratios(flux_ratios, flux_ratio_measurement_uncertainties_percentage):
    """

    :param flux_ratios:
    :param flux_ratio_measurement_uncertainties_percentage:
    :return:
    """
    if flux_ratios.ndim == 1:
        flux_ratios_perturbed = [np.random.normal(flux_ratios[i],
                                        flux_ratios[i] *
                                        flux_ratio_measurement_uncertainties_percentage[i]) for i in range(0, 3)]
    else:
        flux_ratios_perturbed = deepcopy(flux_ratios)
        for i in range(0,3):
            flux_ratios_perturbed[:, i] += np.random.normal(0.0,
                                                            flux_ratios_perturbed[:, i] *
                                                            flux_ratio_measurement_uncertainties_percentage[i])
    return np.array(flux_ratios_perturbed)

def perturbed_flux_ratios_from_fluxes(fluxes, flux_measurement_uncertainties_percentage):
    """

    :param fluxes:
    :param flux_measurement_uncertainties_percentage:
    :return:
    """
    fluxes_perturbed = perturbed_fluxes_from_fluxes(fluxes, flux_measurement_uncertainties_percentage)
    fluxes = np.array(fluxes)
    if fluxes.ndim == 1:
        flux_ratios = fluxes_perturbed[1:] / fluxes_perturbed[0]
    else:
        flux_ratios = fluxes_perturbed[:, 1:] / fluxes_perturbed[:,0,np.newaxis]
    return flux_ratios

def perturbed_fluxes_from_fluxes(fluxes, flux_measurement_uncertainties_percentage):
    """

    :param fluxes:
    :param flux_measurement_uncertainties_percentage:
    :return:
    """
    fluxes = np.array(fluxes)
    if fluxes.ndim == 1:
        fluxes_perturbed = []
        for i in range(0, 4):
            df = np.random.normal(0.0, fluxes[i] * flux_measurement_uncertainties_percentage[i])
            fluxes_perturbed.append(fluxes[i] + df)
        fluxes_perturbed = np.array(fluxes_perturbed)
    else:
        fluxes_perturbed = np.empty_like(fluxes)
        for i in range(0, 4):
            df = np.random.normal(0.0, fluxes[:, i] * flux_measurement_uncertainties_percentage[i])
            fluxes_perturbed[:, i] = fluxes[:, i] + df
    return fluxes_perturbed


def magnification_finite_decoupled_v2(source_model, kwargs_source, x_image, y_image,
                                   lens_model_init, kwargs_lens_init, kwargs_lens, index_lens_split,
                                   grid_size, grid_resolution, lens_model_full,
                                   elliptical_ray_tracing_grid,
                                   grid_increment_factor=15.0,
                                   setup_decoupled_multiplane_lens_model_output=None):
    """
    """
    if setup_decoupled_multiplane_lens_model_output is None:
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free, z_source, z_split, cosmo_bkg = \
            setup_lens_model(lens_model_init, kwargs_lens_init, index_lens_split)
    else:
        (lens_model_fixed, lens_model_free, kwargs_lens_fixed,
         kwargs_lens_free, z_source, z_split, cosmo_bkg) = setup_decoupled_multiplane_lens_model_output
    grid_x_large, grid_y_large, interp_points_large, npix_large = setup_grids(grid_size,
                                                                              5*grid_resolution,
                                                                              0.0, 0.0)
    grid_x_large = grid_x_large.ravel()
    grid_y_large = grid_y_large.ravel()
    r_step = grid_size / grid_increment_factor
    magnifications = []
    flux_arrays = []
    ext = LensModelExtensions(lens_model_full)
    for (x_img, y_img) in zip(x_image, y_image):
        if elliptical_ray_tracing_grid:
            try:
                w1, w2, v11, v12, v21, v22 = ext.hessian_eigenvectors(
                    x_img, y_img, kwargs_lens
                )
                _v = [np.array([v11, v12]), np.array([v21, v22])]
                _w = [abs(w1), abs(w2)]
                idx = int(np.argmax(_w))
                v = _v[idx]
                rotation_angle = np.arctan(v[1] / v[0]) - np.pi / 2
                grid_x, grid_y = util.rotate(grid_x_large, grid_y_large,
                                             rotation_angle)
                sort = np.argsort(_w)
                q_eigenvalue = _w[sort[0]] / _w[sort[1]]
                q = max(0.1, q_eigenvalue)
                grid_r = np.hypot(grid_x, grid_y / q).ravel()
            except:
                print('q eigenvalue not defined; computing image magifications on a circular grid.')
                grid_r = np.hypot(grid_x_large, grid_y_large).ravel()
        else:
            grid_r = np.hypot(grid_x_large, grid_y_large).ravel()
        # mag, flux_array = mag_finite_single_image(source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
        #                     kwargs_lens_free, kwargs_lens, z_split, z_source,
        #                     cosmo_bkg, x_img, y_img, grid_x_large, grid_y_large,
        #                     grid_r, r_step, grid_resolution, grid_size, z_split, z_source)
        mag, flux_array = mag_finite_single_image_v2(source_model, kwargs_source, lens_model_fixed,
                                                     lens_model_free, kwargs_lens_fixed, kwargs_lens_free, kwargs_lens,
                                                     z_split, z_source, cosmo_bkg, x_img, y_img, grid_resolution, grid_size,
                                                     z_split, z_source)
        magnifications.append(mag)
        flux_arrays.append(flux_array.reshape(npix_large, npix_large))
    return np.array(magnifications), flux_arrays

def magnification_finite_decoupled(source_model, kwargs_source, x_image, y_image,
                                   lens_model_init, kwargs_lens_init, kwargs_lens, index_lens_split,
                                   grid_size, grid_resolution, lens_model_full,
                                   elliptical_ray_tracing_grid,
                                   grid_increment_factor=15.0,
                                   setup_decoupled_multiplane_lens_model_output=None):
    """
    """
    if setup_decoupled_multiplane_lens_model_output is None:
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free, z_source, z_split, cosmo_bkg = \
            setup_lens_model(lens_model_init, kwargs_lens_init, index_lens_split)
    else:
        (lens_model_fixed, lens_model_free, kwargs_lens_fixed,
         kwargs_lens_free, z_source, z_split, cosmo_bkg) = setup_decoupled_multiplane_lens_model_output
    magnifications = []
    flux_arrays = []
    # use_method_2 = True
    grid_x_large, grid_y_large, interp_points_large, npix_large = setup_grids(grid_size,
                                                                              grid_resolution,
                                                                              0.0, 0.0)
    grid_x_large = grid_x_large.ravel()
    grid_y_large = grid_y_large.ravel()
    grid_r = np.hypot(grid_x_large, grid_y_large).ravel()
    r_step = grid_size / grid_increment_factor
    for (x_img, y_img) in zip(x_image, y_image):
        mag, flux_array = mag_finite_single_image_v2(source_model, kwargs_source, lens_model_fixed,
                                                         lens_model_free, kwargs_lens_fixed, kwargs_lens_free, kwargs_lens,
                                                         z_split, z_source, cosmo_bkg, x_img, y_img, grid_resolution, grid_size,
                                                         z_split, z_source)
        magnifications.append(mag)
        flux_arrays.append(flux_array)
        # else:
        #     mag, flux_array = mag_finite_single_image(source_model, kwargs_source, lens_model_fixed, lens_model_free,
        #                                               kwargs_lens_fixed,
        #                                               kwargs_lens_free, kwargs_lens, z_split, z_source,
        #                                               cosmo_bkg, x_img, y_img, grid_x_large, grid_y_large,
        #                                               grid_r, r_step, grid_resolution, grid_size, z_split, z_source)
        #     magnifications.append(mag)
        #     flux_arrays.append(flux_array.reshape(npix_large, npix_large))
    return np.array(magnifications), flux_arrays

def mag_finite_single_image_v2(source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                            kwargs_lens_free, kwargs_lens, z_split, z_source,
                            cosmo_bkg, x_image, y_image, grid_resolution, grid_size_max,
                               zlens, zsource, intial_resolution_reduction_factor=4,flux_threshold_factor=100,
                               distance_factor=20):
    """

    """
    Td = cosmo_bkg.T_xy(0, zlens)
    Ts = cosmo_bkg.T_xy(0, zsource)
    Tds = cosmo_bkg.T_xy(zlens, zsource)
    reduced_to_phys = cosmo_bkg.d_xy(0, zsource) / cosmo_bkg.d_xy(zlens, zsource)

    # initialize low-res flux array
    deltapix_init = intial_resolution_reduction_factor * grid_resolution # lower res by a factor 5
    numPix_init = int(grid_size_max / deltapix_init)
    (grid_x_large_init, grid_y_large_init, ra_at_xy_0, dec_at_xy_0,
     x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix) = (
        make_grid_with_coordtransform(numPix_init, deltapix_init))
    coordinates_lowres = Coordinates(Mpix2coord, ra_at_xy_0, dec_at_xy_0)
    grid_r = np.hypot(grid_x_large_init, grid_y_large_init).ravel()

    # setup ray tracing info
    xD = np.zeros_like(grid_x_large_init)
    yD = np.zeros_like(grid_y_large_init)
    alpha_x_foreground = np.zeros_like(grid_x_large_init)
    alpha_y_foreground = np.zeros_like(grid_y_large_init)
    alpha_x_background = np.zeros_like(grid_x_large_init)
    alpha_y_background = np.zeros_like(grid_y_large_init)
    inds_compute = np.where(grid_r <= grid_size_max/2)
    grid_x_large_init = grid_x_large_init[inds_compute]
    grid_y_large_init = grid_y_large_init[inds_compute]
    x_points_temp = grid_x_large_init + x_image
    y_points_temp = grid_y_large_init + y_image
    _xD, _yD, _alpha_x_foreground, _alpha_y_foreground, _alpha_x_background, _alpha_y_background = \
        coordinates_and_deflections(lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                    x_points_temp, y_points_temp, z_split, z_source, cosmo_bkg)
    xD[inds_compute] = _xD
    yD[inds_compute] = _yD
    alpha_x_foreground[inds_compute] = _alpha_x_foreground
    alpha_y_foreground[inds_compute] = _alpha_y_foreground
    alpha_x_background[inds_compute] = _alpha_x_background
    alpha_y_background[inds_compute] = _alpha_y_background
    beta_x, beta_y = calc_source_sb(xD.ravel(),
                                      yD.ravel(),
                                      alpha_x_foreground.ravel(),
                                      alpha_y_foreground.ravel(),
                                      alpha_x_background.ravel(),
                                      alpha_y_background.ravel(),
                                      Td, Tds, Ts, reduced_to_phys,
                                      lens_model_free,
                                      kwargs_lens)
    flux_array = source_model.surface_brightness(beta_x, beta_y, kwargs_source).reshape(numPix_init, numPix_init)

    dist = grid_size_max / distance_factor
    flux_array_threshold = np.max(flux_array) / flux_threshold_factor
    bright_indexes = np.where(flux_array >= flux_array_threshold)
    bright_coords = coordinates_lowres.map_pix2coord(bright_indexes[0], bright_indexes[1])

    # import matplotlib.pyplot as plt
    # plt.imshow(flux_array, origin='upper')
    # plt.show()
    #
    # plt.imshow(flux_array, origin='upper')
    # plt.scatter(bright_indexes[1], bright_indexes[0], color='r',alpha=0.3,marker='+')
    # plt.show()

    # NOW AT HIGH RESOLUTION
    # initialize high-res flux array
    deltapix = grid_resolution
    numPix = int(grid_size_max / deltapix)
    (grid_x_large, grid_y_large, ra_at_xy_0, dec_at_xy_0,
     x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix) = (
        make_grid_with_coordtransform(numPix, deltapix))

    coordinates_highres = Coordinates(Mpix2coord, ra_at_xy_0, dec_at_xy_0)
    pixel_x_large, pixel_y_large = [], []
    inds_compute_array = np.zeros_like(grid_x_large)
    grid_r = np.hypot(grid_x_large, grid_y_large)
    bright_coords_x, bright_coords_y = bright_coords[1], bright_coords[0]
    for grid_index, (coord_x, coord_y, r_coord) in enumerate(zip(grid_x_large, grid_y_large, grid_r)):
        if r_coord > grid_size_max / 2:
            continue
        else:
            dx, dy = coord_x - bright_coords_x, coord_y - bright_coords_y
            dr = np.sqrt(dx ** 2 + dy ** 2)
            if np.any(dr <= dist):
                inds_compute_array[grid_index] = 1
            # for bright_coord_x, bright_coord_y in zip(bright_coords[1], bright_coords[0]):
            #     dx, dy = coord_x - bright_coord_x, coord_y - bright_coord_y
            #     dr = np.sqrt(dx ** 2 + dy ** 2)
            #     if dr <= dist:
            #         inds_compute_array[grid_index] = 1
            #         pix_x, pix_y = coordinates_highres.map_coord2pix(coord_x, coord_y)
            #         pixel_x_large.append(pix_x)
            #         pixel_y_large.append(pix_y)
            #         break
    inds_compute_array = inds_compute_array.reshape(numPix,numPix)[::-1,::-1]
    #plt.imshow(inds_compute_array, origin='upper'); plt.show()
    inds_compute_highres = np.where(inds_compute_array.ravel()==1)
    x_points_temp = grid_x_large[inds_compute_highres] + x_image
    y_points_temp = grid_y_large[inds_compute_highres] + y_image
    # setup ray tracing info
    xD = np.zeros_like(grid_x_large)
    yD = np.zeros_like(grid_y_large)
    alpha_x_foreground = np.zeros_like(grid_x_large)
    alpha_y_foreground = np.zeros_like(grid_y_large)
    alpha_x_background = np.zeros_like(grid_x_large)
    alpha_y_background = np.zeros_like(grid_y_large)
    _xD, _yD, _alpha_x_foreground, _alpha_y_foreground, _alpha_x_background, _alpha_y_background = \
        coordinates_and_deflections(lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                    x_points_temp, y_points_temp, z_split, z_source, cosmo_bkg)
    xD[inds_compute_highres] = _xD
    yD[inds_compute_highres] = _yD
    alpha_x_foreground[inds_compute_highres] = _alpha_x_foreground
    alpha_y_foreground[inds_compute_highres] = _alpha_y_foreground
    alpha_x_background[inds_compute_highres] = _alpha_x_background
    alpha_y_background[inds_compute_highres] = _alpha_y_background
    beta_x_highres, beta_y_highres = calc_source_sb(xD.ravel(),
                                    yD.ravel(),
                                    alpha_x_foreground.ravel(),
                                    alpha_y_foreground.ravel(),
                                    alpha_x_background.ravel(),
                                    alpha_y_background.ravel(),
                                    Td, Tds, Ts, reduced_to_phys,
                                    lens_model_free,
                                    kwargs_lens)
    flux_array_highres = source_model.surface_brightness(beta_x_highres, beta_y_highres, kwargs_source).reshape(numPix, numPix)
    magnification_highres = np.sum(flux_array_highres) * grid_resolution ** 2
    flux_array_highres = flux_array_highres.reshape(numPix, numPix)
    #
    #plt.imshow(flux_array_highres, origin='upper');
    #plt.scatter(pixel_x_large, pixel_y_large, color='r',alpha=0.1,s=5)
    #plt.show()
    #a = input('continue')
    return magnification_highres, flux_array_highres

def calc_source_sb(x, y, alpha_x_foreground, alpha_y_foreground, alpha_x_background, alpha_y_background,
                  Td, Tds, Ts, reduced_to_phys, lens_model_free, kwargs_lens):

    # compute the deflection angles from the main deflector
    deflection_x_main, deflection_y_main = lens_model_free.alpha(
        x / Td, y / Td, kwargs_lens
    )
    deflection_x_main *= reduced_to_phys
    deflection_y_main *= reduced_to_phys

    # add the main deflector to the deflection field
    alpha_x = alpha_x_foreground - deflection_x_main
    alpha_y = alpha_y_foreground - deflection_y_main

    # combine deflections
    alpha_background_x = alpha_x + alpha_x_background
    alpha_background_y = alpha_y + alpha_y_background

    # ray propagation to the source plane with the small angle approximation
    beta_x = x / Ts + alpha_background_x * Tds / Ts
    beta_y = y / Ts + alpha_background_y * Tds / Ts

    return beta_x, beta_y

def _inds_compute_grid_v2(grid_r, r_min, r_max, inds_compute):
    condition1 = grid_r >= r_min
    condition2 = grid_r < r_max
    condition = np.logical_and(condition1, condition2)
    inds_compute_new = np.where(condition)[0]
    inds_outside_r = np.where(grid_r > r_max)[0]
    inds_computed = np.append(inds_compute, inds_compute_new).astype(int)
    return inds_compute_new, inds_outside_r, inds_computed


def mag_finite_single_image(source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                            kwargs_lens_free, kwargs_lens, z_split, z_source,
                            cosmo_bkg, x_image, y_image, grid_x_large, grid_y_large,
                            grid_r, r_step, grid_resolution, grid_size_max, zlens, zsource):
    """

    """
    # initalize flux array
    flux_array = np.zeros(len(grid_x_large))
    # setup ray tracing info
    xD = np.zeros_like(flux_array)
    yD = np.zeros_like(flux_array)
    alpha_x_foreground = np.zeros_like(flux_array)
    alpha_y_foreground = np.zeros_like(flux_array)
    alpha_x_background = np.zeros_like(flux_array)
    alpha_y_background = np.zeros_like(flux_array)
    r_min = 0.0
    r_max = r_min + r_step
    magnification_last = 0.0
    inds_compute = np.array([])
    Td = cosmo_bkg.T_xy(0, zlens)
    Ts = cosmo_bkg.T_xy(0, zsource)
    Tds = cosmo_bkg.T_xy(zlens, zsource)
    reduced_to_phys = cosmo_bkg.d_xy(0, zsource) / cosmo_bkg.d_xy(zlens, zsource)
    #flux_array_test = np.zeros(len(grid_x_large))
    while True:
        # select new coordinates to ray-trace through
        inds_compute, inds_outside_r, inds_computed = _inds_compute_grid(grid_r, r_min, r_max, inds_compute)
        x_points_temp = grid_x_large[inds_compute] + x_image
        y_points_temp = grid_y_large[inds_compute] + y_image

        # compute lensing stuff at these coordinates
        _xD, _yD, _alpha_x_foreground, _alpha_y_foreground, _alpha_x_background, _alpha_y_background = \
            coordinates_and_deflections(lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                        x_points_temp, y_points_temp, z_split, z_source, cosmo_bkg)
        # update the master grids with the new information
        xD[inds_compute] = _xD
        yD[inds_compute] = _yD
        alpha_x_foreground[inds_compute] = _alpha_x_foreground
        alpha_y_foreground[inds_compute] = _alpha_y_foreground
        alpha_x_background[inds_compute] = _alpha_x_background
        alpha_y_background[inds_compute] = _alpha_y_background

        # ray trace to source plane
        x = xD[inds_computed]
        y = yD[inds_computed]
        # compute the deflection angles from the main deflector
        deflection_x_main, deflection_y_main = lens_model_free.alpha(
            x / Td, y / Td, kwargs_lens
        )
        deflection_x_main *= reduced_to_phys
        deflection_y_main *= reduced_to_phys

        # add the main deflector to the deflection field
        alpha_x = alpha_x_foreground[inds_computed] - deflection_x_main
        alpha_y = alpha_y_foreground[inds_computed] - deflection_y_main

        # combine deflections
        alpha_background_x = alpha_x + alpha_x_background[inds_computed]
        alpha_background_y = alpha_y + alpha_y_background[inds_computed]

        # ray propagation to the source plane with the small angle approximation
        beta_x = x / Ts + alpha_background_x * Tds / Ts
        beta_y = y / Ts + alpha_background_y * Tds / Ts

        sb = source_model.surface_brightness(beta_x, beta_y, kwargs_source)
        flux_array[inds_computed] = sb
        flux_array[inds_outside_r] = 0.0

        #flux_array_test[inds_computed] = 1.0
        #flux_array_test[inds_outside_r] = 0.0

        magnification_temp = np.sum(flux_array) * grid_resolution ** 2
        diff = (
            abs(magnification_temp - magnification_last) / magnification_temp
        )
        r_min += r_step
        r_max += r_step
        if r_max >= grid_size_max:
            break
        elif diff < 0.001 and magnification_temp > 0.0001:  # we want to avoid situations with zero flux
            break
        else:
            magnification_last = magnification_temp
    return magnification_temp, flux_array

def _inds_compute_grid(grid_r, r_min, r_max, inds_compute):
    condition1 = grid_r >= r_min
    condition2 = grid_r < r_max
    condition = np.logical_and(condition1, condition2)
    inds_compute_new = np.where(condition)[0]
    inds_outside_r = np.where(grid_r > r_max)[0]
    inds_computed = np.append(inds_compute, inds_compute_new).astype(int)
    return inds_compute_new, inds_outside_r, inds_computed

def setup_gaussian_source(source_fwhm_pc, source_x, source_y, astropy_cosmo, z_source):

    if astropy_cosmo is None:
        from lenstronomy.Cosmo.background import Background
        astropy_cosmo = Background().cosmo
    kpc_per_arcsec = 1/astropy_cosmo.arcsec_per_kpc_proper(z_source).value
    source_sigma = 1e-3 * source_fwhm_pc / 2.354820 / kpc_per_arcsec
    kwargs_source_light = [{'amp': 1.0, 'center_x': source_x, 'center_y': source_y, 'sigma': source_sigma}]
    return LightModel(['GAUSSIAN']), kwargs_source_light

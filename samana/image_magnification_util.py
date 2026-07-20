from lenstronomy.LightModel.light_model import LightModel
from copy import deepcopy
from samana.forward_model_util import batch_lens_profiles
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

def magnification_finite_decoupled(source_model, kwargs_source, x_image, y_image,
                                   lens_model_init, kwargs_lens_init, kwargs_lens, index_lens_split,
                                   grid_size_list, grid_resolution,
                                   grid_increment_factor=15.0,
                                   setup_decoupled_multiplane_lens_model_output=None,
                                   magnification_method='CIRCULAR_APERTURE',
                                   rotation_angle_list=None,
                                   hessian_eigenvalue_list=None):
    """

    :param source_model:
    :param kwargs_source:
    :param x_image:
    :param y_image:
    :param lens_model_init:
    :param kwargs_lens_init:
    :param kwargs_lens:
    :param index_lens_split:
    :param grid_size_list:
    :param grid_resolution:
    :param grid_increment_factor:
    :param setup_decoupled_multiplane_lens_model_output:
    :param magnification_method:
    :param rotation_angle_list:
    :param hessian_eigenvalue_list:
    :return:
    """
    if setup_decoupled_multiplane_lens_model_output is None:
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free, z_source, z_split, cosmo_bkg = \
            setup_lens_model(lens_model_init, kwargs_lens_init, index_lens_split)
    else:
        (lens_model_fixed, lens_model_free, kwargs_lens_fixed,
         kwargs_lens_free, z_source, z_split, cosmo_bkg) = setup_decoupled_multiplane_lens_model_output
    magnifications = []
    flux_arrays = []

    for j, (x_img, y_img) in enumerate(zip(x_image, y_image)):

        grid_x_large, grid_y_large, interp_points_large, npix_large = setup_grids(grid_size_list[j],
                                                                                  grid_resolution,
                                                                                  0.0, 0.0)
        grid_x_large = grid_x_large.ravel()
        grid_y_large = grid_y_large.ravel()

        if magnification_method == 'ADAPTIVE':
            mag, flux_array = mag_finite_single_image_adaptive(source_model, kwargs_source, lens_model_fixed,
                                                         lens_model_free, kwargs_lens_fixed, kwargs_lens_free, kwargs_lens,
                                                         z_split, z_source, cosmo_bkg, x_img, y_img, grid_resolution,
                                                         grid_size_list[j],
                                                         z_split, z_source)
            magnifications.append(mag)
            flux_arrays.append(flux_array.T)

        elif magnification_method in ['CIRCULAR_APERTURE', 'ELLIPTICAL_APERTURE']:
            if magnification_method == 'ELLIPTICAL_APERTURE':
                grid_x, grid_y = util.rotate(grid_x_large, grid_y_large,
                                             rotation_angle_list[j])
                grid_r = np.hypot(grid_x, grid_y / hessian_eigenvalue_list[j]).ravel()
            else:
                grid_r = np.hypot(grid_x_large, grid_y_large).ravel()

            r_step = grid_size_list[j] / grid_increment_factor
            mag, flux_array = mag_finite_single_image(source_model, kwargs_source, lens_model_fixed, lens_model_free,
                                                      kwargs_lens_fixed,
                                                      kwargs_lens_free, kwargs_lens, z_split, z_source,
                                                      cosmo_bkg, x_img, y_img, grid_x_large, grid_y_large,
                                                      grid_r, r_step, grid_resolution, grid_size_list[j], z_split, z_source)
            magnifications.append(mag)
            flux_arrays.append(flux_array.reshape(npix_large, npix_large))
        else:
            raise Exception('magnification_method must be either CIRCULAR_APERTURE, ELLIPTICAL_APERTURE, or ADAPTIVE. '
                            'You specified magnification_method '+str(magnification_method))

    return np.array(magnifications), flux_arrays

def mag_finite_single_image_adaptive(source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                            kwargs_lens_free, kwargs_lens, z_split, z_source,
                            cosmo_bkg, x_image, y_image, grid_resolution, grid_size_max,
                               zlens, zsource, intial_resolution_reduction_factor=4,flux_threshold_factor=200,
                               distance_factor=30):
    """

    :param source_model:
    :param kwargs_source:
    :param lens_model_fixed:
    :param lens_model_free:
    :param kwargs_lens_fixed:
    :param kwargs_lens_free:
    :param kwargs_lens:
    :param z_split:
    :param z_source:
    :param cosmo_bkg:
    :param x_image:
    :param y_image:
    :param grid_resolution:
    :param grid_size_max:
    :param zlens:
    :param zsource:
    :param intial_resolution_reduction_factor:
    :param flux_threshold_factor:
    :param distance_factor:
    :return:
    """
    Td = cosmo_bkg.T_xy(0, zlens)
    Ts = cosmo_bkg.T_xy(0, zsource)
    Tds = cosmo_bkg.T_xy(zlens, zsource)
    reduced_to_phys = cosmo_bkg.d_xy(0, zsource) / cosmo_bkg.d_xy(zlens, zsource)

    # initialize low-res flux array
    deltapix_init = intial_resolution_reduction_factor * grid_resolution
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

    inds_compute_array = np.zeros_like(grid_x_large)
    grid_r = np.hypot(grid_x_large, grid_y_large)
    bright_coords_x, bright_coords_y = bright_coords[1], bright_coords[0]

    # shape: (N_highres_pixels, N_bright_coords)
    dx = grid_x_large[:, None] - bright_coords_x[None, :]
    dy = grid_y_large[:, None] - bright_coords_y[None, :]
    dr = np.sqrt(dx ** 2 + dy ** 2)
    within_radius = np.any(dr <= dist, axis=1)  # shape: (N_highres_pixels,)
    within_bounds = grid_r <= grid_size_max / 2
    inds_compute_highres = np.where(within_radius & within_bounds)


    # for grid_index, (coord_x, coord_y, r_coord) in enumerate(zip(grid_x_large, grid_y_large, grid_r)):
    #     if r_coord > grid_size_max / 2:
    #         continue
    #     else:
    #         dx, dy = coord_x - bright_coords_x, coord_y - bright_coords_y
    #         dr = np.sqrt(dx ** 2 + dy ** 2)
    #         if np.any(dr <= dist):
    #             inds_compute_array[grid_index] = 1


    inds_compute_array = inds_compute_array.reshape(numPix,numPix)[::-1,::-1]
    #plt.imshow(inds_compute_array, origin='upper'); plt.show()
    #inds_compute_highres = np.where(inds_compute_array.ravel()==1)
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


def decoupled_multiplane_rayshooting(grid_r, r_min, r_max, inds_compute,
                                     grid_x_large, grid_y_large, x_image, y_image,
                                     lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
                                     z_split, z_source, cosmo_bkg, xD, yD, alpha_x_foreground, alpha_y_foreground,
                                     alpha_x_background, alpha_y_background, Td, kwargs_lens, reduced_to_phys,
                                     Ts, Tds):
    """
    Ray propagation with the decoupled multiplane formalism, evaluated only at
    the coordinates newly added in the annulus [r_min, r_max). Returns the
    source-plane coordinates of the new points and their indices; previously
    computed points do not change between annulus iterations, so nothing needs
    to be re-evaluated for them.
    """
    # select new coordinates to ray-trace through
    inds_compute, inds_outside_r, inds_computed = _inds_compute_grid(grid_r, r_min, r_max, inds_compute)
    x_points_temp = grid_x_large[inds_compute] + x_image
    y_points_temp = grid_y_large[inds_compute] + y_image

    # compute lensing stuff at the new coordinates (the expensive halo-field call)
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

    # ray trace the NEW points to the source plane
    x = _xD
    y = _yD
    # compute the deflection angles from the main deflector
    deflection_x_main, deflection_y_main = lens_model_free.alpha(
        x / Td, y / Td, kwargs_lens
    )
    deflection_x_main *= reduced_to_phys
    deflection_y_main *= reduced_to_phys

    # add the main deflector to the deflection field
    alpha_x = _alpha_x_foreground - deflection_x_main
    alpha_y = _alpha_y_foreground - deflection_y_main

    # combine deflections
    alpha_background_x = alpha_x + _alpha_x_background
    alpha_background_y = alpha_y + _alpha_y_background

    # ray propagation to the source plane with the small angle approximation
    beta_x = x / Ts + alpha_background_x * Tds / Ts
    beta_y = y / Ts + alpha_background_y * Tds / Ts
    return beta_x, beta_y, inds_compute, inds_outside_r

def mag_finite_single_image(source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                            kwargs_lens_free, kwargs_lens, z_split, z_source,
                            cosmo_bkg, x_image, y_image, grid_x_large, grid_y_large,
                            grid_r, r_step, grid_resolution, grid_size_max, zlens, zsource):
    """
    Compute the magnification of a single lensed image with the decoupled multiplane formalism.
    """
    # initialize flux array
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
    flux_total = 0.0

    while True:

        beta_x_new, beta_y_new, inds_new, _ = decoupled_multiplane_rayshooting(
            grid_r, r_min, r_max, inds_compute,
            grid_x_large, grid_y_large, x_image, y_image,
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
            z_split, z_source, cosmo_bkg, xD, yD, alpha_x_foreground, alpha_y_foreground,
            alpha_x_background, alpha_y_background, Td, kwargs_lens, reduced_to_phys,
            Ts, Tds)

        # surface brightness only at the newly computed coordinates
        sb_new = source_model.surface_brightness(beta_x_new, beta_y_new, kwargs_source)
        flux_array[inds_new] = sb_new
        flux_total += np.sum(sb_new)

        magnification_temp = flux_total * grid_resolution ** 2
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

"""
PROTOTYPE: near/far split for the finite-source magnification.

Idea (per-image):
  1. Split the FOREGROUND + main-plane substructure (z <= z_split) into
       near = within R_max of the image,   far = beyond R_max.
  2. Compute the source-plane landing point of the image-center ray through the
     FULL fixed model and through the NEAR-only fixed model; their difference is
     a constant source-plane shift vector.
  3. Ray-trace the finite-source grid through the NEAR-only model, then add the
     shift so the image center lands at the correct source position.

Why it's correct: the far halos collectively act like a smooth external
convergence + shear across the box, so a CONSTANT shift alone is NOT enough --
tested on mg0414, constant-shift leaves ~40-60% of the magnification-relevant
gradient. But the far field is almost perfectly AFFINE over a 0.05" box, so we
approximate it to FIRST order: a constant shift (registration) plus a constant
Jacobian J = d(beta)/d(theta) (the far-field kappa,gamma). That drops the error
to ~1e-5 (only far-field flexion is discarded). Background halos (z > z_split)
are untouched -- they're handled by the decoupled model already.

Delta_beta and J are fit from a tiny 5-point stencil (center +/- dx, +/- dy) of
the FULL model vs the NEAR-only model -- a handful of full ray-traces per image,
negligible next to the thousands of grid points, and multi-plane-consistent.

Drop-in replacement for the magnification_finite_decoupled(...) call. Choose
R_max by convergence: sweep it and pick the smallest value where the flux ratios
are stable to tolerance.

NOTE on batching: this prototype assumes the fixed model holds individual halo
profiles (each with center_x/center_y). If you batch halos into MULTI_HALO_BATCH
*before* this step, do the near/far split first (on the raw halos) and batch only
the near set -- a batched entry mixes near and far halos at one redshift and
can't be culled as a unit. Profiles without a center (mass sheets, etc.) are
always kept.
"""
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Util.decouple_multi_plane_util import (
    setup_grids, coordinates_and_deflections, setup_lens_model,
)
from lenstronomy.Util import util
from samana.image_magnification_util import calc_source_sb, decoupled_multiplane_rayshooting


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _subset_fixed_model(lens_model_fixed, kwargs_lens_fixed, keep_mask):
    """Build a fixed LensModel containing only the kept profiles, preserving the
    multi-plane setup."""
    idx = np.where(keep_mask)[0]
    lm = LensModel(
        lens_model_list=[lens_model_fixed.lens_model_list[i] for i in idx],
        lens_redshift_list=[lens_model_fixed.redshift_list[i] for i in idx],
        z_source=lens_model_fixed.z_source,
        cosmo=lens_model_fixed.cosmo,
        multi_plane=True,
    )
    kw = [kwargs_lens_fixed[i] for i in idx]
    return lm, kw


def _beta_at(lens_model_fixed, kwargs_lens_fixed, lens_model_free, kwargs_lens_free,
             kwargs_lens, xpts, ypts, z_split, z_source, cosmo_bkg):
    """Source-plane positions of rays at absolute coords (xpts, ypts) through a fixed model."""
    Td = cosmo_bkg.T_xy(0, z_split)
    Ts = cosmo_bkg.T_xy(0, z_source)
    Tds = cosmo_bkg.T_xy(z_split, z_source)
    reduced_to_phys = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(z_split, z_source)
    xpts = np.atleast_1d(xpts).astype(float); ypts = np.atleast_1d(ypts).astype(float)
    xD, yD, a_fg_x, a_fg_y, a_bg_x, a_bg_y = coordinates_and_deflections(
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
        xpts, ypts, z_split, z_source, cosmo_bkg)
    return calc_source_sb(xD, yD, a_fg_x, a_fg_y, a_bg_x, a_bg_y,
                          Td, Tds, Ts, reduced_to_phys, lens_model_free, kwargs_lens)


def _farfield_correction(lm_full, kw_full, lm_near, kw_near, lens_model_free, kwargs_lens_free,
                         kwargs_lens, x_img, y_img, z_split, z_source, cosmo_bkg,
                         h=5e-3, order=2):
    """Fit the far field's contribution to beta over the box as a Taylor expansion about
    the image center, from a 9-point stencil of (beta_full - beta_near).

    Returns:
      dbeta_center (2,)   : far-field shift at center
      J (2,2)             : J[c,i] = d beta_c / d theta_i          (shift + shear)
      K (2,2,2)           : K[c,i,j] = d^2 beta_c / d theta_i d theta_j  (flexion)
    so for a grid offset dth=(dx,dy) from the image center:
      beta_corrected = beta_near + dbeta_center + J@dth + 0.5 * dth^T K[c] dth (per c).
    order=1 zeroes K (recovers the affine correction). h ~ flux/curvature scale in arcsec.
    """
    # stencil: center, +/-x, +/-y, four diagonals (for the cross derivative)
    ox = np.array([0, +1, -1, 0, 0, +1, +1, -1, -1], float) * h
    oy = np.array([0, 0, 0, +1, -1, +1, -1, +1, -1], float) * h
    xs = x_img + ox;
    ys = y_img + oy
    bxf, byf = _beta_at(lm_full, kw_full, lens_model_free, kwargs_lens_free, kwargs_lens,
                        xs, ys, z_split, z_source, cosmo_bkg)
    bxn, byn = _beta_at(lm_near, kw_near, lens_model_free, kwargs_lens_free, kwargs_lens,
                        xs, ys, z_split, z_source, cosmo_bkg)
    d = np.array([bxf - bxn, byf - byn])  # (2, 9) far-field contribution to beta
    dbeta_center = d[:, 0].copy()
    J = np.empty((2, 2));
    K = np.zeros((2, 2, 2))
    for c in range(2):
        f = d[c]
        J[c, 0] = (f[1] - f[2]) / (2 * h)  # d/dx
        J[c, 1] = (f[3] - f[4]) / (2 * h)  # d/dy
        if order >= 2:
            K[c, 0, 0] = (f[1] - 2 * f[0] + f[2]) / h ** 2  # d2/dx2
            K[c, 1, 1] = (f[3] - 2 * f[0] + f[4]) / h ** 2  # d2/dy2
            K[c, 0, 1] = K[c, 1, 0] = (f[5] - f[6] - f[7] + f[8]) / (4 * h ** 2)  # d2/dxdy
    return dbeta_center, J, K


def _mag_single_image_shifted(source_model, kwargs_source, lens_model_fixed, lens_model_free,
                              kwargs_lens_fixed, kwargs_lens_free, kwargs_lens,
                              z_split, z_source, cosmo_bkg, x_image, y_image,
                              grid_x_large, grid_y_large, grid_r, r_step,
                              grid_resolution, grid_size_max, zlens, zsource,
                              dbeta_center=(0.0, 0.0), J=None, K=None):
    """mag_finite_single_image, but with the NEAR-only fixed model and a far-field
    correction  beta += dbeta_center + J@dth + 0.5 dth^T K[c] dth  applied before
    sampling the source. dth=(dx,dy) are the grid offsets from the image center."""
    if J is None:
        J = np.zeros((2, 2))
    if K is None:
        K = np.zeros((2, 2, 2))
    flux_array = np.zeros(len(grid_x_large))
    xD = np.zeros_like(flux_array);
    yD = np.zeros_like(flux_array)
    a_fg_x = np.zeros_like(flux_array);
    a_fg_y = np.zeros_like(flux_array)
    a_bg_x = np.zeros_like(flux_array);
    a_bg_y = np.zeros_like(flux_array)
    r_min, r_max = 0.0, r_step
    mag_last, flux_total = 0.0, 0.0
    inds_compute = np.array([])
    Td = cosmo_bkg.T_xy(0, zlens);
    Ts = cosmo_bkg.T_xy(0, zsource)
    Tds = cosmo_bkg.T_xy(zlens, zsource)
    reduced_to_phys = cosmo_bkg.d_xy(0, zsource) / cosmo_bkg.d_xy(zlens, zsource)
    dbx, dby = dbeta_center

    while True:
        beta_x_new, beta_y_new, inds_new, _ = decoupled_multiplane_rayshooting(
            grid_r, r_min, r_max, inds_compute,
            grid_x_large, grid_y_large, x_image, y_image,
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
            z_split, z_source, cosmo_bkg, xD, yD, a_fg_x, a_fg_y, a_bg_x, a_bg_y,
            Td, kwargs_lens, reduced_to_phys, Ts, Tds)

        # <<< far-field correction: shift + shear (J) + flexion (K), (dx,dy)=grid offset >>>
        dx = grid_x_large[inds_new];
        dy = grid_y_large[inds_new]
        beta_x_new = (beta_x_new + dbx + J[0, 0] * dx + J[0, 1] * dy
                      + 0.5 * (K[0, 0, 0] * dx * dx + 2 * K[0, 0, 1] * dx * dy + K[0, 1, 1] * dy * dy))
        beta_y_new = (beta_y_new + dby + J[1, 0] * dx + J[1, 1] * dy
                      + 0.5 * (K[1, 0, 0] * dx * dx + 2 * K[1, 0, 1] * dx * dy + K[1, 1, 1] * dy * dy))

        sb_new = source_model.surface_brightness(beta_x_new, beta_y_new, kwargs_source)
        flux_array[inds_new] = sb_new
        flux_total += np.sum(sb_new)
        mag_temp = flux_total * grid_resolution ** 2
        diff = abs(mag_temp - mag_last) / mag_temp if mag_temp > 0 else 1.0
        r_min += r_step;
        r_max += r_step
        if r_max >= grid_size_max:
            break
        elif diff < 0.001 and mag_temp > 0.0001:
            break
        else:
            mag_last = mag_temp
    return mag_temp, flux_array


# ---------------------------------------------------------------------------
# main prototype
# ---------------------------------------------------------------------------
def magnification_finite_decoupled_nearfar(
    source_model, kwargs_source, x_image, y_image,
    lens_model_init, kwargs_lens_init, kwargs_lens, index_lens_split,
    grid_size_list, grid_resolution, R_max,
    halo_masses=None, M_ref=1e8, mass_exponent=0.5,
    farfield_order=2, farfield_stencil_h=5e-3,
    grid_increment_factor=15.0,
    setup_decoupled_multiplane_lens_model_output=None,
    magnification_method='ELLIPTICAL_APERTURE',
    rotation_angle_list=None, hessian_eigenvalue_list=None):
    """Near/far-split version of magnification_finite_decoupled.

    Foreground + main-plane halos beyond a (mass-dependent) radius of the image are
    demoted to an affine (shift + kappa,gamma) far-field correction; background halos
    (z > z_split) are untouched.

    :param R_max: base near/far split radius in arcsec (the radius used for a halo of
        mass M_ref). Convergence-test on the flux ratios.
    :param halo_masses: array aligned with kwargs_lens_fixed giving each profile's halo
        mass [M_sun] (np.nan for non-halo profiles / mass sheets). If None, every
        cullable halo uses the base R_max. Get these from pyHalo (halo.mass) in the same
        order lensing_quantities emits the profiles.
    :param M_ref: reference mass for the radius scaling.
    :param mass_exponent: per-halo radius = R_max * (M / M_ref) ** mass_exponent.
        0.5 (sqrt, Einstein-radius/shear footprint) is a safe, conservative default;
        1/3 is the value matched to what the affine correction actually discards
        (flexion ~ M/r^3). Set to 0 to recover a fixed R_max.
    """
    if setup_decoupled_multiplane_lens_model_output is None:
        (lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
         z_source, z_split, cosmo_bkg) = setup_lens_model(
            lens_model_init, kwargs_lens_init, index_lens_split)
    else:
        (lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
         z_source, z_split, cosmo_bkg) = setup_decoupled_multiplane_lens_model_output

    # per-profile redshift and (projected) position of the fixed model
    z_list = np.asarray(lens_model_fixed.redshift_list, dtype=float)
    cx = np.array([kw.get('center_x', np.nan) if isinstance(kw, dict) else np.nan
                   for kw in kwargs_lens_fixed])
    cy = np.array([kw.get('center_y', np.nan) if isinstance(kw, dict) else np.nan
                   for kw in kwargs_lens_fixed])
    cullable = (z_list <= z_split) & np.isfinite(cx) & np.isfinite(cy)  # foreground + subhalos

    # mass-dependent near/far radius: rmax_i = R_max * (M_i / M_ref) ** mass_exponent
    if halo_masses is not None:
        M = np.full(len(kwargs_lens_fixed), np.nan)  # 12452
        M[:len(halo_masses)] = halo_masses  # 12319 halos fill first; 133 sheets stay nan
        with np.errstate(invalid='ignore'):
            rmax_per_halo = R_max * (M / M_ref) ** mass_exponent
        rmax_per_halo = np.where(np.isfinite(rmax_per_halo), rmax_per_halo, R_max)  # sheets -> R_max
        rmax_per_halo = np.maximum(np.max(grid_size_list) / 2, rmax_per_halo)
    else:
        rmax_per_halo = np.full(len(cx), float(R_max))

    magnifications, flux_arrays = [], []
    for j, (x_img, y_img) in enumerate(zip(x_image, y_image)):
        # 1) near/far split for THIS image (per-halo, mass-scaled radius)
        dist = np.hypot(cx - x_img, cy - y_img)
        far = cullable & (dist > rmax_per_halo)
        keep = ~far
        lm_near, kw_near = _subset_fixed_model(lens_model_fixed, kwargs_lens_fixed, keep)

        # 2) far-field correction (shift + shear J + flexion K) from a 9-point stencil
        dbeta_center, J, K = _farfield_correction(
            lens_model_fixed, kwargs_lens_fixed, lm_near, kw_near,
            lens_model_free, kwargs_lens_free, kwargs_lens,
            x_img, y_img, z_split, z_source, cosmo_bkg,
            h=farfield_stencil_h, order=farfield_order)

        # 3) finite-source magnification with near-only model + affine correction
        grid_x_large, grid_y_large, interp_points_large, npix_large = setup_grids(
            grid_size_list[j], grid_resolution, 0.0, 0.0)
        grid_x_large = grid_x_large.ravel();
        grid_y_large = grid_y_large.ravel()
        if magnification_method == 'ELLIPTICAL_APERTURE':
            gx, gy = util.rotate(grid_x_large, grid_y_large, rotation_angle_list[j])
            grid_r = np.hypot(gx, gy / hessian_eigenvalue_list[j]).ravel()
        else:
            grid_r = np.hypot(grid_x_large, grid_y_large).ravel()
        r_step = grid_size_list[j] / grid_increment_factor

        mag, flux_array = _mag_single_image_shifted(
            source_model, kwargs_source, lm_near, lens_model_free, kw_near, kwargs_lens_free,
            kwargs_lens, z_split, z_source, cosmo_bkg, x_img, y_img,
            grid_x_large, grid_y_large, grid_r, r_step, grid_resolution,
            grid_size_list[j], z_split, z_source, dbeta_center=dbeta_center, J=J, K=K)
        magnifications.append(mag)
        flux_arrays.append(flux_array.reshape(npix_large, npix_large))

    return np.array(magnifications), flux_arrays

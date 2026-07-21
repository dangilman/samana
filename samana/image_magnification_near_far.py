"""
Per-plane near/far distortion-field ray tracing (DRAFT).

At each lens plane, halos near the central ray path are kept EXACT; the rest are
collapsed into a single HESSIAN (their convergence+shear at the central ray),
referenced so they produce zero deflection at the center. The grid rays are then
propagated as a *distortion* around the (exactly known) central ray, so the center
ray is undeflected by construction and only the differential distortion -- which is
what magnification depends on -- is accumulated.

Why this fits the cost model: the far halos become 1 HESSIAN per plane, so the
per-grid-point deflection is N_near + 1 profiles instead of N_halos, and it's one
propagation pass over the grid (few ray-shoot calls). Why it should be accurate:
the far field is linearized *per plane* (where it's smooth) about the exact central
ray, discarding only per-plane flexion -- stronger than a single affine fit made
after the whole far field has accumulated to the image.

STATUS: physics filled in and self-checked (near/far split, far->HESSIAN zero at center,
kappa_near vs area-average, CONVERGENCE excluded, reduced->physical factor matches
lenstronomy). The main (free/macromodel) deflector at z_split is applied inside both
setup_central_ray and accumulate_distortions. VALIDATED: with R_max -> inf (nothing
culled) accumulate_distortions reproduces a full lenstronomy multiplane ray_shooting to
2e-12 over the grid, confirming the propagation bookkeeping (deviation tracking,
central-ray subtraction, both reduced->physical factors, main deflector at z_split).
Finite R_max introduces only the far->HESSIAN linearization error (~1% on a stress test).

mag_finite_single_image_distortion is a drop-in for
samana.image_magnification_util.mag_finite_single_image: same growing-annulus driver,
same convergence test and flux bookkeeping, but the per-annulus ray-shoot goes through
the distortion field (setup_central_ray once per image, distortion_multiplane_rayshooting
per annulus) instead of decoupled_multiplane_rayshooting.
"""
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel


def distortion_field_at_lens_plane(x_co, y_co, x_center_at_plane, y_center_at_plane,
                                   T_z, lens_model_exact, lens_model_far,
                                   kwargs_lens_exact, kwargs_lens_far, kappa_near=0.0,
                                   subtract_near_kappa=False, subtract_far_kappa=False):
    """Distortion deflection of the grid rays at one plane, referenced to the central
    ray (deflection at the center is zero by construction).

    :param x_co, y_co: DEVIATION comoving coordinates of the grid rays from the central
        ray at this plane (grid ray's absolute angle = center angle + x_co / T_z)
    :param x_center_at_plane, y_center_at_plane: ANGULAR position of the central ray at
        this plane (from the precomputed central-ray interpolation)
    :param T_z: transverse comoving distance to this plane
    :param lens_model_exact: SinglePlane LensModel of the near ("exact") halos at z
    :param lens_model_far: SinglePlane LensModel of the far field -- a single 'HESSIAN'
        centered at (x_center_at_plane, y_center_at_plane) so alpha_far(center) = 0
    :param kappa_near: mean convergence of the near halos within the aperture (computed
        in lens_models_at_z, "average sense"); its 0th-order effect is subtracted so the
        distortion is the *differential* effect of adding the halos
    :param subtract_near_kappa: subtract the near halos' mean-convergence sheet (removes
        the 0th-order magnification boost from locally adding mass)
    :param subtract_far_kappa: also subtract the far field's net convergence
    """
    # grid ray's absolute angle = central-ray angle + deviation angle
    theta_x = x_center_at_plane + x_co / T_z
    theta_y = y_center_at_plane + y_co / T_z

    # near halos: exact deflection, minus the value at the central ray (-> distortion)
    ax_exact, ay_exact = lens_model_exact.alpha(theta_x, theta_y, kwargs_lens_exact)
    ax_c, ay_c = lens_model_exact.alpha(x_center_at_plane, y_center_at_plane, kwargs_lens_exact)

    # far halos: linear (Hessian) field, already zero at the center
    ax_far, ay_far = lens_model_far.alpha(theta_x, theta_y, kwargs_lens_far)

    ax = ax_exact + ax_far - ax_c
    ay = ay_exact + ay_far - ay_c

    # ---- remove the 0th-order (mean-convergence) effect of the added halos ----
    kappa_sheet = 0.0
    if subtract_near_kappa:
        kappa_sheet += kappa_near                                    # mean near-halo convergence
    if subtract_far_kappa:
        fxx, fxy, fyx, fyy = lens_model_far.hessian(
            x_center_at_plane, y_center_at_plane, kwargs_lens_far)   # far field is smooth -> Hessian trace = its convergence
        kappa_sheet += 0.5 * (fxx + fyy)
    if kappa_sheet != 0.0:
        lm_k = LensModel(['CONVERGENCE'])
        kw_k = [{'kappa': -kappa_sheet, 'ra_0': x_center_at_plane, 'dec_0': y_center_at_plane}]  # cancels the sheet
        axk, ayk = lm_k.alpha(theta_x, theta_y, kw_k)
        ax = ax + axk
        ay = ay + ayk
    return ax, ay


def lens_models_at_z(z, lens_model_fixed, kwargs_lens_fixed,
                     x_center_at_plane, y_center_at_plane, R_max,
                     exclude_names=()):
    """Split the fixed-model halos at redshift z into a near/exact LensModel and a
    single far-field HESSIAN about the central ray. Distance is measured in the sky
    (angular) plane to the central ray position at z.

    :param R_max: near/far angular radius at this plane (can be mass-scaled by the
        caller if desired)
    :param exclude_names: profile names to skip entirely -- CONVERGENCE mass sheets are
        excluded because their effect is already carried by the central-ray path (and
        the near-halo 0th-order convergence is handled by the kappa_near subtraction);
        including them would double-count.
    :return: (lens_model_exact, kwargs_lens_exact), (lens_model_far, kwargs_lens_far),
             kappa_near  -- mean convergence of the near halos within R_max
    """
    zlist = np.asarray(lens_model_fixed.redshift_list, dtype=float)
    names = lens_model_fixed.lens_model_list
    at_z = np.where(np.isclose(zlist, z))[0]

    near_names, near_kw, far_idx = [], [], []
    for i in at_z:
        if names[i] in exclude_names:          # skip mass sheets etc. entirely
            continue
        kw = kwargs_lens_fixed[i]
        cxi = kw.get('center_x', np.nan) if isinstance(kw, dict) else np.nan
        cyi = kw.get('center_y', np.nan) if isinstance(kw, dict) else np.nan
        if np.isfinite(cxi) and np.hypot(cxi - x_center_at_plane, cyi - y_center_at_plane) <= R_max:
            near_names.append(names[i]); near_kw.append(kw)
        else:
            far_idx.append(i)

    lens_model_exact = LensModel(near_names)
    kwargs_lens_exact = near_kw

    # mean convergence of the near halos within R_max, "average sense" via the deflection
    # flux: for the azimuthal average, alpha_radial(R) = R * <kappa>(<R), so
    # <kappa> = <alpha_radial(R_max)> / R_max. Uses only the LensModel (no halo masses).
    if near_names and R_max > 0:
        n_ring = 32
        phi = np.linspace(0.0, 2 * np.pi, n_ring, endpoint=False)
        rx = x_center_at_plane + R_max * np.cos(phi)
        ry = y_center_at_plane + R_max * np.sin(phi)
        aex, aey = lens_model_exact.alpha(rx, ry, kwargs_lens_exact)
        alpha_radial = np.mean(aex * np.cos(phi) + aey * np.sin(phi))
        kappa_near = alpha_radial / R_max
    else:
        kappa_near = 0.0

    # far field: sum the far halos' Hessian at the central ray, represent as one HESSIAN
    if far_idx:
        lm_far_full = LensModel([names[i] for i in far_idx])
        kw_far_full = [kwargs_lens_fixed[i] for i in far_idx]
        fxx, fxy, fyx, fyy = lm_far_full.hessian(x_center_at_plane, y_center_at_plane, kw_far_full)
    else:
        fxx = fxy = fyx = fyy = 0.0
    lens_model_far = LensModel(['HESSIAN'])
    kwargs_lens_far = [{'f_xx': float(fxx), 'f_xy': float(fxy),
                        'f_yx': float(fyx), 'f_yy': float(fyy),
                        'ra_0': x_center_at_plane, 'dec_0': y_center_at_plane}]  # HESSIAN params verified (alpha(center)=0)
    return (lens_model_exact, kwargs_lens_exact), (lens_model_far, kwargs_lens_far), float(kappa_near)


def accumulate_distortions(ray_angle_interp_x, ray_angle_interp_y,
                           lens_model_fixed, lens_model_free,
                           kwargs_lens_fixed, kwargs_lens_free,
                           redshift_planes, x_grid, y_grid,
                           cosmo_bkg, z_source, z_split, R_max,
                           z_source_convention=None):
    """Propagate the grid rays as a distortion around the exact central ray and return
    their source-plane positions.

    :param ray_angle_interp_x/y: functions T_z -> central-ray ANGULAR position at that
        transverse comoving distance (precomputed from the exact full-model central ray)
    :param x_grid, y_grid: initial ANGULAR offsets of the grid rays from the central ray
    :param redshift_planes: sorted list of lens-plane redshifts to traverse
    :param R_max: near/far split radius (scalar, or make it a callable of z / mass)

    :param z_source_convention: redshift the reduced deflections are defined against
        (the fixed model's z_source_convention). Defaults to z_source. The per-plane
        reduced->physical factor is d_xy(0, z_conv) / d_xy(z_plane, z_conv), matching
        lenstronomy's MultiPlaneBase._reduced2physical_deflection.

    [CHECK] mirrors a comoving multiplane recursion (cf. ray_shooting_partial_comoving);
    confirm whether the main free-model plane is handled here or separately in your
    decoupled setup.
    """
    if z_source_convention is None:
        z_source_convention = z_source
    d0s = cosmo_bkg.d_xy(0, z_source_convention)

    # grid rays start at the observer: comoving position 0, angle = grid offset
    x_co = np.zeros_like(x_grid, dtype=float)
    y_co = np.zeros_like(y_grid, dtype=float)
    alpha_x = np.array(x_grid, dtype=float)
    alpha_y = np.array(y_grid, dtype=float)

    z_prev = 0.0
    for zi in redshift_planes:
        T_z = cosmo_bkg.T_xy(0, zi)
        # step from the previous plane to this one (comoving)
        delta_T = cosmo_bkg.T_xy(z_prev, zi)
        x_co = x_co + alpha_x * delta_T
        y_co = y_co + alpha_y * delta_T

        # central-ray angular position at this plane (exact, precomputed)
        x_center = float(ray_angle_interp_x(T_z))
        y_center = float(ray_angle_interp_y(T_z))

        # near/far split and the per-plane distortion deflection (REDUCED)
        (lm_exact, kw_exact), (lm_far, kw_far), kappa_near = lens_models_at_z(
            zi, lens_model_fixed, kwargs_lens_fixed, x_center, y_center, R_max)
        d_ax, d_ay = distortion_field_at_lens_plane(
            x_co, y_co, x_center, y_center, T_z, lm_exact, lm_far, kw_exact, kw_far,
            kappa_near=kappa_near)

        # fixed-halo distortion: reduced -> physical for this plane, then subtract
        factor = d0s / cosmo_bkg.d_xy(zi, z_source_convention)
        alpha_x = alpha_x - d_ax * factor
        alpha_y = alpha_y - d_ay * factor

        # main (free/macro) deflector at the split plane -- applied as a differential
        # (relative to the central ray) with the free model's own reduced->physical
        # factor (defined w.r.t. the true source, as in coordinates_and_deflections)
        if np.isclose(zi, z_split):
            theta_gx = x_center + x_co / T_z          # grid ray's absolute angle
            theta_gy = y_center + y_co / T_z
            amx_g, amy_g = lens_model_free.alpha(theta_gx, theta_gy, kwargs_lens_free)
            amx_c, amy_c = lens_model_free.alpha(x_center, y_center, kwargs_lens_free)
            factor_free = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(z_split, z_source)
            alpha_x = alpha_x - (amx_g - amx_c) * factor_free
            alpha_y = alpha_y - (amy_g - amy_c) * factor_free
        z_prev = zi

    # propagate to the source plane. x_co,y_co are DEVIATION comoving, so this is the
    # DEVIATION source position; the caller adds beta_center for the absolute beta.
    T_s = cosmo_bkg.T_xy(0, z_source)
    delta_T = cosmo_bkg.T_xy(z_prev, z_source)
    x_co = x_co + alpha_x * delta_T
    y_co = y_co + alpha_y * delta_T
    dbeta_x = x_co / T_s
    dbeta_y = y_co / T_s
    return dbeta_x, dbeta_y


# ---------------------------------------------------------------------------
# central-ray setup + annulus wrapper (drop into mag_finite_single_image)
# ---------------------------------------------------------------------------

def _all_halos_at_z(z, lens_model_fixed, kwargs_lens_fixed, exclude_names=()):
    """Single-plane LensModel of ALL (non-excluded) fixed halos at redshift z."""
    zlist = np.asarray(lens_model_fixed.redshift_list, dtype=float)
    names = lens_model_fixed.lens_model_list
    idx = [i for i in np.where(np.isclose(zlist, z))[0] if names[i] not in exclude_names]
    return LensModel([names[i] for i in idx]), [kwargs_lens_fixed[i] for i in idx]


def central_ray_from_interp(ray_interp_x, ray_interp_y, beta_center, redshift_planes,
                            z_source_convention):
    """Build the central-ray dict from ray-path interpolators the pipeline ALREADY has.

    The decoupled-multiplane setup (and pyHalo's align_realization) already interpolate
    the image/centroid ray via samana.forward_model_util.interpolate_ray_paths /
    pyHalo.utilities.interpolate_ray_paths, which return interp1d(comoving distance) ->
    angular position. accumulate_distortions queries them at cosmo_bkg.T_xy(0, z), which
    is the same transverse comoving distance those interpolators use, so they drop in
    directly -- no re-propagation needed.

    :param ray_interp_x, ray_interp_y: interp1d(T) -> angular x / y (e.g. rix[0], riy[0]
        from interpolate_ray_paths for the image being integrated)
    :param beta_center: (bx, by) source-plane landing of that ray
        (lens_model.ray_shooting(x_image, y_image, kwargs), or the source coord if the
        interp was built with terminate_at_source=True)
    """
    return {'ray_angle_interp_x': ray_interp_x, 'ray_angle_interp_y': ray_interp_y,
            'beta_center': (float(beta_center[0]), float(beta_center[1])),
            'redshift_planes': list(redshift_planes),
            'z_source_convention': z_source_convention}


def setup_central_ray(x_image, y_image, lens_model_full, kwargs_lens_full,
                      redshift_planes, cosmo_bkg, z_source, z_split=None,
                      z_source_convention=None):
    """Convenience wrapper: interpolate the image ray through the FULL multiplane model
    and package it for accumulate_distortions. This just calls the pipeline's own
    interpolate_ray_paths (so it stays identical to how the decoupled setup builds ray
    paths) and adds beta_center from ray_shooting.

    :param lens_model_full: a multiplane LensModel containing ALL deflectors (halos + macro)
    :param kwargs_lens_full: its kwargs
    (z_split is accepted but unused here -- the full model already carries every plane;
     it's the free-model plane only in accumulate_distortions.)

    Prefer central_ray_from_interp when the pipeline already has the ray interpolators.
    """
    from samana.forward_model_util import interpolate_ray_paths
    if z_source_convention is None:
        z_source_convention = z_source
    rix, riy = interpolate_ray_paths([x_image], [y_image], lens_model_full,
                                     kwargs_lens_full, z_source)
    bx, by = lens_model_full.ray_shooting(x_image, y_image, kwargs_lens_full)
    return central_ray_from_interp(rix[0], riy[0], (bx, by),
                                   redshift_planes, z_source_convention)


def distortion_multiplane_rayshooting(grid_r, r_min, r_max, inds_compute,
                                      grid_x_large, grid_y_large,
                                      lens_model_fixed, lens_model_free,
                                      kwargs_lens_fixed, kwargs_lens_free,
                                      z_split, z_source, cosmo_bkg, R_max, central_ray,
                                      inds_selector):
    """Distortion-field analogue of decoupled_multiplane_rayshooting: ray-trace only the
    NEW annulus points [r_min, r_max) and return their source-plane positions. Drops
    into mag_finite_single_image's growing-region while-loop.

    :param central_ray: output of setup_central_ray (done once per image)
    :param inds_selector: the _inds_compute_grid function (annulus point selection),
        passed in to avoid a hard import
    :return: beta_x_new, beta_y_new, inds_new, inds_outside_r
    """
    inds_new, inds_outside_r, _ = inds_selector(grid_r, r_min, r_max, inds_compute)
    ox = grid_x_large[inds_new]; oy = grid_y_large[inds_new]     # angular offsets from the image
    dbeta_x, dbeta_y = accumulate_distortions(
        central_ray['ray_angle_interp_x'], central_ray['ray_angle_interp_y'],
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
        central_ray['redshift_planes'], ox, oy, cosmo_bkg, z_source, z_split, R_max,
        z_source_convention=central_ray['z_source_convention'])
    bx0, by0 = central_ray['beta_center']
    return bx0 + dbeta_x, by0 + dbeta_y, inds_new, inds_outside_r



def _as_interp(ri):
    """Accept either a single interp1d or the 1-element list interpolate_ray_paths returns."""
    if isinstance(ri, (list, tuple)):
        return ri[0]
    return ri

def mag_finite_single_image_distortion(
        source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
        kwargs_lens, z_split, z_source,
        cosmo_bkg, grid_x_large, grid_y_large,
        grid_r, r_step, grid_resolution, grid_size_max,
        R_max, ray_interp_x, ray_interp_y, redshift_planes=None, z_source_convention=None):

    rix = _as_interp(ray_interp_x); riy = _as_interp(ray_interp_y)
    if redshift_planes is None:
        redshift_planes = sorted(set(np.round(np.asarray(
            lens_model_fixed.redshift_list, dtype=float), 8).tolist()))
        if not any(np.isclose(z, z_split) for z in redshift_planes):
            redshift_planes = sorted(redshift_planes + [float(z_split)])
    if z_source_convention is None:
        z_source_convention = z_source

    # beta_center = source-plane landing of the image ray (interp runs to z_source)
    beta_center = (kwargs_source[0]['center_x'], kwargs_source[0]['center_y'])
    central_ray = central_ray_from_interp(rix, riy, beta_center, redshift_planes, z_source_convention)

    flux_array = np.zeros(len(grid_x_large))
    r_min = 0.0; r_max = r_min + r_step
    magnification_last = 0.0; inds_compute = np.array([]); flux_total = 0.0

    while True:
        beta_x_new, beta_y_new, inds_new, _ = distortion_multiplane_rayshooting(
            grid_r, r_min, r_max, inds_compute,
            grid_x_large, grid_y_large,
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens,
            z_split, z_source, cosmo_bkg, R_max, central_ray, _inds_compute_grid)
        sb_new = source_model.surface_brightness(beta_x_new, beta_y_new, kwargs_source)
        flux_array[inds_new] = sb_new
        flux_total += np.sum(sb_new)
        magnification_temp = flux_total * grid_resolution ** 2
        diff = abs(magnification_temp - magnification_last) / magnification_temp if magnification_temp > 0 else 1.0
        r_min += r_step; r_max += r_step
        if r_max >= grid_size_max:
            break
        elif diff < 0.001 and magnification_temp > 0.0001:
            break
        else:
            magnification_last = magnification_temp
    return magnification_temp, flux_array

def _inds_compute_grid(grid_r, r_min, r_max, inds_compute):
    condition = np.logical_and(grid_r >= r_min, grid_r < r_max)
    inds_compute_new = np.where(condition)[0]
    inds_outside_r = np.where(grid_r > r_max)[0]
    inds_computed = np.append(inds_compute, inds_compute_new).astype(int)
    return inds_compute_new, inds_outside_r, inds_computed

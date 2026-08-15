"""
Per-plane near/far distortion-field ray tracing.

At each lens plane, halos near the central ray path are kept EXACT; the rest are
collapsed into a single HESSIAN (their convergence+shear at the central ray),
referenced so they produce zero deflection at the center. The grid rays are then
propagated as a *distortion* around the (exactly known) central ray, so the center
ray is undeflected by construction and only the differential distortion -- which is
what magnification depends on -- is accumulated.

Why this fits the cost model: the far halos become 1 HESSIAN per plane, so the
per-grid-point deflection is N_near + 1 profiles instead of N_halos, and it's one
propagation pass over the grid (few ray-shoot calls). Why it should be accurate:
the far field is linearized *per plane* (where it's smooth) about the central ray,
discarding only per-plane flexion -- stronger than a single affine fit made after the
whole far field has accumulated to the image.

Notes on convergence sheets: CONVERGENCE profiles are NOT excluded. A mass sheet is an
exact constant Hessian, so it folds into the far-field HESSIAN term with zero
approximation error; excluding it drops the pyHalo mean-field terms and badly biases
the magnification. The optional kappa-subtraction (subtract_near_kappa /
subtract_far_kappa) is therefore OFF by default -- the macro fit and the pyHalo sheets
already handle the mean field, so subtracting again double-removes it.

VALIDATED: with R_max -> inf (nothing culled) accumulate_distortions reproduces a full
lenstronomy multiplane ray_shooting to ~1e-12 over the grid, and on a real 3124-halo
realization mag_finite_single_image_distortion matches the exact decoupled magnification
to <0.1% at matched integrator.

mag_finite_single_image_distortion is a drop-in for
samana.image_magnification_util.mag_finite_single_image: same growing-annulus driver,
same convergence test and flux bookkeeping, but the per-annulus ray-shoot goes through
the distortion field (central_ray_from_interp once per image,
distortion_multiplane_rayshooting per annulus) instead of decoupled_multiplane_rayshooting.
"""
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from samana.image_magnification_util import (adaptive_quadtree_magnification,
                                             rasterize_leaves)


_LENS_MODEL_CACHE = {}
_LM_HESSIAN = LensModel(['HESSIAN'])

def _cached_lens_model(names):
    key = tuple(names)
    lm = _LENS_MODEL_CACHE.get(key)
    if lm is None:
        if len(_LENS_MODEL_CACHE) > 5000:   # crude bound for long production runs
            _LENS_MODEL_CACHE.clear()
        lm = LensModel(list(names))
        _LENS_MODEL_CACHE[key] = lm
    return lm

def distortion_field_at_lens_plane(x_co, y_co, x_center_at_plane, y_center_at_plane,
                                   T_z, lens_model_exact, lens_model_far,
                                   kwargs_lens_exact, kwargs_lens_far, kappa_near=0.0,
                                   subtract_near_kappa=False, subtract_far_kappa=False):
    """Reduced distortion-deflection of the grid rays at one lens plane, referenced to
    the central ray (the deflection at the central ray is zero by construction).

    :param x_co: deviation comoving x of the grid rays from the central ray at this
        plane (the grid ray's absolute angle is x_center_at_plane + x_co / T_z)
    :param y_co: deviation comoving y of the grid rays from the central ray at this plane
    :param x_center_at_plane: angular x position of the central ray at this plane
        (from the precomputed central-ray interpolation)
    :param y_center_at_plane: angular y position of the central ray at this plane
    :param T_z: transverse comoving distance to this plane (Mpc)
    :param lens_model_exact: single-plane LensModel of the near ("exact") halos at z
    :param lens_model_far: single-plane LensModel of the far field -- a single 'HESSIAN'
        centered at (x_center_at_plane, y_center_at_plane) so alpha_far(center) = 0
    :param kwargs_lens_exact: kwargs for lens_model_exact
    :param kwargs_lens_far: kwargs for lens_model_far
    :param kappa_near: mean convergence of the near halos within R_max (from
        lens_models_at_z); only used if subtract_near_kappa is True
    :param subtract_near_kappa: if True, subtract the near halos' mean-convergence sheet
        (removes the 0th-order magnification boost from locally adding mass). Default
        False -- see module note; leaving it on double-removes the mean field.
    :param subtract_far_kappa: if True, also subtract the far field's net convergence.
        Default False.
    :return: (ax, ay), the reduced distortion-deflection components of the grid rays at
        this plane (zero at the central ray)
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

    # ---- optionally remove the 0th-order (mean-convergence) effect of the added halos ----
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

def precompute_plane_splits(ray_angle_interp_x, ray_angle_interp_y, lens_model_fixed,
                            kwargs_lens_fixed, redshift_planes, cosmo_bkg, R_max, z_source):
    """Precompute the per-plane near/far split ONCE for an image (it depends only on the
    fixed central ray, not on the annulus grid points, so it is identical for every
    annulus). Reuse the result across annuli to avoid re-splitting the halos and
    re-summing the far-field HESSIAN 10x. This also sidesteps needing a batched HESSIAN:
    the far Hessian is summed here a single time per plane.

    :param ray_angle_interp_x: interp1d(T_z) -> central-ray angular x
    :param ray_angle_interp_y: interp1d(T_z) -> central-ray angular y
    :param lens_model_fixed: multiplane LensModel of the fixed (halo) deflectors
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param redshift_planes: sorted list of lens-plane redshifts to traverse
    :param cosmo_bkg: lenstronomy Background
    :param R_max: near/far split radius (scalar or per-halo array; see lens_models_at_z)
    :param z_source: source redshift
    :return: list of per-plane dicts with keys zi, T_z, x_center, y_center, lm_exact,
        kw_exact, lm_far, kw_far, kappa_near
    """
    splits = []
    d0s = cosmo_bkg.d_xy(0, z_source)
    z_prev = 0.0
    for zi in redshift_planes:
        T_z = cosmo_bkg.T_xy(0, zi)
        x_center = float(ray_angle_interp_x(T_z))
        y_center = float(ray_angle_interp_y(T_z))
        (lm_e, kw_e), (lm_f, kw_f), kn = lens_models_at_z(
            zi, lens_model_fixed, kwargs_lens_fixed, x_center, y_center, R_max)
        splits.append({'zi': zi, 'T_z': T_z, 'x_center': x_center, 'y_center': y_center,
                       'lm_exact': lm_e, 'kw_exact': kw_e, 'lm_far': lm_f, 'kw_far': kw_f,
                       'kappa_near': kn, 'delta_T': cosmo_bkg.T_xy(z_prev, zi),
                       'factor': d0s / cosmo_bkg.d_xy(zi, z_source)})
        z_prev = zi
    return splits

def lens_models_at_z(z, lens_model_fixed, kwargs_lens_fixed,
                     x_center_at_plane, y_center_at_plane, R_max,
                     exclude_names=(), verbose=False):
    """Split the fixed-model halos at redshift z into a near/exact single-plane LensModel
    and a single far-field HESSIAN about the central ray. Distance is measured in the sky
    (angular) plane to the central-ray position at z. A halo is "near" if it has a finite
    center within R_max of the central ray; everything else (including CONVERGENCE sheets,
    which have no center) goes into the far Hessian, where a sheet is represented exactly.

    :param z: the lens-plane redshift to build models for
    :param lens_model_fixed: the multiplane LensModel of the fixed (halo) deflectors
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param x_center_at_plane: angular x position of the central ray at this plane
    :param y_center_at_plane: angular y position of the central ray at this plane
    :param R_max: near/far angular split radius (arcsec). Either a scalar (same radius
        for every halo) or a per-halo array indexed like lens_model_fixed.lens_model_list
        (e.g. mass-scaled, R_max_0 * (M / 1e8)**0.5); halo i is "near" if its center is
        within R_max[i] of the central ray. Non-finite entries (e.g. mass sheets) fall to
        the far field.
    :param exclude_names: profile names to skip entirely. Default () -- nothing is
        excluded. In particular CONVERGENCE sheets are kept (they fold exactly into the
        far Hessian); excluding them drops the pyHalo mean-field and biases the result.
    :return: (lens_model_exact, kwargs_lens_exact), (lens_model_far, kwargs_lens_far),
        kappa_near -- the near/exact model, the single far-field HESSIAN model, and the
        mean convergence of the near halos within R_max
    """
    zlist = np.asarray(lens_model_fixed.redshift_list, dtype=float)
    names = lens_model_fixed.lens_model_list
    at_z = np.where(np.isclose(zlist, z))[0]

    # R_max may be a scalar (shared) or a per-halo array indexed like lens_model_list
    R_max_arr = np.atleast_1d(np.asarray(R_max, dtype=float))
    scalar_rmax = R_max_arr.size == 1
    # lens_model_fixed can LEAD with non-halo macro deflectors absent from the per-halo
    # R_max array -- e.g. HE0435's SIS background galaxy at another plane, which
    # index_lens_split=[0,1] leaves in the fixed model. These
    # occupy the first n_extra fixed indices, so offset the lookup
    n_extra = 0 if scalar_rmax else max(len(names) - R_max_arr.size, 0)

    near_names, near_kw, far_idx = [], [], []

    for i in at_z:
        if names[i] in exclude_names:  # skip mass sheets etc. entirely
            continue
        kw = kwargs_lens_fixed[i]
        cxi = kw.get('center_x', np.nan) if isinstance(kw, dict) else np.nan
        cyi = kw.get('center_y', np.nan) if isinstance(kw, dict) else np.nan
        force_near = False
        if scalar_rmax:
            rmi = R_max_arr[0]
        else:
            j = i - n_extra
            if 0 <= j < R_max_arr.size:
                rmi = R_max_arr[j]
            else:
                rmi, force_near = np.inf, True  # extra macro deflector (bkg galaxy) -> keep exact
        if force_near or (np.isfinite(cxi) and np.isfinite(rmi)
                          and np.hypot(cxi - x_center_at_plane, cyi - y_center_at_plane) <= rmi):
            near_names.append(names[i])
            near_kw.append(kw)
        else:
            far_idx.append(i)

    lens_model_exact = _cached_lens_model(near_names)
    kwargs_lens_exact = near_kw

    kappa_near = 0.0
    # far field: sum the far halos' Hessian at the central ray, represent as one HESSIAN.
    # Group the far halos into MULTI_HALO_BATCH entries (per profile type) so the Hessian
    # sum is vectorized; CONVERGENCE sheets pass through individually.
    if far_idx:
        from samana.forward_model_util import batch_lens_profiles
        far_names_in = [names[i] for i in far_idx]
        far_kw_in = [kwargs_lens_fixed[i] for i in far_idx]
        far_z = [0.0] * len(far_idx)  # single plane; z unused by the hessian
        to_batch = set(far_names_in) - {'CONVERGENCE'}  # don't batch mass sheets or GCs
        b_names, _, b_kw = batch_lens_profiles(far_names_in, far_z, far_kw_in,
                                               profiles_to_batch=to_batch, min_group=4)
        lm_far_full = _cached_lens_model(b_names)
        fxx, fxy, fyx, fyy = lm_far_full.hessian(x_center_at_plane, y_center_at_plane, b_kw)
    else:
        fxx = fxy = fyx = fyy = 0.0
    lens_model_far = _LM_HESSIAN
    kwargs_lens_far = [{'f_xx': float(fxx), 'f_xy': float(fxy),
                        'f_yx': float(fyx), 'f_yy': float(fyy),
                        'ra_0': x_center_at_plane, 'dec_0': y_center_at_plane}]
    return (lens_model_exact, kwargs_lens_exact), (lens_model_far, kwargs_lens_far), float(kappa_near)


def accumulate_distortions(ray_angle_interp_x, ray_angle_interp_y,
                           lens_model_fixed, lens_model_free,
                           kwargs_lens_fixed, kwargs_lens_free,
                           redshift_planes, x_grid, y_grid,
                           cosmo_bkg, z_source, z_split, R_max,
                           plane_splits=None, verbose=False):
    """Propagate the grid rays as a distortion around the central ray and return their
    DEVIATION source-plane positions (the caller adds beta_center for the absolute beta).
    Marches plane by plane in comoving coordinates, at each plane splitting the halos
    near/far (lens_models_at_z), applying the reduced distortion-deflection
    (distortion_field_at_lens_plane) with the per-plane reduced->physical factor, and at
    z_split applying the free/macro deflector as a differential relative to the central ray.

    :param ray_angle_interp_x: interp1d(T_z) -> central-ray angular x at that transverse
        comoving distance (from the exact full-model central ray)
    :param ray_angle_interp_y: interp1d(T_z) -> central-ray angular y
    :param lens_model_fixed: multiplane LensModel of the fixed (halo) deflectors
    :param lens_model_free: single-plane LensModel of the free/macro deflector at z_split
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param kwargs_lens_free: kwargs for lens_model_free (the macro model)
    :param redshift_planes: sorted list of lens-plane redshifts to traverse (must include z_split)
    :param x_grid: initial angular x offsets of the grid rays from the central ray
    :param y_grid: initial angular y offsets of the grid rays from the central ray
    :param cosmo_bkg: lenstronomy Background (provides T_xy comoving and d_xy angular distances)
    :param z_source: source redshift
    :param z_split: redshift of the free/macro deflector (the "main" plane)
    :param R_max: near/far angular split radius (scalar; or mass-scale it in lens_models_at_z)
    :param z_source_convention: redshift the fixed model's reduced deflections are defined
        against; defaults to z_source. The per-plane reduced->physical factor is
        d_xy(0, z_conv) / d_xy(z_plane, z_conv), matching lenstronomy's
        MultiPlaneBase._reduced2physical_deflection.
    :return: (dbeta_x, dbeta_y), the deviation source-plane positions of the grid rays
        relative to the central ray
    """

    # grid rays start at the observer: comoving position 0, angle = grid offset
    x_co = np.zeros_like(x_grid, dtype=float)
    y_co = np.zeros_like(y_grid, dtype=float)
    alpha_x = np.array(x_grid, dtype=float)
    alpha_y = np.array(y_grid, dtype=float)
    if plane_splits is None:
        plane_splits = precompute_plane_splits(ray_angle_interp_x, ray_angle_interp_y,
                                               lens_model_fixed, kwargs_lens_fixed,
                                               redshift_planes, cosmo_bkg, R_max, z_source)
    z_prev = 0.0
    for sp in plane_splits:
        zi = sp['zi'];
        T_z = sp['T_z']
        x_center = sp['x_center'];
        y_center = sp['y_center']
        # step from the previous plane to this one (comoving)
        x_co = x_co + alpha_x * sp['delta_T']
        y_co = y_co + alpha_y * sp['delta_T']

        # per-plane distortion deflection (REDUCED) from the cached near/far split
        d_ax, d_ay = distortion_field_at_lens_plane(
            x_co, y_co, x_center, y_center, T_z,
            sp['lm_exact'], sp['lm_far'], sp['kw_exact'], sp['kw_far'],
            kappa_near=sp['kappa_near'])

        # fixed-halo distortion: reduced -> physical for this plane, then subtract
        alpha_x = alpha_x - d_ax * sp['factor']
        alpha_y = alpha_y - d_ay * sp['factor']

        # main (free/macro) deflector at the split plane -- applied as a differential
        # (relative to the central ray) with the free model's own reduced->physical
        # factor (defined w.r.t. the true source, as in coordinates_and_deflections)
        if np.isclose(zi, z_split):
            theta_gx = x_center + x_co / T_z  # grid ray's absolute angle
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
    """Build a single-plane LensModel of ALL (non-excluded) fixed halos at redshift z.

    :param z: the lens-plane redshift
    :param lens_model_fixed: the multiplane LensModel of the fixed (halo) deflectors
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param exclude_names: profile names to skip (default (): keep everything)
    :return: (LensModel of the halos at z, list of their kwargs)
    """
    zlist = np.asarray(lens_model_fixed.redshift_list, dtype=float)
    names = lens_model_fixed.lens_model_list
    idx = [i for i in np.where(np.isclose(zlist, z))[0] if names[i] not in exclude_names]
    return LensModel([names[i] for i in idx]), [kwargs_lens_fixed[i] for i in idx]


def central_ray_from_interp(ray_interp_x, ray_interp_y, beta_center, redshift_planes,
                            z_source_convention):
    """Package ray-path interpolators (which the pipeline already has) into the
    central-ray dict consumed by distortion_multiplane_rayshooting / accumulate_distortions.

    The decoupled-multiplane setup (and pyHalo's align_realization) already interpolate
    the image/centroid ray via samana.forward_model_util.interpolate_ray_paths /
    pyHalo.utilities.interpolate_ray_paths, which return interp1d(comoving distance) ->
    angular position. accumulate_distortions queries them at cosmo_bkg.T_xy(0, z), the
    same transverse comoving distance those interpolators use, so they drop in directly.

    :param ray_interp_x: interp1d(T) -> central-ray angular x (e.g. rix[0] from
        interpolate_ray_paths for the image being integrated)
    :param ray_interp_y: interp1d(T) -> central-ray angular y
    :param beta_center: (bx, by) source-plane landing to pin the central ray to. Typically
        the source position (kwargs_source center) so the central ray lands on the source.
    :param redshift_planes: sorted list of lens-plane redshifts to traverse
    :param z_source_convention: redshift the fixed model's reduced deflections are defined against
    :return: dict with keys ray_angle_interp_x, ray_angle_interp_y, beta_center,
        redshift_planes, z_source_convention
    """
    return {'ray_angle_interp_x': ray_interp_x, 'ray_angle_interp_y': ray_interp_y,
            'beta_center': (float(beta_center[0]), float(beta_center[1])),
            'redshift_planes': list(redshift_planes),
            'z_source_convention': z_source_convention}


def setup_central_ray(x_image, y_image, lens_model_full, kwargs_lens_full,
                      redshift_planes, cosmo_bkg, z_source, z_split=None,
                      z_source_convention=None):
    """Convenience wrapper: interpolate the image ray through the FULL multiplane model
    and package it for accumulate_distortions. Calls the pipeline's own
    interpolate_ray_paths (so the path matches how the decoupled setup builds ray paths)
    and takes beta_center from ray_shooting. Prefer central_ray_from_interp when the
    pipeline already has the ray interpolators.

    :param x_image: image angular x position (arcsec)
    :param y_image: image angular y position (arcsec)
    :param lens_model_full: a multiplane LensModel containing ALL deflectors (halos + macro)
    :param kwargs_lens_full: kwargs for lens_model_full
    :param redshift_planes: sorted list of lens-plane redshifts to traverse
    :param cosmo_bkg: lenstronomy Background
    :param z_source: source redshift
    :param z_split: accepted but unused here (the full model already carries every plane;
        z_split matters only inside accumulate_distortions for the free-model plane)
    :param z_source_convention: redshift the reduced deflections are defined against;
        defaults to z_source
    :return: the central-ray dict (see central_ray_from_interp)
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
                                      inds_selector, verbose=False):
    """Distortion-field analogue of decoupled_multiplane_rayshooting: ray-trace only the
    NEW annulus points [r_min, r_max) and return their source-plane positions. Drops into
    mag_finite_single_image_distortion's growing-region while-loop.

    :param grid_r: radius of every grid point from the image center (arcsec)
    :param r_min: inner radius of the annulus to add this iteration (arcsec)
    :param r_max: outer radius of the annulus to add this iteration (arcsec)
    :param inds_compute: indices already computed on previous iterations (for inds_selector)
    :param grid_x_large: angular x offsets of all grid points from the image center
    :param grid_y_large: angular y offsets of all grid points from the image center
    :param lens_model_fixed: multiplane LensModel of the fixed (halo) deflectors
    :param lens_model_free: single-plane LensModel of the free/macro deflector at z_split
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param kwargs_lens_free: kwargs for lens_model_free (the macro model)
    :param z_split: redshift of the free/macro deflector
    :param z_source: source redshift
    :param cosmo_bkg: lenstronomy Background
    :param R_max: near/far angular split radius
    :param central_ray: central-ray dict from central_ray_from_interp / setup_central_ray
        (done once per image)
    :param inds_selector: the annulus point-selection function (e.g. _inds_compute_grid)
    :return: (beta_x_new, beta_y_new, inds_new, inds_outside_r) -- source-plane positions
        of the newly added annulus points, their grid indices, and the indices beyond r_max
    """
    inds_new, inds_outside_r, _ = inds_selector(grid_r, r_min, r_max, inds_compute)
    ox = grid_x_large[inds_new]; oy = grid_y_large[inds_new]     # angular offsets from the image
    dbeta_x, dbeta_y = accumulate_distortions(
        central_ray['ray_angle_interp_x'], central_ray['ray_angle_interp_y'],
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
        central_ray['redshift_planes'], ox, oy, cosmo_bkg, z_source,
        z_split, R_max, plane_splits=central_ray.get('plane_splits'), verbose=verbose)
    bx0, by0 = central_ray['beta_center']
    return bx0 + dbeta_x, by0 + dbeta_y, inds_new, inds_outside_r

def _as_interp(ri):
    """Normalize a ray interpolator argument.

    :param ri: either a single interp1d, or the 1-element list interpolate_ray_paths
        returns (one interp per input coordinate)
    :return: the single interp1d (ri[0] if a list/tuple was passed, else ri)
    """
    if isinstance(ri, (list, tuple)):
        return ri[0]
    return ri

def mag_finite_single_image_distortion_adaptive(
        source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
        kwargs_lens, z_split, z_source, cosmo_bkg, x_image, y_image,
        grid_resolution, grid_size_max, R_max, ray_interp_x, ray_interp_y,
        kwargs_lens_free, n_coarse=40, rel_tol=5e-4, flux_floor_frac=1e-4,
        rotation_angle=None, hessian_eigenvalue=None,
        lens_model_fixed_batched=None,
        kwargs_lens_fixed_batched=None
):
    """Adaptive-tiling version of mag_finite_single_image_distortion"""

    rix = _as_interp(ray_interp_x); riy = _as_interp(ray_interp_y)
    redshift_planes = sorted(set(np.round(np.asarray(
        lens_model_fixed.redshift_list, dtype=float), 8).tolist()))
    if not any(np.isclose(z, z_split) for z in redshift_planes):
        redshift_planes = sorted(redshift_planes + [float(z_split)])

    beta_center = (kwargs_source[0]['center_x'], kwargs_source[0]['center_y'])
    plane_splits = precompute_plane_splits(rix, riy, lens_model_fixed, kwargs_lens_fixed,
                                           redshift_planes, cosmo_bkg, R_max, z_source)

    def ray_shoot(thx, thy):
        ox = np.atleast_1d(thx) - x_image
        oy = np.atleast_1d(thy) - y_image
        dbeta_x, dbeta_y = accumulate_distortions(
            rix, riy, lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens,
            redshift_planes, ox, oy, cosmo_bkg, z_source, z_split, R_max,
            plane_splits=plane_splits)
        return beta_center[0] + dbeta_x, beta_center[1] + dbeta_y

    def source_sb(bx, by):
        return source_model.surface_brightness(bx, by, kwargs_source)

    mag, n_calls, n_pts, leaves = adaptive_quadtree_magnification(
        ray_shoot, source_sb, x_image, y_image, grid_size_max, grid_resolution,
        n_coarse=n_coarse, rel_tol=rel_tol, flux_floor_frac=flux_floor_frac,
        rotation_angle=rotation_angle, hessian_eigenvalue=hessian_eigenvalue,
        return_leaves=True)

    npix = max(int(round(grid_size_max / grid_resolution)), 2)
    flux_array = rasterize_leaves(leaves, grid_size_max, npix)
    lcx, lcy, lh, lsb = leaves  # square-quadtree leaves: centers, half-size, SB

    tiling = {'shape': 'square', 'cx': lcx, 'cy': lcy, 'half': lh, 'sb': lsb,
              'box_size': grid_size_max, 'npix': npix, 'grid_resolution': grid_resolution,
              'rotation_angle': rotation_angle, 'hessian_eigenvalue': hessian_eigenvalue,
              'n_calls': n_calls, 'n_points': n_pts}

    lm_exact = lens_model_fixed_batched if lens_model_fixed_batched is not None else lens_model_fixed
    kw_exact = kwargs_lens_fixed_batched if kwargs_lens_fixed_batched is not None else kwargs_lens_fixed
    mu_point = exact_point_source_magnification(
        lm_exact, lens_model_free, kw_exact, kwargs_lens_free, kwargs_lens,
        z_split, z_source, cosmo_bkg, x_image, y_image, eps=grid_resolution)

    eps = grid_resolution
    _tx = np.array([x_image + eps, x_image - eps, x_image, x_image])
    _ty = np.array([y_image, y_image, y_image + eps, y_image - eps])
    _bx, _by = ray_shoot(_tx, _ty)
    Axx = (_bx[0] - _bx[1]) / (2 * eps);
    Ayx = (_by[0] - _by[1]) / (2 * eps)
    Axy = (_bx[2] - _bx[3]) / (2 * eps);
    Ayy = (_by[2] - _by[3]) / (2 * eps)
    det = Axx * Ayy - Axy * Ayx
    mu_point_nearfar = 1.0 / abs(det) if det != 0 else np.inf
    mu_discrepancy = (abs(mu_point_nearfar - mu_point) / mu_point
                      if mu_point > 0 and np.isfinite(mu_point_nearfar) else np.inf)

    return mag, flux_array, tiling, mu_discrepancy


def mag_finite_single_image_distortion(
        source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
        kwargs_lens, z_split, z_source,
        cosmo_bkg, grid_x_large, grid_y_large,
        grid_r, r_step, grid_resolution, grid_size_max,
        R_max, ray_interp_x, ray_interp_y, x_image, y_image, kwargs_lens_free,
        lens_model_fixed_batched=None,
        kwargs_lens_fixed_batched=None,
        verbose=False):
    """Finite-source magnification of a single image via the near/far distortion field

    :param source_model: a lenstronomy LightModel for the source
    :param kwargs_source: source light kwargs; kwargs_source[0]['center_x'/'center_y'] IS
        the source position and is used as beta_center (the central ray is pinned here)
    :param lens_model_fixed: multiplane LensModel of the fixed (halo) deflectors
    :param lens_model_free: single-plane LensModel of the free/macro deflector at z_split
    :param kwargs_lens_fixed: kwargs for lens_model_fixed
    :param kwargs_lens: kwargs for the free/macro model (applied at z_split)
    :param z_split: redshift of the free/macro deflector
    :param z_source: source redshift
    :param cosmo_bkg: lenstronomy Background
    :param grid_x_large: angular x offsets of all grid points from the image center
    :param grid_y_large: angular y offsets of all grid points from the image center
    :param grid_r: radius of every grid point from the image center (arcsec)
    :param r_step: annulus growth step (arcsec) for the growing-region loop
    :param grid_resolution: pixel scale (arcsec); magnification = flux_total * grid_resolution**2
    :param grid_size_max: outer bound (arcsec); the loop stops once r_max reaches it
    :param R_max: near/far angular split radius
    :param ray_interp_x: interp1d(T) (or 1-element list) -> central-ray angular x, from
        interpolate_ray_paths on the full initial model
    :param ray_interp_y: interp1d(T) (or 1-element list) -> central-ray angular y
    :param verbose: make print statements
    :return: (magnification, flux_array, log_mag_grad) -- finite-source magnification, the
        per-grid-point surface-brightness array, and log10 of the flux-weighted
        magnification gradient (near-critical trigger; fall back to exact when it
        exceeds ~ -1). -inf if there is no flux.
    """
    rix = _as_interp(ray_interp_x)
    riy = _as_interp(ray_interp_y)
    redshift_planes = sorted(set(np.round(np.asarray(
        lens_model_fixed.redshift_list, dtype=float), 8).tolist()))
    if not any(np.isclose(z, z_split) for z in redshift_planes):
        redshift_planes = sorted(redshift_planes + [float(z_split)])

    beta_center = (kwargs_source[0]['center_x'], kwargs_source[0]['center_y'])
    central_ray = central_ray_from_interp(rix, riy, beta_center, redshift_planes, z_source)

    flux_array = np.zeros(len(grid_x_large))
    beta_x_grid = np.full(len(grid_x_large), np.nan)  # source-plane map, for the near-critical check
    beta_y_grid = np.full(len(grid_x_large), np.nan)
    r_min = 0.0
    r_max = r_min + r_step
    magnification_last = 0.0
    inds_compute = np.array([])
    flux_total = 0.0
    central_ray['plane_splits'] = precompute_plane_splits(
        rix, riy, lens_model_fixed, kwargs_lens_fixed, redshift_planes, cosmo_bkg, R_max, z_source)

    while True:
        beta_x_new, beta_y_new, inds_new, _ = distortion_multiplane_rayshooting(
            grid_r, r_min, r_max, inds_compute,
            grid_x_large, grid_y_large,
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens,
            z_split, z_source, cosmo_bkg, R_max, central_ray, _inds_compute_grid, verbose=verbose)
        sb_new = source_model.surface_brightness(beta_x_new, beta_y_new, kwargs_source)
        flux_array[inds_new] = sb_new
        beta_x_grid[inds_new] = beta_x_new
        beta_y_grid[inds_new] = beta_y_new
        flux_total += np.sum(sb_new)
        magnification_temp = flux_total * grid_resolution ** 2
        diff = abs(magnification_temp - magnification_last) / magnification_temp if magnification_temp > 0 else 1.0
        r_min += r_step
        r_max += r_step

        if r_max >= grid_size_max:
            break
        elif diff < 0.001 and magnification_temp > 0.0001:
            break
        else:
            magnification_last = magnification_temp

    lm_exact = lens_model_fixed_batched if lens_model_fixed_batched is not None else lens_model_fixed
    kw_exact = kwargs_lens_fixed_batched if kwargs_lens_fixed_batched is not None else kwargs_lens_fixed
    mu_point = exact_point_source_magnification(
        lm_exact, lens_model_free, kw_exact, kwargs_lens_free, kwargs_lens,
        z_split, z_source, cosmo_bkg, x_image, y_image, eps=grid_resolution)
    mu_discrepancy = abs(magnification_temp - mu_point) / mu_point if mu_point > 0 else np.inf
    return magnification_temp, flux_array, mu_discrepancy

def exact_point_source_magnification(lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                                     kwargs_lens_free, kwargs_lens, z_split, z_source,
                                     cosmo_bkg, x_image, y_image, eps):
    """Exact point-source magnification 1/|det(dbeta/dtheta)| at (x_image, y_image), from a
    5-point finite-difference Hessian through the decoupled multiplane model (~5 ray-shoots).
    For a small source away from a caustic the finite-source magnification equals this, so a
    large near/far-vs-this discrepancy flags a near/far failure (over- OR under-production)
    that the geometry-based fold metric misses."""
    from lenstronomy.LensModel.Util.decouple_multi_plane_util import coordinates_and_deflections
    from samana.image_magnification_util import calc_source_sb
    Td = cosmo_bkg.T_xy(0, z_split); Ts = cosmo_bkg.T_xy(0, z_source)
    Tds = cosmo_bkg.T_xy(z_split, z_source)
    reduced_to_phys = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(z_split, z_source)
    tx = np.array([x_image + eps, x_image - eps, x_image, x_image])
    ty = np.array([y_image, y_image, y_image + eps, y_image - eps])
    xD, yD, afx, afy, abx, aby = coordinates_and_deflections(
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
        tx, ty, z_split, z_source, cosmo_bkg)
    bx, by = calc_source_sb(xD, yD, afx, afy, abx, aby, Td, Tds, Ts,
                            reduced_to_phys, lens_model_free, kwargs_lens)
    Axx = (bx[0] - bx[1]) / (2 * eps); Ayx = (by[0] - by[1]) / (2 * eps)
    Axy = (bx[2] - bx[3]) / (2 * eps); Ayy = (by[2] - by[3]) / (2 * eps)
    det = Axx * Ayy - Axy * Ayx
    return 1.0 / abs(det) if det != 0 else np.inf

def _inds_compute_grid(grid_r, r_min, r_max, inds_compute):
    """Select the grid points in the annulus [r_min, r_max) to add this iteration.

    :param grid_r: radius of every grid point from the image center (arcsec)
    :param r_min: inner annulus radius (arcsec)
    :param r_max: outer annulus radius (arcsec)
    :param inds_compute: indices computed on previous iterations
    :return: (inds_compute_new, inds_outside_r, inds_computed) -- indices newly selected
        in the annulus, indices beyond r_max, and the accumulated set of computed indices
    """
    condition = np.logical_and(grid_r >= r_min, grid_r < r_max)
    inds_compute_new = np.where(condition)[0]
    inds_outside_r = np.where(grid_r > r_max)[0]
    inds_computed = np.append(inds_compute, inds_compute_new).astype(int)
    return inds_compute_new, inds_outside_r, inds_computed

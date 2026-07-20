
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

def _beta_at_batch(lens_model_fixed, kwargs_lens_fixed, lens_model_free, kwargs_lens_free,
                   kwargs_lens, xpts, ypts, z_split, z_source, cosmo_bkg):
    Td = cosmo_bkg.T_xy(0, z_split)
    Ts = cosmo_bkg.T_xy(0, z_source)
    Tds = cosmo_bkg.T_xy(z_split, z_source)
    reduced_to_phys = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(z_split, z_source)
    xpts = np.atleast_1d(xpts).astype(float)
    ypts = np.atleast_1d(ypts).astype(float)
    xD, yD, a_fg_x, a_fg_y, a_bg_x, a_bg_y = coordinates_and_deflections(
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
        xpts, ypts, z_split, z_source, cosmo_bkg)
    return calc_source_sb(xD, yD, a_fg_x, a_fg_y, a_bg_x, a_bg_y,
                          Td, Tds, Ts, reduced_to_phys, lens_model_free, kwargs_lens)


def _fit_stencil(d, h, order):
    """d: (2, 9) far-field beta difference on the stencil; returns dbeta, J, K."""
    dbeta = d[:, 0].copy()
    J = np.empty((2, 2))
    K = np.zeros((2, 2, 2))
    for c in range(2):
        f = d[c]
        J[c, 0] = (f[1] - f[2]) / (2 * h)
        J[c, 1] = (f[3] - f[4]) / (2 * h)
        if order >= 2:
            K[c, 0, 0] = (f[1] - 2 * f[0] + f[2]) / h ** 2
            K[c, 1, 1] = (f[3] - 2 * f[0] + f[4]) / h ** 2
            K[c, 0, 1] = K[c, 1, 0] = (f[5] - f[6] - f[7] + f[8]) / (4 * h ** 2)
    return dbeta, J, K

def _mag_single_image_shifted(source_model, kwargs_source, lens_model_fixed, lens_model_free,
                              kwargs_lens_fixed, kwargs_lens_free, kwargs_lens,
                              z_split, z_source, cosmo_bkg, x_image, y_image,
                              grid_x_large, grid_y_large, grid_r, r_step,
                              grid_resolution, grid_size_max, zlens, zsource,
                              dbeta_center=(0.0, 0.0), J=None, K=None):
    """Annulus-based finite magnification with the near model plus the far-field
    correction beta += dbeta + J@dth + 0.5 dth^T K dth."""
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
    r_min, r_max_ann = 0.0, r_step
    mag_last, flux_total = 0.0, 0.0
    inds_compute = np.array([])
    Td = cosmo_bkg.T_xy(0, zlens);
    Ts = cosmo_bkg.T_xy(0, zsource)
    Tds = cosmo_bkg.T_xy(zlens, zsource)
    reduced_to_phys = cosmo_bkg.d_xy(0, zsource) / cosmo_bkg.d_xy(zlens, zsource)
    dbx, dby = dbeta_center
    while True:
        beta_x_new, beta_y_new, inds_new, _ = decoupled_multiplane_rayshooting(
            grid_r, r_min, r_max_ann, inds_compute,
            grid_x_large, grid_y_large, x_image, y_image,
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
            z_split, z_source, cosmo_bkg, xD, yD, a_fg_x, a_fg_y, a_bg_x, a_bg_y,
            Td, kwargs_lens, reduced_to_phys, Ts, Tds)
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
        r_max_ann += r_step
        if r_max_ann >= grid_size_max:
            break
        elif diff < 0.001 and mag_temp > 0.0001:
            break
        else:
            mag_last = mag_temp
    return mag_temp, flux_array

def _build_near_model_with_shifts(lens_model_fixed, kwargs_lens_fixed, keep, z_split):
    """Subset model with SHIFT profiles at: a leading plane before everything, every
    populated plane, and a plane right behind z_split (start of the background leg)."""
    idx = np.where(keep)[0]
    z_all = np.asarray(lens_model_fixed.redshift_list, dtype=float)
    z_kept = np.round(z_all[idx], 10)
    kept_planes = np.unique(z_kept)
    z_pre = 0.5 * float(np.min(z_all))
    z_bg_all = z_all[z_all > z_split]
    if len(z_bg_all):
        z_bg0 = z_split + 0.25 * (float(np.min(z_bg_all)) - z_split)
    else:
        z_bg0 = 0.5 * (z_split + lens_model_fixed.z_source)
    shift_planes = np.unique(np.concatenate([[z_pre, z_bg0], kept_planes]))
    names, zs, kw, shift_index = [], [], [], {}
    for zp in shift_planes:
        shift_index[zp] = len(names)
        names.append('SHIFT'); zs.append(float(zp))
        kw.append({'alpha_x': 0.0, 'alpha_y': 0.0})
        for i in idx[z_kept == zp]:
            names.append(lens_model_fixed.lens_model_list[i])
            zs.append(lens_model_fixed.redshift_list[i])
            kw.append(kwargs_lens_fixed[i])
    lm = LensModel(lens_model_list=names, lens_redshift_list=zs,
                   z_source=lens_model_fixed.z_source, cosmo=lens_model_fixed.cosmo,
                   multi_plane=True)
    planes_fg = shift_planes[shift_planes <= z_split]
    planes_bg = shift_planes[shift_planes > z_split]
    return lm, kw, shift_index, planes_fg, planes_bg


def _leg_trajectory(mp, kwargs, x0, y0, ax0, ay0, z_start, stops):
    """Comoving (x, y) at arrival at each stop. include_z_start=True from the second
    segment on, so every plane's deflection is applied exactly once and recorded
    positions are pre-deflection at arrival."""
    x = np.atleast_1d(np.array(x0, dtype=float) * 1.0)
    y = np.atleast_1d(np.array(y0, dtype=float) * 1.0)
    ax = np.atleast_1d(float(ax0)); ay = np.atleast_1d(float(ay0))
    z_prev = z_start
    traj = {}
    started = False
    for zk in stops:
        if zk <= z_prev + 1e-12 and started:
            # zero-length segment (duplicate stop): do not re-apply plane deflections
            traj[zk] = (float(x[0]), float(y[0]))
            continue
        x, y, ax, ay = mp.ray_shooting_partial_comoving(
            x, y, ax, ay, z_prev, zk, kwargs, include_z_start=started,
            check_convention=False)
        traj[zk] = (float(x[0]), float(y[0]))
        z_prev = zk
        started = True
    return traj, (float(x[0]), float(y[0]), float(ax[0]), float(ay[0]))

def _register_leg(lm_near, kw_near, shift_index, planes, traj_full, start_near, z_start,
                  z_end, cosmo_bkg, zs_model, REGISTRATION_PASSES=3):
    """Calibrate SHIFTs on one leg so the near trajectory matches traj_full at every
    plane in `planes`. A shift at plane k adds the same slope to all downstream
    segments, so each correction is the INCREMENT of the segment-slope mismatch."""
    x0, y0, ax0, ay0 = start_near
    stops = list(planes) + [z_end]
    Tz = {zp: cosmo_bkg.T_xy(0, zp) for zp in stops}
    # empirical response at the first plane of this leg
    probe = 1e-4
    z0, z1 = stops[0], stops[1]
    dT01 = Tz[z1] - Tz[z0]
    traj_b, _ = _leg_trajectory(lm_near.lens_model, kw_near, x0, y0, ax0, ay0, z_start, stops)
    kw_near[shift_index[z0]]['alpha_x'] += probe
    traj_a, _ = _leg_trajectory(lm_near.lens_model, kw_near, x0, y0, ax0, ay0, z_start, stops)
    kw_near[shift_index[z0]]['alpha_x'] -= probe
    resp0 = (((traj_a[z1][0] - traj_a[z0][0]) - (traj_b[z1][0] - traj_b[z0][0])) / dT01) / probe
    fac0 = cosmo_bkg.d_xy(0, zs_model) / cosmo_bkg.d_xy(z0, zs_model)
    for _ in range(REGISTRATION_PASSES):
        traj_near, _ = _leg_trajectory(lm_near.lens_model, kw_near, x0, y0, ax0, ay0,
                                       z_start, stops)
        mx_prev, my_prev = 0.0, 0.0
        for k, zp in enumerate(planes):
            zn = stops[k + 1]
            dT = Tz[zn] - Tz[zp]
            if dT <= 0:
                # zero-length segment (e.g. a kept plane exactly at the leg boundary):
                # this plane's shift has no lever arm in this leg -- leave it at zero
                continue
            mx = ((traj_full[zn][0] - traj_full[zp][0]) -
                  (traj_near[zn][0] - traj_near[zp][0])) / dT
            my = ((traj_full[zn][1] - traj_full[zp][1]) -
                  (traj_near[zn][1] - traj_near[zp][1])) / dT
            resp_k = resp0 * (cosmo_bkg.d_xy(0, zs_model) / cosmo_bkg.d_xy(zp, zs_model)) / fac0
            si = shift_index[zp]
            kw_near[si]['alpha_x'] += (mx - mx_prev) / resp_k
            kw_near[si]['alpha_y'] += (my - my_prev) / resp_k
            mx_prev, my_prev = mx, my
    return kw_near

def _register(lm_near, kw_near, shift_index, planes_fg, planes_bg,
              lens_model_fixed, kwargs_lens_fixed, lens_model_free, kwargs_lens_free,
              theta_x, theta_y, z_split, z_source, cosmo_bkg):
    """Two-leg registration replicating coordinates_and_deflections' propagation."""
    zs_model = lens_model_fixed.z_source
    Td = cosmo_bkg.T_xy(0, z_split)
    reduced_to_phys = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(z_split, z_source)

    # ---- leg 1: 0 -> z_split (fixed model only) ----
    stops_fg = list(planes_fg) + [z_split]
    traj_full_fg, (xDf, yDf, axf, ayf) = _leg_trajectory(
        lens_model_fixed.lens_model, kwargs_lens_fixed, 0.0, 0.0, theta_x, theta_y,
        0.0, stops_fg)
    kw_near = _register_leg(lm_near, kw_near, shift_index, list(planes_fg),
                            traj_full_fg, (0.0, 0.0, theta_x, theta_y), 0.0,
                            z_split, cosmo_bkg, zs_model)
    # near leg-1 end state (positions now match; angles may differ slightly)
    _, (xDn, yDn, axn, ayn) = _leg_trajectory(
        lm_near.lens_model, kw_near, 0.0, 0.0, theta_x, theta_y, 0.0, stops_fg)

    if len(planes_bg) == 0:
        return kw_near

    # ---- leg 2: z_split -> z_source, angle = alpha_fg - alpha_main (as in
    # coordinates_and_deflections) ----
    amx, amy = lens_model_free.alpha(xDf / Td, yDf / Td, kwargs_lens_free)
    ang_fx = axf - float(np.atleast_1d(amx)[0]) * reduced_to_phys
    ang_fy = ayf - float(np.atleast_1d(amy)[0]) * reduced_to_phys
    amxn, amyn = lens_model_free.alpha(xDn / Td, yDn / Td, kwargs_lens_free)
    ang_nx = axn - float(np.atleast_1d(amxn)[0]) * reduced_to_phys
    ang_ny = ayn - float(np.atleast_1d(amyn)[0]) * reduced_to_phys

    stops_bg = list(planes_bg) + [z_source]
    traj_full_bg, _ = _leg_trajectory(lens_model_fixed.lens_model, kwargs_lens_fixed,
                                      xDf, yDf, ang_fx, ang_fy, z_split, stops_bg)
    kw_near = _register_leg(lm_near, kw_near, shift_index, list(planes_bg),
                            traj_full_bg, (xDn, yDn, ang_nx, ang_ny), z_split,
                            z_source, cosmo_bkg, zs_model)
    return kw_near

def magnification_finite_decoupled_nearfar(
        source_model, kwargs_source, x_image, y_image,
        lens_model_init, kwargs_lens_init, kwargs_lens, index_lens_split,
        grid_size_list, grid_resolution, R_max, halo_masses,
        farfield_order=2, farfield_stencil_h=5e-3, source_x=None, source_y=None,
        grid_increment_factor=15.0,
        setup_decoupled_multiplane_lens_model_output=None,
        magnification_method='ELLIPTICAL_APERTURE',
        rotation_angle_list=None, hessian_eigenvalue_list=None):
    """Near/far split with background culling and two-leg ray registration.
    Sheets and other centerless profiles are always culled INTO the far field
    together with the halos (the registration shifts make this safe)."""

    M_ref = 1e8
    mass_exponent = 0.5
    floor_buffer = 3.0

    if setup_decoupled_multiplane_lens_model_output is None:
        (lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
         z_source, z_split, cosmo_bkg) = setup_lens_model(
            lens_model_init, kwargs_lens_init, index_lens_split)
    else:
        (lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
         z_source, z_split, cosmo_bkg) = setup_decoupled_multiplane_lens_model_output

    z_list = np.asarray(lens_model_fixed.redshift_list, dtype=float)
    cx = np.array([kw.get('center_x', np.nan) if isinstance(kw, dict) else np.nan
                   for kw in kwargs_lens_fixed])
    cy = np.array([kw.get('center_y', np.nan) if isinstance(kw, dict) else np.nan
                   for kw in kwargs_lens_fixed])
    has_center = np.isfinite(cx) & np.isfinite(cy)

    # per-halo cull radius, floored at floor_buffer * grid half-size
    n_fixed = len(kwargs_lens_fixed)
    if halo_masses is not None:
        M = np.full(n_fixed, np.nan)
        M[:len(halo_masses)] = halo_masses
        with np.errstate(invalid='ignore'):
            rmax_per_halo = R_max * (M / M_ref) ** mass_exponent
        rmax_per_halo = np.where(np.isfinite(rmax_per_halo), rmax_per_halo, R_max)
    else:
        rmax_per_halo = np.full(n_fixed, float(R_max))
    rmax_per_halo = np.maximum(floor_buffer * np.max(grid_size_list) / 2, rmax_per_halo)

    # source estimate for the background ray path
    if source_x is None or source_y is None:
        bx, by = _beta_at_batch(lens_model_fixed, kwargs_lens_fixed, lens_model_free,
                                kwargs_lens_free, kwargs_lens, x_image, y_image,
                                z_split, z_source, cosmo_bkg)
        source_x, source_y = np.mean(bx), np.mean(by)

    Td = cosmo_bkg.T_xy(0, z_split)
    Ts = cosmo_bkg.T_xy(0, z_source)
    Tz = np.array([cosmo_bkg.T_xy(0, z) for z in z_list])
    behind = z_list > z_split
    w = np.zeros(n_fixed)
    w[behind] = (Tz[behind] - Td) / (Ts - Td)

    # stencils for all images through the full model, one call
    h = farfield_stencil_h
    ox = np.array([0, 1, -1, 0, 0, 1, 1, -1, -1], float) * h
    oy = np.array([0, 0, 0, 1, -1, 1, -1, 1, -1], float) * h
    n_img = len(x_image)
    xs_all = np.concatenate([x_image[j] + ox for j in range(n_img)])
    ys_all = np.concatenate([y_image[j] + oy for j in range(n_img)])
    bxf, byf = _beta_at_batch(lens_model_fixed, kwargs_lens_fixed, lens_model_free,
                              kwargs_lens_free, kwargs_lens, xs_all, ys_all,
                              z_split, z_source, cosmo_bkg)

    magnifications, flux_arrays = [], []
    for j in range(n_img):
        # near/far split for this image (halos by distance to ray path; sheets always far)
        px = np.full(n_fixed, x_image[j]); py = np.full(n_fixed, y_image[j])
        px[behind] = (x_image[j]*Td + w[behind]*(source_x*Ts - x_image[j]*Td)) / Tz[behind]
        py[behind] = (y_image[j]*Td + w[behind]*(source_y*Ts - y_image[j]*Td)) / Tz[behind]
        dist = np.hypot(cx - px, cy - py)
        far = (has_center & (dist > rmax_per_halo)) | (~has_center)
        keep = ~far

        # near model with registered SHIFT profiles
        lm_near, kw_near, shift_index, planes_fg, planes_bg = \
            _build_near_model_with_shifts(lens_model_fixed, kwargs_lens_fixed, keep, z_split)
        kw_near = _register(lm_near, kw_near, shift_index, planes_fg, planes_bg,
                            lens_model_fixed, kwargs_lens_fixed, lens_model_free,
                            kwargs_lens_free, x_image[j], y_image[j], z_split,
                            z_source, cosmo_bkg)

        # far-field correction from the registered difference field
        sl = slice(9*j, 9*(j+1))
        bxn, byn = _beta_at_batch(lm_near, kw_near, lens_model_free, kwargs_lens_free,
                                  kwargs_lens, xs_all[sl], ys_all[sl],
                                  z_split, z_source, cosmo_bkg)
        d = np.array([bxf[sl] - bxn, byf[sl] - byn])
        dbeta_center, J, K = _fit_stencil(d, h, farfield_order)

        # magnification
        grid_x_large, grid_y_large, interp_points_large, npix_large = setup_grids(
            grid_size_list[j], grid_resolution, 0.0, 0.0)
        grid_x_large = grid_x_large.ravel(); grid_y_large = grid_y_large.ravel()
        if magnification_method == 'ELLIPTICAL_APERTURE':
            gx, gy = util.rotate(grid_x_large, grid_y_large, rotation_angle_list[j])
            grid_r = np.hypot(gx, gy / hessian_eigenvalue_list[j]).ravel()
        else:
            grid_r = np.hypot(grid_x_large, grid_y_large).ravel()
        r_step = grid_size_list[j] / grid_increment_factor
        mag, flux_array = _mag_single_image_shifted(
            source_model, kwargs_source, lm_near, lens_model_free, kw_near,
            kwargs_lens_free, kwargs_lens, z_split, z_source, cosmo_bkg,
            x_image[j], y_image[j], grid_x_large, grid_y_large, grid_r, r_step,
            grid_resolution, grid_size_list[j], z_split, z_source,
            dbeta_center=dbeta_center, J=J, K=K)
        magnifications.append(mag)
        flux_arrays.append(flux_array.reshape(npix_large, npix_large))
    return np.array(magnifications), flux_arrays

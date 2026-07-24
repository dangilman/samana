from lenstronomy.LightModel.light_model import LightModel
from copy import deepcopy
from lenstronomy.Util import util
from lenstronomy.LensModel.Util.decouple_multi_plane_util import (
    setup_grids, coordinates_and_deflections, setup_lens_model,
)

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
                                   grid_size_list, grid_resolution_list,
                                   grid_increment_factor=20.0,
                                   setup_decoupled_multiplane_lens_model_output=None,
                                   magnification_method='CIRCULAR_APERTURE',
                                   rotation_angle_list=None,
                                   hessian_eigenvalue_list=None,
                                   lens_model_batch=None,
                                   kwargs_lens_batch=None,
                                   halo_masses=None,
                                   setup_decoupled_multiplane_lens_model_output_batch=None,
                                   fallback='ELLIPTICAL_APERTURE',
                                   MU_TOLERANCE=0.05,
                                   verbose=False):
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

    if setup_decoupled_multiplane_lens_model_output_batch is not None:
        (lens_model_fixed_batch, _, kwargs_lens_fixed_batch,
         _, _, _, _) = setup_decoupled_multiplane_lens_model_output_batch
    else:
        lens_model_fixed_batch = lens_model_fixed
        kwargs_lens_fixed_batch = kwargs_lens_fixed

    magnifications = []
    flux_arrays = []

    for j, (x_img, y_img) in enumerate(zip(x_image, y_image)):

        grid_x_large, grid_y_large, interp_points_large, npix_large = setup_grids(grid_size_list[j],
                                                                                  grid_resolution_list[j],
                                                                                  0.0, 0.0)
        grid_x_large = grid_x_large.ravel()
        grid_y_large = grid_y_large.ravel()

        if magnification_method == 'ADAPTIVE':
            # batching is slower here, don't use it
            mag, flux_array, tiling = mag_finite_single_image_adaptive(
                source_model, kwargs_source, lens_model_fixed, lens_model_free,
                kwargs_lens_fixed, kwargs_lens_free, kwargs_lens,
                z_split, z_source, cosmo_bkg, x_img, y_img, grid_resolution_list[j],
                grid_size_list[j], z_split, z_source, rotation_angle_list[j],
                hessian_eigenvalue_list[j],
                n_coarse=40,
                rel_tol=5e-4,
                flux_floor_frac=1e-4)
            magnifications.append(mag)
            flux_arrays.append([flux_array, tiling])

        elif magnification_method in ['CIRCULAR_APERTURE', 'ELLIPTICAL_APERTURE']:
            if magnification_method == 'ELLIPTICAL_APERTURE':
                grid_x, grid_y = util.rotate(grid_x_large, grid_y_large,
                                             rotation_angle_list[j])
                grid_r = np.hypot(grid_x, grid_y / hessian_eigenvalue_list[j]).ravel()
            else:
                grid_r = np.hypot(grid_x_large, grid_y_large).ravel()
            r_step = grid_size_list[j] / grid_increment_factor
            # batching is slower here, don't use it
            mag, flux_array = mag_finite_single_image(
                source_model, kwargs_source, lens_model_fixed,
                lens_model_free, kwargs_lens_fixed,
                kwargs_lens_free, kwargs_lens, z_split, z_source,
                cosmo_bkg, x_img, y_img, grid_x_large, grid_y_large,
                grid_r, r_step, grid_resolution_list[j], grid_size_list[j],
                z_split, z_source
            )
            magnifications.append(mag)
            flux_arrays.append(flux_array.reshape(npix_large, npix_large))
        elif magnification_method in ['NEAR_FAR_SPLITTING', 'NEAR_FAR_SPLITTING_ADAPTIVE']:

            from samana.image_magnification_near_far import (mag_finite_single_image_distortion,
                                                             mag_finite_single_image_distortion_adaptive)
            from samana.forward_model_util import interpolate_ray_paths
            R_max_0 = 0.4
            if halo_masses is None:
                R_max = R_max_0
            else:
                R_max = np.maximum(R_max_0 * (np.array(halo_masses) / 10 ** 8) ** 0.333,
                                   2.0*grid_size_list[j])
                R_max[np.where(np.log10(halo_masses) > 9)[0]] = 1e9
            grid_r = np.hypot(grid_x_large, grid_y_large).ravel()
            r_step = grid_size_list[j] / grid_increment_factor
            # batching is faster here, use it
            ray_interp_x_list, ray_interp_y_list = interpolate_ray_paths([x_img], [y_img],
                                                               lens_model_batch, kwargs_lens_batch, z_source,
                                                               terminate_at_source=False
                                                               )
            tiling = None
            if magnification_method == 'NEAR_FAR_SPLITTING':
                mag, flux_array, mu_discrepancy = mag_finite_single_image_distortion(
                    source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                    kwargs_lens, z_split, z_source,
                    cosmo_bkg, grid_x_large, grid_y_large,
                    grid_r, r_step, grid_resolution_list[j], grid_size_list[j],
                    R_max, ray_interp_x_list[0], ray_interp_y_list[0], x_img, y_img, kwargs_lens_free,
                    lens_model_fixed_batched=lens_model_fixed_batch,
                    kwargs_lens_fixed_batched=kwargs_lens_fixed_batch,
                    verbose=verbose
                )

            else:
                mag, flux_array, tiling, mu_discrepancy = mag_finite_single_image_distortion_adaptive(
                    source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                    kwargs_lens, z_split, z_source, cosmo_bkg, x_img, y_img,
                    grid_resolution_list[j], grid_size_list[j], R_max, ray_interp_x_list[0], ray_interp_y_list[0],
                    kwargs_lens_free,
                    n_coarse=40,
                    rel_tol=5e-4,
                    flux_floor_frac=1e-4,
                    rotation_angle=rotation_angle_list[j],
                    hessian_eigenvalue=hessian_eigenvalue_list[j],
                    lens_model_fixed_batched=lens_model_fixed_batch,
                    kwargs_lens_fixed_batched=kwargs_lens_fixed_batch
                )
            if verbose: print('point-source mag discrepancy: ', mu_discrepancy)
            # FLAGS IMAGES WHERE APPROXIMATION BREAKS DOWN
            if mu_discrepancy > MU_TOLERANCE:
                if verbose:
                    print('image '+str(j+1)+' exceeds threshold '+str(MU_TOLERANCE)+', using exact ray tracing')

                if fallback == 'ADAPTIVE':
                    mag, flux_array, tiling = mag_finite_single_image_adaptive(
                        source_model, kwargs_source, lens_model_fixed_batch, lens_model_free,
                        kwargs_lens_fixed_batch, kwargs_lens_free, kwargs_lens,
                        z_split, z_source, cosmo_bkg, x_img, y_img, grid_resolution_list[j],
                        grid_size_list[j], z_split, z_source, rotation_angle_list[j],
                        hessian_eigenvalue_list[j],
                        n_coarse=40,
                        rel_tol=5e-4,
                        flux_floor_frac=1e-4)
                else:

                    if hessian_eigenvalue_list is not None:
                        grid_x, grid_y = util.rotate(grid_x_large, grid_y_large,
                                                     rotation_angle_list[j])
                        grid_r = np.hypot(grid_x, grid_y / hessian_eigenvalue_list[j]).ravel()
                    else:
                        grid_r = np.hypot(grid_x_large, grid_y_large).ravel()
                    r_step = grid_size_list[j] / grid_increment_factor
                    mag, flux_array = mag_finite_single_image(source_model, kwargs_source, lens_model_fixed_batch,
                                                              lens_model_free, kwargs_lens_fixed_batch,
                                                              kwargs_lens_free, kwargs_lens, z_split, z_source,
                                                              cosmo_bkg, x_img, y_img, grid_x_large, grid_y_large,
                                                              grid_r, r_step, grid_resolution_list[j],
                                                              grid_size_list[j],
                                                              z_split, z_source)
                    tiling = None  # don't return tiling if we use the fallback

            magnifications.append(mag)
            if tiling is not None:
                flux_arrays.append([flux_array, tiling])
            else:
                flux_arrays.append(flux_array.reshape(npix_large, npix_large))

        else:
            raise Exception('magnification_method must be either CIRCULAR_APERTURE, ELLIPTICAL_APERTURE, or ADAPTIVE. '
                            'You specified magnification_method '+str(magnification_method))

    return np.array(magnifications), flux_arrays


"""
Edge-refining breadth-first quadtree finite-source magnification.

mag = integral over the box of SB(beta(theta)) d^2theta.

Algorithm (as specified):
  0. Coarse scan: a grid at 1/reduction res -- each pixel is a Cell. One batched
     ray-shoot; record SB at each cell center.
  1. Split every active Cell into 4 children; ray-shoot all children in ONE batched
     call (breadth-first -> one call per level, which is what the cost model rewards:
     per-call overhead ~8ms dominates, so few big calls beats many small ones).
  2. Refine a Cell further iff it has FLUX and a LARGE CHANGE on subdivision:
        flux gate:   max(SB over the cell) > flux_floor
        change gate: |sum(child estimates) - parent estimate| > thresh
     Empty cells (no flux) and smooth cells (no change) are accepted as-is; only
     edges / under-resolved peaks keep splitting.
  3. Iterate until no cell needs refining, or cells reach the base resolution.

The magnification is the sum of accepted-leaf estimates (each = cell-center SB x cell
area), using the finer child estimate for a cell whenever it was subdivided.

The integrator is lens-agnostic: pass ray_shoot(thx,thy)->(bx,by) and
source_sb(bx,by)->SB. `n_calls` (levels) is returned alongside the eval count because
that is the real cost driver in the decoupled model.
"""
import numpy as np

class _Counter:
    def __init__(self, ray_shoot):
        self._f = ray_shoot; self.n_pts = 0; self.n_calls = 0

    def __call__(self, thx, thy):
        thx = np.atleast_1d(thx); thy = np.atleast_1d(thy)
        self.n_pts += thx.size; self.n_calls += 1
        return self._f(thx, thy)

def _aperture_radius(cx, cy, rotation_angle=None, hessian_eigenvalue=None):
    """Aperture metric radius for offsets (cx, cy) from the image center.
    Circle (default) -> hypot(cx, cy). Ellipse (both args given) -> rotate by
    rotation_angle then hypot(x_rot, y_rot / hessian_eigenvalue), matching samana's
    ELLIPTICAL_APERTURE grid_r. A point is inside the aperture if this <= box_size/2."""
    if rotation_angle is None or hessian_eigenvalue is None:
        return np.hypot(cx, cy)
    ca = np.cos(rotation_angle); sa = np.sin(rotation_angle)
    xr = cx * ca + cy * sa               # lenstronomy util.rotate convention
    yr = -cx * sa + cy * ca
    return np.hypot(xr, yr / hessian_eigenvalue)


def adaptive_quadtree_magnification(
        ray_shoot, source_sb, x_image, y_image, box_size, grid_resolution,
        n_coarse, rel_tol, flux_floor_frac, max_depth=None,
        empty_rescan=True, rotation_angle=None, hessian_eigenvalue=None,
        return_leaves=False):
    """
    :param ray_shoot: callable (thx, thy) -> (beta_x, beta_y)  [absolute angles]
    :param source_sb: callable (beta_x, beta_y) -> surface brightness
    :param x_image, y_image: box center (image position), arcsec
    :param box_size: full box width, arcsec
    :param grid_resolution: target finest pixel scale (refinement stops here)
    :param n_coarse: number of coarse cells PER SIDE at level 0 (coarse cell =
        box_size / n_coarse). Scale-invariant: ~20 always gives enough resolution to
        catch and resolve a pre-image (~box/5) while staying cheap (~n_coarse^2 points,
        one call). Prefer this over a "reduction * base-pixel" rule, which for a small
        box collapses to just a handful of cells across the whole box.
    :param rel_tol: PER-CELL accuracy knob -- refine a cell while subdividing it changes
        its own estimate by more than rel_tol, i.e. |child - parent| / |child| > rel_tol
    :param flux_floor_frac: a cell must have SB > flux_floor_frac * peak_SB somewhere
        to be eligible for refinement (the "there is flux" gate)
    :param max_depth: cap on refinement levels; default set so the finest cell reaches
        ~grid_resolution from the coarse cell
    :param empty_rescan: if the coarse scan finds no flux, halve the coarse cell and
        rescan (up to a few times) -- guards against skipping a tiny low-mag pre-image.
    :param rotation_angle, hessian_eigenvalue: if BOTH given, the aperture is the ELLIPSE
        hypot(x_rot, y_rot/hessian_eigenvalue) <= box_size/2 (rotated by rotation_angle),
        matching samana's ELLIPTICAL_APERTURE. If either is None the aperture is the
        CIRCLE of radius box_size/2 inscribed in the box (the default). Cells whose
        center falls outside the aperture are dropped (not evaluated, not integrated).
    :param return_leaves: also return accepted-leaf (cx, cy, half, sb) for rasterizing
    :return: (magnification, n_calls, n_points[, leaves])
    """
    rs = _Counter(ray_shoot)
    r_ap = box_size / 2.0

    def in_aperture(cx, cy):
        return _aperture_radius(cx, cy, rotation_angle, hessian_eigenvalue) <= r_ap

    def coarse_scan(n0):
        n0 = max(int(n0), 2)
        cell = box_size / n0
        c = (np.arange(n0) + 0.5) * cell - box_size / 2.0
        cxg, cyg = np.meshgrid(c, c)
        cx = cxg.ravel(); cy = cyg.ravel()
        bx, by = rs(cx + x_image, cy + y_image)
        return cx, cy, source_sb(bx, by), cell

    n0 = max(int(round(n_coarse)), 2)
    cx, cy, sbc, cell0 = coarse_scan(n0)
    # empty-scan fallback: if nothing lit up, the grid may have skipped the pre-image
    tries = 0
    while empty_rescan and float(np.max(sbc)) <= 0.0 and cell0 > grid_resolution and tries < 4:
        n0 *= 2
        cx, cy, sbc, cell0 = coarse_scan(n0)
        tries += 1

    half = np.full(cx.shape, cell0 / 2.0)
    if max_depth is None:
        max_depth = int(np.ceil(np.log2(max(cell0 / grid_resolution, 2))))

    peak = 1e-300           # tracked as the brightest sample seen (updated from children)

    leaf_cx, leaf_cy, leaf_h, leaf_sb = [], [], [], []
    total = 0.0

    for level in range(max_depth + 1):
        if cx.size == 0:
            break
        # aperture mask: drop cells whose center is outside the circle/ellipse
        inside = in_aperture(cx, cy)
        if not np.all(inside):
            cx = cx[inside]; cy = cy[inside]; sbc = sbc[inside]; half = half[inside]
        if cx.size == 0:
            break
        area = (2.0 * half) ** 2
        q = half / 2.0
        # 4 children per cell, one batched call
        childx = np.concatenate([cx - q, cx + q, cx - q, cx + q])
        childy = np.concatenate([cy - q, cy - q, cy + q, cy + q])
        bxc, byc = rs(childx + x_image, childy + y_image)
        sbch = source_sb(bxc, byc).reshape(4, cx.size)

        # keep the flux floor relative to the brightest sample seen so far (the coarse
        # scan can underestimate the peak; the denser child samples correct it)
        peak = max(peak, float(sbch.max()))
        flux_floor = flux_floor_frac * peak

        coarse_est = sbc * area                       # parent midpoint estimate
        fine_est = sbch.sum(axis=0) * (area / 4.0)     # 4-child midpoint estimate
        disagree = np.abs(fine_est - coarse_est)

        has_flux = np.maximum(sbc, sbch.max(axis=0)) > flux_floor
        # PER-CELL relative change on subdivision: refine a cell while subdividing it
        # changes ITS OWN estimate by more than rel_tol (the "large derivative" test).
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_change = disagree / np.maximum(np.abs(fine_est), 1e-300)
        at_base = (2.0 * q) <= grid_resolution * 1.0001    # children would be <= base res
        if level >= max_depth:
            refine = np.zeros(cx.size, bool)
        elif level == 0:
            # every coarse cell is subdivided at least once (the coarse scan can step
            # over a compact pre-image); the flux/change gates only act from level 1 on
            refine = ~at_base
        else:
            refine = has_flux & (rel_change > rel_tol) & (~at_base)

        acc = ~refine
        total += float(np.sum(fine_est[acc]))
        if return_leaves and np.any(acc):
            leaf_cx.append(cx[acc]); leaf_cy.append(cy[acc])
            leaf_h.append(half[acc]); leaf_sb.append(sbch[:, acc].mean(axis=0))
        if not np.any(refine):
            break

        r = np.where(refine)[0]
        qc = q[r]
        cx = np.concatenate([cx[r] - qc, cx[r] + qc, cx[r] - qc, cx[r] + qc])
        cy = np.concatenate([cy[r] - qc, cy[r] - qc, cy[r] + qc, cy[r] + qc])
        sbc = np.concatenate([sbch[0, r], sbch[1, r], sbch[2, r], sbch[3, r]])
        half = np.concatenate([qc, qc, qc, qc])

    if return_leaves:
        if leaf_cx:
            leaves = (np.concatenate(leaf_cx), np.concatenate(leaf_cy),
                      np.concatenate(leaf_h), np.concatenate(leaf_sb))
        else:
            z = np.array([]); leaves = (z, z, z, z)
        return total, rs.n_calls, rs.n_pts, leaves
    return total, rs.n_calls, rs.n_pts


def rasterize_leaves(leaves, box_size, npix):
    """Paint accepted quadtree leaves onto an (npix, npix) image. Leaf edges are
    rounded to pixel boundaries so adjacent leaves share the same integer edge --
    no fp cracks at cell seams (the box center is a seam at every level)."""
    lcx, lcy, lh, lsb = leaves
    img = np.zeros((npix, npix))
    px = box_size / npix
    half_box = box_size / 2.0
    for k in range(len(lcx)):
        i0 = max(int(round((lcx[k] - lh[k] + half_box) / px)), 0)
        i1 = min(int(round((lcx[k] + lh[k] + half_box) / px)), npix)
        j0 = max(int(round((lcy[k] - lh[k] + half_box) / px)), 0)
        j1 = min(int(round((lcy[k] + lh[k] + half_box) / px)), npix)
        img[j0:j1, i0:i1] = lsb[k]
    return img


# ---------------------------------------------------------------------------
# samana drop-in: replaces mag_finite_single_image_adaptive (same call signature)
# ---------------------------------------------------------------------------

def mag_finite_single_image_adaptive(
        source_model, kwargs_source, lens_model_fixed, lens_model_free, kwargs_lens_fixed,
        kwargs_lens_free, kwargs_lens, z_split, z_source,
        cosmo_bkg, x_image, y_image, grid_resolution, grid_size_max, zlens, zsource,
        rotation_angle=None, hessian_eigenvalue=None,
        n_coarse=16, rel_tol=1e-3, flux_floor_frac=1e-3):
    """Edge-refining quadtree magnification through the exact decoupled multiplane model.
    Drop-in for samana.image_magnification_util.mag_finite_single_image_adaptive.

    :param rotation_angle, hessian_eigenvalue: per-image ELLIPSE aperture parameters
        (scalars). If BOTH given, cells are kept inside
        hypot(x_rot, y_rot/hessian_eigenvalue) <= grid_size_max/2 (samana's
        ELLIPTICAL_APERTURE). If either is None, the aperture defaults to the CIRCLE of
        radius grid_size_max/2 inscribed in the box. In magnification_finite_decoupled's
        loop pass the per-image values, e.g. rotation_angle_list[j], hessian_eigenvalue_list[j].
    :return: (magnification, flux_array, tiling)
        flux_array : (npix, npix) block image of the subdivided cells
                     (npix = round(grid_size_max / grid_resolution)),
                     imshow(flux_array, origin='lower').
        tiling     : dict describing the quadtree leaves + aperture -- pass to plot_image()
                     to overlay the cell boundaries and aperture outline. Keys:
                       cx, cy   leaf-center offsets from the image (arcsec)
                       half     leaf half-size (arcsec)
                       sb       leaf mean surface brightness
                       box_size, npix, grid_resolution
                       rotation_angle, hessian_eigenvalue   (None -> circle)
                       n_calls, n_points   (ray-shoot diagnostics)
    """
    from lenstronomy.LensModel.Util.decouple_multi_plane_util import coordinates_and_deflections
    from samana.image_magnification_util import calc_source_sb

    Td = cosmo_bkg.T_xy(0, zlens)
    Ts = cosmo_bkg.T_xy(0, zsource)
    Tds = cosmo_bkg.T_xy(zlens, zsource)
    reduced_to_phys = cosmo_bkg.d_xy(0, zsource) / cosmo_bkg.d_xy(zlens, zsource)

    def ray_shoot(thx, thy):
        xD, yD, afx, afy, abx, aby = coordinates_and_deflections(
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
            np.atleast_1d(thx), np.atleast_1d(thy), z_split, z_source, cosmo_bkg)
        # calc_source_sb returns the SOURCE-plane coords (beta), despite the name
        return calc_source_sb(xD, yD, afx, afy, abx, aby, Td, Tds, Ts,
                              reduced_to_phys, lens_model_free, kwargs_lens)

    def source_sb(bx, by):
        return source_model.surface_brightness(bx, by, kwargs_source)

    mag, n_calls, n_pts, leaves = adaptive_quadtree_magnification(
        ray_shoot, source_sb, x_image, y_image, grid_size_max, grid_resolution,
        n_coarse=n_coarse, rel_tol=rel_tol, flux_floor_frac=flux_floor_frac,
        rotation_angle=rotation_angle, hessian_eigenvalue=hessian_eigenvalue,
        return_leaves=True)

    npix = max(int(round(grid_size_max / grid_resolution)), 2)
    flux_array = rasterize_leaves(leaves, grid_size_max, npix)
    lcx, lcy, lh, lsb = leaves
    tiling = {'cx': lcx, 'cy': lcy, 'half': lh, 'sb': lsb,
              'box_size': grid_size_max, 'npix': npix,
              'grid_resolution': grid_resolution,
              'rotation_angle': rotation_angle, 'hessian_eigenvalue': hessian_eigenvalue,
              'n_calls': n_calls, 'n_points': n_pts}
    return mag, flux_array, tiling

def plot_tiled_image(flux_array, tiling, ax=None, cmap='inferno',
                     show_cells=False, cell_color='white', cell_lw=0.3, cell_alpha=0.35,
               show_aperture=True, log=False):
    """Show the finite-source image with the adaptive quadtree cell boundaries and the
    aperture (circle or ellipse) overlaid.

    :param flux_array: the (npix, npix) block image returned by
        mag_finite_single_image_adaptive
    :param tiling: the tiling dict returned alongside it
    :param ax: optional matplotlib Axes; a new figure is made if None
    :param show_cells: draw the leaf-cell rectangles (the subdivided grid)
    :param show_aperture: draw the circular/elliptical aperture outline
    :param log: display log10(flux) (clipped) instead of linear
    :return: the Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    box = tiling['box_size']; h = box / 2.0
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    img = flux_array
    if log:
        pos = img[img > 0]
        floor = (pos.min() if pos.size else 1.0) * 1e-3
        img = np.log10(np.clip(img, floor, None))
    ax.imshow(img, origin='lower', extent=[-h, h, -h, h], cmap=cmap)

    if show_cells:
        cx = tiling['cx']; cy = tiling['cy']; ch = tiling['half']
        for k in range(len(cx)):
            ax.add_patch(Rectangle((cx[k] - ch[k], cy[k] - ch[k]), 2 * ch[k], 2 * ch[k],
                                   fill=False, edgecolor=cell_color,
                                   linewidth=cell_lw, alpha=cell_alpha))
    if show_aperture:
        ang = tiling.get('rotation_angle'); qe = tiling.get('hessian_eigenvalue')
        phi = np.linspace(0, 2 * np.pi, 200)
        xr = h * np.cos(phi); yr = (h if qe is None else h * qe) * np.sin(phi)
        if ang is None or qe is None:
            ax_bx, ax_by = xr, yr
        else:
            ca, sa = np.cos(ang), np.sin(ang)
            ax_bx = xr * ca - yr * sa            # inverse of util.rotate
            ax_by = xr * sa + yr * ca
        ax.plot(ax_bx, ax_by, color='cyan', lw=1.0, alpha=0.8)

    ax.set_xlim(-h, h); ax.set_ylim(-h, h)
    ax.set_xlabel('arcsec'); ax.set_ylabel('arcsec')
    ax.set_title('%d cells, %d calls, %d ray-shoots'
                 % (len(tiling['cx']), tiling['n_calls'], tiling['n_points']))
    return ax

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

# save_class_exact.py  -- ship alongside the pickle
class SaveClassExact:
    fields = ('source_model', 'kwargs_source', 'lens_model_fixed', 'lens_model_free',
              'kwargs_lens_fixed', 'kwargs_lens_free', 'kwargs_lens', 'z_split', 'z_source',
              'cosmo_bkg', 'x_image', 'y_image', 'grid_x_large', 'grid_y_large',
              'grid_r', 'r_step', 'grid_resolution', 'grid_size_max', 'zlens', 'zsource')

    def __init__(self, inputs):
        self.inputs = tuple(inputs)
        for name, val in zip(self.fields, self.inputs):
            setattr(self, name, val)

    def run_exact(self):
        from samana.image_magnification_util import mag_finite_single_image
        return mag_finite_single_image(*self.inputs)

# save_class.py  -- keep this file alongside the pickle
class SaveClass:
    fields = ('source_model', 'kwargs_source', 'lens_model_fixed', 'lens_model_free',
              'kwargs_lens_fixed', 'kwargs_lens', 'z_split', 'z_source', 'cosmo_bkg',
              'grid_x_large', 'grid_y_large', 'grid_r', 'r_step', 'grid_resolution',
              'grid_size', 'R_max', 'ray_interp_x', 'ray_interp_y')

    def __init__(self, inputs):
        self.inputs = tuple(inputs)
        for name, val in zip(self.fields, self.inputs):
            setattr(self, name, val)

    def run_nearfar(self, verbose=False):
        from samana.image_magnification_near_far import mag_finite_single_image_distortion
        return mag_finite_single_image_distortion(*self.inputs, verbose=verbose)

    def run_adaptive(self):
        from samana.image_magnification_util import mag_finite_single_image_adaptive
        # pass only the args the adaptive full-model function actually takes
        return mag_finite_single_image_adaptive(
            self.source_model, self.kwargs_source, self.lens_model_fixed, self.lens_model_free,
            self.kwargs_lens_fixed, self.kwargs_lens, self.z_split, self.z_source,
            self.cosmo_bkg, self.grid_x_large, self.grid_y_large,  # x_img,y_img in your adaptive branch
            self.grid_resolution, self.grid_size, self.z_split, self.z_source)

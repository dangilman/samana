import numpy as np
from scipy.interpolate import interp1d

_ARCSEC = np.pi / (180 * 3600)
_C_KMS = 299792.458
_CACHE = {}


def _efficiency_interp(z_source, astropy_cosmo, z_min, z_max, n_grid=512):
    """
    Interpolator for z_lens given the lensing efficiency D_ds/D_s, cached per
    (z_source, z_min, z_max). D_ds/D_s decreases monotonically with z_lens, so it inverts.
    """
    key = (round(z_source, 4), round(z_min, 4), round(z_max, 4), id(astropy_cosmo))
    if key not in _CACHE:
        z = np.linspace(z_min, z_max, n_grid)
        d_s = astropy_cosmo.angular_diameter_distance(z_source).value
        ratio = np.array([astropy_cosmo.angular_diameter_distance_z1z2(zi, z_source).value
                          for zi in z]) / d_s
        _CACHE[key] = interp1d(ratio[::-1], z[::-1], bounds_error=False, fill_value=np.nan)
    return _CACHE[key]

def sample_z_lens_from_theta_E(theta_E, z_source, astropy_cosmo,
                               sigma_v_mean=254.1, sigma_v_sigma=43.6,
                               sigma_v_min=150.0, sigma_v_max=400.0,
                               z_min=0.2, z_source_buffer=0.3,
                               f_sie=1.0, max_draws=1000, decimals=2):
    """
    Draw one deflector redshift from the Einstein radius and a velocity dispersion prior.

    For a singular isothermal sphere theta_E = 4 pi (sigma_v / c)^2 D_ds / D_s, so a draw
    of sigma_v fixes the lensing efficiency and hence z_lens. Rejection is used for the
    prior truncation and the z_lens bounds -- clipping either one would stack the rejected
    tail onto a boundary value and put a spike in the z_lens distribution.

    :param theta_E: Einstein radius of the main deflector [arcsec]
    :param z_source: source redshift
    :param astropy_cosmo: astropy cosmology instance
    :param sigma_v_mean: mean of the Gaussian velocity dispersion prior [km/s]
    :param sigma_v_sigma: standard deviation of the prior [km/s]
    :param sigma_v_min: lower truncation of the prior [km/s]
    :param sigma_v_max: upper truncation of the prior [km/s]
    :param z_min: lowest deflector redshift accepted
    :param z_source_buffer: deflector redshifts above z_source - this are rejected
    :param f_sie: sigma_SIS / sigma_star, if the prior is on the stellar dispersion
    :param max_draws: proposals before raising
    :param decimals: if not None, round the returned redshift to this many decimals
    :return: z_lens
    """
    z_max = z_source - z_source_buffer
    if z_max <= z_min:
        raise Exception('z_source = %.2f leaves no room for a deflector between %.2f and '
                        'z_source - %.2f' % (z_source, z_min, z_source_buffer))
    z_of_ratio = _efficiency_interp(z_source, astropy_cosmo, z_min, z_max)
    for _ in range(int(max_draws)):
        sigma_v = float(np.random.normal(sigma_v_mean, sigma_v_sigma))
        if sigma_v < sigma_v_min or sigma_v > sigma_v_max:
            continue
        ratio = theta_E * _ARCSEC * _C_KMS ** 2 / (4 * np.pi * (sigma_v * f_sie) ** 2)
        z_lens = z_of_ratio(ratio)
        if np.isfinite(z_lens):
            return round(float(z_lens), decimals) if decimals is not None else float(z_lens)
    raise Exception('no acceptable velocity dispersion in %d draws for theta_E = %.3f", '
                    'z_source = %.2f, z_lens in [%.2f, %.2f]. The prior N(%.0f, %.0f) '
                    'truncated to [%.0f, %.0f] km/s may be inconsistent with theta_E'
                    % (max_draws, theta_E, z_source, z_min, z_max, sigma_v_mean,
                       sigma_v_sigma, sigma_v_min, sigma_v_max))

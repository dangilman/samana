"""
Fit lens and source light profiles at a fixed lens model
"""
from lenstronomy.ImSim.Numerics.grid import RegularGrid
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from copy import deepcopy
import os
import pickle

def setup_light_reconstruction(output_class_filename,
                               measured_flux_ratios,
                               flux_ratio_uncertainties,
                               n_keep_best=10000,
                               n_keep_random=10000):

    with open(output_class_filename, 'rb') as f:
        output = pickle.load(f).clean()
    f.close()
    flux_ratios = output.flux_ratios
    fr_chi2 = 0
    for i in range(0, len(measured_flux_ratios)):
        if flux_ratio_uncertainties[i] == -1:
            continue
        fr_chi2 += 0.5 * (flux_ratios[:, i] - measured_flux_ratios[i]) ** 2 / flux_ratio_uncertainties[i] ** 2
    print('number reduced chi^2 < 1: ', np.sum(fr_chi2/3 < 1))
    index_best = np.argsort(fr_chi2)
    index_random = np.random.randint(0, len(fr_chi2), n_keep_random)
    index_best = index_best[0:n_keep_best]
    seed_array_baseline = output.seed[index_random]
    seed_array_best = output.seed[index_best]
    print('flux ratio chi2: ', fr_chi2)
    print('median chi2: ', np.median(fr_chi2))
    print('worst chi2: ', max(fr_chi2))
    print('best seeds: ', seed_array_best)
    return seed_array_best, seed_array_baseline

def setup_params_light_fitting(kwargs_params, source_x, source_y):
    """

    :param kwargs_params:
    :param source_x:
    :param source_y:
    :return:
    """
    kwargs_params_out = deepcopy(kwargs_params)
    kwargs_params_out['lens_model'] = [[{}], [{}], [{}], [{}], [{}]]
    # fix the source light centroids to (source_x, source_y)
    source_params = kwargs_params['source_model']
    # [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
    #                          kwargs_upper_source]
    source_params_fixed = deepcopy(source_params[2])
    source_params_init = source_params[0]
    for i, kwargs_source in enumerate(source_params_init):
        if 'center_x' in kwargs_source.keys():
            source_params_fixed[i]['center_x'] = source_y
            source_params_fixed[i]['center_y'] = source_x
    kwargs_params_out['source_model'][2] = source_params_fixed

    return kwargs_params_out

class FixedLensModelNew(object):
    """

    """
    def __init__(self, data_class, lens_model, kwargs_lens, super_sample_factor=1):
        """

        :param data_class:
        :param lens_model:
        :param kwargs_lens:
        :param super_sample_factor:
        """
        nx, ny = data_class.kwargs_data['image_data'].shape
        nx = int(nx)
        ny = int(ny)
        super_sample_factor = int(super_sample_factor)
        deltaPix, ra_at_xy_0, dec_at_xy_0, transform_pix2angle, window_size = data_class.coordinate_properties
        grid = RegularGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, super_sample_factor)
        ra_coords, dec_coords = grid.coordinates_evaluate
        _ra_coords = np.linspace(np.min(ra_coords), np.max(ra_coords), nx)
        _dec_coords = np.linspace(np.min(dec_coords), np.max(dec_coords), ny)
        ra_coords, dec_coords = np.meshgrid(_ra_coords, _dec_coords)
        alpha_x, alpha_y = lens_model.alpha(ra_coords.ravel(), dec_coords.ravel(), kwargs_lens)
        points = (ra_coords[0, :], dec_coords[:, 0])
        self._interp_x = RegularGridInterpolator(points, alpha_x.reshape(nx, ny), bounds_error=False, fill_value=None)
        self._interp_y = RegularGridInterpolator(points, alpha_y.reshape(nx, ny), bounds_error=False, fill_value=None)

    def __call__(self, x, y, *args, **kwargs):
        """

        :param x:
        :param y:
        :param args:
        :param kwargs:
        :return:
        """
        point = (y, x)
        alpha_x = self._interp_x(point)
        alpha_y = self._interp_y(point)

        if isinstance(x, float) or isinstance(x, int) and isinstance(y, float) or isinstance(y, int):
            alpha_x = float(alpha_x)
            alpha_y = float(alpha_y)
        else:
            alpha_x = np.squeeze(alpha_x)
            alpha_y = np.squeeze(alpha_y)

        return alpha_x, alpha_y

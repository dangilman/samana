"""
Fit lens and source light profiles at a fixed lens model
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from copy import deepcopy
import pickle
from lenstronomy.Util.class_creator import create_class_instances, create_image_model


def setup_light_reconstruction(output_class_filename,
                               measured_flux_ratios,
                               flux_ratio_uncertainties,
                               n_keep_best=10000,
                               n_keep_random=10000,
                               seed_filename=None):

    if seed_filename is not None:
        seed_array_best = np.loadtxt(seed_filename + '_best_seeds.txt')
        seed_array_baseline = np.loadtxt(seed_filename + '_random_seeds.txt')
    else:
        with open(output_class_filename, 'rb') as f:
            output = pickle.load(f).clean()
        f.close()
        flux_ratios = output.flux_ratios
        fr_chi2 = 0
        for i in range(0, len(measured_flux_ratios)):
            if flux_ratio_uncertainties[i] == -1:
                continue
            fr_chi2 += (flux_ratios[:, i] - measured_flux_ratios[i]) ** 2 / flux_ratio_uncertainties[i] ** 2
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
            source_params_fixed[i]['center_x'] = source_x
            source_params_fixed[i]['center_y'] = source_y
    kwargs_params_out['source_model'][2] = source_params_fixed
    return kwargs_params_out


class FixedLensModel(object):
    """
    Fixed mapping from a coordinate on the image plane to the source plane
    """

    def __init__(self, kwargs_model, kwargs_data, kwargs_psf, kwargs_numerics, kwargs_lens):
        """

        :param kwargs_model:
        :param kwargs_data:
        :param kwargs_psf:
        :param kwargs_numerics:
        :param kwargs_lens:
        """
        nx, ny = kwargs_data['image_data'].shape
        lens_model = create_class_instances(**kwargs_model)[0]
        image_model = create_image_model(kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model)
        ra_grid, dec_grid = image_model.Data.coordinate_grid(nx, ny)
        alpha_x, alpha_y = lens_model.alpha(ra_grid.ravel(), dec_grid.ravel(), kwargs_lens)
        points = (ra_grid.ravel(), dec_grid.ravel())
        self._interp_x = LinearNDInterpolator(points, alpha_x)
        self._interp_y = LinearNDInterpolator(points, alpha_y)

    def __call__(self, x, y, *args, **kwargs):
        """

        :param x:
        :param y:
        :param args:
        :param kwargs:
        :return:
        """
        point = (x, y)
        alpha_x = self._interp_x(point)
        alpha_y = self._interp_y(point)
        if isinstance(x, float) or isinstance(x, int) and isinstance(y, float) or isinstance(y, int):
            alpha_x = float(alpha_x)
            alpha_y = float(alpha_y)
        else:
            alpha_x = np.squeeze(alpha_x)
            alpha_y = np.squeeze(alpha_y)
        return alpha_x, alpha_y

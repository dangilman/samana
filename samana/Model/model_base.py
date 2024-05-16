from lenstronomy.LensModel.Util.decouple_multi_plane_util import *
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from copy import deepcopy
from samana.image_magnification_util import magnification_finite_decoupled
import numpy as np


class ModelBase(object):

    def __init__(self, data_class, kde_sampler=None):

        self._data = data_class
        self.kde_sampler = kde_sampler

    def gaussian_source_clump(self, center_x, center_y, kwargs_init=None):

        source_model_list = ['GAUSSIAN_ELLIPSE']
        if kwargs_init is None:
            kwargs_source = [{'amp': 1.0, 'sigma': 0.05, 'center_x': center_x, 'center_y': center_y, 'e1': 0.0, 'e2': 0.0}]
        else:
            kwargs_source = deepcopy(kwargs_init)
            kwargs_source['center_x'], kwargs_source['center_y'] = center_x, center_y
        kwargs_source_sigma = [
            {'amp': 10.0, 'sigma': 0.05, 'center_x': 0.025, 'center_y': 0.025, 'e1': 0.1, 'e2': 0.1}]
        kwargs_lower_source = [
            {'amp': 0.00001, 'sigma': 0.0, 'center_x': center_x - 0.2, 'center_y': center_y - 0.2,
             'e1': -0.5, 'e2': -0.5}]
        kwargs_upper_source = [
            {'amp': 100, 'sigma': 0.75, 'center_x': center_x + 0.2, 'center_y': center_y + 0.2,
             'e1': 0.5, 'e2': 0.5}]
        kwargs_source_fixed = [{}]
        return source_model_list, kwargs_source, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source, kwargs_upper_source

    def shapelet_source_clump(self, center_x, center_y, n_max_clump=4, beta_clump=0.05, kwargs_init=None):

        source_model_list = ['SHAPELETS']
        if kwargs_init is None:
            kwargs_source = [{'amp': 1.0, 'beta': beta_clump, 'center_x': center_x, 'center_y': center_y, 'n_max': n_max_clump}]
        else:
            kwargs_source = deepcopy(kwargs_init)
            kwargs_source['center_x'], kwargs_source['center_y'] = center_x, center_y
            kwargs_source['n_max'] = n_max_clump
            kwargs_source['beta'] = beta_clump
        kwargs_source_sigma = [{'amp': 1.0, 'beta': 0.2 * beta_clump, 'center_x': 0.05, 'center_y': 0.05, 'n_max': 1}]
        kwargs_lower_source = [
            {'amp': 1e-9, 'beta': 0.0, 'center_x': center_x - 0.2, 'center_y': center_y - 0.2, 'n_max': 0}]
        kwargs_upper_source = [{'amp': 100.0, 'beta': beta_clump * 20, 'center_x': center_x + 0.2, 'center_y': center_y + 0.2,
                                'n_max': n_max_clump + 1}]
        kwargs_source_fixed = [{'n_max': n_max_clump}]
        return source_model_list, kwargs_source, kwargs_source_sigma, kwargs_source_fixed, \
               kwargs_lower_source, kwargs_upper_source

    def setup_point_source_model(self, fix_image_positions=True):
        point_source_model_list = ['LENSED_POSITION']
        kwargs_ps_init = [{'ra_image': self._data.x_image, 'dec_image': self._data.y_image}]
        kwargs_ps_sigma = [{'ra_image': [5e-3] * 4, 'dec_image': [5e-3] * 4}]
        if fix_image_positions:
            kwargs_ps_fixed = [{'ra_image': self._data.x_image,
                                'dec_image': self._data.y_image}]
        else:
            kwargs_ps_fixed = [{}]
        kwargs_lower_ps = [{'ra_image': -10 + np.zeros_like(self._data.x_image), 'dec_image': -10 + np.zeros_like(self._data.y_image)}]
        kwargs_upper_ps = [{'ra_image': 10 + np.zeros_like(self._data.x_image), 'dec_image': 10 + np.zeros_like(self._data.y_image)}]
        ps_params = [kwargs_ps_init, kwargs_ps_sigma, kwargs_ps_fixed, kwargs_lower_ps, kwargs_upper_ps]
        return point_source_model_list, ps_params

    def update_kwargs_fixed_macro(self, lens_model_list_macro, kwargs_lens_fixed, kwargs_lens_init, macromodel_samples_fixed=None):

        if macromodel_samples_fixed is not None:
            for param_fixed in macromodel_samples_fixed:
                kwargs_lens_fixed[0][param_fixed] = macromodel_samples_fixed[param_fixed]
                kwargs_lens_init[0][param_fixed] = macromodel_samples_fixed[param_fixed]
        return kwargs_lens_fixed, kwargs_lens_init

    def image_magnification_gaussian(self, source_model_quasar, kwargs_source, lens_model_init, kwargs_lens_init,
                            kwargs_lens, grid_size, grid_resolution):

        _, _, index_lens_split, _ = self.setup_lens_model()
        mags = magnification_finite_decoupled(source_model_quasar, kwargs_source,
                                              self._data.x_image, self._data.y_image,
                                              lens_model_init, kwargs_lens_init,
                                              kwargs_lens, index_lens_split, grid_size, grid_resolution)
        return mags

    def setup_kwargs_model(self, decoupled_multiplane=False, lens_model_list_halos=None,
                           redshift_list_halos=None, kwargs_halos=None, kwargs_lens_macro_init=None,
                           grid_resolution=0.05, verbose=False, macromodel_samples_fixed=None,
                           observed_convention_index=None):

        lens_model_list_macro, redshift_list_macro, _, _ = self.setup_lens_model(kwargs_lens_macro_init,
                                                                                 macromodel_samples_fixed)
        source_model_list, _ = self.setup_source_light_model()
        lens_light_model_list, _ = self.setup_lens_light_model()
        point_source_list, _ = self.setup_point_source_model()
        if observed_convention_index is not None:
            assert decoupled_multiplane is False
        kwargs_model = {'lens_model_list': lens_model_list_macro,
                        'lens_redshift_list': redshift_list_macro,
                        'multi_plane': True,
                        'decouple_multi_plane': False,
                        'z_source': self._data.z_source,
                        'kwargs_lens_eqn_solver': {'arrival_time_sort': False},
                        'source_light_model_list': source_model_list,
                        'lens_light_model_list': lens_light_model_list,
                        'point_source_model_list': point_source_list,
                        'additional_images_list': [False],
                        # check what fixed_magnification list does
                        'fixed_magnification_list': [False] * len(point_source_list),
                        'observed_convention_index': observed_convention_index}
        lens_model_init, kwargs_lens_init, index_lens_split = None, None, None
        if decoupled_multiplane:
            if verbose:
                print('setting up decoupled multi-plane approximation...')
            kwargs_decoupled_class_setup, lens_model_init, kwargs_lens_init, index_lens_split = self._setup_decoupled_multiplane_model(
                lens_model_list_halos,
                redshift_list_halos,
                kwargs_halos,
                kwargs_lens_macro_init,
                grid_resolution,
                macromodel_samples_fixed)
            if verbose:
                print('done.')
            kwargs_model['kwargs_multiplane_model'] = kwargs_decoupled_class_setup['kwargs_multiplane_model']
            kwargs_model['decouple_multi_plane'] = True
        return kwargs_model, lens_model_init, kwargs_lens_init, index_lens_split

    def setup_special_params(self, delta_x_image=None, delta_y_image=None):

        if delta_x_image is None:
            delta_x_image = [0.0] * len(self._data.x_image)
        if delta_y_image is None:
            delta_y_image = [0.0] * len(self._data.y_image)
        special_init = {'delta_x_image': delta_x_image,
                        'delta_y_image': delta_y_image}
        special_sigma = {'delta_x_image': [0.005] * 4,
                         'delta_y_image': [0.005] * 4}
        special_lower = {'delta_x_image': [-1.0] * 4,
                         'delta_y_image': [-1.0] * 4}
        special_upper = {'delta_x_image': [1.0] * 4,
                         'delta_y_image': [1.0] * 4}
        special_fixed = [{}]
        kwargs_special = [special_init, special_sigma, special_fixed, special_lower, special_upper]
        return kwargs_special

    def kwargs_params(self, kwargs_lens_macro_init=None,
                      delta_x_image=None,
                      delta_y_image=None,
                      macromodel_samples_fixed=None):

        _, _, _, lens_params = self.setup_lens_model(kwargs_lens_macro_init, macromodel_samples_fixed)
        _, source_params = self.setup_source_light_model()
        lens_light_model_list, lens_light_params = self.setup_lens_light_model()
        _, ps_params = self.setup_point_source_model()
        kwargs_params = {'lens_model': lens_params,
                         'source_model': source_params,
                         'lens_light_model': lens_light_params,
                         'point_source_model': ps_params}
        if self.kwargs_constraints['point_source_offset']:
            special_params = self.setup_special_params(delta_x_image, delta_y_image)
            kwargs_params['special'] = special_params

        return kwargs_params

    def _setup_decoupled_multiplane_model(self, lens_model_list_halos, redshift_list_halos, kwargs_halos,
                                         kwargs_macro_init=None, grid_resolution=0.05, macromodel_samples_fixed=None):

        deltaPix, _, _, _, window_size = self._data.coordinate_properties
        x_grid, y_grid, interp_points, npix = setup_grids(window_size, grid_resolution)
        lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params = \
            self.setup_lens_model(kwargs_macro_init, macromodel_samples_fixed)
        kwargs_lens_macro = lens_model_params[0]
        lens_model_init = LensModel(lens_model_list_macro + lens_model_list_halos,
                                          lens_redshift_list=list(redshift_list_macro) + list(redshift_list_halos),
                                          z_source=self._data.z_source,
                                          multi_plane=True)
        kwargs_lens_init = kwargs_lens_macro + kwargs_halos
        lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free, z_source, z_split, cosmo_bkg = \
            setup_lens_model(lens_model_init, kwargs_lens_init, index_lens_split)
        xD, yD, alpha_x_foreground, alpha_y_foreground, alpha_beta_subx, alpha_beta_suby = coordinates_and_deflections(
            lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
            x_grid, y_grid, z_split, z_source, cosmo_bkg)
        kwargs_class_setup = decoupled_multiplane_class_setup(lens_model_free, xD, yD, alpha_x_foreground, \
                                         alpha_y_foreground, alpha_beta_subx, \
                                         alpha_beta_suby, z_split, \
                                         coordinate_type='GRID', \
                                         interp_points=interp_points)

        return kwargs_class_setup, lens_model_init, kwargs_lens_init, index_lens_split

    def setup_lens_model(self, *args, **kwargs):
        raise Exception('must define a setup_lens_model function in the model class')

    def setup_lens_light_model(self):
        raise Exception('must define a setup_lens_light_model function in the model class')

    def setup_source_light_model(self):
        raise Exception('must define a setup_source_light_model function in the model class')

    @property
    def coordinate_properties(self):
        raise Exception('must define a coordinate_properties property in the model class')

    @property
    def kwargs_constraints(self):
        raise Exception('must specify kwargs_constraints in model class')

    @property
    def kwargs_likelihood(self):
        raise Exception('must specify kwargs_likelihood in model class')

    @property
    def prior_lens(self):
        return None

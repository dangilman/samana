from lenstronomy.LensModel.Util.decouple_multi_plane_util import *
from lenstronomy.Util.param_util import ellipticity2phi_q
from samana.param_managers import EPLMultipole134FreeShear
from samana.image_magnification_util import magnification_finite_decoupled
from samana.forward_model_util import macromodel_readout_function_eplshear
import numpy as np


class EPLModelBase(object):

    _spherical_multipole = False

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=1):

        self._shapelets_order = shapelets_order
        self._data = data_class
        self._shapelets_scale_factor = shapelets_scale_factor

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_eplshear

    def param_class_4pointsolver(self, lens_model_list_macro,
                                 kwargs_lens_init,
                                 macromodel_samples_fixed_dict):

        param_class = EPLMultipole134FreeShear(kwargs_lens_init,
                                                            macromodel_samples_fixed_dict['a1_a'],
                                                            macromodel_samples_fixed_dict['a4_a'],
                                                            macromodel_samples_fixed_dict['a3_a'],
                                                            macromodel_samples_fixed_dict['delta_phi_m1'],
                                                            macromodel_samples_fixed_dict['delta_phi_m3'],
                                                            macromodel_samples_fixed_dict['delta_phi_m4'])
        return param_class

    @property
    def beta_min(self):
        return self._shapelets_scale_factor * self.beta_scale_param(self._shapelets_order) / 2.5

    @property
    def beta_max(self):
        #return self.beta_min * 7.0
        return self.beta_min * 5

    def beta_scale_param(self, n_max):

        pixel_scale = self._data.coordinate_properties[0]
        return pixel_scale * np.sqrt(n_max+1)

    def add_shapelets_lens(self, n_max):

        n_max = int(n_max)
        source_model_list = ['SHAPELETS']
        beta_lower_bound = 0.05
        beta_upper_bound = 0.5
        beta_sigma = 2.0 * beta_lower_bound
        beta_init = 3.0 * beta_lower_bound
        kwargs_source_init = [{'amp': 1.0, 'beta': beta_init, 'center_x': 0.0, 'center_y': 0.0,
                                'n_max': n_max}]
        kwargs_source_sigma = [{'amp': 10.0, 'beta': beta_sigma, 'center_x': 0.1, 'center_y': 0.1, 'n_max': 1}]
        kwargs_lower_source = [{'amp': 10.0, 'beta': beta_lower_bound, 'center_x': -0.2, 'center_y': -0.2, 'n_max': 0}]
        kwargs_upper_source = [{'amp': 10.0, 'beta': beta_upper_bound, 'center_x': 0.2, 'center_y': 0.2, 'n_max': n_max + 1}]
        kwargs_source_fixed = [{'n_max': n_max}]
        return source_model_list, kwargs_source_init, kwargs_source_sigma, \
               kwargs_source_fixed, kwargs_lower_source, kwargs_upper_source

    def add_shapelets_source(self, n_max, beta_init=None):

        n_max = int(n_max)
        source_model_list = ['SHAPELETS']
        beta_lower_bound = self.beta_min / self._data.kwargs_numerics['supersampling_factor']
        beta_upper_bound = self.beta_max
        if beta_init is None:
            beta_sigma = 2.0 * beta_lower_bound
            beta_init = 3.0 * beta_lower_bound
        else:
            assert beta_init > beta_lower_bound * 1.1
            assert beta_init < beta_upper_bound * 0.9
            beta_sigma = 0.5 * beta_init

        kwargs_source_init = [{'amp': 1.0, 'beta': beta_init, 'center_x': 0.0, 'center_y': 0.0,
                                'n_max': n_max}]
        kwargs_source_sigma = [{'amp': 10.0, 'beta': beta_sigma, 'center_x': 0.1, 'center_y': 0.1, 'n_max': 1}]
        kwargs_lower_source = [{'amp': 10.0, 'beta': beta_lower_bound, 'center_x': -0.2, 'center_y': -0.2, 'n_max': 0}]
        kwargs_upper_source = [{'amp': 10.0, 'beta': beta_upper_bound, 'center_x': 0.2, 'center_y': 0.2, 'n_max': n_max + 1}]
        kwargs_source_fixed = [{'n_max': n_max}]
        return source_model_list, kwargs_source_init, kwargs_source_sigma, \
               kwargs_source_fixed, kwargs_lower_source, kwargs_upper_source

    def add_uniform_lens_light(self, amp_init=0.0, amp_sigma=1.0):

        kwargs_light = [{'amp': amp_init}]
        kwargs_light_sigma = [{'amp': amp_sigma}]
        kwargs_light_fixed = [{}]
        kwargs_lower_light = [{'amp': -100}]
        kwargs_upper_light = [{'amp': 100}]

        return kwargs_light, kwargs_light_sigma, kwargs_light_fixed, \
               kwargs_lower_light, kwargs_upper_light

    def gaussian_source_clump(self, center_x, center_y, sigma):

        source_model_list = ['GAUSSIAN_ELLIPSE']
        kwargs_source = [{'amp': 1.0, 'sigma': sigma, 'center_x': center_x, 'center_y': center_y, 'e1': 0.0, 'e2': 0.0}]
        kwargs_source_sigma = [
            {'amp': 10.0, 'sigma': 0.25 * kwargs_source[0]['sigma'], 'center_x': 0.05, 'center_y': 0.05, 'e1': 0.1, 'e2': 0.1}]
        kwargs_lower_source = [
            {'amp': 0.00001, 'sigma': 0.0, 'center_x': center_x - 0.2, 'center_y': center_y - 0.2,
             'e1': -0.5, 'e2': -0.5}]
        kwargs_upper_source = [
            {'amp': 100, 'sigma': 0.2, 'center_x': center_x + 0.2, 'center_y': center_y + 0.2,
             'e1': 0.5, 'e2': 0.5}]
        kwargs_source_fixed = [{}]
        return source_model_list, kwargs_source, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source, kwargs_upper_source

    @property
    def population_gamma_prior(self):
        return []

    def axis_ratio_masslight_alignment(self, kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source):

        q_prior = self.axis_ratio_prior(kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source)

        alignment_prior = self.joint_lens_with_light_prior(kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source)
        return q_prior + alignment_prior

    def joint_lens_with_light_prior(self, kwargs_lens,
                                  kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                                  kwargs_extinction, kwargs_tracer_source, max_offset=0.2):

        center_x_lens, center_y_lens = kwargs_lens[0]['center_x'], kwargs_lens[0]['center_y']
        center_x_light, center_y_light = kwargs_lens_light[0]['center_x'], kwargs_lens_light[0]['center_y']
        sigma = 0.025
        dr_squared = (center_x_lens - center_x_light)**2 + (center_y_lens - center_y_light)**2
        if np.sqrt(dr_squared) > max_offset:
            return -1e9
        return -0.5 * (dr_squared / sigma ** 2)

    def axis_ratio_prior(self, kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source):

        e1, e2 = kwargs_lens[0]['e1'], kwargs_lens[0]['e2']
        _, q = ellipticity2phi_q(e1, e2)
        if q < 0.4:
            return -1e10
        else:
            return -0.5 * (q - 0.8) ** 2 / 0.3 ** 2

    def shapelet_source_clump(self, center_x, center_y, n_max_clump=4, beta_clump=0.05):

        source_model_list = ['SHAPELETS']
        kwargs_source = [{'amp': 1.0, 'beta': beta_clump,
                              'center_x': center_x, 'center_y': center_y, 'n_max': n_max_clump}]
        kwargs_source_sigma = [{'amp': 1.0, 'beta': 0.2 * beta_clump,
                                'center_x': 0.05, 'center_y': 0.05, 'n_max': 1}]
        kwargs_lower_source = [
            {'amp': 1e-9, 'beta': 0.0,
             'center_x': center_x - 10.0,
             'center_y': center_y - 10.0,
             'n_max': 0}]
        kwargs_upper_source = [{'amp': 100.0,
                                'beta': beta_clump * 20,
                                'center_x': center_x + 10.0,
                                'center_y': center_y + 10.0,
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
                            kwargs_lens, grid_size, grid_resolution, lens_model, elliptical_ray_tracing_grid=True):

        _, _, index_lens_split, _ = self.setup_lens_model()
        mags = magnification_finite_decoupled(source_model_quasar, kwargs_source,
                                              self._data.x_image, self._data.y_image,
                                              lens_model_init, kwargs_lens_init,
                                              kwargs_lens, index_lens_split,
                                              grid_size, grid_resolution, lens_model,
                                              elliptical_ray_tracing_grid)
        return mags

    def setup_kwargs_model(self, decoupled_multiplane=False, lens_model_list_halos=None,
                           redshift_list_halos=None, kwargs_halos=None, kwargs_lens_macro_init=None,
                           grid_resolution=0.05, verbose=False, macromodel_samples_fixed=None,
                           observed_convention_index=None, astropy_cosmo=None):

        lens_model_list_macro, redshift_list_macro, _, lens_model_params = self.setup_lens_model(kwargs_lens_macro_init,
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
                        'observed_convention_index': observed_convention_index,
                        'cosmo': astropy_cosmo}

        if kwargs_halos is not None:
            kwargs_lens_init = [lens_model_params[0] + kwargs_halos]
            lm_list = lens_model_list_macro + lens_model_list_halos
            z_list = list(redshift_list_macro) + list(redshift_list_halos)
        else:
            kwargs_lens_init = [lens_model_params[0]]
            lm_list = lens_model_list_macro
            z_list = list(redshift_list_macro)

        index_lens_split = None
        lens_model_init = LensModel(lm_list,
                                    lens_redshift_list=z_list,
                                    cosmo=astropy_cosmo,
                                    z_source=self._data.z_source,
                                    multi_plane=True)
        if decoupled_multiplane:
            if verbose:
                print('setting up decoupled multi-plane approximation...')
            kwargs_decoupled_class_setup, lens_model_init, kwargs_lens_init, index_lens_split = self._setup_decoupled_multiplane_model(
                lens_model_list_halos,
                redshift_list_halos,
                kwargs_halos,
                kwargs_lens_macro_init,
                grid_resolution,
                macromodel_samples_fixed,
                astropy_cosmo)
            if verbose:
                print('done.')
            kwargs_model['kwargs_multiplane_model'] = kwargs_decoupled_class_setup['kwargs_multiplane_model']
            kwargs_model['decouple_multi_plane'] = True
            kwargs_model['lens_model_list'] = kwargs_decoupled_class_setup['lens_model_list']
            kwargs_model['lens_redshift_list'] = kwargs_decoupled_class_setup['lens_redshift_list']
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
                                         kwargs_macro_init=None, grid_resolution=0.05,
                                          macromodel_samples_fixed=None, astropy_cosmo=None,
                                          scale_window_size=1.25):

        deltaPix, _, _, _, window_size = self._data.coordinate_properties
        grid_size = window_size * scale_window_size
        x_grid, y_grid, interp_points, npix = setup_grids(grid_size, grid_resolution)
        lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params = \
            self.setup_lens_model(kwargs_macro_init, macromodel_samples_fixed)
        kwargs_lens_macro = lens_model_params[0]
        lens_model_init = LensModel(lens_model_list_macro + lens_model_list_halos,
                                        cosmo=astropy_cosmo,
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

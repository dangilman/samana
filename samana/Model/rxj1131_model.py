from samana.Model.model_base import EPLModelBase
import numpy as np
from samana.forward_model_util import macromodel_readout_function_eplshear_satellite


class _RXJ1131ModelBase(EPLModelBase):

    def update_kwargs_fixed_macro(self, lens_model_list_macro, kwargs_lens_fixed, kwargs_lens_init, macromodel_samples_fixed=None):

        if macromodel_samples_fixed is not None:
            for param_fixed in macromodel_samples_fixed:
                if param_fixed == 'satellite_1_theta_E':
                    kwargs_lens_fixed[2]['theta_E'] = macromodel_samples_fixed[param_fixed]
                    kwargs_lens_init[2]['theta_E'] = macromodel_samples_fixed[param_fixed]
                elif param_fixed == 'satellite_1_x':
                    kwargs_lens_fixed[2]['center_x'] = macromodel_samples_fixed[param_fixed]
                    kwargs_lens_init[2]['center_x'] = macromodel_samples_fixed[param_fixed]
                elif param_fixed == 'satellite_1_y':
                    kwargs_lens_fixed[2]['center_y'] = macromodel_samples_fixed[param_fixed]
                    kwargs_lens_init[2]['center_y'] = macromodel_samples_fixed[param_fixed]
                else:
                    kwargs_lens_fixed[0][param_fixed] = macromodel_samples_fixed[param_fixed]
                    kwargs_lens_init[0][param_fixed] = macromodel_samples_fixed[param_fixed]
        return kwargs_lens_fixed, kwargs_lens_init

    @property
    def kwargs_constraints(self):
        joint_source_with_point_source = [[0, 0]]
        kwargs_constraints = {'joint_source_with_point_source': joint_source_with_point_source,
                              'num_point_source_list': [len(self._data.x_image)],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True,
                              'joint_lens_with_light': [[1, 2, ['center_x', 'center_y']]]
                              }
        if self._shapelets_order is not None:
            kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        kwargs_constraints['image_plane_source_list'] = self._image_plane_source_list
        return kwargs_constraints

    def setup_source_light_model(self):

        self._image_plane_source_list = [False]
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 37.84232574657596, 'R_sersic': 0.3698572919076755, 'n_sersic': 3.7617972953749863,
             'e1': 0.031165960173117565, 'e2': 0.41419937730384127, 'center_x': 0.1300199654460757,
             'center_y': 0.00894567472215557}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 5.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]

        if self._shapelets_order is not None:
            self._image_plane_source_list += [False]
            n_max = int(self._shapelets_order)
            shapelets_source_list, kwargs_shapelets_init, kwargs_shapelets_sigma, \
            kwargs_shapelets_fixed, kwargs_lower_shapelets, kwargs_upper_shapelets = \
                self.add_shapelets_source(n_max)
            source_model_list += shapelets_source_list
            kwargs_source_init += kwargs_shapelets_init
            kwargs_source_fixed += kwargs_shapelets_fixed
            kwargs_source_sigma += kwargs_shapelets_sigma
            kwargs_lower_source += kwargs_lower_shapelets
            kwargs_upper_source += kwargs_upper_shapelets

        # self._image_plane_source_list += [True]
        # x1 = -0.2
        # y1 = 2.2
        # source_model_list_gaussian, kwargs_source_gaussian, kwargs_source_sigma_gaussian, \
        # kwargs_source_fixed_gaussian, kwargs_lower_source_gaussian, kwargs_upper_source_gaussian = \
        #     self.shapelet_source_clump(x1, y1, 4, 0.1)
        # source_model_list += source_model_list_gaussian
        # kwargs_source_init += kwargs_source_gaussian
        # kwargs_source_fixed += kwargs_source_fixed_gaussian
        # kwargs_source_sigma += kwargs_source_sigma_gaussian
        # kwargs_lower_source += kwargs_lower_source_gaussian
        # kwargs_upper_source += kwargs_upper_source_gaussian
        #
        # self._image_plane_source_list += [True]
        # x2 = 0.25
        # y2 = 2.4
        # source_model_list_gaussian, kwargs_source_gaussian, kwargs_source_sigma_gaussian, \
        # kwargs_source_fixed_gaussian, kwargs_lower_source_gaussian, kwargs_upper_source_gaussian = \
        #     self.shapelet_source_clump(x2, y2, 4, 0.1)
        # source_model_list += source_model_list_gaussian
        # kwargs_source_init += kwargs_source_gaussian
        # kwargs_source_fixed += kwargs_source_fixed_gaussian
        # kwargs_source_sigma += kwargs_source_sigma_gaussian
        # kwargs_lower_source += kwargs_lower_source_gaussian
        # kwargs_upper_source += kwargs_upper_source_gaussian
        #
        # self._image_plane_source_list += [True]
        # x3 = 1.4
        # y3 = 1.95
        # source_model_list_gaussian, kwargs_source_gaussian, kwargs_source_sigma_gaussian, \
        # kwargs_source_fixed_gaussian, kwargs_lower_source_gaussian, kwargs_upper_source_gaussian = \
        #     self.shapelet_source_clump(x3, y3, 4, 0.1)
        # source_model_list += source_model_list_gaussian
        # kwargs_source_init += kwargs_source_gaussian
        # kwargs_source_fixed += kwargs_source_fixed_gaussian
        # kwargs_source_sigma += kwargs_source_sigma_gaussian
        # kwargs_lower_source += kwargs_lower_source_gaussian
        # kwargs_upper_source += kwargs_upper_source_gaussian

        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]

        return source_model_list, source_params

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE', 'SERSIC']
        kwargs_lens_light_init = [
                {'amp': 62.36710261156405, 'R_sersic': 1.1844711774610077, 'n_sersic': 3.769270717843491,
                 'e1': 0.03636972680087215, 'e2': -0.04611601158547492, 'center_x': -0.43657989585448514,
                 'center_y': 0.1344524555733173},
                {'amp': 29.11652334765286, 'R_sersic': 0.10649631588821883, 'n_sersic': 1.1906176282594878,
                 'center_x': self._data.g2x, 'center_y': self._data.g2y}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.025, 'center_y': 0.025}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': self._data.g2x - 0.15, 'center_y': self._data.g2y - 0.15}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
            {'R_sersic': 10, 'n_sersic': 10.0, 'center_x': self._data.g2x + 0.15, 'center_y': self._data.g2y + 0.15}]
        kwargs_lens_light_fixed = [{}, {}]

        add_uniform_light = False
        if add_uniform_light:
            kwargs_uniform, kwargs_uniform_sigma, kwargs_uniform_fixed, \
            kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light(-12.6447)
            lens_light_model_list += ['UNIFORM']
            kwargs_lens_light_init += kwargs_uniform
            kwargs_lens_light_sigma += kwargs_uniform_sigma
            kwargs_lens_light_fixed += kwargs_uniform_fixed
            kwargs_lower_lens_light += kwargs_uniform_lower
            kwargs_upper_lens_light += kwargs_uniform_upper


        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

        return lens_light_model_list, lens_light_params

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 5e-3,
                             'prior_lens': self.prior_lens,
                             'source_position_tolerance': 0.0001,
                             'custom_logL_addition': self.axis_ratio_prior,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

class RXJ1131ModelEPLM3M4Shear(_RXJ1131ModelBase):

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_eplshear_satellite

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS']
        if self._data.band == 'hst814w':
            kwargs_lens_macro = [
                {'theta_E': 1.477157571763176, 'gamma': 2.1957003973362847, 'e1': 0.061215895321337525,
                 'e2': -0.1146635085009388, 'center_x': -0.4268989865545886, 'center_y': 0.04152677365715801, 'a3_a': 0.0,
                 'a1_a': 0.0, 'delta_phi_m1': 0.0,'delta_phi_m3': -0.29527050533980004, 'a4_a': 0.0, 'delta_phi_m4': -0.6210803143805403},
                {'gamma1': -0.132352186696442, 'gamma2': 0.03510715565395326, 'ra_0': 0.0, 'dec_0': 0.0},
                {'theta_E': 0.4319041914345371, 'center_x': -0.29581713909339646, 'center_y': 0.5501448601891118}
            ]
        else:
            kwargs_lens_macro = [
                {'theta_E': 1.477157571763176, 'gamma': 2.1957003973362847, 'e1': 0.061215895321337525,
                 'e2': -0.1146635085009388, 'center_x': -0.4268989865545886, 'center_y': 0.04152677365715801,
                 'a1_a': 0.0, 'delta_phi_m1': 0.0,'a3_a': 0.0,
                 'delta_phi_m3': -0.29527050533980004, 'a4_a': 0.0, 'delta_phi_m4': -0.6210803143805403},
                {'gamma1': -0.132352186696442, 'gamma2': 0.03510715565395326, 'ra_0': 0.0, 'dec_0': 0.0},
                {'theta_E': 0.4319041914345371, 'center_x': self._data.g2x, 'center_y': self._data.g2y}
            ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1, 'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05}, {'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.001, 'center_x': self._data.g2x - 0.15, 'center_y': self._data.g2y - 0.15}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': self._data.g2x + 0.15, 'center_y': self._data.g2x + 0.15, 'e1': 0.5, 'e2': 0.5, 'gamma': 3.5, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

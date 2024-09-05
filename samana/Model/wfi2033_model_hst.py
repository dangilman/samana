from samana.Model.model_base import ModelBase
import numpy as np
import pickle


class _WFI2033ModelBase(ModelBase):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None):
        self._shapelets_order = shapelets_order
        super(_WFI2033ModelBase, self).__init__(data_class, kde_sampler)

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
                elif param_fixed == 'satellite_2_theta_E':
                    kwargs_lens_fixed[3]['theta_E'] = macromodel_samples_fixed[param_fixed]
                    kwargs_lens_init[3]['theta_E'] = macromodel_samples_fixed[param_fixed]
                elif param_fixed == 'satellite_2_x':
                    kwargs_lens_fixed[3]['center_x'] = macromodel_samples_fixed[param_fixed]
                    kwargs_lens_init[3]['center_x'] = macromodel_samples_fixed[param_fixed]
                elif param_fixed == 'satellite_2_y':
                    kwargs_lens_fixed[3]['center_y'] = macromodel_samples_fixed[param_fixed]
                    kwargs_lens_init[3]['center_y'] = macromodel_samples_fixed[param_fixed]
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
                              'joint_lens_with_light': [[1, 2, ['center_x', 'center_y']],
                                                       # [2, 3, ['center_x', 'center_y']]
                                                        ]
                              }
        if self._shapelets_order is not None:
            kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1]]

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 1.4598659553398432, 'R_sersic': 1.7273232630291777, 'n_sersic': 2.298874191784458,
             'e1': -0.33373534412646283, 'e2': 0.08083436630711226, 'center_x': -1.0227453894267913,
             'center_y': 0.025397526392554957}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05,
                                'center_y': 0.05}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]

        if self._shapelets_order is not None:
            n_max = int(self._shapelets_order)
            source_model_list += ['SHAPELETS']
            kwargs_source_init += [{'amp': 1.0, 'beta': 0.33, 'center_x': 0.018, 'center_y': -0.031,
                                    'n_max': n_max}]
            kwargs_source_sigma += [{'amp': 10.0, 'beta': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'n_max': 1}]
            kwargs_lower_source += [{'amp': 1e-9, 'beta': 0.0, 'center_x': -10.0, 'center_y': -10.0, 'n_max': 0}]
            kwargs_upper_source += [{'amp': 100.0, 'beta': 1.0, 'center_x': 10.0, 'center_y': 10.0, 'n_max': n_max+1}]
            kwargs_source_fixed += [{'n_max': n_max}]

        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]

        return source_model_list, source_params

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE', 'SERSIC',
                                # 'SERSIC'
                                 ]
        kwargs_lens_light_init = [
            {'amp': 2.988675202076953, 'R_sersic': 2.312873512068243, 'n_sersic': 5.229966623889376,
             'e1': -0.07879614332928109, 'e2': 0.07352985080432621, 'center_x': -0.04825428196460036,
             'center_y': -0.04868561928572901},
            {'amp': 6362.943163277013, 'R_sersic': 0.009497235147221821, 'n_sersic': 3.734262409628431,
             'center_x': 0.21228905644270246, 'center_y': 1.9896170594129898},
         #   {'amp': 1.943163277013, 'R_sersic': 2.0, 'n_sersic': 4.0,
         #    'center_x': self._satellite_x, 'center_y': self._satellite_y}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
            {'R_sersic': 0.01, 'n_sersic': 0.25, 'center_x': 0.1, 'center_y': 0.1},
        #    {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}
        ]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
            {'R_sersic': 0.0001, 'n_sersic': 0.5, 'center_x': -10.0, 'center_y': -10.0},
         #   {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': -10.0, 'center_y': -10.0}
        ]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
       # {'R_sersic': 5.0, 'n_sersic': 10.0, 'center_x': 10.0, 'center_y': 10.0}
            ]
        kwargs_lens_light_fixed = [{},
                                   {}
                                   #{}
                                   ]

        include_uniform_comp = True
        if include_uniform_comp:
            kwargs_light_uniform, kwargs_light_sigma_uniform, kwargs_light_fixed_uniform, \
            kwargs_lower_light_uniform, kwargs_upper_light_uniform = \
                self.add_uniform_lens_light(4.0, 1.0)
            lens_light_model_list += ['UNIFORM']
            kwargs_lens_light_init += kwargs_light_uniform
            kwargs_lens_light_sigma += kwargs_light_sigma_uniform
            kwargs_lower_lens_light += kwargs_lower_light_uniform
            kwargs_upper_lens_light += kwargs_upper_light_uniform
            kwargs_lens_light_fixed += kwargs_light_fixed_uniform

        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

        return lens_light_model_list, lens_light_params

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 0.005,
                             'source_position_likelihood': False,
                             #'check_matched_source_position': False,
                             'source_position_sigma': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

class WFI2033ModelEPLM3M4ShearObservedConvention(_WFI2033ModelBase):

    _satellite_x = -4.0
    _satellite_y = -0.08

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1], [2, 'center_x', 0.245, 0.05],
                [2, 'center_y', 2.037, 0.05], [2, 'theta_E', 0.05, 0.05],
                #[3, 'center_x', -4.0, 0.1],
                #[3, 'center_y', -0.08, 0.1],
                #[3, 'theta_E', 0.9, 0.1]
                ]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        # 0.245 2.037
        # -4.0 -0.08
        # satellite observed position: -4.0 -0.08
        # satellite inferred position from lens model: -3.7056, -0.14765
        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4', 'SHEAR', 'SIS', 'SIS']
        kwargs_lens_macro = [{'theta_E': 1.0132433996923365, 'gamma': 2.249988059286743, 'e1': -0.06752859713433747,
                              'e2': 0.14688963289111892, 'center_x': -0.0733386322208671,
                              'center_y': -0.016497479550054307, 'a3_a': 0.0,
                              'delta_phi_m3': -0.1293835622154349, 'a4_a': 0.0, 'delta_phi_m4': -0.018710557972236738},
                             {'gamma1': 0.18064156252999108, 'gamma2': -0.08382923416962305, 'ra_0': 0.0, 'dec_0': 0.0},
                             {'theta_E': 0.09557984210599564, 'center_x': 0.2109876011562729, 'center_y': 1.986792353941162},
                             {'theta_E': 0.8874, 'center_x': self._satellite_x, 'center_y': self._satellite_y}]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens,
                               self._data.z_lens, 0.745]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05},
                             {'theta_E': 0.01, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.001, 'center_x': -10, 'center_y': -10},
            {'theta_E': 0.5, 'center_x': -10, 'center_y': -10}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 3.5, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.5, 'center_x': 10, 'center_y': 10},
            {'theta_E': 1.5, 'center_x': 10, 'center_y': 10}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class WFI2033ModelEPLM3M4Shear(WFI2033ModelEPLM3M4ShearObservedConvention):

    _satellite_x = -3.873537518
    _satellite_y = -0.159589891

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1], [0, 'a4_a', 0.0, 0.01], [0, 'a3_a', 0.0, 0.005], [2, 'center_x', 0.245, 0.05],
                [2, 'center_y', 2.037, 0.05], [2, 'theta_E', 0.05, 0.05],
                # [3, 'center_x', -3.7056, 0.2],
                # [3, 'center_y', -0.14765, 0.2],
                # [3, 'theta_E', 0.9, 0.1]
                ]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        # 0.245 2.037
        # -4.0 -0.08
        # satellite observed position: -4.0 -0.08
        # satellite inferred position from lens model: -3.7056, -0.14765
        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4_ELL', 'SHEAR', 'SIS', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 0.9912126548542992, 'gamma': 2.154473001360359, 'e1': -0.1490547449226145,
             'e2': 0.18157594473479932, 'center_x': -0.08101002574825644, 'center_y': -0.02869132259069882, 'a3_a': 0.0,
             'delta_phi_m3': 0.16709974394986865, 'a4_a': 0.0, 'delta_phi_m4': -0.14810669089991801},
            {'gamma1': 0.16684542942450115, 'gamma2': -0.04997559877244375, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.09491605731769136, 'center_x': 0.21228905644270246, 'center_y': 1.9896170594129898},
            {'theta_E': 0.8874, 'center_x': self._satellite_x, 'center_y': self._satellite_y}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens,
                               self._data.z_lens, 0.745]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi / 12, 'delta_phi_m4': np.pi / 16},
                             {'gamma1': 0.05, 'gamma2': 0.05},
                             {'theta_E': 0.01, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi / 6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.001, 'center_x': -10, 'center_y': -10},
            {'theta_E': 0.5, 'center_x': -10, 'center_y': -10}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi / 6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
            {'theta_E': 0.5, 'center_x': 10, 'center_y': 10},
            {'theta_E': 1.5, 'center_x': 10, 'center_y': 10}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params


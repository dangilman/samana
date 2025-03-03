from samana.Model.model_base import EPLModelBase
import numpy as np
from samana.forward_model_util import macromodel_readout_function_2033


class _WFI2033ModelNircamBase(EPLModelBase):

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
        joint_lens_with_light = [[1, 2, ['center_x', 'center_y']]]
        kwargs_constraints = {'joint_source_with_point_source': joint_source_with_point_source,
                              'num_point_source_list': [len(self._data.x_image)],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True,
                              'joint_lens_with_light': joint_lens_with_light,
                              #'joint_lens_light_with_lens_light': joint_lens_light_with_lens_light,
                              'image_plane_source_list': self._image_plane_source_list
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
            {'amp': -4.2172017403304665, 'R_sersic': 1.4072048028204784, 'n_sersic': 1.9472450764190539,
             'e1': 0.008038756473571228, 'e2': 0.0706953392676213, 'center_x': -0.7440302190692125,
             'center_y': 0.11167600073774643}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 1.0,
                                'center_y': 1.0}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -4, 'center_y': -4.0}]
        kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 4.0, 'center_y': 4.0}]
        kwargs_source_fixed = [{}]
        self._image_plane_source_list = [False]

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

        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]

        return source_model_list, source_params

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE',
                                 'SERSIC_ELLIPSE',
                                 'SERSIC',
                                 'SERSIC']
        kwargs_lens_light_init = [
            {'amp': 1.0, 'R_sersic': 2.267929030468463, 'n_sersic': 4.527214657123246,
             'e1': -0.329886025950307, 'e2': 0.05232357447738071, 'center_x': 0.0,
             'center_y': 0.0},
            {'amp': 1.0, 'R_sersic': 1.0, 'n_sersic': 3.0,
             'e1': -0.0, 'e2': 0.0,'center_x': 0.0, 'center_y': 0.0},
            {'amp': 9295.352803369131, 'R_sersic': 0.027114349427617816, 'n_sersic': 2.9632338353717778,
             'center_x': 0.2732173973136928, 'center_y': 2.0044491965512194},
            {'amp': 9295.352803369131, 'R_sersic': 1.0, 'n_sersic': 4.0,
             'center_x': -3.68422, 'center_y': 0.125}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.5, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05, 'center_y': 0.05},
            {'R_sersic': 0.5, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05, 'center_y': 0.05},
            {'R_sersic': 0.01, 'n_sersic': 0.25, 'center_x': 0.025, 'center_y': 0.025},
            {'R_sersic': 0.5, 'n_sersic': 0.5, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -0.5, 'center_y': -0.5},
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -0.5, 'center_y': -0.5},
            {'R_sersic': 0.0001, 'n_sersic': 0.5, 'center_x': 0.2732-0.3, 'center_y': 2.00444-0.3},
            {'R_sersic': 0.01, 'n_sersic': 0.25, 'center_x': -3.6842-0.25, 'center_y': 0.125-0.25}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.5, 'center_y': 0.5},
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.5, 'center_y': 0.5},
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.2732+0.3, 'center_y': 2.00444+0.3},
            {'R_sersic': 10, 'n_sersic':10, 'center_x': -3.6842+0.25, 'center_y': 0.125+0.25}]
        kwargs_lens_light_fixed = [{}, {} ,{}, {}]
        add_uniform_light = True
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
                             'image_position_uncertainty': 0.005,
                             'source_position_likelihood': False,
                             #'check_matched_source_position': False,
                             'source_position_sigma': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

class WFI2033NircamModelEPLM3M4Shear(_WFI2033ModelNircamBase):

    gx2_phys = -3.7016
    gy2_phys = 0.0414

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_2033

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1], [0, 'a4_a', 0.0, 0.01], [0, 'a3_a', 0.0, 0.005],
                [2, 'center_x', self._data.gx1, 0.05],
                [2, 'center_y', self._data.gy1, 0.05],
                [2, 'theta_E', 0.05, 0.1],
                #[3, 'center_x', self.gx2_phys, 0.1],
                #[3, 'center_y', self.gy2_phys, 0.1],
                #[3, 'theta_E', 0.9, 0.1]
                ]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4', 'SHEAR', 'SIS', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 0.9364303494726072, 'gamma': 2.1331494713908254, 'e1': -0.036123261307694034,
             'e2': 0.12458938833908947, 'center_x': 0.016305142064483028, 'center_y': -0.008652493157235784,
             'a1_a': 0.0, 'delta_phi_m1': 0.0, 'a3_a': 0.0, 'delta_phi_m3': 0.0, 'a4_a': 0.0, 'delta_phi_m4': 0.0},
            {'gamma1': 0.22093939223207587, 'gamma2': -0.08277495517430819, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.07123176071640608, 'center_x': 0.03648402114393852, 'center_y': -0.03552860923950886},
            {'theta_E': 1.0394906734030547, 'center_x': -3.8579819047525516, 'center_y': 0.10711299834239145}
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
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05},
                             {'theta_E': 0.01, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -np.pi/8},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.001, 'center_x': -10, 'center_y': -10},
            {'theta_E': 0.5, 'center_x': -3.857 - 0.3, 'center_y': 0.1071 - 0.3}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': np.pi/8},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.6, 'center_x': 10, 'center_y': 10},
            {'theta_E': 1.2, 'center_x': -3.857 + 0.3, 'center_y': 0.1071 + 0.3}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params


class WFI2033NircamModelEPLM3M4ShearObservedConvention(_WFI2033ModelNircamBase):

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_2033

    @property
    def prior_lens(self):
        return self.population_gamma_prior + [[0, 'a4_a', 0.0, 0.01], [0, 'a3_a', 0.0, 0.005],
                [2, 'center_x', self._data.gx1, 0.05],
                [2, 'center_y', self._data.gy1, 0.05],
                [2, 'theta_E', 0.05, 0.05],
                [3, 'center_x', self._data.gx2, 0.1],
                [3, 'center_y', self._data.gy2, 0.1],
                [3, 'theta_E', 0.9, 0.1]
                ]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        # observed positions
        # -1.578481764045944, 1.3689577497404388 first satellite
        # -2.165625453981066 -3.3645306348834603 second bigger satellite

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 0.9819352178051077, 'gamma': 2.1840487253312193, 'e1': -0.060421075894487926,
             'e2': -0.08953109900982839, 'center_x': -0.009227587610902884, 'center_y': 0.010009574939230213,
             'a1_a': 0.01, 'delta_phi_m1': 0.1,'a3_a': 0.0, 'delta_phi_m3': -0.08915130660163478, 'a4_a': 0.0, 'delta_phi_m4': 1.5213918804074946},
            {'gamma1': -0.0029842967957686463, 'gamma2': 0.213368145715242, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.07669619849915402, 'center_x': self._data.gx1, 'center_y': self._data.gy1},
            {'theta_E': 1.0, 'center_x': self._data.gx2, 'center_y': self._data.gy2}
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
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05},
                             {'theta_E': 0.01, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.001, 'center_x': -10, 'center_y': -10},
            {'theta_E': 0.5, 'center_x': -10, 'center_y': -10}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.6, 'center_x': 10, 'center_y': 10},
            {'theta_E': 1.2, 'center_x': 10, 'center_y': 10}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

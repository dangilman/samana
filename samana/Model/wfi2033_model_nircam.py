from samana.Model.model_base import EPLModelBase
import numpy as np
from samana.forward_model_util import macromodel_readout_function_2033


class _WFI2033ModelNircamBase(EPLModelBase):

    @property
    def kwargs_constraints(self):
        joint_source_with_point_source = [[0, 0]]
        joint_lens_with_light = [[2, 2, ['center_x', 'center_y']]]
        joint_lens_light_with_lens_light = [[0,1,['center_x', 'center_y']]]
        kwargs_constraints = {'joint_source_with_point_source': joint_source_with_point_source,
                              'num_point_source_list': [len(self._data.x_image)],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True,
                              'joint_lens_with_light': joint_lens_with_light,
                              'joint_lens_light_with_lens_light': joint_lens_light_with_lens_light,
                              'image_plane_source_list': self._image_plane_source_list
                              }
        if self._shapelets_order is not None:
            kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints

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
            shapelets_source_list, _, kwargs_shapelets_sigma, \
            kwargs_shapelets_fixed, kwargs_lower_shapelets, _ = \
                self.add_shapelets_source(n_max)
            kwargs_shapelets_init = [{'amp': 1.0, 'beta': 0.08, 'center_x': 0.0, 'center_y': 0.0, 'n_max': n_max}]
            kwargs_upper_shapelets = [{'amp': 10.0, 'beta': 0.25, 'center_x': 0.2, 'center_y': 0.2, 'n_max': n_max + 1}]
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
            {'amp': 14.367596437761906, 'R_sersic': 2.4446239259884908, 'n_sersic': 4.52750478994104,
             'e1': -0.3261215351821478, 'e2': 0.11359000754780076, 'center_x': 0.03628615325086386,
             'center_y': -0.037257014675854905},
            {'amp': 482.19262607326374, 'R_sersic': 0.42935963846539227, 'n_sersic': 2.215942829809199,
             'e1': 0.07410165465513978, 'e2': 0.09055838026648412, 'center_x': 0.03628615325086386,
             'center_y': -0.037257014675854905},
            {'amp': 10752.43019065742, 'R_sersic': 0.02406909577122646, 'n_sersic': 2.8475827251302825,
             'center_x': 0.27996727180682557, 'center_y': 2.0044560444528003},
            {'amp': 12.2627078822764, 'R_sersic': 2.331743953299096, 'n_sersic': 3.6452704216864427,
             'center_x': -3.584537539932441, 'center_y': 0.10774749984573875}
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
            kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light(60)
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
                             'prior_lens': self.prior_lens,
                             'source_position_tolerance': 0.00001,
                             'source_position_likelihood': True,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.axis_ratio_prior_with_light,
                             }
        return kwargs_likelihood

class WFI2033NircamModelEPLM3M4Shear(_WFI2033ModelNircamBase):

    gx1, gy1 = 0.28, 2.02
    gx2_phys = -3.62
    gy2_phys = -0.118

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_2033

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 0.005,
                             'prior_lens': self.prior_lens,
                             'source_position_tolerance': 0.0001,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             #'custom_logL_addition': self.shear_prior
                             }
        return kwargs_likelihood

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4', 'SHEAR', 'SIS', 'SIS']
        # used in forward modeling v1
        # kwargs_lens_macro = [
        #     {'theta_E': 1.0534360481718583, 'gamma': 2.171762510566768, 'e1': 0.1020681308697495, 'e2': -0.1443852511147689,
        #      'center_x': 0.10864691301522046, 'center_y': -0.027685070187009064,
        #      'a1_a': 0.0, 'delta_phi_m1': 0.0, 'a3_a': 0.0, 'delta_phi_m3': 0.0,
        #      'a4_a': 0.0, 'delta_phi_m4': 0.0},
        #     {'gamma1': 0.15493703472766196, 'gamma2': -0.14691208945427703, 'ra_0': 0.0, 'dec_0': 0.0},
        #     {'theta_E': 0.10385813844247266, 'center_x': 0.27996727180682557, 'center_y': 2.0044560444528003},
        #     {'theta_E': 0.5004528970016093, 'center_x': -3.596830704357751, 'center_y': -0.21494848752766837}
        # ]
        kwargs_lens_macro = [
            {'theta_E': 1.0321027224447663, 'gamma': 2.072978345285635, 'e1': -0.035056879216832815,
             'e2': 0.14118073989452923, 'center_x': 0.015104732586054595, 'center_y': -0.02997087202454787, 'a1_a': 0.0,
             'delta_phi_m1': 0.0, 'a3_a': 0.0, 'delta_phi_m3': 0.0, 'a4_a': 0.0, 'delta_phi_m4': 0.0},
            {'gamma1': 0.15628116029629033, 'gamma2': -0.04746184092482805, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.07471026075145329, 'center_x': 0.2821279374833845, 'center_y': 2.0055514296524075},
            {'theta_E': 0.7682870829485742, 'center_x': -3.7280301576075505, 'center_y': -0.3672325791192176}
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
            {'theta_E': 0.001, 'center_x': self.gx1-0.3, 'center_y': self.gy1-0.3},
            {'theta_E': 0.5, 'center_x': self.gx2_phys - 0.5, 'center_y': self.gy2_phys - 0.5}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': np.pi/8},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.6, 'center_x': self.gx1+0.3, 'center_y': self.gy1+0.3},
            {'theta_E': 1.2, 'center_x': self.gx2_phys + 0.5, 'center_y': self.gy2_phys + 0.5}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params


class WFI2033NircamModelEPLM3M4ShearObservedConvention(_WFI2033ModelNircamBase):

    gx1, gy1 = 0.28, 2.02
    gx2, gy2 = -3.9, -0.05

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_2033

    @property
    def prior_lens(self):
        return self.population_gamma_prior + [[0, 'a4_a', 0.0, 0.01], [0, 'a3_a', 0.0, 0.005],
                [2, 'center_x', self.gx1, 0.05],
                [2, 'center_y', self.gy1, 0.05],
                [2, 'theta_E', 0.05, 0.05],
                [3, 'center_x', self.gx2, 0.1],
                [3, 'center_y', self.gy2, 0.1],
                [3, 'theta_E', 0.6, 0.06]
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
            {'theta_E': 0.07669619849915402, 'center_x': self.gx1, 'center_y': self.gy1},
            {'theta_E': 0.7, 'center_x': self.gx2, 'center_y': self.gy2}
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
            {'theta_E': 0.001, 'center_x': self.gx1-0.3, 'center_y': self.gy1-0.3},
            {'theta_E': 0.5, 'center_x': self.gx2-0.3, 'center_y': self.gy2-0.3}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.6, 'center_x': self.gx1+0.3, 'center_y': self.gy1+0.3},
            {'theta_E': 1.2, 'center_x': self.gx2+0.3, 'center_y': self.gy2+0.3}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

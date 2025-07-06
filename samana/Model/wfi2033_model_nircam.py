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
            {'amp': -6.480064692809521, 'R_sersic': 0.9553143696244061, 'n_sersic': 0.8399156121224464,
             'e1': -0.03757389106203387, 'e2': 0.1325327354650878, 'center_x': -0.6539482386077969,
             'center_y': -0.017628777046964382}
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
    #
    # def add_shapelets_lens(self, n_max):
    #     """
    #     Here we fix beta ~ 0.08
    #     :param n_max:
    #     :return:
    #     """
    #     n_max = int(n_max)
    #     source_model_list = ['SHAPELETS']
    #     beta_lower_bound = 0.06
    #     beta_upper_bound = 0.1
    #     beta_sigma = 2.0 * beta_lower_bound
    #     beta_init = 3.0 * beta_lower_bound
    #     kwargs_source_init = [{'amp': 1.0, 'beta': beta_init, 'center_x': 0.0, 'center_y': 0.0,
    #                             'n_max': n_max}]
    #     kwargs_source_sigma = [{'amp': 10.0, 'beta': beta_sigma, 'center_x': 0.1, 'center_y': 0.1, 'n_max': 1}]
    #     kwargs_lower_source = [{'amp': 10.0, 'beta': beta_lower_bound, 'center_x': -0.2, 'center_y': -0.2, 'n_max': 0}]
    #     kwargs_upper_source = [{'amp': 10.0, 'beta': beta_upper_bound, 'center_x': 0.2, 'center_y': 0.2, 'n_max': n_max + 1}]
    #     kwargs_source_fixed = [{'n_max': n_max}]
    #     return source_model_list, kwargs_source_init, kwargs_source_sigma, \
    #            kwargs_source_fixed, kwargs_lower_source, kwargs_upper_source

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE',
                                 'SERSIC_ELLIPSE',
                                 'SERSIC',
                                 'SERSIC']
        kwargs_lens_light_init = [
            {'amp': 45.12991487626987, 'R_sersic': 2.585014860911036, 'n_sersic': 3.833788528555177,
             'e1': -0.13214747768783427, 'e2': 0.14162871359840715, 'center_x': -0.007027939355085939,
             'center_y': 0.00023195049536191997},
            {'amp': 2454.3939753026434, 'R_sersic': 0.09701263451025438, 'n_sersic': 3.673793158438429,
             'e1': -0.011360160342285229, 'e2': 0.10625977774348269, 'center_x': -0.007027939355085939,
             'center_y': 0.00023195049536191997},
            {'amp': 12362.217363128877, 'R_sersic': 0.02204367014637297, 'n_sersic': 2.9000162494130053,
             'center_x': 0.23850365135651908, 'center_y': 2.0486099940367724},
            {'amp': 15.293544377293621, 'R_sersic': 2.2255092276598734, 'n_sersic': 3.363971228734671,
             'center_x': -3.5484989729122516, 'center_y': 0.16753003165282}
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
        add_uniform_light = False
        print('WARNING: NOT ADDING UNIFORM LENS LIGHT SUBTRACTION!!!')
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
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4', 'SHEAR', 'SIS', 'SIS']
        # used in forward modeling before swithcing to exact dataset from https://arxiv.org/pdf/2503.00099
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
            {'theta_E': 1.0140790106976274, 'gamma': 1.8705312654746549, 'e1': -0.0467082787555793,
             'e2': 0.10796984573471909, 'center_x': -0.022432523430405006, 'center_y': -0.004210965016325727,
             'a1_a': 0.0, 'delta_phi_m1': 0.0, 'a3_a': 0.0, 'delta_phi_m3': 0.0, 'a4_a': 0.0, 'delta_phi_m4': 0.0},
            {'gamma1': 0.13236433794126734, 'gamma2': -0.02229694844946458, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.052837846356017376, 'center_x': 0.23850365135651908, 'center_y': 2.0486099940367724},
            {'theta_E': 0.8044480530004041, 'center_x': -3.72289481865213, 'center_y': -0.41425281668886754}
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

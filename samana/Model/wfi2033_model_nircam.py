from samana.Model.model_base import ModelBase
import numpy as np
import pickle


class _WFI2033ModelNircamBase(ModelBase):

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
        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05,
                                'center_y': 0.05}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
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
                                 'SERSIC',
                                 'SERSIC_ELLIPSE']
        kwargs_lens_light_init = [
            {'amp': -0.9109199773152901, 'R_sersic': 2.267929030468463, 'n_sersic': 4.527214657123246,
             'e1': -0.329886025950307, 'e2': 0.05232357447738071, 'center_x': 0.3837970795906828,
             'center_y': -0.10588018026389344},
            {'amp': 9295.352803369131, 'R_sersic': 0.027114349427617816, 'n_sersic': 2.9632338353717778,
             'center_x': 0.2732173973136928, 'center_y': 2.0044491965512194},
            {'amp': 47.80297973362198, 'R_sersic': 2.3173100018030532, 'n_sersic': 5.3255548535305035,
             'e1': -0.06316829190379669, 'e2': 0.095078858184226, 'center_x': 0.02669387346426591,
             'center_y': -0.032771162997085004}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
            {'R_sersic': 0.01, 'n_sersic': 0.25, 'center_x': 0.1, 'center_y': 0.1},
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
            {'R_sersic': 0.0001, 'n_sersic': 0.5, 'center_x': -10.0, 'center_y': -10.0},
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -0.5, 'center_y': -0.5}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
            {'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.5, 'center_y': 0.5}]
        kwargs_lens_light_fixed = [{}, {}, {}]
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

        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4', 'SHEAR', 'SIS', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 1.00073793658843, 'gamma': 2.117963495872616, 'e1': -0.0431186975201338,
             'e2': 0.10661592549492657, 'center_x': 0.02864739269529095, 'center_y': -0.01591947145914082, 'a3_a': 0.0,
             'delta_phi_m3': 0.3650468716845697, 'a4_a': 0.0, 'delta_phi_m4': 0.8765892899103462},
            {'gamma1': 0.18179565065440584, 'gamma2': -0.08206855255720198, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.07397118742374756, 'center_x': 0.2732173973136928, 'center_y': 2.0044491965512194},
            {'theta_E': 0.9620899082928352, 'center_x': -3.6842267732585654, 'center_y': 0.12452244355145467}
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
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05},
                             {'theta_E': 0.01, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.001, 'center_x': -10, 'center_y': -10},
            {'theta_E': 0.5, 'center_x': -10, 'center_y': -10}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.6, 'center_x': 10, 'center_y': 10},
            {'theta_E': 1.2, 'center_x': 10, 'center_y': 10}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params


class WFI2033NircamModelEPLM3M4ShearObservedConvention(_WFI2033ModelNircamBase):

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

        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4', 'SHEAR', 'SIS', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 0.9819352178051077, 'gamma': 2.1840487253312193, 'e1': -0.060421075894487926,
             'e2': -0.08953109900982839, 'center_x': -0.009227587610902884, 'center_y': 0.010009574939230213,
             'a3_a': 0.0, 'delta_phi_m3': -0.08915130660163478, 'a4_a': 0.0, 'delta_phi_m4': 1.5213918804074946},
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
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05},
                             {'theta_E': 0.01, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.001, 'center_x': -10, 'center_y': -10},
            {'theta_E': 0.5, 'center_x': -10, 'center_y': -10}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.6, 'center_x': 10, 'center_y': 10},
            {'theta_E': 1.2, 'center_x': 10, 'center_y': 10}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

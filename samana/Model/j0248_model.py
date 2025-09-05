from samana.Model.model_base import EPLModelBase
import numpy as np
import pickle
from samana.forward_model_util import macromodel_readout_function_eplshear_satellite

class _J0248ModelBase(EPLModelBase):

    @property
    def kwargs_constraints(self):
        joint_source_with_point_source = [[0, 0]]
        kwargs_constraints = {'joint_source_with_point_source': joint_source_with_point_source,
                              'num_point_source_list': [len(self._data.x_image)],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True
                              }
        if self._shapelets_order is not None:
           kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints

    @property
    def prior_lens(self):
        return None

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 1, 'R_sersic': 0.280303246950735, 'n_sersic': 5.048041153908457, 'e1': 0.12521229719430846,
             'e2': -0.24205937613299833, 'center_x': -0.02125468977932954, 'center_y': -0.028971998131811488}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 5.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]

        if self._shapelets_order is not None:
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

        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [
            {'amp': 1, 'R_sersic': 1.9279258629791503, 'n_sersic': 5.0,
             'e1': -0.3944652014214787, 'e2': 0.08458272000048936,
             'center_x': -0.00848560108466855, 'center_y': 0.015171022786425209}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 8.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_light_fixed = [{}]
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

        add_uniform_light = True
        if add_uniform_light:
            kwargs_uniform, kwargs_uniform_sigma, kwargs_uniform_fixed, \
                kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light()
            lens_light_model_list += ['UNIFORM']
            kwargs_lens_light_init += kwargs_uniform
            kwargs_lens_light_sigma += kwargs_uniform_sigma
            kwargs_lens_light_fixed += kwargs_uniform_fixed
            kwargs_lower_lens_light += kwargs_uniform_lower
            kwargs_upper_lens_light += kwargs_uniform_upper

        return lens_light_model_list, lens_light_params

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 5e-3,
                             'source_position_tolerance': 0.00001,
                             'source_position_likelihood': True,
                             'prior_lens': self.prior_lens,
                             #'custom_logL_addition': self.axis_ratio_prior_with_light,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

# class J0248ModelEPLM3M4Shear(_J0248ModelBase):
#
#     def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.5/2):
#
#         super(J0248ModelEPLM3M4Shear, self).__init__(data_class, shapelets_order, shapelets_scale_factor)
#
#     @property
#     def prior_lens(self):
#         return self.population_gamma_prior
#
#     def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):
#
#         lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR']
#         kwargs_lens_macro = [
#             {'theta_E': 0.7688125079574597, 'gamma': 2.258826043132833, 'e1': -0.34426981106975874,
#              'e2': 0.12025360595770614, 'center_x': -0.016575767345683192, 'center_y': 0.02121922223861687, 'a3_a': 0.0,
#              'a1_a': 0.0, 'delta_phi_m1': 0.0,'delta_phi_m3': 0.5038333224393988, 'a4_a': 0.0, 'delta_phi_m4': 2.3321217222116664},
#             {'gamma1': -0.2288961659289795, 'gamma2': -0.04663614767679, 'ra_0': 0.0, 'dec_0': 0.0}
#         ]
#         redshift_list_macro = [self._data.z_lens, self._data.z_lens]
#         index_lens_split = [0, 1]
#         if kwargs_lens_macro_init is not None:
#             for i in range(0, len(kwargs_lens_macro_init)):
#                 for param_name in kwargs_lens_macro_init[i].keys():
#                     kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
#         kwargs_lens_init = kwargs_lens_macro
#         kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
#                               'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
#                              {'gamma1': 0.1, 'gamma2': 0.1}]
#         kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}]
#         kwargs_lower_lens = [
#             {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
#              'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
#             {'gamma1': -0.5, 'gamma2': -0.5}]
#         kwargs_upper_lens = [
#             {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'a4_a': 0.1,
#              'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
#             {'gamma1': 0.5, 'gamma2': 0.5}]
#         kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
#                                                                              kwargs_lens_init, macromodel_samples_fixed)
#         lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
#                              kwargs_upper_lens]
#         return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class J0248ModelEPLM3M4ShearSatellite(_J0248ModelBase):
    # the first and second satellites are far enough to apparently not influence the lens model
    #satellite_x1 = 2.09
    #satellite_y1 = 0.86
    #satellite_x2 = -1.21
    #satellite_y2 = -1.84
    satellite_x3 = 0.99
    satellite_y3 = -1.46
    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.5/2):

        super(J0248ModelEPLM3M4ShearSatellite, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_eplshear_satellite

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 0.7363618405268003, 'gamma': 2.2906343004154914, 'e1': -0.20931342953058993,
             'e2': 0.021340502852150202, 'center_x': -0.008376572257326336, 'center_y': 0.016159930073583264,
             'a1_a': 0.0, 'delta_phi_m1': -0.08394381880942713, 'a3_a': 0.0, 'delta_phi_m3': 0.24613898913602586,
             'a4_a': 0.0, 'delta_phi_m4': 2.42844281423497},
            {'gamma1': -0.13894767969137736, 'gamma2': -0.12258445591789656, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.16809823375040833, 'center_x': 1.2817186290414526, 'center_y': -1.279008400017075}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                             #{'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05},
                             #{'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.04, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}, {}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
       # {'theta_E': 0.0, 'center_x': self.satellite_x1-0.3, 'center_y': self.satellite_y1-0.3},
           # {'theta_E': 0.0, 'center_x': self.satellite_x2 - 0.3, 'center_y': self.satellite_y2 - 0.3},
        {'theta_E': 0.0, 'center_x': self.satellite_x3-0.3, 'center_y': self.satellite_y3-0.3}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        #{'theta_E': 1.0, 'center_x': self.satellite_x1+0.3, 'center_y': self.satellite_y1+0.3},
           # {'theta_E': 1.0, 'center_x': self.satellite_x2 + 0.3, 'center_y': self.satellite_y2 + 0.3},
        {'theta_E': 0.3, 'center_x': self.satellite_x3+0.3, 'center_y': self.satellite_y3+0.3}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

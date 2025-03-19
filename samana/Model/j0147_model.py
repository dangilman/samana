from samana.Model.model_base import EPLModelBase
import numpy as np
from samana.param_managers import EPLMultipole134FreeShearLensMassPrior

class _J0147ModelBase(EPLModelBase):

    def param_class_4pointsolver(self, lens_model_list_macro,
                                 kwargs_lens_init,
                                 macromodel_samples_fixed_dict):
        center_x = -0.190679  # 0.0718
        center_y = -0.8101  # -0.22
        sigma_xy = 0.05
        param_class = EPLMultipole134FreeShearLensMassPrior(kwargs_lens_init,
                                                            macromodel_samples_fixed_dict['a1_a'],
                                                            macromodel_samples_fixed_dict['a4_a'],
                                                            macromodel_samples_fixed_dict['a3_a'],
                                                            macromodel_samples_fixed_dict['delta_phi_m1'],
                                                            macromodel_samples_fixed_dict['delta_phi_m3'],
                                                            macromodel_samples_fixed_dict['delta_phi_m4'],
                                                            center_x, center_y, sigma_xy)
        return param_class

    @property
    def kwargs_constraints(self):
        joint_source_with_point_source = [[0, 0]]
        kwargs_constraints = {
            'joint_source_with_point_source': joint_source_with_point_source,
                              'num_point_source_list': [len(self._data.x_image)],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True,
                              }
        if self._shapelets_order is not None:
           kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 46.01163509645058, 'R_sersic': 0.5541138887969872, 'n_sersic': 3.0026362463305007,
             'e1': -0.49974666597962486, 'e2': 0.37983365962053117, 'center_x': -0.11168535036243356,
             'center_y': -0.15938352531829392}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 1.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
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
            {'amp': 226.8787640801956, 'R_sersic': 0.2981038197494028, 'n_sersic': 2.373668072070351,
             'e1': -0.15495299117022826, 'e2': 0.3049476041783155, 'center_x': -0.11293789715104738,
             'center_y': -0.8}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -0.4, 'center_y': -1.1}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.2, 'center_y': -0.4}]
        kwargs_lens_light_fixed = [{}]
        add_uniform_light = True
        if add_uniform_light:
            kwargs_uniform, kwargs_uniform_sigma, kwargs_uniform_fixed, \
            kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light(42.49465, 4.0)
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
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.joint_lens_with_light_prior,
                             }
        return kwargs_likelihood

class J0147ModelEPLM3M4Shear(_J0147ModelBase):

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.5 / 2):
        super(J0147ModelEPLM3M4Shear, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR']
        kwargs_lens_macro = [
            {'theta_E': 1.899454293735385, 'gamma': 2.0464024429737595, 'e1': -0.1517631013044443,
             'e2': -0.009782793928123423, 'center_x': -0.19428252146944783, 'center_y': -0.8596508190368513,
             'a1_a': 0.0, 'delta_phi_m1': 0.0, 'a3_a': 0.0, 'delta_phi_m3': 0., 'a4_a': 0.0, 'delta_phi_m4': 0.},
            {'gamma1': 0.1101796804653667, 'gamma2': -0.059733757683208405, 'ra_0': 0.0, 'dec_0': 0.0}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.1, 'e2': 0.1, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -np.pi/8},
            {'gamma1': -0.5, 'gamma2': -0.5}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.6, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': np.pi/8},
            {'gamma1': 0.5, 'gamma2': 0.5}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class J0147ModelEPLM3M4ShearSatellite(_J0147ModelBase):
    satellite_x = -0.7
    satellite_y = 1.6
    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.5 / 2):
        super(J0147ModelEPLM3M4ShearSatellite, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 5e-3,
                             'source_position_likelihood': False,
                             # 'check_matched_source_position': False,
                             'source_position_tolerance': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.joint_lens_with_light_prior,
                             }
        return kwargs_likelihood

    @property
    def kwargs_constraints(self):
        joint_source_with_point_source = [[0, 0]]
        joint_lens_with_light = [[1, 2, ['center_x', 'center_y']]]
        kwargs_constraints = {
            'joint_source_with_point_source': joint_source_with_point_source,
            'num_point_source_list': [len(self._data.x_image)],
            'solver_type': 'PROFILE_SHEAR',
            'point_source_offset': True,
            'joint_lens_with_light': joint_lens_with_light
        }
        if self._shapelets_order is not None:
            kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE',
                                 'SERSIC'
                                 ]
        kwargs_lens_light_init = [
            {'amp': 226.8787640801956, 'R_sersic': 0.2981038197494028, 'n_sersic': 2.373668072070351,
             'e1': -0.15495299117022826, 'e2': 0.3049476041783155, 'center_x': -0.11293789715104738,
             'center_y': -0.8},
            {'amp': 1.8787640801956, 'R_sersic': 0.05, 'n_sersic': 2.0,
             'center_x': self.satellite_x, 'center_y': self.satellite_y}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05, 'center_y': 0.05},
            {'amp': 1., 'R_sersic': 0.025, 'n_sersic': 0.5, 'center_x': 0.1, 'center_y': 0.1}
        ]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -0.4, 'center_y': -1.1},
            {'amp': 1.0, 'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': self.satellite_x-0.4, 'center_y':self.satellite_y-0.4}
            ]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.2, 'center_y': -0.4},
            {'amp': 1.0, 'R_sersic': 0.5, 'n_sersic': 10.0, 'center_x': self.satellite_x+0.4, 'center_y':self.satellite_y+0.4}
            ]
        kwargs_lens_light_fixed = [{},
                                   {}
                                   ]
        add_uniform_light = True
        if add_uniform_light:
            kwargs_uniform, kwargs_uniform_sigma, kwargs_uniform_fixed, \
            kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light(42.49465, 4.0)
            lens_light_model_list += ['UNIFORM']
            kwargs_lens_light_init += kwargs_uniform
            kwargs_lens_light_sigma += kwargs_uniform_sigma
            kwargs_lens_light_fixed += kwargs_uniform_fixed
            kwargs_lower_lens_light += kwargs_uniform_lower
            kwargs_upper_lens_light += kwargs_uniform_upper
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

        return lens_light_model_list, lens_light_params

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR','SIS']
        kwargs_lens_macro = [
            {'theta_E': 1.899454293735385, 'gamma': 2.0464024429737595, 'e1': -0.1517631013044443,
             'e2': -0.009782793928123423, 'center_x': -0.19428252146944783, 'center_y': -0.8596508190368513,
             'a1_a': 0.0, 'delta_phi_m1': 0.0, 'a3_a': 0.0, 'delta_phi_m3': 0., 'a4_a': 0.0, 'delta_phi_m4': 0.},
            {'gamma1': 0.1101796804653667, 'gamma2': -0.059733757683208405, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.1, 'center_x': self.satellite_x, 'center_y': self.satellite_y}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.1, 'e2': 0.1, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': np.pi/3,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05},
                             {'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -np.pi/8},
            {'gamma1': -0.5, 'gamma2': -0.5},
        {'theta_E': 0.0, 'center_x': self.satellite_x-0.3, 'center_y': self.satellite_y-0.3}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.6, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': np.pi/8},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 1.0, 'center_x': self.satellite_x+0.3, 'center_y': self.satellite_y+0.3}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

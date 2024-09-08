from samana.Model.model_base import ModelBase
import numpy as np
import pickle


class _J2344ModelBase(ModelBase):

    def __init__(self, data_class, kde_sampler, shapelets_order):
        self._shapelets_order = shapelets_order
        super(_J2344ModelBase, self).__init__(data_class, kde_sampler)

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
                              'point_source_offset': True
                              }
        if self._shapelets_order is not None:
           kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.2]]

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 0.8921707344537865, 'R_sersic': 6.973228764573325, 'n_sersic': 3.83897557910403,
             'e1': -0.006035362309378849, 'e2': -0.23171682525353335, 'center_x': 0.03722291590649773,
             'center_y': -0.034204091049480204}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]

        if self._shapelets_order is not None:
            n_max = int(self._shapelets_order)
            source_model_list += ['SHAPELETS']
            kwargs_source_init += [{'amp': 1.0, 'beta': 0.1, 'center_x': 0.018, 'center_y': -0.031,
                                    'n_max': n_max}]
            kwargs_source_sigma += [{'amp': 10.0, 'beta': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'n_max': 1}]
            kwargs_lower_source += [{'amp': 10.0, 'beta': 0.0, 'center_x': -10.0, 'center_y': -10.0, 'n_max': 0}]
            kwargs_upper_source += [{'amp': 10.0, 'beta': 0.5, 'center_x': 10.0, 'center_y': 10.0, 'n_max': n_max+1}]
            kwargs_source_fixed += [{'n_max': n_max}]

        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]

        return source_model_list, source_params

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [{'amp': 4.367520306982096, 'R_sersic': 1.8072200192311154,
                                   'n_sersic': 8.952979461993934, 'e1': 0.08790722967451502,
                                   'e2': -0.018565037868028, 'center_x': 0.03992889452141417,
                                   'center_y': -0.04261938784509421}]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.1, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -0.3, 'center_y': -0.3}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.3, 'center_y': 0.3}]
        kwargs_lens_light_fixed = [{}]
        add_uniform_light = True
        if add_uniform_light:
            kwargs_uniform, kwargs_uniform_sigma, kwargs_uniform_fixed, \
            kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light(-35.1075339404, 2.0)
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
                             'source_position_likelihood': True,
                             #'check_matched_source_position': False,
                             'source_position_sigma': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

class J2344ModelEPLM3M4Shear(_J2344ModelBase):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None):
        super(J2344ModelEPLM3M4Shear, self).__init__(data_class, kde_sampler, shapelets_order)

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1]]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4_ELL', 'SHEAR']
        kwargs_lens_macro = [
            {'theta_E': 0.5041533930022769, 'gamma': 2.1580810010904954, 'e1': -0.14036807629821524,
             'e2': 0.12356389649570605, 'center_x': 0.03171191861205254, 'center_y': -0.03485422259012721, 'a3_a': 0.0,
             'delta_phi_m3': -0.2419525334571311, 'a4_a': 0.0, 'delta_phi_m4': 2.1133464477475252},
            {'gamma1': -0.09145390101961005, 'gamma2': 0.08020633200117362, 'ra_0': 0.0, 'dec_0': 0.0}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 3.5, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

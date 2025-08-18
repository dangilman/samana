from samana.Model.model_base import EPLModelBase
import numpy as np
from samana.forward_model_util import macromodel_readout_function_2033, macromodel_readout_function_3satellite, macromodel_readout_function_eplshear_satellite
import pickle


class _J2145(EPLModelBase):

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

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 38.66894438005693, 'R_sersic': 0.42564909776328186, 'n_sersic': 4.0,
             'e1': 0.4999972930169189, 'e2': -0.2857266536463178, 'center_x': -0.1676034477343039,
             'center_y': 0.10228595373813372}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 5.0, 'n_sersic': 8.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
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

        lens_light_model_list = ['SERSIC']
        kwargs_lens_light_init = [
            {'amp': 2.768739867400874, 'R_sersic': 1.9993500387238252, 'n_sersic': 3.8344444084993423,
             'center_x': -0.29516658229291076, 'center_y': 0.08857532078862981}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.025, 'center_y': 0.025}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': -0.3, 'center_y': 0.0}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 8.0, 'center_x': 0.0, 'center_y': 0.5}]
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

    def axis_ratio_masslight_alignment(self, kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source):

        q_prior = self.axis_ratio_prior(kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source)

        alignment_prior = self.lens_mass_lens_light_alignment_prior(kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source)

        return q_prior + alignment_prior

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 5e-3,
                             'source_position_tolerance': 0.00001,
                             'source_position_likelihood': True,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.axis_ratio_masslight_alignment,
                             }
        return kwargs_likelihood

class J2145ModelEPLM3M4Shear(_J2145):

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.5 / 2):

        super(J2145ModelEPLM3M4Shear, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR']
        kwargs_lens_macro = [
            {'theta_E': 0.9985879870024106, 'gamma': 2.05, 'e1': -0.035261127487310716, 'e2': 0.011588120038880656,
             'center_x': -0.18418458630048956, 'center_y': 0.21737688338539365, 'a1_a': 0.0,
             'delta_phi_m1': 0.006207400490477667, 'a3_a': 0.0, 'delta_phi_m3': -0.26051449314623903, 'a4_a': 0.0,
             'delta_phi_m4': 0.31883685774623705},
            {'gamma1': 0.052477788068817734, 'gamma2': 0.147380730485384, 'ra_0': 0.0, 'dec_0': 0.0}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -np.pi/8},
            {'gamma1': -0.5, 'gamma2': -0.5}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': np.pi/8},
            {'gamma1': 0.5, 'gamma2': 0.5}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class J2145ModelEPLM1M3M4Shear2Satellite(_J2145):

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.5 / 2):

        super(J2145ModelEPLM1M3M4Shear2Satellite, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_2033

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS', 'SIS']
        sat_x = self._data.x_image[0] + self._data.sat1_x_wrt0
        sat_y = self._data.y_image[0] + self._data.sat1_y_wrt0
        sat_x2 = self._data.x_image[0] + self._data.sat2_x_wrt0
        sat_y2 = self._data.y_image[0] + self._data.sat2_y_wrt0
        kwargs_lens_macro = [
            {'theta_E': 0.9985879870024106, 'gamma': 2.05, 'e1': -0.035261127487310716, 'e2': 0.011588120038880656,
             'center_x': -0.18418458630048956, 'center_y': 0.21737688338539365, 'a1_a': 0.0,
             'delta_phi_m1': 0.006207400490477667, 'a3_a': 0.0, 'delta_phi_m3': -0.26051449314623903, 'a4_a': 0.0,
             'delta_phi_m4': 0.31883685774623705},
            {'gamma1': 0.052477788068817734, 'gamma2': 0.147380730485384, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.45, 'center_x': sat_x, 'center_y': sat_y},
            {'theta_E': 0.22, 'center_x': sat_x2, 'center_y': sat_y2}
        ]
        redshift_list_macro = [self._data.z_lens] * len(kwargs_lens_macro)
        index_lens_split = [0, 1, 2, 3]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1, 'a4_a': 0.01, 'a3_a': 0.005,
                              'delta_phi_m3': np.pi / 12, 'delta_phi_m4': np.pi / 16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}
                             ]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi, 'a3_a': -0.1, 'delta_phi_m3': -np.pi / 6,
             'delta_phi_m4': -np.pi / 8},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.0, 'center_x': sat_x - 0.2, 'center_y': sat_y - 0.2},
            {'theta_E': 0.0, 'center_x': sat_x2 - 0.2, 'center_y': sat_y2 - 0.2}
        ]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi, 'a3_a': 0.1, 'delta_phi_m3': np.pi / 6, 'delta_phi_m4': np.pi / 8},
            {'gamma1': 0.5, 'gamma2': 0.5},
            {'theta_E': 3.0, 'center_x': sat_x + 0.2, 'center_y': sat_y + 0.2},
            {'theta_E': 3.0, 'center_x': sat_x2 + 0.2, 'center_y': sat_y2 + 0.2}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class J2145ModelEPLM1M3M4ShearSatellite(_J2145):

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.5 / 2):

        super(J2145ModelEPLM1M3M4ShearSatellite, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    def setup_lens_light_model(self):

        sat3_x = self._data.x_image[0] + self._data.sat3_x_wrt0
        sat3_y = self._data.y_image[0] + self._data.sat3_y_wrt0
        lens_light_model_list = ['SERSIC', 'SERSIC']
        kwargs_lens_light_init = [
            {'amp': 2.768739867400874, 'R_sersic': 1.9993500387238252, 'n_sersic': 3.8344444084993423,
             'center_x': -0.29516658229291076, 'center_y': 0.08857532078862981},
            {'amp': 8.736684618863357, 'R_sersic': 1.9026498835104646, 'n_sersic': 3.9892094430729927,
             'center_x': 0.060804930772608366, 'center_y': -1.1649399969148475}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.025, 'center_y': 0.025},
        {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.025, 'center_y': 0.025}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': -0.3, 'center_y': 0.0},
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': sat3_x - 0.2, 'center_y': sat3_y - 0.2}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 8.0, 'center_x': 0.0, 'center_y': 0.5},
        {'R_sersic': 10, 'n_sersic': 8.0, 'center_x': sat3_x + 0.2, 'center_y': sat3_y + 0.2}]
        kwargs_lens_light_fixed = [{}, {}]
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
    def prior_lens(self):
        return None

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_eplshear_satellite

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS']
        sat_x3 = self._data.x_image[0] + self._data.sat3_x_wrt0
        sat_y3 = self._data.y_image[0] + self._data.sat3_y_wrt0
        kwargs_lens_macro = [
            {'theta_E': 0.96, 'gamma': 2.0, 'e1': -0.035261127487310716, 'e2': 0.011588120038880656,
             'center_x': -0.18418458630048956, 'center_y': 0.21737688338539365, 'a1_a': 0.0,
             'delta_phi_m1': 0.006207400490477667, 'a3_a': 0.0, 'delta_phi_m3': -0.26051449314623903, 'a4_a': 0.0,
             'delta_phi_m4': 0.31883685774623705},
            {'gamma1': 0.052477788068817734, 'gamma2': 0.147380730485384, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.3, 'center_x': sat_x3, 'center_y': sat_y3}
        ]
        redshift_list_macro = [self._data.z_lens] * len(kwargs_lens_macro)
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1, 'a4_a': 0.01, 'a3_a': 0.005,
                              'delta_phi_m3': np.pi / 12, 'delta_phi_m4': np.pi / 16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}
                             ]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi, 'a3_a': -0.1, 'delta_phi_m3': -np.pi / 6,
             'delta_phi_m4': -np.pi / 8},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.0, 'center_x': sat_x3 - 0.2, 'center_y': sat_y3 - 0.2},
        ]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi, 'a3_a': 0.1, 'delta_phi_m3': np.pi / 6, 'delta_phi_m4': np.pi / 8},
            {'gamma1': 0.5, 'gamma2': 0.5},
            {'theta_E': 1.0, 'center_x': sat_x3 + 0.2, 'center_y': sat_y3 + 0.2}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params


class J2145_WIDEFIELD_ModelEPLM1M3M4ShearSatellite(_J2145):

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.5 / 2):

        super(J2145_WIDEFIELD_ModelEPLM1M3M4ShearSatellite, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    @property
    def prior_lens(self):
        return None

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_3satellite

    def setup_lens_light_model(self):
        sat_x = self._data.x_image[0] + self._data.sat1_x_wrt0
        sat_y = self._data.y_image[0] + self._data.sat1_y_wrt0
        sat_x2 = self._data.x_image[0] + self._data.sat2_x_wrt0
        sat_y2 = self._data.y_image[0] + self._data.sat2_y_wrt0
        sat_x3 = self._data.x_image[0] + self._data.sat3_x_wrt0
        sat_y3 = self._data.y_image[0] + self._data.sat3_y_wrt0
        lens_light_model_list = ['SERSIC', 'SERSIC', 'SERSIC', 'SERSIC']
        kwargs_lens_light_init = [
            {'amp': 1.7454240902431188, 'R_sersic': 2.2299147056839135, 'n_sersic': 4.0,
             'center_x': -0.2, 'center_y': 0.25},
            {'amp': 1.7454240902431188, 'R_sersic': 1.0, 'n_sersic': 4.0,
             'center_x': sat_x, 'center_y': sat_y},
            {'amp': 1.7454240902431188, 'R_sersic': 1.0, 'n_sersic': 4.0,
             'center_x': sat_x2, 'center_y': sat_y2},
            {'amp': 0.4063795423148451, 'R_sersic': 1.0, 'n_sersic': 4.0,
             'center_x': sat_x3, 'center_y': sat_y3}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.025, 'center_y': 0.025},
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.05, 'center_y': 0.05},
        {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.05, 'center_y': 0.05},
        {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': -0.3, 'center_y': 0.0},
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': sat_x-0.1, 'center_y': sat_y-0.1},
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': sat_x2-0.1, 'center_y': sat_y2-0.1},
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': sat_x3-0.1, 'center_y': sat_y3-0.1}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 6.0, 'center_x': 0.0, 'center_y': 0.5},
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': sat_x+0.1, 'center_y': sat_y+0.1},
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': sat_x2+0.1, 'center_y': sat_y2+0.1},
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': sat_x3+0.1, 'center_y': sat_y3+0.1}]
        kwargs_lens_light_fixed = [{}, {},{}, {}]
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

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS', 'SIS', 'SIS']
        sat_x = self._data.x_image[0] + self._data.sat1_x_wrt0
        sat_y = self._data.y_image[0] + self._data.sat1_y_wrt0
        sat_x2 = self._data.x_image[0] + self._data.sat2_x_wrt0
        sat_y2 = self._data.y_image[0] + self._data.sat2_y_wrt0
        sat_x3 = self._data.x_image[0] + self._data.sat3_x_wrt0
        sat_y3 = self._data.y_image[0] + self._data.sat3_y_wrt0
        kwargs_lens_macro = [
            {'theta_E': 0.96, 'gamma': 2.05, 'e1': -0.035261127487310716, 'e2': 0.011588120038880656,
             'center_x': -0.18418458630048956, 'center_y': 0.21737688338539365, 'a1_a': 0.0,
             'delta_phi_m1': 0.006207400490477667, 'a3_a': 0.0, 'delta_phi_m3': -0.26051449314623903, 'a4_a': 0.0,
             'delta_phi_m4': 0.31883685774623705},
            {'gamma1': 0.052477788068817734, 'gamma2': 0.147380730485384, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.36, 'center_x': sat_x, 'center_y': sat_y},
            {'theta_E': 0.18, 'center_x': sat_x2, 'center_y': sat_y2},
            {'theta_E': 0.24, 'center_x': sat_x3, 'center_y': sat_y3}
        ]
        redshift_list_macro = [self._data.z_lens] * len(kwargs_lens_macro)
        index_lens_split = [0, 1, 2, 3, 4]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1, 'a4_a': 0.01, 'a3_a': 0.005,
                              'delta_phi_m3': np.pi / 12, 'delta_phi_m4': np.pi / 16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}
                             ]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}, {}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi, 'a3_a': -0.1, 'delta_phi_m3': -np.pi / 6,
             'delta_phi_m4': -np.pi / 8},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.0, 'center_x': sat_x - 0.2, 'center_y': sat_y - 0.2},
            {'theta_E': 0.0, 'center_x': sat_x2 - 0.2, 'center_y': sat_y2 - 0.2},
            {'theta_E': 0.0, 'center_x': sat_x3 - 0.2, 'center_y': sat_y3 - 0.2},
        ]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi, 'a3_a': 0.1, 'delta_phi_m3': np.pi / 6, 'delta_phi_m4': np.pi / 8},
            {'gamma1': 0.5, 'gamma2': 0.5},
            {'theta_E': 3.0, 'center_x': sat_x + 0.2, 'center_y': sat_y + 0.2},
            {'theta_E': 3.0, 'center_x': sat_x2 + 0.2, 'center_y': sat_y2 + 0.2},
            {'theta_E': 3.0, 'center_x': sat_x3 + 0.2, 'center_y': sat_y3 + 0.2}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params


from samana.Model.model_base import ModelBase
import numpy as np
from lenstronomy.Util.param_util import ellipticity2phi_q


class _J0659ModelBase(ModelBase):

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
        return [[0, 'gamma', 2.0, 0.2]]

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 0.004229672145693155, 'R_sersic': 6.534741563565758, 'n_sersic': 3.5686539728600524,
             'e1': -0.0570713847800167, 'e2': -0.18864599652652417, 'center_x': -0.3097977563684708,
             'center_y': -0.18641225641777434}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -11, 'center_y': -11.0}]
        kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 11.0, 'center_y': 11.0}]
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

        star_x, star_y = self._data.satellite_or_star_coords
        lens_light_model_list = ['SERSIC_ELLIPSE', 'SERSIC']
        kwargs_lens_light_init = [
            {'amp': 3.7534296013695, 'R_sersic': 1.6786882388346782, 'n_sersic': 4.968104162146947,
             'e1': -0.04218609491776261, 'e2': 0.01221099338904454, 'center_x': 0.06635121513182367,
             'center_y': -0.06761315264140716},
            {'amp': 185.9911238011588, 'R_sersic': 0.0647064660040841, 'n_sersic': 2.117934331728986,
             'center_x': star_x, 'center_y': star_y}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
        {'R_sersic': 0.05, 'n_sersic': 1, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
        {'R_sersic': 0.0, 'n_sersic': 0.1, 'center_x': -10.0, 'center_y': -10.0}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
        {'R_sersic': 1.0, 'n_sersic': 10.0, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_lens_light_fixed = [{}, {}]

        include_uniform_comp = True
        if include_uniform_comp:
            kwargs_light_uniform, kwargs_light_sigma_uniform, kwargs_light_fixed_uniform, \
            kwargs_lower_light_uniform, kwargs_upper_light_uniform = \
                self.add_uniform_lens_light(3.7, 1.0)
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
                             'image_position_uncertainty': 5e-3,
                             'source_position_likelihood': True,
                             #'check_matched_source_position': False,
                             'source_position_sigma': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.q_prior
                             }
        return kwargs_likelihood

class J0659ModelEPLM3M4Shear_AssumeStar(_J0659ModelBase):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None):
        super(J0659ModelEPLM3M4Shear_AssumeStar, self).__init__(data_class, kde_sampler, shapelets_order)

    def q_prior(self, kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source):
        e1, e2 = kwargs_lens[0]['e1'], kwargs_lens[0]['e2']
        if abs(e1) > 0.7 or abs(e1) > 0.7:
            return -1e9
        else:
            return 0.0

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        star_x, star_y = self._data.satellite_or_star_coords
        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4', 'SHEAR', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 2.1605631755867893, 'gamma': 1.9560591845864295, 'e1': -0.037496027239971266,
              'e2': -0.05399604866482186, 'center_x': 0.046135644852520676, 'center_y': -0.22132533038219016,
              'a3_a': 0.0, 'delta_phi_m3': -0.36971825331534836, 'a4_a': 0.0, 'delta_phi_m4': 0.51992745655839},
             {'gamma1': 0.04347299847870509, 'gamma2': 0.06341935451688109, 'ra_0': 0.0, 'dec_0': 0.0},
             {'theta_E': 0.0, 'center_x': star_x, 'center_y': star_y}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi / 12, 'delta_phi_m4': np.pi / 16},
                             {'gamma1': 0.1, 'gamma2': 0.1}, {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}]

        kwargs_lens_fixed = [{},
                             {'ra_0': 0.0, 'dec_0': 0.0},
                             {'theta_E': 0.0}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.7, 'e2': -0.7, 'gamma': 1.5, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi / 6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.2, 'center_x': star_x - 0.4, 'center_y': star_y - 0.4}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.7, 'e2': 0.7, 'gamma': 2.2, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi / 6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
            {'theta_E': 0.7, 'center_x': star_x + 0.4, 'center_y': star_y + 0.4}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params


class J0659ModelEPLM3M4Shear(_J0659ModelBase):

    def q_prior(self, kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source):

        e1, e2 = kwargs_lens[0]['e1'], kwargs_lens[0]['e2']
        _, q = ellipticity2phi_q(e1, e2)
        if q < 0.5:
            return -1e9
        else:
            return -0.5 * (q - 0.95) ** 2 / 0.1 ** 2

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1],
                [2, 'theta_E', 0.25, 0.2]]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None, assume_star=False):

        star_x, star_y = self._data.satellite_or_star_coords
        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4_ELL', 'SHEAR', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 2.0698946907784523, 'gamma': 2.1, 'e1': 0.009009195764101342,
             'e2': 0.3323437241761038, 'center_x': -0.15872251846486193, 'center_y': -0.5782514700137745,
             'a3_a': 0.0, 'delta_phi_m3': -0.10113786153466237, 'a4_a': 0.0, 'delta_phi_m4': 2.01420533512725},
            {'gamma1': 0.03644470797475445, 'gamma2': 0.22723101935610268,
             'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.25, 'center_x': 0.3232445684962102, 'center_y': 1.5607906855285532}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi / 12, 'delta_phi_m4': np.pi / 16},
                             {'gamma1': 0.1, 'gamma2': 0.1}, {'theta_E': 0.1, 'center_x': 0.05, 'center_y': 0.05}]

        kwargs_lens_fixed = [{},
                             {'ra_0': 0.0, 'dec_0': 0.0},
                             {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi / 6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.0, 'center_x': star_x - 0.4, 'center_y': star_y - 0.4}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.2, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi / 6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
            {'theta_E': 0.7, 'center_x': star_x + 0.4, 'center_y': star_y + 0.4}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

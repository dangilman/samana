from samana.Model.model_base import EPLModelBase
import numpy as np
import pickle
from lenstronomy.Util.param_util import ellipticity2phi_q

class _HE1113(EPLModelBase):

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
            {'amp': 130.69647908292973, 'R_sersic': 0.20642549995946133, 'n_sersic': 6.7061875334997545,
             'e1': -0.3195751342552877, 'e2': -0.07275627531709833, 'center_x': -0.07024920320165151,
             'center_y': -0.06335753077104779}
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

        lens_light_model_list = ['SERSIC']
        kwargs_lens_light_init = [
            {'amp': 19.345277702072867, 'R_sersic': 0.8491740518771395, 'n_sersic': 4.2622267147848545,
             #'e1': 0.4053141855213741, 'e2': -0.4095417323775894,
             'center_x': 0,
             'center_y': 0}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25,
            # 'e1': 0.1, 'e2': 0.1,
             'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5,
             #'e1': -0.5, 'e2': -0.5,
             'center_x': -0.2, 'center_y': -0.2}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0,
             #'e1': 0.5, 'e2': 0.5,
             'center_x': 0.2, 'center_y': 0.2}]
        kwargs_lens_light_fixed = [{}]
        add_uniform_light = True
        if add_uniform_light:
            kwargs_uniform, kwargs_uniform_sigma, kwargs_uniform_fixed, \
            kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light(20.0, 2.0)
            lens_light_model_list += ['UNIFORM']
            kwargs_lens_light_init += kwargs_uniform
            kwargs_lens_light_sigma += kwargs_uniform_sigma
            kwargs_lens_light_fixed += kwargs_uniform_fixed
            kwargs_lower_lens_light += kwargs_uniform_lower
            kwargs_upper_lens_light += kwargs_uniform_upper
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

        return lens_light_model_list, lens_light_params

class HE1113ModelEPLM3M4Shear(_HE1113):

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.5/2):

        super(HE1113ModelEPLM3M4Shear, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 5e-3,
                             'source_position_tolerance': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.axis_ratio_masslight_alignment,
                             }
        return kwargs_likelihood

    def joint_lens_with_light_prior(self, kwargs_lens,
                                  kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                                  kwargs_extinction, kwargs_tracer_source):

        center_x_lens, center_y_lens = kwargs_lens[0]['center_x'], kwargs_lens[0]['center_y']
        center_x_light, center_y_light = kwargs_lens_light[0]['center_x'], kwargs_lens_light[0]['center_y']
        sigma = 0.005

        # penalize high ellipticity
        e1, e2 = kwargs_lens[0]['e1'], kwargs_lens[0]['e2']
        _, q = ellipticity2phi_q(e1, e2)
        if q < 0.2:
            return -1e9
        q_prior = -0.5 * (q - 0.9) ** 2 / 0.1**2
        return q_prior + \
               -0.5 * ((center_x_lens - center_x_light)**2 / sigma ** 2 + (center_y_lens - center_y_light)**2 / sigma**2)


    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR']
        kwargs_lens_macro = [
            {'theta_E': 0.3536519255418004, 'gamma': 1.9758557647300843, 'e1': -0.12765273133916696,
             'e2': 0.4009752016571935, 'center_x': -0.050765861657884534, 'center_y': -0.024413071923722968,
             'a1_a': 0.0, 'delta_phi_m1': -0.043994984823729585, 'a3_a': 0.0, 'delta_phi_m3': -0.32721968983036276,
             'a4_a': 0.0, 'delta_phi_m4': -0.39941320567259203},
            {'gamma1': -0.04143178211377716, 'gamma2': 0.11000978389143896, 'ra_0': 0.0, 'dec_0': 0.0}
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
                             {'gamma1': 0.05, 'gamma2': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.4, 'e2': -0.4, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.4, 'e2': 0.4, 'gamma': 2.6, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

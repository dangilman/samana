from samana.Model.model_base import ModelBase
import numpy as np
import pickle


class _HE0435ModelBase(ModelBase):

    def __init__(self, data_class, kde_sampler, shapelets_order, include_source_blobs):
        self._shapelets_order = shapelets_order
        self._include_souce_blobs = include_source_blobs
        self._image_plane_source_list = None
        super(_HE0435ModelBase, self).__init__(data_class, kde_sampler)

    def update_kwargs_fixed_macro(self, lens_model_list_macro, kwargs_lens_fixed, kwargs_lens_init,
                                  macromodel_samples_fixed=None):

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
        kwargs_constraints = {
                        'joint_source_with_point_source': joint_source_with_point_source,
                              'num_point_source_list': [len(self._data.x_image)],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True
                              }
        if self._shapelets_order is not None:
            kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        kwargs_constraints['image_plane_source_list'] = self._image_plane_source_list
        return kwargs_constraints

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.2]]

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 19.64101030043808, 'R_sersic': 0.14029734050422163, 'n_sersic': 0.54126896396064,
             'e1': 0.4610724942007716, 'e2': -0.21741965386966786,
             'center_x': -0.23567361310239077, 'center_y': -0.39359767377131005}
            ]
        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05,
                                'center_y': 0.05}]
        kwargs_lower_source = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [
            {'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]
        self._image_plane_source_list = [False]

        if self._shapelets_order is not None:
            n_max = int(self._shapelets_order)
            source_model_list += ['SHAPELETS']
            kwargs_source_init += [{'amp': 1.0, 'beta': 0.34, 'center_x': 0.018, 'center_y': -0.031,
                                    'n_max': n_max}]
            kwargs_source_sigma += [{'amp': 10.0, 'beta': 0.1, 'center_x': 0.1, 'center_y': 0.1, 'n_max': 1}]
            kwargs_lower_source += [{'amp': 1e-9, 'beta': 0.01, 'center_x': -10.0, 'center_y': -10.0, 'n_max': 0}]
            kwargs_upper_source += [{'amp': 100.0, 'beta': 1.0, 'center_x': 10.0, 'center_y': 10.0, 'n_max': n_max + 1}]
            kwargs_source_fixed += [{'n_max': n_max}]
            self._image_plane_source_list += [False]

        if self._include_souce_blobs:
            self._image_plane_source_list += [True]
            point_of_interest_x1 = -0.42
            point_of_interest_y1 = -1.02
            source_model_list_clump, kwargs_source_clump, kwargs_source_sigma_clump, kwargs_source_fixed_clump, \
            kwargs_lower_source_clump, kwargs_upper_source_clump = self.shapelet_source_clump(point_of_interest_x1,
                                                                                              point_of_interest_y1,
                                                                                              n_max_clump=6,
                                                                                              beta_clump=0.1)
            source_model_list += source_model_list_clump
            kwargs_source_init += kwargs_source_clump
            kwargs_source_sigma += kwargs_source_sigma_clump
            kwargs_lower_source += kwargs_lower_source_clump
            kwargs_upper_source += kwargs_upper_source_clump
            kwargs_source_fixed += kwargs_source_fixed_clump

        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                             kwargs_upper_source]

        return source_model_list, source_params

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [
            {'amp': 24.8285061030015, 'R_sersic': 0.922145233299516, 'n_sersic': 4.233061033032083,
             'e1': -0.08065215326353539, 'e2': -0.060455206113180796, 'center_x': -0.00020941488474152307,
             'center_y': -0.01818193971930611}
            ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_light_fixed = [{}, {}]
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed,
                             kwargs_lower_lens_light,
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


class HE0435ModelEPLM3M4ShearObservedConvention(_HE0435ModelBase):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None, include_source_blobs=False):
        super(HE0435ModelEPLM3M4ShearObservedConvention, self).__init__(data_class, kde_sampler,
                                                                        shapelets_order, include_source_blobs)

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1], [0, 'a4_a', 0.0, 0.01], [0, 'a3_a', 0.0, 0.005],
                [2, 'center_x', -2.6, 0.2],
                [2, 'center_y', -3.65, 0.2],
                [2, 'theta_E', 0.25, 0.1]
                ]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        # satellite observed position: -2.6 -3.65
        # satellite inferred position from lens mdoel: -2.4501, -3.223
        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4', 'SHEAR', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 1.1741891086217888, 'gamma': 1.9506480641206367, 'e1': -0.09657889891955888,
             'e2': -0.042687865798191844, 'center_x': -0.0002228599671922995, 'center_y': -0.018656741492927596,
             'a3_a': 0.0, 'delta_phi_m3': -0.12605275164268495, 'a4_a': 0.0, 'delta_phi_m4': 0.835345646895161},
            {'gamma1': 0.01051459539863377, 'gamma2': 0.038909734445549675, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.48257693675612023, 'center_x': -2.5130483846833798, 'center_y': -4.288159799886844}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens,
                               0.78]
        index_lens_split = [0, 1]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.05,
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi / 12, 'delta_phi_m4': np.pi / 16},
                             {'gamma1': 0.05, 'gamma2': 0.05}, {'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi / 6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.001, 'center_x': -10, 'center_y': -10}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi / 6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
            {'theta_E': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params


class HE0435ModelEPLM3M4Shear(HE0435ModelEPLM3M4ShearObservedConvention):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None, include_source_blobs=False):
        super(HE0435ModelEPLM3M4Shear, self).__init__(data_class, kde_sampler, shapelets_order, include_source_blobs)

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1], [0, 'a4_a', 0.0, 0.01], [0, 'a3_a', 0.0, 0.005],
                ]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        # satellite observed position: -2.6 -3.65
        # satellite inferred position from lens mdoel: -2.4501, -3.223
        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4_ELL', 'SHEAR', 'SIS']
        kwargs_lens_macro = [{'theta_E': 1.183879459825199, 'gamma': 2.1932908582668946, 'e1': 0.036681457593259845,
                              'e2': 0.03307337024314603, 'center_x': -0.006229479709622775,
                              'center_y': -0.0016358162588540096, 'a3_a': 0.0,
                              'delta_phi_m3': -0.1584880442965335, 'a4_a': 0.0,
                              'delta_phi_m4': -0.051442949784854586},
                             {'gamma1': -0.02777010089244484, 'gamma2': -0.04988889392094336,
                              'ra_0': 0.0, 'dec_0': 0.0},
                             {'theta_E': 0.37122592808914573,
                              'center_x': -2.1380, 'center_y': -3.1583}]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens,
                               0.78]
        index_lens_split = [0, 1]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi / 12, 'delta_phi_m4': np.pi / 16},
                             {'gamma1': 0.05, 'gamma2': 0.05}, {'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi / 6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.001, 'center_x': -10, 'center_y': -10}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi / 6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
            {'theta_E': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params


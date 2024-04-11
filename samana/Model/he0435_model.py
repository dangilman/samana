from samana.Model.model_base import ModelBase
import numpy as np
import pickle


class _HE0435ModelBase(ModelBase):

    def __init__(self, data_class, kde_sampler, shapelets_order):
        self._shapelets_order = shapelets_order
        super(_HE0435ModelBase, self).__init__(data_class, kde_sampler)

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
        kwargs_source_init = [{'amp': 10.820966460179898, 'R_sersic': 0.3331897774564813, 'n_sersic': 4.6641799992836095,
                               'e1': 0.12145758662327957, 'e2': -0.05831861796908974, 'center_x': -0.2269010853766411,
                               'center_y': -0.26496947342308347}]
        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05,
                                'center_y': 0.05}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]

        if self._shapelets_order is not None:
            n_max = int(self._shapelets_order)
            source_model_list += ['SHAPELETS']
            kwargs_source_init += [{'amp': 1.0, 'beta': 0.3, 'center_x': 0.018, 'center_y': -0.031,
                                    'n_max': n_max}]
            kwargs_source_sigma += [{'amp': 10.0, 'beta': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'n_max': 1}]
            kwargs_lower_source += [{'amp': 1e-9, 'beta': 0.0, 'center_x': -10.0, 'center_y': -10.0, 'n_max': 0}]
            kwargs_upper_source += [{'amp': 100.0, 'beta': 1.0, 'center_x': 10.0, 'center_y': 10.0, 'n_max': n_max+1}]
            kwargs_source_fixed += [{'n_max': n_max}]

        # beta_x_clump = [0.0]
        # beta_y_clump = [0.0]
        # x_clump_sigma = [0.2]
        # y_clump_sigma = [0.2]
        # clump_model = ['GAUSSIAN_ELLIPSE'] * len(beta_x_clump)
        # for (clump_x, clump_y, dx, dy, clump_model) in zip(beta_x_clump, beta_y_clump, x_clump_sigma, y_clump_sigma, clump_model):
        #     if clump_model == 'GAUSSIAN_ELLIPSE':
        #         clump_model_list, kwargs_clump, clump_kwargs_sigma, kwargs_clump_fixed, \
        #             kwargs_lower_clump, kwargs_upper_clump = self.gaussian_source_clump(clump_x, clump_y, dx, dy)
        #     elif clump_model == 'SHAPELETS':
        #         clump_model_list, kwargs_clump, clump_kwargs_sigma, kwargs_clump_fixed, \
        #             kwargs_lower_clump, kwargs_upper_clump = self.shapetlet_source_clump(clump_x, clump_y, dx, dy)
        #     else:
        #         raise Exception('clumps must be GAUSSIAN or SHAPELETS')
        #     source_model_list += clump_model_list
        #     kwargs_source_init += kwargs_clump
        #     kwargs_source_sigma += clump_kwargs_sigma
        #     kwargs_source_fixed += kwargs_clump_fixed
        #     kwargs_lower_source += kwargs_lower_clump
        #     kwargs_upper_source += kwargs_upper_clump

        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]

        return source_model_list, source_params

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE', 'UNIFORM']
        kwargs_lens_light_init = [{'amp': 22.70814537558557, 'R_sersic': 0.9675848101146824,
                                   'n_sersic': 4.397374544488779, 'e1': -0.10138625101281855,
                                   'e2': -0.05019253943570243, 'center_x': -0.0014625368200421815,
                                   'center_y': -0.020496884983990455}, {'amp': 1.0}]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
            {'amp': 5}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
            {'amp': -100}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}, {'amp': 100}]
        kwargs_lens_light_fixed = [{}, {}]
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

        return lens_light_model_list, lens_light_params

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': True,
                             'source_marg': False,
                             'image_position_uncertainty': 0.005,
                             'source_position_likelihood': False,
                             'check_matched_source_position': True,
                             'source_position_sigma': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

class HE0435ModelEPLM3M4ShearObservedConvention(_HE0435ModelBase):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None):
        super(HE0435ModelEPLM3M4ShearObservedConvention, self).__init__(data_class, kde_sampler, shapelets_order)

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.2], [0, 'a4_a', 0.0, 0.01], [0, 'a3_a', 0.0, 0.005],
                #[2, 'center_x', -2.6, 0.1],
                #[2, 'center_y', -3.65, 0.1],
                #[2, 'theta_E', 0.25, 0.1]
                ]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        # satellite observed position: -2.6 -3.65
        # satellite inferred position from lens mdoel: -1.915, -3.0173
        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4', 'SHEAR', 'SIS']
        kwargs_lens_macro = [{'theta_E': 1.2363967422373972, 'gamma': 2.2217392289109097, 'e1': -0.3695749167777887,
                              'e2': -0.1894447241763164, 'center_x': 0.008717948372366814,
                              'center_y': -0.04343534216241877, 'a3_a': 0.0, 'delta_phi_m3': -0.5080741201623616,
                              'a4_a': 0.0, 'delta_phi_m4': 0.0},
                             {'gamma1': -0.01935293564446043, 'gamma2': 0.01759784833305446, 'ra_0': 0.0,
                              'dec_0': 0.0}, {'theta_E': 0.35, 'center_x': -2.539340372689504,
                                              'center_y': -3.592817067434825}]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens,
                               0.78]
        index_lens_split = [0, 1]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05}, {'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.001, 'center_x': -10, 'center_y': -10}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 3.5, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

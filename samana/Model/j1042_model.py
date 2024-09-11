from samana.Model.model_base import ModelBase
import numpy as np
import pickle


class _J1042ModelBase(ModelBase):

    def __init__(self, data_class, kde_sampler, shapelets_order):
        self._shapelets_order = shapelets_order
        super(_J1042ModelBase, self).__init__(data_class, kde_sampler)

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
        #joint_lens_light_with_lens_light = [[0, 1, ['center_x', 'center_y']]]
        kwargs_constraints = {
                            'joint_source_with_point_source': joint_source_with_point_source,
                              #'joint_lens_light_with_lens_light': joint_lens_light_with_lens_light,
                              'num_point_source_list': [len(self._data.x_image)],
                                'joint_lens_with_light': [[1, 2, ['center_x', 'center_y']]],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True,
            'image_plane_source_list': self._image_plane_source_list
                              }
        if self._shapelets_order is not None:
            kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.2]]

    def setup_source_light_model(self):

        self._image_plane_source_list = [False]
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 1, 'R_sersic': 6.185699807429242, 'n_sersic': 3.714144525940761, 'e1': -0.33146582654012263,
             'e2': 0.25777142083278803, 'center_x': 0.004471372006364254, 'center_y': 0.08963660479313908}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]

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

        # self._image_plane_source_list += [True]
        # point_of_interest_x1 = -0.95
        # point_of_interest_y1 = 0.0
        # source_model_list_clump, kwargs_source_clump, kwargs_source_sigma_clump, kwargs_source_fixed_clump, \
        # kwargs_lower_source_clump, kwargs_upper_source_clump = self.gaussian_source_clump(point_of_interest_x1,
        #                                                                                   point_of_interest_y1,
        #                                                                                   0.05)
        # source_model_list += source_model_list_clump
        # kwargs_source_init += kwargs_source_clump
        # kwargs_source_sigma += kwargs_source_sigma_clump
        # kwargs_lower_source += kwargs_lower_source_clump
        # kwargs_upper_source += kwargs_upper_source_clump
        # kwargs_source_fixed += kwargs_source_fixed_clump

        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]

        return source_model_list, source_params

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE',
                                 #'SERSIC_ELLIPSE',
                                 'SERSIC_ELLIPSE']
        kwargs_lens_light_init = [
            {'amp': 1, 'R_sersic': 0.8751746135510492, 'n_sersic': 4.032484946977566, 'e1': 0.03706113533782985,
             'e2': -0.03907980039597373, 'center_x': 0.05279135524046345, 'center_y': 0.019180403175134753},
            #{'amp': 1, 'R_sersic': 1.5228517698717252, 'n_sersic': 3.9012745982754726, 'e1': -0.09275713906510573,
            # 'e2': 0.16998845939896, 'center_x': 0.05279135524046345, 'center_y': 0.019180403175134753},
            {'amp': 1.0, 'R_sersic': 0.11902191994454238,
             'n_sersic': 4.0, 'e1': 0.1, 'e2': 0.1, 'center_x': self._data.gx, 'center_y': self._data.gy}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
            #{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
        {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.1, 'e2': 0.1,}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
           # {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': self._data.gx - 0.3, 'center_y': self._data.gy - 0.3}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
           # {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
        {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': self._data.gx + 0.3, 'center_y': self._data.gx + 0.3}]
        kwargs_lens_light_fixed = [{},
                                  # {},
                                   {}]

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

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 5e-3,
                             'source_position_likelihood': False,
                             #'check_matched_source_position': False,
                             'source_position_sigma': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

class J1042ModelEPLM3M4Shear(_J1042ModelBase):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None):
        super(J1042ModelEPLM3M4Shear, self).__init__(data_class, kde_sampler, shapelets_order)

    @property
    def prior_lens(self):
        return self.population_gamma_prior + [
                [2, 'center_x', self._data.gx, 0.2],
                [2, 'center_y', self._data.gx, 0.2],
                [2, 'theta_E', 0.05, 0.1]]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4_ELL', 'SHEAR', 'SIS']
        kwargs_lens_macro = [{'theta_E': 0.882739551123883, 'gamma': 1.9883321182031855, 'e1': -0.18132040488439743,
                              'e2': 0.07420431700424802, 'center_x': 0.019086787194673605, 'center_y': 0.0671528454081338,
                              'a3_a': 0.0, 'delta_phi_m3': -0.1006435539748753, 'a4_a': 0.0,
                              'delta_phi_m4': -0.7592985744255293},
                             {'gamma1': -0.08343148556542614, 'gamma2': -0.015054336151458486,
                              'ra_0': 0.0, 'dec_0': 0.0},
                             {'theta_E': 0.05,
                              'center_x': self._data.gx,
                              'center_y': self._data.gy}]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2]
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
            {'theta_E': 0.0, 'center_x': -10, 'center_y': -10}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.2, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

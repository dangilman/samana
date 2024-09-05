from samana.Model.model_base import ModelBase
import numpy as np
import pickle


class _HE0435NircamModelBase(ModelBase):

    def __init__(self, data_class, kde_sampler, shapelets_order, include_source_blobs, n_max_blobs):
        self._shapelets_order = shapelets_order
        self._include_souce_blobs = include_source_blobs
        self._n_max_blobs = n_max_blobs
        self._image_plane_source_list = None
        super(_HE0435NircamModelBase, self).__init__(data_class, kde_sampler)

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
        kwargs_constraints['image_plane_source_list'] = self._image_plane_source_list
        return kwargs_constraints

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.2]]

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [{'amp': -17.637157609965968, 'R_sersic': 0.33100213287380237, 'n_sersic': 5.544368831579415,
                               'e1': 0.4579884763037544, 'e2': -0.013601747968084579, 'center_x': -0.17942909396556667,
                               'center_y': -0.27771056590074705}]
        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05,
                                'center_y': 0.05}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]
        self._image_plane_source_list = [False]

        if self._shapelets_order is not None:
            n_max = int(self._shapelets_order)
            source_model_list += ['SHAPELETS']
            kwargs_source_init += [{'amp': 1.0, 'beta': 0.34, 'center_x': 0.018, 'center_y': -0.031,
                                    'n_max': n_max}]
            kwargs_source_sigma += [{'amp': 10.0, 'beta': 0.1, 'center_x': 0.1, 'center_y': 0.1, 'n_max': 1}]
            kwargs_lower_source += [{'amp': 1e-9, 'beta': 0.01, 'center_x': -10.0, 'center_y': -10.0, 'n_max': 0}]
            kwargs_upper_source += [{'amp': 100.0, 'beta': 1.0, 'center_x': 10.0, 'center_y': 10.0, 'n_max': n_max+1}]
            kwargs_source_fixed += [{'n_max': n_max}]
            self._image_plane_source_list += [False]

        if self._include_souce_blobs:
            self._image_plane_source_list += [True]
            point_of_interest_x1 = -1.025
            point_of_interest_y1 = 0.22
            source_model_list_clump, kwargs_source_clump, kwargs_source_sigma_clump, kwargs_source_fixed_clump, \
            kwargs_lower_source_clump, kwargs_upper_source_clump = self.shapelet_source_clump(point_of_interest_x1,
                                                                                              point_of_interest_y1,
                                                                                              n_max_clump=self._n_max_blobs,
                                                                                              beta_clump=0.05)
            source_model_list += source_model_list_clump
            kwargs_source_init += kwargs_source_clump
            kwargs_source_sigma += kwargs_source_sigma_clump
            kwargs_lower_source += kwargs_lower_source_clump
            kwargs_upper_source += kwargs_upper_source_clump
            kwargs_source_fixed += kwargs_source_fixed_clump
            #
            # self._image_plane_source_list += [True]
            # point_of_interest_x2 = 0.475
            # point_of_interest_y2 = -1.2
            # source_model_list_clump, kwargs_source_clump, kwargs_source_sigma_clump, kwargs_source_fixed_clump, \
            # kwargs_lower_source_clump, kwargs_upper_source_clump = self.shapelet_source_clump(point_of_interest_x2,
            #                                                                                   point_of_interest_y2,
            #                                                                                   beta_clump=0.05,
            #                                                                                   n_max_clump=5)
            # source_model_list += source_model_list_clump
            # kwargs_source_init += kwargs_source_clump
            # kwargs_source_sigma += kwargs_source_sigma_clump
            # kwargs_lower_source += kwargs_lower_source_clump
            # kwargs_upper_source += kwargs_upper_source_clump
            # kwargs_source_fixed += kwargs_source_fixed_clump
            #
            # self._image_plane_source_list += [True]
            # point_of_interest_x3 = -0.04
            # point_of_interest_y3 = 1.44
            # source_model_list_clump, kwargs_source_clump, kwargs_source_sigma_clump, kwargs_source_fixed_clump, \
            # kwargs_lower_source_clump, kwargs_upper_source_clump = self.shapelet_source_clump(point_of_interest_x3,
            #                                                                                   point_of_interest_y3,
            #                                                                                   n_max_clump=5,
            #                                                                                   beta_clump=0.05)
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

        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [
            {'amp': 306.15179761817427, 'R_sersic': 1.028061447952791, 'n_sersic': 4.443232182177677,
             'e1': 0.0559315849182795, 'e2': 0.09341343307121905, 'center_x': 0.0013421254629133244,
             'center_y': -0.010411140290881226}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_light_fixed = [{}]
        #
        # kwargs_uniform, kwargs_uniform_sigma, kwargs_uniform_fixed, \
        # kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light()
        # lens_light_model_list += ['UNIFORM']
        # kwargs_lens_light_init += kwargs_uniform
        # kwargs_lens_light_sigma += kwargs_uniform_sigma
        # kwargs_lens_light_fixed += kwargs_uniform_fixed
        # kwargs_lower_lens_light += kwargs_uniform_lower
        # kwargs_upper_lens_light += kwargs_uniform_upper

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


class HE0435ModelNircamEPLM3M4ShearObservedConvention(_HE0435NircamModelBase):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None, include_source_blobs=False,
                 n_max_blobs=8):
        super(HE0435ModelNircamEPLM3M4ShearObservedConvention, self).__init__(data_class,
                                                                              kde_sampler,
                                                                              shapelets_order,
                                                                              include_source_blobs,
                                                                              n_max_blobs)

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1], [0, 'a4_a', 0.0, 0.01], [0, 'a3_a', 0.0, 0.005],
                [2, 'center_x', -4.2, 0.2],
                [2, 'center_y', 1.35, 0.2],
                [2, 'theta_E', 0.25, 0.1]
                ]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4', 'SHEAR', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 1.1778927407630202, 'gamma': 2.1715769335116764, 'e1': 0.05819312551789523,
             'e2': 0.1199644307864586, 'center_x': 0.0007622692384399802, 'center_y': -0.0027688317014858007,
             'a3_a': 0.0, 'delta_phi_m3': -0.13807471874163363, 'a4_a': 0.0, 'delta_phi_m4': -0.0875170451788644},
            {'gamma1': -0.005759199062677984, 'gamma2': -0.05226478062092215, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.36940399665014434, 'center_x': -4.208596548630573, 'center_y': 1.363938783578341}
        ]
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
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class HE0435ModelNircamEPLM3M4Shear(_HE0435NircamModelBase):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None, include_source_blobs=False,
                 n_max_blobs=8):
        super(HE0435ModelNircamEPLM3M4Shear, self).__init__(data_class,
                                                              kde_sampler,
                                                              shapelets_order,
                                                              include_source_blobs,
                                                              n_max_blobs)

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1]]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        # satellite observed position: -2.6 -3.65
        # satellite inferred position from lens mdoel: -2.4501, -3.223
        if self._spherical_multipole:
            lens_model_list_macro = ['EPL_MULTIPOLE_M3M4', 'SHEAR', 'SIS']
        else:
            lens_model_list_macro = ['EPL_MULTIPOLE_M3M4_ELL', 'SHEAR', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 1.1778927407630202, 'gamma': 2.1715769335116764, 'e1': 0.05819312551789523,
             'e2': 0.1199644307864586, 'center_x': 0.0007622692384399802, 'center_y': -0.0027688317014858007,
             'a3_a': 0.0, 'delta_phi_m3': -0.13807471874163363, 'a4_a': 0.0, 'delta_phi_m4': -0.0875170451788644},
            {'gamma1': -0.005759199062677984, 'gamma2': -0.05226478062092215, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.36940399665014434, 'center_x': -3.6631, 'center_y': 1.0098}
            ]
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
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]

        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params


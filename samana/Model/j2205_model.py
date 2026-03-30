from samana.Model.model_base import EPLModelBase
import numpy as np
import pickle


class _J2205ModelBase(EPLModelBase):

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
        return [[0, 'gamma', 2.0, 0.1]]

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 0.0659771157586616, 'R_sersic': 6.540004694391513, 'n_sersic': 4.096910989043003,
             'e1': -0.21182917732567763, 'e2': -0.0027189327773029354,
             'center_x': 0.07313335903091817, 'center_y': 0.016938006487960656}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
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

        gal_x = -1.122
        gal_y = 0.194
        lens_light_model_list = ['SERSIC_ELLIPSE',
                                 #'SERSIC'
                                 ]
        kwargs_lens_light_init = [
            {'amp': 1.2626913540852998, 'R_sersic': 1.8729178532743398, 'n_sersic': 7.87202614104127, 'e1': -0.08893059240200864,
             'e2': -0.07511257013762268, 'center_x': -0.004101287731659908, 'center_y': 0.006985962364087441},
            # {'amp': 10.662270851598919, 'R_sersic': 0.11902191994454238,
            #  'n_sersic': 2.9729265138510583, 'center_x': gal_x,
            #  'center_y': gal_y}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
           # {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.025, 'center_y': 0.025}
        ]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
            #{'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': gal_x - 0.2, 'center_y': gal_y - 0.2}
            ]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
           # {'R_sersic': 10, 'n_sersic': 10.0, 'center_x': gal_x + 0.2, 'center_y': gal_y + 0.2}
        ]
        kwargs_lens_light_fixed = [{},
                                   #{}
                                   ]

        add_uniform_component = True
        if add_uniform_component:
            lens_light_model_list += ['UNIFORM']
            kwargs_light_uniform, kwargs_light_sigma_uniform, kwargs_light_fixed_uniform, \
                kwargs_lower_light_uniform, kwargs_upper_light_uniform = self.add_uniform_lens_light(-1.28, 1.0)

            kwargs_lens_light_init += kwargs_light_uniform
            kwargs_lens_light_sigma += kwargs_light_sigma_uniform
            kwargs_lens_light_fixed += kwargs_light_fixed_uniform
            kwargs_lower_lens_light += kwargs_lower_light_uniform
            kwargs_upper_lens_light += kwargs_upper_light_uniform

        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

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
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.axis_ratio_prior_with_light
                             }
        return kwargs_likelihood

class J2205ModelEPLM3M4Shear(_J2205ModelBase):

    @property
    def prior_lens(self):
        return None

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR'
                                 ]
        kwargs_lens_macro = [
                {'theta_E': 0.7730230170044669, 'gamma': 2.147359102535725, 'e1': -0.23567956628857506,
                 'e2': -0.06287521979067849, 'center_x': -0.03144469295868643, 'center_y': 0.006302096466668356,
                 'a1_a': 0.0, 'delta_phi_m1': 0.0, 'a3_a': 0.0, 'delta_phi_m3': 0.1854133418529986, 'a4_a': 0.0, 'delta_phi_m4': 1.6209400356133206},
            {'gamma1': 0.028751121688242592, 'gamma2': -0.002500086891443659, 'ra_0': 0.0, 'dec_0': 0.0}
            ]

        redshift_list_macro = [self._data.z_lens, self._data.z_lens,
                               #self._data.z_lens
                               ]
        index_lens_split = [0, 1,
                            #2
                            ]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                            # {'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}
                             ]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0},
                             #{}
                             ]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.4, 'e2': -0.4, 'gamma': 1.7, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
        #{'theta_E': 0.0, 'center_x': gal_x - 0.2, 'center_y': gal_y - 0.2}
        ]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.4, 'e2': 0.4, 'gamma': 2.5, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        #{'theta_E': 0.5, 'center_x': gal_x + 0.2, 'center_y': gal_y + 0.2}
            ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class J2205ModelEPLM3M4Shear_NIRCam(J2205ModelEPLM3M4Shear):

    @property
    def kwargs_constraints(self):
        joint_source_with_point_source = [[0, 0]]
        joint_light_with_light = [[0, 1, ['center_x', 'center_y']]]
        kwargs_constraints = {'joint_source_with_point_source': joint_source_with_point_source,
                              'joint_lens_light_with_lens_light': joint_light_with_light,
                              'num_point_source_list': [len(self._data.x_image)],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True
                              }
        if self._shapelets_order is not None:
            kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR'
                                 ]
        kwargs_lens_macro = [
            {'theta_E': 0.7616487831127611, 'gamma': 2.24371988177323, 'e1': -0.101396687225481,
             'e2': -0.0335256031399618, 'center_x': -0.004050554812044212, 'center_y': 0.020543517136614845,
             'a1_a': 0.0, 'delta_phi_m1': -0.7063288941566399, 'a3_a': 0.0, 'delta_phi_m3': 0.26379463085090477,
             'a4_a': 0.0, 'delta_phi_m4': 2.276427260959404},
            {'gamma1': 0.07450676803637994, 'gamma2': 0.007782171470565312, 'ra_0': 0.0, 'dec_0': 0.0}
            ]

        redshift_list_macro = [self._data.z_lens, self._data.z_lens
                               ]
        index_lens_split = [0, 1,
                            #2
                            ]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                            # {'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}
                             ]
        kwargs_lens_fixed = [{},
                             {'ra_0': 0.0, 'dec_0': 0.0},
                             #{}
                             ]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.4, 'e2': -0.4, 'gamma': 1.7, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
        #{'theta_E': 0.0, 'center_x': gal_x - 0.2, 'center_y': gal_y - 0.2}
        ]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.4, 'e2': 0.4, 'gamma': 2.5, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        #{'theta_E': 0.5, 'center_x': gal_x + 0.2, 'center_y': gal_y + 0.2}
            ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE',
                                 'SERSIC_ELLIPSE'
                                 ]
        kwargs_lens_light_init = [
            {'amp': 72.75455873454605, 'R_sersic': 1.9294713090133775, 'n_sersic': 9.627536150484303,
             'e1': -0.09851411370215975, 'e2': -0.0966763917500797, 'center_x': -0.0007789675573719373,
             'center_y': 0.014111857562748182},
            {'amp': -14.705120652740344, 'R_sersic': 1.768326764555597, 'n_sersic': 8.23538544773847,
             'e1': -0.07970078894963104, 'e2': -0.18252363813402986, 'center_x': -0.0007789675573719373,
             'center_y': 0.014111857562748182}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
           {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
        ]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
            ]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
           {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
        ]
        kwargs_lens_light_fixed = [{},
                                   {}
                                   ]

        add_uniform_component = False
        if add_uniform_component:
            lens_light_model_list += ['UNIFORM']
            kwargs_light_uniform, kwargs_light_sigma_uniform, kwargs_light_fixed_uniform, \
                kwargs_lower_light_uniform, kwargs_upper_light_uniform = self.add_uniform_lens_light(-1.28, 1.0)

            kwargs_lens_light_init += kwargs_light_uniform
            kwargs_lens_light_sigma += kwargs_light_sigma_uniform
            kwargs_lens_light_fixed += kwargs_light_fixed_uniform
            kwargs_lower_lens_light += kwargs_lower_light_uniform
            kwargs_upper_lens_light += kwargs_upper_light_uniform

        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

        return lens_light_model_list, lens_light_params

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 7.6629840229670245, 'R_sersic': 6.561028288590462, 'n_sersic': 4.08190207200481,
             'e1': -0.2199696824193018, 'e2': 0.018923867374456796, 'center_x': 0.09917205326327172,
             'center_y': 0.03900899666030728}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
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

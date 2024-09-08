from samana.Model.model_base import ModelBase
import numpy as np
import pickle


class _J2205ModelBase(ModelBase):

    def __init__(self, data_class, kde_sampler, shapelets_order):
        self._shapelets_order = shapelets_order
        super(_J2205ModelBase, self).__init__(data_class, kde_sampler)

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
                              'point_source_offset': True,
                              #'joint_lens_with_light': [[1, 2, ['center_x', 'center_y']]]
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
            source_model_list += ['SHAPELETS']
            kwargs_source_init += [{'amp': 1.0, 'beta': 0.2, 'center_x': 0.018, 'center_y': -0.031,
                                    'n_max': n_max}]
            kwargs_source_sigma += [{'amp': 10.0, 'beta': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'n_max': 1}]
            kwargs_lower_source += [{'amp': 10.0, 'beta': 0.0, 'center_x': -10.0, 'center_y': -10.0, 'n_max': 0}]
            kwargs_upper_source += [{'amp': 10.0, 'beta': 0.5, 'center_x': 10.0, 'center_y': 10.0, 'n_max': n_max+1}]
            kwargs_source_fixed += [{'n_max': n_max}]

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
                             'source_position_likelihood': False,
                             #'check_matched_source_position': False,
                             'source_position_sigma': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

class J2205ModelEPLM3M4Shear(_J2205ModelBase):

    def __init__(self, data_class, kde_sampler=None, shapelets_order=None):
        super(J2205ModelEPLM3M4Shear, self).__init__(data_class, kde_sampler, shapelets_order)

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.1]]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        gal_x = -1.122
        gal_y = 0.194
        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4_ELL', 'SHEAR',
                                 #'SIS'
                                 ]
        kwargs_lens_macro = [
            {'theta_E': 0.7565025832986526, 'gamma': 2.067171862904866, 'e1': -0.02628738394280838,
             'e2': 0.010309735437349511, 'center_x': 0.00294258728356529,
             'center_y': 0.003458909721253912, 'a3_a': 0.0, 'delta_phi_m3': -0.3139572526104714,
             'a4_a': 0.0, 'delta_phi_m4': 1.8154962399165802},
            {'gamma1': 0.0830198688490999, 'gamma2': 0.02321987501579647,
             'ra_0': 0.0, 'dec_0': 0.0},
            #{'theta_E': 0.0058908649559920164, 'center_x': -1.1201501675379746, 'center_y': 0.19552617530411562}
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
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                            # {'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}
                             ]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0},
                             #{}
                             ]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.4, 'e2': -0.4, 'gamma': 1.7, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
        #{'theta_E': 0.0, 'center_x': gal_x - 0.2, 'center_y': gal_y - 0.2}
        ]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.4, 'e2': 0.4, 'gamma': 2.5, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        #{'theta_E': 0.5, 'center_x': gal_x + 0.2, 'center_y': gal_y + 0.2}
            ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

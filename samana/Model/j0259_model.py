from samana.Model.model_base import EPLModelBase
import numpy as np
import pickle


class _J0259ModelBase(EPLModelBase):

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
            {'amp': 2.5366893078821025, 'R_sersic': 0.4166095652205182, 'n_sersic': 3.563437423048624,
             'e1': -0.06601957042669272, 'e2': -0.00226484917443781,
             'center_x': -0.048041075911909224, 'center_y': 0.014409420481205734}
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

        if self._data.band == 'F814W':
            lens_light_model_list = ['SERSIC_ELLIPSE']
            kwargs_lens_light_init = [
                {'amp': 0.20074624938399432, 'R_sersic': 2.097226414051843,
                 'n_sersic': 4.0, 'e1': -0.2769940052424607,
                 'e2': 0.2530155023529464, 'center_x': -0.008330245657409108,
                 'center_y': 0.002286411273696616}
            ]
            kwargs_lens_light_sigma = [
                {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.025, 'center_y': 0.025}]
            kwargs_lower_lens_light = [
                {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -0.15, 'center_y': -0.15}]
            kwargs_upper_lens_light = [
                {'R_sersic': 10, 'n_sersic': 8.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.15, 'center_y': 0.15}]
            kwargs_lens_light_fixed = [{}]
        elif self._data.band == 'F475X':
            lens_light_model_list = ['SERSIC']
            kwargs_lens_light_init = [
                {'amp': 1, 'R_sersic': 2.2467253226957458,
                 'n_sersic': 3.3911670131456577, 'center_x': 0.0025886976171241058,
                 'center_y': 0.0226415303675161}
            ]
            kwargs_lens_light_sigma = [
                {'R_sersic': 0.05, 'n_sersic': 0.25,
                 #'e1': 0.1, 'e2': 0.1,
                 'center_x': 0.025, 'center_y': 0.025}]
            kwargs_lower_lens_light = [
                {'R_sersic': 0.001, 'n_sersic': 0.5,
                 #'e1': -0.5, 'e2': -0.5,
                 'center_x': -0.15, 'center_y': -0.15}]
            kwargs_upper_lens_light = [
                {'R_sersic': 10, 'n_sersic': 10.0,
                # 'e1': 0.5, 'e2': 0.5,
                 'center_x': 0.15, 'center_y': 0.15}]
            kwargs_lens_light_fixed = [{}]
        else:
            raise Exception('image data band must be F814W or F475X')

        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                                     kwargs_upper_lens_light]


        return lens_light_model_list, lens_light_params

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 5e-3,
                             'source_position_tolerance': 0.0001,
                             'prior_lens': self.prior_lens,
                             'custom_logL_addition': self.axis_ratio_prior,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

class J0259ModelEPLM3M4Shear(_J0259ModelBase):

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR']
        if self._data.band == 'F814W':
            kwargs_lens_macro = [
                {'theta_E': 0.7414035561215695, 'gamma': 2.0221867187111067, 'e1': -0.12416891142507401,
                 'e2': 0.05220093206376522, 'center_x': -0.004921626192296812, 'center_y': 0.008559895185872239,
                 'a1_a': 0.0, 'delta_phi_m1': 0.0, 'a3_a': 0.0, 'delta_phi_m3': -0.10157639936931348, 'a4_a': 0.0, 'delta_phi_m4': 0.911096138562145},
                {'gamma1': 0.00643783384552091, 'gamma2': -0.03510346169910469, 'ra_0': 0.0, 'dec_0': 0.0}
            ]
        elif self._data.band == 'F475X':
            kwargs_lens_macro = [
                {'theta_E': 0.7373590543163482, 'gamma': 2.2765451155672305, 'e1': -0.10130446214725602,
                 'e2': 0.023262328547683935, 'center_x': -0.020987216615909077, 'center_y': 0.012317620381194251,
                 'a1_a': 0.0, 'delta_phi_m1': 0.0,'a3_a': 0.0, 'delta_phi_m3': 0.10127688417675478, 'a4_a': 0.0, 'delta_phi_m4': 0.4911663020271606},
                {'gamma1': 0.03831713538022406, 'gamma2': -0.06042265400072665, 'ra_0': 0.0, 'dec_0': 0.0}
            ]
        else:
            raise Exception('imaging data band must be either F814W or F475X')
        redshift_list_macro = [self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1, 'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi, 'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -np.pi/8},
            {'gamma1': -0.5, 'gamma2': -0.5}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.8, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': np.pi/8},
            {'gamma1': 0.5, 'gamma2': 0.5}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

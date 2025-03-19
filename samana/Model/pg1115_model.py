from samana.Model.model_base import EPLModelBase
import numpy as np
import pickle

class _PG1115ModelBase(EPLModelBase):

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
        if self._data.band == 'NIRCAM115W':
            kwargs_constraints['joint_lens_light_with_lens_light'] = [[0,1,  ['center_x', 'center_y']]]
        return kwargs_constraints

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 498.6338483531917, 'R_sersic': 0.1611063359090809, 'n_sersic': 4.587266000722903,
             'e1': -0.029061317690800247, 'e2': -0.024550267213920576, 'center_x': 0.0004111450202642533,
             'center_y': 0.15368021757678293}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 1.0, 'e1': 0.2, 'e2': 0.2, 'center_x': 0.1,
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

        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [
            {'amp': 1, 'R_sersic': 0.931857213603357, 'n_sersic': 6.6831217688402145, 'e1': 0.022950513140317954,
             'e2': -0.062492598644636396, 'center_x': -0.01708814312274429, 'center_y': 0.0040634619638040105}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_light_fixed = [{}]

        if self._data.band == 'NIRCAM115W':
            n_max = 5
            shapelets_list, kwargs_shapelets_init, kwargs_shapelets_sigma, \
                kwargs_shapelets_fixed, kwargs_lower_shapelets, kwargs_upper_shapelets = \
                self.add_shapelets_lens(n_max)
            lens_light_model_list += shapelets_list
            kwargs_lens_light_init += kwargs_shapelets_init
            kwargs_lens_light_fixed += kwargs_shapelets_fixed
            kwargs_lens_light_sigma += kwargs_shapelets_sigma
            kwargs_lower_lens_light += kwargs_lower_shapelets
            kwargs_upper_lens_light += kwargs_upper_shapelets

        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed,
                             kwargs_lower_lens_light,
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
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

class PG1115ModelEPLM1M3M4Shear(_PG1115ModelBase):

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR']
        if self._data.band == 'NIRCAMF115W':
            kwargs_lens_macro = [
                {'theta_E': 1.1433368143806257, 'gamma': 2.195267260231437, 'e1': 0.07033069421094822,
                 'e2': -0.05353523618672113, 'center_x': -0.03722268271968586, 'center_y': 0.02046786026808224,
                 'a1_a': 0.0, 'delta_phi_m1': 0.33823761435824107, 'a3_a': 0.0, 'delta_phi_m3': -0.33307053415483157,
                 'a4_a': 0.0, 'delta_phi_m4': 0.11898793424445868},
                {'gamma1': -0.04566477118765903, 'gamma2': -0.13016987476487257, 'ra_0': 0.0, 'dec_0': 0.0}
            ]
        else:
            kwargs_lens_macro = [
                {'theta_E': 1.1466296178228925, 'gamma': 2.0014133180207025, 'e1': 0.07143218306427672,
                 'e2': -0.03166973868176683, 'center_x': -0.05379887696364931, 'center_y': 0.008165180380356397,
                  'a1_a': 0.01, 'delta_phi_m1': 0.1,'a3_a': 0.0, 'delta_phi_m3': -0.005969269433780858, 'a4_a': 0.0, 'delta_phi_m4': 0.20840045113672415},
                {'gamma1': -0.028738087587014873, 'gamma2': -0.10888032218076507, 'ra_0': 0.0, 'dec_0': 0.0}
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
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 3.5, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

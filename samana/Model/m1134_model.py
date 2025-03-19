from samana.Model.model_base import EPLModelBase
import numpy as np
import pickle
from samana.forward_model_util import macromodel_readout_function_eplshear_satellite


class _M1134ModelBase(EPLModelBase):

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
        if self._data.band == 'HST814W':
            kwargs_source_init = [
                {'amp': 2.1185751163399558, 'R_sersic': 1.4776136937531457, 'n_sersic': 3.9437525404186053,
                 'e1': -0.09472006972243252, 'e2': -0.23604200716782295, 'center_x': 0.10381092761860801,
                 'center_y': -0.18719810606822243}
            ]
        else:
            kwargs_source_init = [
                {'amp': 2.1185751163399558, 'R_sersic': 1.4776136937531457, 'n_sersic': 3.9437525404186053,
                 'e1': -0.09472006972243252, 'e2': -0.23604200716782295, 'center_x': 0.10381092761860801,
                 'center_y': -0.18719810606822243}
            ]

        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
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

        lens_light_model_list = ['SERSIC_ELLIPSE']
        if self._data.band == 'HST814W':
            kwargs_lens_light_init = [
                {'amp': 2.3757548657801983, 'R_sersic': 1.2905859494604435, 'n_sersic': 7.605471337428463,
                 'e1': -0.1202898797182632, 'e2': 0.0067593098640644994, 'center_x': 0.009134769676769624,
                 'center_y': 0.015374389413448142}
            ]
        else:
            kwargs_lens_light_init = [
                {'amp': 1, 'R_sersic': 1.2942967883289467, 'n_sersic': 7.697747494985299, 'e1': -0.36132931126429374,
                 'e2': -0.1062427449413843, 'center_x': -0.06085505726650964, 'center_y': 0.12707496720879105}
            ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_light_fixed = [{}]

        if self._data.band == 'HST814W':
            add_uniform_light = False
        else:
            add_uniform_light = True
        if add_uniform_light:
            kwargs_uniform, kwargs_uniform_sigma, kwargs_uniform_fixed, \
            kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light(0.0, 2.0)
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
                             'source_position_tolerance': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.joint_lens_with_light_prior
                             }
        return kwargs_likelihood

class M1134ModelEPLM3M4ShearSatellite(_M1134ModelBase):

    satellite_x = 3.208
    satellite_y = -3.962
    # https://arxiv.org/pdf/1803.07175

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=1.,z_satellite=None):
        self.z_satellite = z_satellite
        super(M1134ModelEPLM3M4ShearSatellite, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_eplshear_satellite

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS']
        if self._data.band == 'HST814W':
            kwargs_lens_macro = [
                {'theta_E': 1.2386858327327679, 'gamma': 2.197949349105164, 'e1': -0.07181599131960777,
                 'e2': -0.148121156860746, 'center_x': -0.06371822765700086, 'center_y': 0.09752899927115061,
                 'a1_a': 0.0, 'delta_phi_m1': 0.013569351203598552, 'a3_a': 0.0, 'delta_phi_m3': -0.48464944547015704,
                 'a4_a': 0.0, 'delta_phi_m4': 0.6786230184849307},
                {'gamma1': -0.0038159207840149597, 'gamma2': 0.3609340651280836, 'ra_0': 0.0, 'dec_0': 0.0},
                {'theta_E': 0.050479483337966896, 'center_x': 3.3527689955490216, 'center_y': -3.995395025902885}
            ]
        elif self._data.band == 'MIRI560W':
            kwargs_lens_macro = [
                {'theta_E': 1.2386858327327679, 'gamma': 2.197949349105164, 'e1': -0.07181599131960777,
                 'e2': -0.148121156860746, 'center_x': -0.06371822765700086, 'center_y': 0.09752899927115061,
                 'a1_a': 0.0, 'delta_phi_m1': 0.013569351203598552, 'a3_a': 0.0, 'delta_phi_m3': -0.48464944547015704,
                 'a4_a': 0.0, 'delta_phi_m4': 0.6786230184849307},
                {'gamma1': -0.0038159207840149597, 'gamma2': 0.3609340651280836, 'ra_0': 0.0, 'dec_0': 0.0},
                {'theta_E': 0.050479483337966896, 'center_x': 3.3527689955490216, 'center_y': -3.995395025902885}
            ]
        if self.z_satellite is None:
            z_satellite = self._data.z_lens
        else:
            z_satellite = self.z_satellite
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, z_satellite]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                             {'theta_E': 0.2, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.7, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.05, 'center_x': self.satellite_x - 0.3, 'center_y': self.satellite_y - 0.3}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.6, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 1.0, 'center_x': self.satellite_x + 0.3, 'center_y': self.satellite_y + 0.3}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

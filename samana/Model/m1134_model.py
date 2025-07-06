from samana.Model.model_base import EPLModelBase
import numpy as np
import pickle
from samana.forward_model_util import macromodel_readout_function_eplshear_satellite


class _M1134ModelBase(EPLModelBase):

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
                             'source_position_tolerance': 0.00001,
                             'source_position_likelihood': True,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.axis_ratio_masslight_alignment,
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
        return None

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS']
        if self._data.band == 'HST814W':
            kwargs_lens_macro = [
                {'theta_E': 1.1977973522694216, 'gamma': 2.1811804820363756, 'e1': -0.049259366198138886,
                 'e2': 0.03393966515095142, 'center_x': 0.01310572691186353, 'center_y': -0.040971570963843316,
                 'a1_a': 0.0, 'delta_phi_m1': 0.0946244679093586, 'a3_a': 0.0, 'delta_phi_m3': -0.5205287918514736,
                 'a4_a': 0.0, 'delta_phi_m4': 0.7240750564861568},
                {'gamma1': -0.001539234313987411, 'gamma2': 0.3660613471494284, 'ra_0': 0.0, 'dec_0': 0.0},
                {'theta_E': 0.23750596785356515, 'center_x': 3.3841857914879068, 'center_y': -3.998138778021614}
            ]
        elif self._data.band == 'MIRI560W':
            kwargs_lens_macro = [
                {'theta_E': 1.2128051442284533, 'gamma': 2.3417510269959703, 'e1': -0.07613658453691555,
                 'e2': -0.04677832515324652, 'center_x': -0.05029154884342433, 'center_y': 0.07887971250297662,
                 'a1_a': 0.0, 'delta_phi_m1': 0.18167336876426537, 'a3_a': 0.0, 'delta_phi_m3': -0.03053852529246417,
                 'a4_a': 0.0, 'delta_phi_m4': 0.7978670578128553},
                {'gamma1': 0.0006014103799520669, 'gamma2': 0.4273562498399638, 'ra_0': 0.0, 'dec_0': 0.0},
                {'theta_E': 0.035528174387371776, 'center_x': 3.3548647214696916, 'center_y': -3.9578092339596487}
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
            {'theta_E': 0.0, 'center_x': self.satellite_x - 0.3, 'center_y': self.satellite_y - 0.3}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.6, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.4, 'center_x': self.satellite_x + 0.3, 'center_y': self.satellite_y + 0.3}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

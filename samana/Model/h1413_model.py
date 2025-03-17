from samana.Model.model_base import EPLModelBase
import numpy as np
from samana.forward_model_util import macromodel_readout_function_eplshear_satellite


class _H1413ModelBase(EPLModelBase):

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

    def setup_source_light_model(self):

        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 1, 'R_sersic': 0.02796380070166966, 'n_sersic': 3.9367467127355806, 'e1': 0.10746901603615666,
             'e2': 0.31715572162154915, 'center_x': 0.22411118522966256, 'center_y': 0.49605189911239933}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.01, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 2.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
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
            {'amp': 1, 'R_sersic': 0.3783137419156091, 'n_sersic': 7.910625151165032, 'e1': 0.4821410087473546,
             'e2': 0.09032354893207806, 'center_x': -0.0670697263992593, 'center_y': -0.05753689652059006}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25,
             'e1': 0.1, 'e2': 0.1,
             'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5,
             'e1': -0.5, 'e2': -0.5,
             'center_x': -0.25, 'center_y': -0.25}]
        kwargs_upper_lens_light = [
            {'R_sersic': 5, 'n_sersic': 10.0,
             'e1': 0.5, 'e2': 0.5,
             'center_x': 0.25, 'center_y': 0.25}]
        kwargs_lens_light_fixed = [{}]
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

        add_uniform_light = True
        if add_uniform_light:
            kwargs_uniform, kwargs_uniform_sigma, kwargs_uniform_fixed, \
                kwargs_uniform_lower, kwargs_uniform_upper = self.add_uniform_lens_light()
            lens_light_model_list += ['UNIFORM']
            kwargs_lens_light_init += kwargs_uniform
            kwargs_lens_light_sigma += kwargs_uniform_sigma
            kwargs_lens_light_fixed += kwargs_uniform_fixed
            kwargs_lower_lens_light += kwargs_uniform_lower
            kwargs_upper_lens_light += kwargs_uniform_upper
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
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.custom_prior
                             }
        return kwargs_likelihood

class H1413ModelEPLM3M4Shear(_H1413ModelBase):

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.0):
        # shapelets scale factor set to 2; lens model changes with increasing nmax suggesting
        # shapelets are fitting psf noise
        super(H1413ModelEPLM3M4Shear, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    def custom_prior(self, kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source):

        prior = self.hard_cut_axis_ratio_prior(kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source)
        prior += self.joint_lens_with_light_prior(kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source)
        return prior


    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_eplshear_satellite

    @property
    def prior_lens(self):
        return [[2, 'center_x', self._data.g2x, 0.05], [2, 'center_y', self._data.g2y, 0.05], [2, 'theta_E', 0.5, 0.1]]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 0.5572237697796922, 'gamma': 1.7590114081399728, 'e1': -0.11445566775779056,
             'e2': -0.005456834152063129, 'center_x': 0.022068352330487328, 'center_y': 0.06987126178817833,
             'a1_a': 0.0, 'delta_phi_m1': -0.2816284758597358, 'a3_a': 0.0, 'delta_phi_m3': 0.27025495271440064,
             'a4_a': 0.0, 'delta_phi_m4': -0.10611421863617054},
            {'gamma1': -0.06105938233582977, 'gamma2': -0.020669935655420205, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.6099876560738409, 'center_x': 1.5150160867635873, 'center_y': 3.737785802018212}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12,
                              'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                             {'theta_E': 0.2, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0},{}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -np.pi/8},
            {'gamma1': -0.5, 'gamma2': -0.5},
        {'theta_E': 0.0, 'center_x': self._data.g2x - 0.25, 'center_y': self._data.g2y - 0.25}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': np.pi/8},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 1.2, 'center_x': self._data.g2x + 0.25, 'center_y': self._data.g2y + 0.25}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class H1413ModelEPLM3M4ShearTwoSatellites(_H1413ModelBase):
    # neither companion appears to affect the lens model very much
    satellite_x1 = 1.70482
    satellite_y1 = 3.64
    satellite_x2 = -4.4
    satellite_y2 = 0.2
    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=2.0):
        # shapelets scale factor set to 2; lens model changes with increasing nmax suggesting
        # shapelets are fitting psf noise
        super(H1413ModelEPLM3M4ShearTwoSatellites, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_eplshear_satellite

    @property
    def prior_lens(self):
        return None
        #return [[2, 'center_x', self._data.g2x, 0.05], [2, 'center_y', self._data.g2y, 0.05]]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 0.5945561181171953, 'gamma': 1.9118152489969296, 'e1': -0.27220780635663133,
             'e2': 0.14288396199766928, 'center_x': 0.027739302935571437, 'center_y': 0.08194389978787217,
             'a1_a': 0.0, 'delta_phi_m1': -0.2709690505908811, 'a3_a': 0.0, 'delta_phi_m3': -0.2709690505908811,
             'a4_a': 0.0, 'delta_phi_m4': 1.9112051591872898},
            {'gamma1': -0.08767098811462325, 'gamma2': 0.018614730158481117, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.5161391605843233, 'center_x': self.satellite_x1, 'center_y': self.satellite_y1},
            {'theta_E': 0.5161391605843233, 'center_x': self.satellite_x2, 'center_y': self.satellite_y2}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2, 3]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.1, 'gamma2': 0.1},
                             {'theta_E': 0.2, 'center_x': 0.05, 'center_y': 0.05},
                             {'theta_E': 0.2, 'center_x': 0.05, 'center_y': 0.05}
                             ]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0},{},{}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
        {'theta_E': 0.0, 'center_x': self.satellite_x1 - 0.3, 'center_y': self.satellite_y1 - 0.3},
        {'theta_E': 0.0, 'center_x': self.satellite_x2 - 0.3, 'center_y': self.satellite_y2 - 0.3}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
            {'theta_E': 1.0, 'center_x': self.satellite_x1 + 0.3, 'center_y': self.satellite_y1 + 0.3},
            {'theta_E': 1.0, 'center_x': self.satellite_x2 + 0.3, 'center_y': self.satellite_y2 + 0.3}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

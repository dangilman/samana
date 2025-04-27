from samana.Model.model_base import EPLModelBase
import numpy as np
from lenstronomy.Util.class_creator import create_class_instances
from samana.forward_model_util import macromodel_readout_function_eplshear_satellite
from lenstronomy.Util.param_util import ellipticity2phi_q


class _J1042ModelBase(EPLModelBase):

    @property
    def kwargs_constraints(self):
        joint_source_with_point_source = [[0, 0]]
        kwargs_constraints = {
                            'joint_source_with_point_source': joint_source_with_point_source,
                              'num_point_source_list': [len(self._data.x_image)],
                                'joint_lens_with_light': [[1, 2, ['center_x', 'center_y']]],
                              'solver_type': 'PROFILE_SHEAR',
                              'point_source_offset': True,
                                'image_plane_source_list': self._image_plane_source_list
                              }
        if self._shapelets_order is not None:
            kwargs_constraints['joint_source_with_source'] = [[0, 1, ['center_x', 'center_y']]]
        return kwargs_constraints

    def setup_source_light_model(self):

        self._image_plane_source_list = [False]
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 3.3744972356729663, 'R_sersic': 0.4474039432568018, 'n_sersic': 5.05467376415875,
             'e1': 0.10114424434535499, 'e2': 0.3036593003967822, 'center_x': -0.024591454450440663,
             'center_y': 0.08165022745322371}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 1.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
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
        #                                                                                   0.1)
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
            {'amp': 20.26840868517619, 'R_sersic': 1.0944366759485558, 'n_sersic': 4.356436990256856,
             'e1': -0.14179393969864268, 'e2': 0.11376265630262214, 'center_x': 0.053111677261926483,
             'center_y': 0.012181545978949382},
            {'amp': 3.122706493583511, 'R_sersic': 0.15748965221512662, 'n_sersic': 4.427836211253151,
             'e1': 0.2508985183627598, 'e2': -0.2788941645631706, 'center_x': 1.8818250941196057,
             'center_y': -0.24600847097834963}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
            #{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
        {'R_sersic': 0.05, 'n_sersic': 0.25, 'center_x': 0.05, 'center_y': 0.05, 'e1': 0.1, 'e2': 0.1,}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
           # {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0},
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': self._data.gx - 0.1, 'center_y': self._data.gy - 0.1}]
        kwargs_upper_lens_light = [
            {'R_sersic': 4.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
           # {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
        {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': self._data.gx + 0.1, 'center_y': self._data.gy + 0.1}]
        kwargs_lens_light_fixed = [{},
                                  # {},
                                   {}]

        add_uniform_light = False
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

class J1042ModelEPLM3M4Shear(_J1042ModelBase):

    def __init__(self, data_class, shapelets_order=None, shapelets_scale_factor=1):

        super(J1042ModelEPLM3M4Shear, self).__init__(data_class, shapelets_order, shapelets_scale_factor)

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
                             'custom_logL_addition': self.axis_ratio_masslight_alignment
                             }
        return kwargs_likelihood

    def custom_logL(self, kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source, kwargs_model):
        """
        Note: this only works with a modified version of lenstronomy where kwargs_model gets passed to custom_logL_addtion
        """
        lens_model = create_class_instances(**kwargs_model, only_lens_model=True)
        point_x = -0.95
        point_y = 0.
        beta_x, beta_y = lens_model.ray_shooting(point_x, point_y, kwargs_lens)
        counter_x = 0.8
        counter_y = 0.8
        beta_x_counter, beta_y_counter = lens_model.ray_shooting(counter_x, counter_y, kwargs_lens)
        dx, dy = beta_x - beta_x_counter, beta_y - beta_y_counter
        return -0.5 * (dx ** 2 + dy ** 2) / 0.001 ** 2

    @property
    def macromodel_readout_function(self):
        return macromodel_readout_function_eplshear_satellite

    @property
    def prior_lens(self):
        # note the prior on the lens position is implicitely also a prior on the light position
        return self.population_gamma_prior + [
                [2, 'center_x', self._data.gx, 0.05],
                [2, 'center_y', self._data.gy, 0.05],
                [2, 'theta_E', 0.05, 0.2]]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 0.8849554512395507, 'gamma': 2.082322949717141, 'e1': -0.15888570155198645,
             'e2': 0.15857160380293264, 'center_x': 0.007422379850661033, 'center_y': 0.02648838109905435, 'a1_a': 0.0,
             'delta_phi_m1': 0.42386516901676236, 'a3_a': 0.0, 'delta_phi_m3': 0.5042549203165205, 'a4_a': 0.0,
             'delta_phi_m4': 0.0646589723529078},
            {'gamma1': -0.053097552455548126, 'gamma2': 0.008165121330997928, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.05675346094302038, 'center_x': 1.8818250941196057, 'center_y': -0.24600847097834963}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.05, 'e2': 0.05, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.1,'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05}, {'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.7, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi,'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -np.pi/8},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.0, 'center_x': self._data.gx - 0.2, 'center_y': self._data.gy - 0.2}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.6, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': np.pi/8},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.25, 'center_x': self._data.gx + 0.2, 'center_y': self._data.gy + 0.2}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class J1042ModelEPLM3M4ShearExpShapelets(J1042ModelEPLM3M4Shear):

    def setup_source_light_model(self):

        self._image_plane_source_list = [False]
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [
            {'amp': 19.65100967474721, 'R_sersic': 0.3623232419635632, 'n_sersic': 4.631856837954162,
             'e1': 0.11925205084786344, 'e2': 0.06692796091466355, 'center_x': 0.011226501439929489,
             'center_y': 0.07155978396187809}
        ]
        kwargs_source_sigma = [{'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [{'R_sersic': 1.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]

        if self._shapelets_order is not None:
            self._image_plane_source_list += [False]
            n_max = int(self._shapelets_order)
            shapelets_source_list, kwargs_shapelets_init, kwargs_shapelets_sigma, \
            kwargs_shapelets_fixed, kwargs_lower_shapelets, kwargs_upper_shapelets = \
                self.add_exp_shapelets_source(n_max)
            source_model_list += shapelets_source_list
            kwargs_source_init += kwargs_shapelets_init
            kwargs_source_fixed += kwargs_shapelets_fixed
            kwargs_source_sigma += kwargs_shapelets_sigma
            kwargs_lower_source += kwargs_lower_shapelets
            kwargs_upper_source += kwargs_upper_shapelets

        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]

        return source_model_list, source_params

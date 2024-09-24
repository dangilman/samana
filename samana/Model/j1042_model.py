from samana.Model.model_base import ModelBase
import numpy as np
from lenstronomy.Util.class_creator import create_class_instances

class _J1042ModelBase(ModelBase):

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
            {'amp': 35.006813219064206, 'R_sersic': 0.7465437659406482, 'n_sersic': 4.094928696529352,
             'e1': -0.1900170413877109, 'e2': 0.13709164288916312, 'center_x': 0.04939959340021617,
             'center_y': 0.010484697289207538},
            {'amp': 4.913981821321209, 'R_sersic': 0.18737736861280985, 'n_sersic': 3.8362876729343975,
             'e1': 0.1858321164784092, 'e2': -0.2753944179112829, 'center_x': self._data.gx,
             'center_y': self._data.gy}
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
                             'source_position_likelihood': False,
                             # 'check_matched_source_position': False,
                             'source_position_sigma': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': None
                             }
        return kwargs_likelihood

    def custom_logL(self, kwargs_lens,
                kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                kwargs_extinction, kwargs_tracer_source, kwargs_model):

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
    def prior_lens(self):
        # note the prior on the lens position is implicitely also a prior on the light position
        return self.population_gamma_prior + [
                [2, 'center_x', self._data.gx, 0.05],
                [2, 'center_y', self._data.gy, 0.05],
                [2, 'theta_E', 0.05, 0.1]]

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4_ELL', 'SHEAR', 'SIS']
        kwargs_lens_macro = [
            {'theta_E': 0.8884654787595039, 'gamma': 1.9389205510769976, 'e1': -0.22945509053278765,
             'e2': 0.15832030781687093, 'center_x': 0.024594789216292615, 'center_y': 0.038319171626879284, 'a3_a': 0.0,
             'delta_phi_m3': 0.40255807918558223, 'a4_a': 0.0, 'delta_phi_m4': -0.5163251300933784},
            {'gamma1': -0.1008630901650493, 'gamma2': 0.016019787962277186, 'ra_0': 0.0, 'dec_0': 0.0},
            {'theta_E': 0.07545918728644968, 'center_x': 1.8699636079097566, 'center_y': -0.4847974949545327}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1, 2]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.05, 'e2': 0.05, 'gamma': 0.1,
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi/12, 'delta_phi_m4': np.pi/16},
                             {'gamma1': 0.05, 'gamma2': 0.05}, {'theta_E': 0.05, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}, {}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.7, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5},
            {'theta_E': 0.0, 'center_x': self._data.gx - 0.2, 'center_y': self._data.gy - 0.2}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.6, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5},
        {'theta_E': 0.25, 'center_x': self._data.gx + 0.2, 'center_y': self._data.gy + 0.2}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

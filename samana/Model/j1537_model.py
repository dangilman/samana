from copy import deepcopy

from samana.Model.model_base import EPLModelBase
import numpy as np
import pickle

class _J1537ModelBase(EPLModelBase):

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
        kwargs_source_init = [{'amp': 1, 'R_sersic': 6.445332536966378, 'n_sersic': 3.6305228276190764,
                               'e1': -0.4155480081962428, 'e2': 0.36638779330275034,
                               'center_x': 0.023093143905461546, 'center_y': -0.054747647240303066}]
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

        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [
            {'amp': 0.2903299253210497, 'R_sersic': 6.156629320813526, 'n_sersic': 9.997002153610675,
             'e1': -0.04839034932783541, 'e2': 0.13399653909148815, 'center_x': -0.02071253106507734,
             'center_y': 0.040168714294272996}
        ]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10.0, 'center_y': -10.0}]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
        kwargs_lens_light_fixed = [{}]
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]

        return lens_light_model_list, lens_light_params

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'source_position_tolerance': 0.0001,
                             'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True
                             }
        return kwargs_likelihood

class J1537ModelEPLM3M4Shear(_J1537ModelBase):

    @property
    def prior_lens(self):
        return self.population_gamma_prior

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR']
        kwargs_lens_macro = [
            {'theta_E': 1.3995740944935433, 'gamma': 2.1119312176212826, 'e1': 0.0034740189666655746,
             'e2': 0.013653891623438216, 'center_x': -0.023300423603577855, 'center_y': 0.01944275113479937,
             'a1_a': 0.0, 'delta_phi_m1': 0.0, 'a3_a': 0.0, 'delta_phi_m3': -0.4373204113065346, 'a4_a': 0.0, 'delta_phi_m4': 0.5970557298331496},
            {'gamma1': 0.0851999128560513, 'gamma2': -0.13903555309738413, 'ra_0': 0.0, 'dec_0': 0.0}
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
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.6, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi, 'a3_a': -0.1, 'delta_phi_m3': -np.pi/6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.4, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi, 'a3_a': 0.1, 'delta_phi_m3': np.pi/6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class J1537ModelNFWSersic(_J1537ModelBase):

    @property
    def kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'image_position_uncertainty': 5e-3,
                             'source_position_likelihood': False,
                             # 'check_matched_source_position': False,
                             'source_position_sigma': 0.0001,
                             #'prior_lens': self.prior_lens,
                             'image_likelihood_mask_list': [self._data.likelihood_mask],
                             'astrometric_likelihood': True,
                             'custom_logL_addition': self.custom_priors
                             }
        return kwargs_likelihood

    def custom_priors(self, kwargs_lens,
                                    kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special,
                                    kwargs_extinction, kwargs_tracer_source, max_offset=0.2):

        from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
        from lenstronomy.LensModel.lens_model import LensModel
        lens_model = LensModel(['NFW_MC_ELLIPSE_SERSIC', 'SHEAR'])
        analysis = LensProfileAnalysis(lens_model)
        log_slope = analysis.profile_slope(kwargs_lens, radius=1.4)
        log_slope_logL = -0.5 * (log_slope - 2.1)**2 / 0.2**2

        aperture = 1.4
        center_x = kwargs_lens[0]['center_x']
        center_y = kwargs_lens[0]['center_y']
        kw_cham = deepcopy(kwargs_lens)
        kw_nfw = deepcopy(kwargs_lens)
        kw_nfw[0]['k_eff'] = 0.0
        kw_cham[0]['logM'] = 0.0
        kappa_nfw = analysis.mass_fraction_within_radius(kw_nfw,
                                                          center_x,
                                                          center_y,
                                                          aperture)[0]
        kappa_cham = analysis.mass_fraction_within_radius(kw_cham,
                                                         center_x,
                                                         center_y,
                                                         aperture)[0]
        mass_fraction_logL = -0.5 * (kappa_nfw/3 - kappa_cham)**2 / 0.1**2
        return mass_fraction_logL + log_slope_logL

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):

        lens_model_list_macro = ['NFW_MC_ELLIPSE_SERSIC', 'SHEAR']
        kwargs_lens_macro = [
            {'logM': 13.3, 'concentration': 4.0, 'e1_nfw': 0.0, 'e2_nfw': 0.0,
             'k_eff': 0.6, 'R_sersic': 1.0, 'n_sersic': 4.0, 'e1_sers': 0.1, 'e2_sers': 0.1,
             'center_x': 0.0, 'center_y': 0.0},
            {'gamma1': 0.0851999128560513, 'gamma2': -0.13903555309738413, 'ra_0': 0.0, 'dec_0': 0.0}
        ]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [
            {'logM': 0.3, 'concentration': 1.0, 'e1_nfw': 0.1, 'e2_nfw': 0.1,
             'k_eff': 0.2, 'R_sersic': 0.25, 'n_sersic': 1.0, 'e1_sers': 0.2, 'e2_sers': 0.2,
             'center_x': 0.1, 'center_y': 0.1},
            {'gamma1': 0.05, 'gamma2': 0.05, 'ra_0': 0.0, 'dec_0': 0.0}
        ]
        kwargs_lens_fixed = [{}, {'ra_0': 0.0, 'dec_0': 0.0}]
        kwargs_lower_lens = [
            {'logM': 12.0, 'concentration': 1.2, 'e1_nfw': -0.2, 'e2_nfw': -0.2,
             'k_eff': 0.0001, 'R_sersic': 0.01, 'n_sersic': 0.5, 'e1_sers': -0.5, 'e2_sers': -0.5,
             'center_x': -0.5, 'center_y': 0.5},
            {'gamma1': -0.5, 'gamma2': -0.5, 'ra_0': 0.0, 'dec_0': 0.0}
            ]
        kwargs_upper_lens = [
            {'logM': 14.0, 'concentration': 10.0, 'e1_nfw': 0.2, 'e2_nfw': 0.2,
             'k_eff': 10.0, 'R_sersic': 10.0, 'n_sersic': 10.0, 'e1_sers': 0.5, 'e2_sers': 0.5,
             'center_x': 0.5, 'center_y': 0.5},
            {'gamma1': 0.5, 'gamma2': 0.5, 'ra_0': 0.0, 'dec_0': 0.0}
        ]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params



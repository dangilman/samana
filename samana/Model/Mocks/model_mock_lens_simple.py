from samana.Model.model_base import EPLModelBase
import numpy as np


class MockModelBase(EPLModelBase):

    def __init__(self, data_class, shapelets_order=None):
        self._shapelets_order = shapelets_order
        super(MockModelBase, self).__init__(data_class, shapelets_order)

    def _setup_true_source_light_model(self, ra_source, dec_source):

        from paltas.Sources.cosmos import COSMOSCatalog
        from pyHalo.Cosmology.cosmology import Cosmology
        from copy import deepcopy
        import os

        cosmo = Cosmology()
        colossus_cosmo = cosmo.colossus
        cosmos_folder = os.getenv('HOME') + '/data/cosmo_catalog/COSMOS_23.5_training_sample/'
        source_parameters = {'minimum_size_in_pixels': 10.0,
                             'faintest_apparent_mag': -18,
                             'max_z': 0.025,
                             'smoothing_sigma': 0.001,
                             'cosmos_folder': cosmos_folder,
                             'random_rotation': 0.0,
                             'min_flux_radius': 0.0,
                             'output_ab_zeropoint': 25.95,
                             'z_source': self._data.z_source,
                             'center_x': ra_source,
                             'center_y': dec_source}
        cosmo = COSMOSCatalog(colossus_cosmo, source_parameters)
        idx_source = 1
        source_model_list, kwargs_source_init, zsource_list = cosmo.draw_source(idx_source)
        kwargs_source_sigma = [{'image': 1.0, 'center_x': 1.0, 'center_y': 1.0, 'phi_G': 1.0, 'scale': 1.0}]
        kwargs_source_fixed = deepcopy(kwargs_source_init)
        kwargs_lower_source = [{'image': 1.0, 'center_x': 1.0, 'center_y': 1.0, 'phi_G': 1.0, 'scale': 1.0}]
        kwargs_upper_source = [{'image': 1.0, 'center_x': 1.0, 'center_y': 1.0, 'phi_G': 1.0, 'scale': 1.0}]
        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]
        if self._shapelets_order is not None:
            source_model_list, source_params = \
                self._add_source_shapelets(self._shapelets_order, source_model_list, source_params)
        return source_model_list, source_params

    @staticmethod
    def _add_source_shapelets(n_max, model_list, source_params):

        kwargs_source_init = [{'amp': 1.0, 'beta': 0.1, 'center_x': 0.0, 'center_y': 0.0, 'n_max': n_max}]
        kwargs_source_sigma = [{'amp': 100.0, 'beta': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'n_max': 1}]
        kwargs_lower_source = [{'amp': 0.0, 'beta': 0.0, 'center_x': -10.0, 'center_y': -10.0, 'n_max': 1}]
        kwargs_upper_source = [{'amp': 10000.0, 'beta': 1.0, 'center_x': 10.0, 'center_y': 10.0, 'n_max': 10}]
        kwargs_source_fixed = [{'n_max': n_max}]
        model_list += ['SHAPELETS']
        source_params[0] += kwargs_source_init
        source_params[1] += kwargs_source_sigma
        source_params[2] += kwargs_source_fixed
        source_params[3] += kwargs_lower_source
        source_params[4] += kwargs_upper_source
        return model_list, source_params

    @property
    def prior_lens(self):
        return [[0, 'gamma', 2.0, 0.2]]

    def setup_lens_light_model(self):

        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light_init = [{'amp': 20.0, 'R_sersic': 0.3, 'n_sersic': 4., 'e1': -0.0,
                                   'e2': 0.05, 'center_x': 0.0, 'center_y': 0.0}]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.05, 'center_y': 0.05}]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.01, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -0.25, 'center_y': -0.25}]
        kwargs_upper_lens_light = [
            {'R_sersic': 5, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 0.25, 'center_y': 0.25}]
        kwargs_lens_light_fixed = [{}]
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

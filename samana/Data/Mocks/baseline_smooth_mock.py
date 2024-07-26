import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.baseline_smooth_mock import image_data
from samana.Model.Mocks.model_mock_lens_simple import MockModelBase

class BaselineSmoothMock(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.5
        z_source = 2.0
        x_image = [-0.92770948,  0.61449948, -0.21134454,  0.75025513]
        y_image = [-0.69051723,  0.76790815,  0.9271273 , -0.45817762]
        magnifications_true = [3.48573513, 7.48256353, 5.05235091, 3.6984534 ]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true =0.0
        self.a4a_true = 0.0
        self.delta_phi_m3_true = 0.0
        self.delta_phi_m4_true = 0.0

        super(BaselineSmoothMock, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

class BaselineSmoothMockModel(MockModelBase):

    def setup_source_light_model(self):
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_init = [{'amp': 5.0, 'center_x': -0.06, 'center_y': -0.04, 'e1': 0.1,
                  'e2': 0.05, 'R_sersic': 0.1, 'n_sersic': 4.2}]
        kwargs_source_sigma = [{'amp': 2.0, 'R_sersic': 0.025, 'n_sersic': 0.25, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1,
                                'center_y': 0.1}]
        kwargs_lower_source = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10.0}]
        kwargs_upper_source = [
            {'R_sersic': 10.0, 'n_sersic': 10.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 10.0, 'center_y': 10.0}]
        kwargs_source_fixed = [{}]
        source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]
        if self._shapelets_order is not None:
            source_model_list, source_params = \
                self._add_source_shapelets(self._shapelets_order, source_model_list, source_params)
        return source_model_list, source_params

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):
        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4', 'SHEAR']
        kwargs_lens_macro = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.1, 'e2': -0.1,
                              'gamma': 1.8894064923991685, 'a4_a': 0.0,
                              'a3_a': 0.0, 'delta_phi_m3': 0.0,
                              'delta_phi_m4': 0.0}, {'gamma1': -0.04, 'gamma2': 0.04}]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.1, 'e2': 0.1, 'gamma': 0.1,
                              'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi / 12, 'delta_phi_m4': np.pi / 16},
                             {'gamma1': 0.05, 'gamma2': 0.05}]
        kwargs_lens_fixed = [{'delta_phi_m4': 0.0}, {'ra_0': 0.0, 'dec_0': 0.0}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a3_a': -0.1, 'delta_phi_m3': -np.pi / 6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 3.5, 'a4_a': 0.1,
             'a3_a': 0.1, 'delta_phi_m3': np.pi / 6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

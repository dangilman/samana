import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Model.Mocks.model_mock_lens_simple import MockModelBase

class BaselineSmoothMock(MockBase):

    def __init__(self, super_sample_factor=1.0):
        z_lens = 0.5
        z_source = 2.0
        x_image = np.array([-0.92770948,  0.61449948, -0.21134454,  0.75025513])
        y_image = np.array([-0.69051723,  0.76790815,  0.9271273 , -0.45817762])
        magnifications_true = np.array([3.48573513, 7.48256353, 5.05235091, 3.6984534 ])
        from samana.Data.ImageData.baseline_smooth_mock import image_data
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.00001] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = -0.0
        self.a4a_true = 0.00
        self.delta_phi_m3_true = 0.0
        self.delta_phi_m4_true = 0.0
        super(BaselineSmoothMock, self).__init__(z_lens, z_source, x_image, y_image,
                                                           magnifications, astrometric_uncertainties,
                                                           flux_ratio_uncertainties, image_data,
                                                           super_sample_factor)

class BaselineSmoothMockCross(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.5
        z_source = 2.0
        x_image = np.array([-0.18095938,  0.56976289,  0.89493927, -0.80173546])
        y_image = np.array([-1.07962642,  0.91282083, -0.35135024,  0.34103346])
        magnifications_true = np.array([4.70890019, 4.91307029, 4.47422769, 2.84484085])
        from samana.Data.ImageData.baseline_smooth_mock_cross import image_data
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.00001] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = -0.0
        self.a4a_true = 0.00
        self.delta_phi_m3_true = 0.0
        self.delta_phi_m4_true = 0.0
        super(BaselineSmoothMockCross, self).__init__(z_lens, z_source, x_image, y_image,
                                                           magnifications, astrometric_uncertainties,
                                                           flux_ratio_uncertainties, image_data,
                                                           super_sample_factor)

class BaselineSmoothMockCusp(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.5
        z_source = 2.0
        x_image = np.array([ 0.21444353,  1.10214829,  0.90068483, -0.55478174])
        y_image = np.array([-1.16268268,  0.20553389, -0.66282107,  0.39551167])
        magnifications_true = np.array([5.96953758, 6.60914698, 8.81613144, 1.21439187])
        from samana.Data.ImageData.baseline_smooth_mock_cusp import image_data
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.00001] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = -0.0
        self.a4a_true = 0.00
        self.delta_phi_m3_true = 0.0
        self.delta_phi_m4_true = 0.0
        super(BaselineSmoothMockCusp, self).__init__(z_lens, z_source, x_image, y_image,
                                                           magnifications, astrometric_uncertainties,
                                                           flux_ratio_uncertainties, image_data,
                                                           super_sample_factor)

class BaselineSmoothMockLowq(MockBase):

    def __init__(self, super_sample_factor=1.0):
        z_lens = 0.5
        z_source = 2.0
        x_image = np.array([ 1.06785869, -0.61402397,  0.12432216, -0.18476854])
        y_image = np.array([-0.41289394, -0.949661  , -1.09617422,  0.64855028])
        magnifications_true = np.array([4.01804612, 5.34630254, 5.49621952, 1.0582571 ])
        from samana.Data.ImageData.baseline_smooth_mock_lowq import image_data
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.00001] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = -0.0
        self.a4a_true = 0.00
        self.delta_phi_m3_true = 0.0
        self.delta_phi_m4_true = 0.0
        super(BaselineSmoothMockLowq, self).__init__(z_lens, z_source, x_image, y_image,
                                                           magnifications, astrometric_uncertainties,
                                                           flux_ratio_uncertainties, image_data,
                                                           super_sample_factor)

class BaselineSmoothMockv2(MockBase):
    """This mock has a relatively large ellipticity q = 0.4"""
    def __init__(self, super_sample_factor=1.0):
        z_lens = 0.5
        z_source = 2.0
        x_image = np.array([ 1.20615909, -0.79343069, -0.16863934, -0.08341527])
        y_image = np.array([ 0.24720679,  0.37340355,  0.82242164, -0.56724028])
        magnifications_true = np.array([1.67270303, 2.68093683, 1.12174305, 0.45608223])
        from samana.Data.ImageData.baseline_smooth_mock_v2 import image_data
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.00001] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = -0.0
        self.a4a_true = 0.00
        self.delta_phi_m3_true = 0.0
        self.delta_phi_m4_true = 0.0
        super(BaselineSmoothMockv2, self).__init__(z_lens, z_source, x_image, y_image,
                                                           magnifications, astrometric_uncertainties,
                                                           flux_ratio_uncertainties, image_data,
                                                           super_sample_factor)

class BaselineSmoothMockMultipole1(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.5
        z_source = 2.0
        x_image = np.array([-0.9829939, 0.9937506, 0.32072365, -0.01401119])
        y_image = np.array([-0.41747989, 0.01509745, -0.91748978, 0.9534751])
        magnifications_true = np.array([5.77372704, 10.64707305, 7.68367305, 5.60619377])
        from samana.Data.ImageData.baseline_smooth_mock_multipole1 import image_data
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.00001] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = -0.01
        self.a4a_true = 0.005
        self.delta_phi_m3_true = np.pi/6/3
        self.delta_phi_m4_true = np.pi/8/3
        self.phi_q_true = 1.3872
        self.q_true = 0.75548
        self.gamma_true = 2.07
        super(BaselineSmoothMockMultipole1, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

class BaselineSmoothMockMultipole2(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.5
        z_source = 2.0
        x_image = np.array([-0.43278781, 0.7020786, 0.8702122, -0.71639362])
        y_image = np.array([-1.00066423, 0.78917751, -0.40830315, 0.55200377])
        magnifications_true = np.array([5.4828351, 5.61039413, 6.0839831, 3.89908285])
        from samana.Data.ImageData.baseline_smooth_mock_multipole2 import image_data
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.00001] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = -0.009
        self.a4a_true = -0.01
        self.delta_phi_m3_true = -np.pi/6/3
        self.delta_phi_m4_true = np.pi/8/6
        self.phi_q_true = -0.78539
        self.q_true = 0.8691
        self.gamma_true = 2.05
        super(BaselineSmoothMockMultipole2, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

class BaselineSmoothMockMultipole3(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.5
        z_source = 2.0
        x_image = np.array([0.96814425, 0.09098276, 0.57690954, -0.71767754])
        y_image = np.array([0.56960341, -1.06182703, -0.85925099, 0.39019487])
        magnifications_true = np.array([4.99642229, 9.47296505, 9.64000799, 2.28187071])
        from samana.Data.ImageData.baseline_smooth_mock_multipole3 import image_data
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.00001] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = 0.004
        self.a4a_true = 0.015
        self.delta_phi_m3_true = 0.1 * np.pi/6
        self.delta_phi_m4_true = -0.2 * np.pi/8
        self.phi_q_true = -0.7853
        self.q_true = 0.8181
        self.gamma_true = 2.05
        super(BaselineSmoothMockMultipole3, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)

class BaselineSmoothMockMultipole4(MockBase):

    def __init__(self, super_sample_factor=1.0):

        z_lens = 0.5
        z_source = 2.0
        x_image = np.array([-0.38584609, 1.04142898, 0.65914844, -0.48699737])
        y_image = np.array([-1.03634905, 0.0660994, -0.80284886, 0.68684917])
        magnifications_true = np.array([5.81689864, 9.54536663, 10.33470303, 2.1830171])
        from samana.Data.ImageData.baseline_smooth_mock_multipole4 import image_data
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.00001] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = -0.01
        self.a4a_true = 0.01
        self.delta_phi_m3_true = np.pi/6/2
        self.delta_phi_m4_true = -np.pi/8/2
        self.phi_q_true = -1.1780972
        self.q_true = 0.7522
        self.gamma_true = 2.05
        super(BaselineSmoothMockMultipole4, self).__init__(z_lens, z_source, x_image, y_image,
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
        lens_model_list_macro = ['EPL_MULTIPOLE_M1M3M4_ELL', 'SHEAR']
        kwargs_lens_macro = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.1, 'e2': -0.1,
                              'gamma': 2.0, 'a4_a': 0.0,
                              'a1_a': 0.0, 'delta_phi_m1': 0.0, 'a3_a': 0.0, 'delta_phi_m3': 0.0,
                              'delta_phi_m4': 0.0}, {'gamma1': -0.04, 'gamma2': 0.04}]
        redshift_list_macro = [self._data.z_lens, self._data.z_lens]
        index_lens_split = [0, 1]
        if kwargs_lens_macro_init is not None:
            for i in range(0, len(kwargs_lens_macro_init)):
                for param_name in kwargs_lens_macro_init[i].keys():
                    kwargs_lens_macro[i][param_name] = kwargs_lens_macro_init[i][param_name]
        kwargs_lens_init = kwargs_lens_macro
        kwargs_lens_sigma = [{'theta_E': 0.05, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.1, 'e2': 0.1, 'gamma': 0.1,
                              'a1_a': 0.01, 'delta_phi_m1': 0.2, 'a4_a': 0.01, 'a3_a': 0.005, 'delta_phi_m3': np.pi / 12, 'delta_phi_m4': np.pi / 16},
                             {'gamma1': 0.05, 'gamma2': 0.05}]
        kwargs_lens_fixed = [{'delta_phi_m4': 0.0}, {'ra_0': 0.0, 'dec_0': 0.0}]
        kwargs_lower_lens = [
            {'theta_E': 0.05, 'center_x': -10.0, 'center_y': -10.0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'a4_a': -0.1,
             'a1_a': -0.1, 'delta_phi_m1': -np.pi, 'a3_a': -0.1, 'delta_phi_m3': -np.pi / 6, 'delta_phi_m4': -10.0},
            {'gamma1': -0.5, 'gamma2': -0.5}]
        kwargs_upper_lens = [
            {'theta_E': 5.0, 'center_x': 10.0, 'center_y': 10.0, 'e1': 0.5, 'e2': 0.5, 'gamma': 3.5, 'a4_a': 0.1,
             'a1_a': 0.1, 'delta_phi_m1': np.pi,'a3_a': 0.1, 'delta_phi_m3': np.pi / 6, 'delta_phi_m4': 10.0},
            {'gamma1': 0.5, 'gamma2': 0.5}]
        kwargs_lens_fixed, kwargs_lens_init = self.update_kwargs_fixed_macro(lens_model_list_macro, kwargs_lens_fixed,
                                                                             kwargs_lens_init, macromodel_samples_fixed)
        lens_model_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens,
                             kwargs_upper_lens]
        return lens_model_list_macro, redshift_list_macro, index_lens_split, lens_model_params

class BaselineSmoothMockModel_EllipticalMultipole(BaselineSmoothMockModel):

    def setup_lens_model(self, kwargs_lens_macro_init=None, macromodel_samples_fixed=None):
        lens_model_list_macro = ['EPL_MULTIPOLE_M3M4_ELL', 'SHEAR']
        kwargs_lens_macro = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.1, 'e2': -0.1,
                              'gamma': 2.0, 'a4_a': 0.0,
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


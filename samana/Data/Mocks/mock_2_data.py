import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData.mock_2_simple import image_data as simple_image_data
from samana.Data.ImageData.MockImageData.mock_2_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.MockImageData.mock_2_cosmos import image_data_new as cosmos_image_data_new
from samana.Data.ImageData.MockImageData.mock_2_2038 import image_data as simulated_2038_image_data
from samana.Data.ImageData.MockImageData.mock_2_cosmos_psf3 import image_data as cosmos_image_data_psf3
from samana.Data.ImageData.MockImageData.mock_2_cosmos_wdm import image_data as cosmos_image_data_wdm

class Mock2Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False, cosmos_source_psf3=False):

        z_lens = 0.4
        z_source = 1.7
        x_image = [ 0.96123035, -0.88968036, -0.70396572,  0.15753392]
        y_image = [ 0.76392051, -0.2642994 ,  0.53126911, -0.80157403]
        magnifications_true = [3.18286653, 6.59852554, 5.19721593, 2.43877   ]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0002277
        self.a4a_true = -0.004348
        self.delta_phi_m3_true = -0.067025
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        elif sim2038_source:
            image_data = simulated_2038_image_data
        elif cosmos_source_psf3:
            image_data = cosmos_image_data_psf3
        else:
            image_data = simple_image_data
        super(Mock2Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties, image_data,
                                        super_sample_factor)


class Mock2DataPSF3(Mock2Data):

    def __init__(self, super_sample_factor=1.0):
        super(Mock2DataPSF3, self).__init__(super_sample_factor, cosmos_source_psf3=True)

    @property
    def kwargs_psf(self):
        fwhm = 0.3
        deltaPix = self.coordinate_properties[0]
        kwargs_psf = {'psf_type': 'GAUSSIAN',
                      'fwhm': fwhm,
                      'pixel_size': deltaPix,
                      'truncation': 5}
        return kwargs_psf

class Mock2DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):
        z_lens = 0.4
        z_source = 1.7
        x_image = [0.92818, -0.91011, -0.72283, 0.18645]
        y_image = [0.81366, -0.25818, 0.53113, -0.80334]
        magnifications_true = [3.07732, 6.39911, 4.895, 2.36147]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = 0.0002277
        self.a4a_true = -0.004348
        self.delta_phi_m3_true = -0.067025
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock2DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                        image_data, super_sample_factor)

class Mock2DataNew(MockBase):
    """
    Created with pyHalo commit d9772f6
    Includes the Galacticus truncation model applied to CDM subhalos
    """
    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.4
        z_source = 1.7
        x_image = np.array([ 0.91094244, -0.87994455, -0.69009147,  0.16632836])
        y_image = np.array([ 0.75072815, -0.24933848,  0.53528928, -0.78094243])
        magnifications_true = np.array([3.02539226, 6.51658887, 4.86756372, 2.3252286 ])
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None
        self.a3a_true = 0.0002277
        self.a4a_true = -0.004348
        self.delta_phi_m3_true = -0.067025
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_new
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock2DataNew, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                        image_data, super_sample_factor)

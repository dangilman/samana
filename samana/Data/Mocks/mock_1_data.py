import numpy as np
from samana.Data.Mocks.base import MockBase
from samana.Data.ImageData.MockImageData.mock_1_simple import image_data as simple_image_data
from samana.Data.ImageData.MockImageData.mock_1_cosmos import image_data as cosmos_image_data
from samana.Data.ImageData.MockImageData.mock_1_cosmos import image_data_new as cosmos_image_data_new
from samana.Data.ImageData.MockImageData.mock_1_cosmos_wdm import image_data as cosmos_image_data_wdm
from samana.Data.ImageData.MockImageData.mock_1_2038 import image_data as simulated_2038_image_data
from samana.Data.ImageData.MockImageData.mock_1_cosmos_psf3 import image_data as cosmos_image_data_psf3

class Mock1Data(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=False, sim2038_source=False,
                 cosmos_source_psf3=False):

        z_lens = 0.5
        z_source = 2.2
        x_image = [-0.92161217,  0.78024418, -0.25056735,  0.70611926]
        y_image = [-0.6914899 ,  0.61451654,  0.86960051, -0.52450815]
        magnifications_true = [ 4.34866761, 10.45921158,  4.45857144,  4.52709824]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004010
        self.a4a_true = -0.004488
        self.delta_phi_m3_true = -0.08689
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data
        elif sim2038_source:
            image_data = simulated_2038_image_data
        elif cosmos_source_psf3:
            image_data = cosmos_image_data_psf3
        else:
            image_data = simple_image_data
        super(Mock1Data, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                        image_data, super_sample_factor)

class Mock1DataPSF3(Mock1Data):

    def __init__(self, super_sample_factor=1.0):
        super(Mock1DataPSF3, self).__init__(super_sample_factor, cosmos_source_psf3=True)

    @property
    def kwargs_psf(self):
        fwhm = 0.3
        deltaPix = self.coordinate_properties[0]
        kwargs_psf = {'psf_type': 'GAUSSIAN',
                      'fwhm': fwhm,
                      'pixel_size': deltaPix,
                      'truncation': 5}
        return kwargs_psf

class Mock1DataNew(MockBase):
    """
    Created with pyHalo commit d9772f6
    Includes the Galacticus truncation model applied to CDM subhalos
    """
    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.5
        z_source = 2.2
        x_image = np.array([-0.89778621,  0.74734589, -0.24599752,  0.70958954])
        y_image = np.array([-0.6912581 ,  0.61500878,  0.84860177, -0.48576415])
        magnifications_true = np.array([3.82156356, 7.17056328, 4.24261308, 4.23036913])
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004010
        self.a4a_true = -0.004488
        self.delta_phi_m3_true = -0.08689
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_new
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock1DataNew, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                        image_data, super_sample_factor)

class Mock1DataWDM(MockBase):

    def __init__(self, super_sample_factor=1.0, cosmos_source=True):

        z_lens = 0.5
        z_source = 2.2
        x_image = [-0.91847, 0.75361, -0.27069, 0.69158]
        y_image = [-0.69266, 0.64057, 0.86346, -0.52606]
        magnifications_true = [4.22817, 6.87895, 4.56096, 4.33764]
        magnification_measurement_errors = 0.0
        magnifications = np.array(magnifications_true) + np.array(magnification_measurement_errors)
        astrometric_uncertainties = [0.003] * 4
        flux_ratio_uncertainties = None

        self.a3a_true = -0.004010
        self.a4a_true = -0.004488
        self.delta_phi_m3_true = -0.08689
        self.delta_phi_m4_true = 0.0
        if cosmos_source:
            image_data = cosmos_image_data_wdm
        else:
            raise Exception('only cosmos source implemented for this class')
        super(Mock1DataWDM, self).__init__(z_lens, z_source, x_image, y_image,
                                    magnifications, astrometric_uncertainties, flux_ratio_uncertainties,
                                        image_data, super_sample_factor)


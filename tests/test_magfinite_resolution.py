import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt
from copy import deepcopy
from pyHalo.single_realization import SingleHalo
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.concentration_models import preset_concentration_models
from samana.analysis_util import quick_setup, numerics_setup
from lenstronomy.LensModel.lens_model import LensModel
from samana.image_magnification_util import setup_gaussian_source
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_resolution, auto_raytracing_grid_size
from lenstronomy.Cosmo.background import Background

class TestMagnificationGridRes(object):
    """
    Test that the initial kwargs_lens approximately satisifes the lens equation for the image positions
    """
    def setup_method(self):

        cosmo = Background()
        self._astropy_cosmo = cosmo.cosmo

        halo_mass = 5 * 10 ** 6
        x_arcsec = 0.
        y_arcsec = 0.0
        z_halo = 0.5

        zsource = 1.5
        zlens = 0.5
        mass_definition = 'NFW'
        subhalo_flag = False
        lens_cosmo = LensCosmo(zlens, zsource)
        astropy_class = lens_cosmo.cosmo
        model, kwargs_concentration_model = preset_concentration_models('DIEMERJOYCE19')
        kwargs_concentration_model['scatter'] = False
        kwargs_concentration_model['cosmo'] = astropy_class
        concentration_model = model(**kwargs_concentration_model)
        halo_truncation_model = None
        kwargs_halo_model = {'truncation_model': halo_truncation_model,
                             'concentration_model': concentration_model,
                             'kwargs_density_profile': {}}
        single_halo = SingleHalo(halo_mass, x_arcsec, y_arcsec, mass_definition, z_halo, zlens, zsource,
                                 subhalo_flag, kwargs_halo_model=kwargs_halo_model)
        self._lens_model_list_halo, self._redshift_array_halo, self._kwargs_lens_halo_base, _ = single_halo.lensing_quantities()

    def _run_test(self, lens_ID, source_size_list, index_max=3,
                  high_res_factor=5.0,
                  tol_mag=0.01, tol_ratio=0.0001):

        #print(lens_ID)
        data, model = quick_setup(lens_ID)
        data_class = data()
        model_class = model(data_class)

        lens_model_list_macro, redshift_list_macro, _, lens_model_params = model_class.setup_lens_model()
        kwargs_lens_init = lens_model_params[0]
        lens_model = LensModel(lens_model_list_macro + self._lens_model_list_halo,
                               lens_redshift_list=list(redshift_list_macro) + list(self._redshift_array_halo),
                               multi_plane=True,
                               z_source=data_class.z_source)
        image_index = 1
        kwargs_halo = deepcopy(self._kwargs_lens_halo_base)
        kwargs_halo[0]['center_x'] = data_class.x_image[image_index] + 0.003
        kwargs_halo[0]['center_y'] = data_class.y_image[image_index] + 0.00
        source_x, source_y = lens_model.ray_shooting(data_class.x_image,
                                                     data_class.y_image,
                                                     kwargs_lens_init + kwargs_halo)

        for source_size_pc in source_size_list:

            rescale_grid_size, rescale_grid_resolution = numerics_setup(lens_ID)
            source_model_quasar, kwargs_source = setup_gaussian_source(source_size_pc,
                                                                       np.mean(source_x), np.mean(source_y),
                                                                       self._astropy_cosmo, data_class.z_source)
            grid_size = rescale_grid_size * auto_raytracing_grid_size(source_size_pc)
            grid_resolution = rescale_grid_resolution * auto_raytracing_grid_resolution(source_size_pc)

            magnifications_default, _ = model_class.image_magnification_gaussian(source_model_quasar,
                                                                              kwargs_source,
                                                                              lens_model,
                                                                              kwargs_lens_init + kwargs_halo,
                                                                              kwargs_lens_init + kwargs_halo,
                                                                              grid_size,
                                                                              grid_resolution)
            flux_ratios_default = magnifications_default[1:]/magnifications_default[0]

            magnifications_high_res, _ = model_class.image_magnification_gaussian(source_model_quasar,
                                                                                 kwargs_source,
                                                                                 lens_model,
                                                                                  kwargs_lens_init + kwargs_halo,
                                                                                  kwargs_lens_init + kwargs_halo,
                                                                                 grid_size,
                                                                                 grid_resolution/high_res_factor)
            flux_ratios_high_res = magnifications_high_res[1:] / magnifications_high_res[0]
            diff = [magnifications_high_res[i] / magnifications_default[i] - 1 for i in range(0,index_max+1)]
            npt.assert_array_less(np.absolute(diff), tol_mag)
            diff = [flux_ratios_high_res[i] / flux_ratios_default[i] - 1 for i in range(0,index_max)]
            npt.assert_array_less(np.absolute(diff), tol_ratio)

    def test_b1422(self):

        lens_ID = 'B1422'
        source_size_list = [50.0]
        self._run_test(lens_ID, source_size_list, index_max=2)

    def test_h1413(self):
        # H1413
        # flux ratios with subhalo: [1.42265396 1.33961941 0.54018591] (5*10^7)
        # flux ratios with subhalo: [1.3527355  1.33995483 0.54023253] (5*10^6)
        # flux ratios with subhalo: [1.32362166 1.33998064 0.54023914] (5*10^5)
        # flux ratios with subhalo: [1.31288459 1.33997555 0.54023995] (5*10^4)
        # flux ratios without subhalo: [1.3084687,  1.3394672,  0.54003725]

        lens_ID = 'H1413'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_he0435(self):

        lens_ID = 'HE0435'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    # def test_0147(self):
    #
    #     lens_ID = 'J0147'
    #     source_size_list = [2.0, 8.0]
    #     self._run_test(lens_ID, source_size_list)

    def test_j0248(self):

        lens_ID = 'J0248'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_j0259(self):

        lens_ID = 'J0259_HST_475X'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list, tol_mag=0.01)
        #
        # lens_ID = 'J0259'
        # source_size_list = [2.0, 8.0]
        # self._run_test(lens_ID, source_size_list, tol_mag=0.012)
    #
    def test_0405(self):

        lens_ID = 'J0405'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_0607(self):

        lens_ID = 'J0607'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_0608(self):

        lens_ID = 'J0608'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_0659(self):

        lens_ID = 'J0659'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_0803(self):

        lens_ID = 'J0803'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_0924(self):

        lens_ID = 'J0924'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_1042(self):

        lens_ID = 'J1042'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_1131(self):

        lens_ID = 'J1131'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_1251(self):

        lens_ID = 'J1251'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_1537(self):

        lens_ID = 'J1537'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_2017(self):

        lens_ID = 'J2017'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_2145(self):

        lens_ID = 'J2145'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_2205(self):

        lens_ID = 'J2205'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_2344(self):

        lens_ID = 'J2344'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    # def test_m1134(self):
    #
    #     lens_ID = 'M1134'
    #     self._test_solution(lens_ID)

    # def test_mg0414(self):
    #
    #     lens_ID = 'MG0414'
    #     self._test_solution(lens_ID)

    def test_PG1115(self):

        lens_ID = 'PG1115'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_1606(self):

        lens_ID = 'PSJ1606'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    # def test_rxj0911(self):
    #
    #     lens_ID = 'RXJ0911'
    #     self._test_solution(lens_ID)

    def test_wfi2033(self):

        lens_ID = 'WFI2033'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)

    def test_2038(self):

        lens_ID = 'WGD2038'
        source_size_list = [2.0, 8.0]
        self._run_test(lens_ID, source_size_list)


if __name__ == "__main__":
    pytest.main()

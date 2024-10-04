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

        halo_mass = 1 * 10 ** 9
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

    def _run_test(self, lens_ID):

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

        source_size_pc = 6.0
        rescale_grid_size, rescale_grid_resolution = numerics_setup(lens_ID)
        source_model_quasar, kwargs_source = setup_gaussian_source(source_size_pc,
                                                                   np.mean(source_x), np.mean(source_y),
                                                                   self._astropy_cosmo, data_class.z_source)
        grid_size = rescale_grid_size * auto_raytracing_grid_size(source_size_pc)
        grid_resolution = rescale_grid_resolution * auto_raytracing_grid_resolution(source_size_pc)

        elliptical_ray_tracing_grid = True
        magnifications_ellipse, _ = model_class.image_magnification_gaussian(source_model_quasar,
                                                                          kwargs_source,
                                                                          lens_model,
                                                                          kwargs_lens_init + kwargs_halo,
                                                                          kwargs_lens_init + kwargs_halo,
                                                                          grid_size,
                                                                          grid_resolution,
                                                                          lens_model,
                                                                          elliptical_ray_tracing_grid)
        elliptical_ray_tracing_grid = False
        magnifications_circle, _ = model_class.image_magnification_gaussian(source_model_quasar,
                                                                             kwargs_source,
                                                                             lens_model,
                                                                             kwargs_lens_init + kwargs_halo,
                                                                             kwargs_lens_init + kwargs_halo,
                                                                             grid_size,
                                                                             grid_resolution,
                                                                             lens_model,
                                                                             elliptical_ray_tracing_grid)
        npt.assert_almost_equal(magnifications_circle, magnifications_ellipse, 3)

    def test_b1422(self):

        lens_ID = 'B1422'
        self._run_test(lens_ID)

    def test_h1413(self):

        lens_ID = 'H1413'
        self._run_test(lens_ID)

    def test_he0435(self):

        lens_ID = 'HE0435'
        self._run_test(lens_ID)

    def test_0147(self):

        lens_ID = 'PSJ0147'
        self._run_test(lens_ID)

    def test_j0248(self):

        lens_ID = 'J0248'
        self._run_test(lens_ID)

    def test_j0259(self):

        lens_ID = 'J0259_HST_475X'
        self._run_test(lens_ID)

        lens_ID = 'J0259'
        self._run_test(lens_ID)
    #
    def test_0405(self):

        lens_ID = 'J0405'
        self._run_test(lens_ID)

    def test_0607(self):

        lens_ID = 'J0607'
        self._run_test(lens_ID)

    def test_0608(self):

        lens_ID = 'J0608'
        self._run_test(lens_ID)

    def test_0659(self):

        lens_ID = 'J0659'
        self._run_test(lens_ID)

    def test_0803(self):

        lens_ID = 'J0803'
        self._run_test(lens_ID)

    def test_0924(self):

        lens_ID = 'J0924'
        self._run_test(lens_ID)

    def test_1042(self):

        lens_ID = 'J1042'
        self._run_test(lens_ID)

    def test_1131(self):

        lens_ID = 'J1131'
        self._run_test(lens_ID)

    def test_1251(self):

        lens_ID = 'J1251'
        self._run_test(lens_ID)

    def test_1537(self):

        lens_ID = 'J1537'
        self._run_test(lens_ID)

    def test_2017(self):

        lens_ID = 'J2017'
        self._run_test(lens_ID)

    def test_2145(self):

        lens_ID = 'J2145'
        self._run_test(lens_ID)

    def test_2205(self):

        lens_ID = 'J2205'
        self._run_test(lens_ID)

    def test_2344(self):

        lens_ID = 'J2344'
        self._run_test(lens_ID)

    def test_m1134(self):

        lens_ID = 'M1134_MIRI'
        self._run_test(lens_ID)

    def test_mg0414(self):

        lens_ID = 'MG0414'
        self._run_test(lens_ID)

    def test_PG1115(self):

        lens_ID = 'PG1115'
        self._run_test(lens_ID)

    def test_1606(self):

        lens_ID = 'PSJ1606'
        self._run_test(lens_ID)

    def test_rxj1131(self):

        lens_ID = 'RXJ1131'
        self._run_test(lens_ID)

    # def test_rxj0911(self):
    #
    #     lens_ID = 'RXJ0911'
    #     self._run_test(lens_ID)

    def test_wfi2033(self):

        lens_ID = 'WFI2033'
        self._run_test(lens_ID)

    def test_2038(self):

        lens_ID = 'WGD2038'
        self._run_test(lens_ID)

    def test_2017(self):

        lens_ID = 'J2017'
        self._run_test(lens_ID)


if __name__ == "__main__":
    pytest.main()

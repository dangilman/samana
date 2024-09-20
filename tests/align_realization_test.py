from samana.forward_model_util import align_realization
from pyHalo.PresetModels.cdm import CDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import numpy.testing as npt
import pytest

class TestAlignment(object):

    def setup_method(self):

        self.lens_model_list = ['EPL', 'SHEAR', 'SIS', 'SIS']
        self.lens_redshift_list = [0.3, 0.3, 1.5, 2.5]
        self.kwargs_lens = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.2, 'e2': -0.5, 'gamma': 2.0},
                       {'gamma1': 0.04, 'gamma2': 0.01},
                       {'theta_E': -0.4, 'center_x': 0.5, 'center_y': -0.5},
                       {'theta_E': 0.4, 'center_x': 0.95, 'center_y': -0.25}]
        z_lens = self.lens_redshift_list[0]
        z_source = 4.0
        self.realization = CDM(z_lens, z_source, sigma_sub=0.0, LOS_normalization=0.0)
        lens_model = LensModel(self.lens_model_list, lens_redshift_list=self.lens_redshift_list, multi_plane=True,
                               z_source=z_source)
        solver = LensEquationSolver(lens_model)
        source_x = 0.2
        source_y = -0.
        self.x_image, self.y_image = solver.image_position_from_source(source_x, source_y, self.kwargs_lens)

    def test_alignment(self):
        realization_aligned, ray_interp_x, ray_interp_y, lens_model, kwargs_lens = align_realization(self.realization,
                                                                                                     self.lens_model_list,
                                                                                                     self.lens_redshift_list,
                                                                                                     self.kwargs_lens,
                                                                                                     self.x_image,
                                                                                                     self.y_image,
                                                                                                     self.realization.lens_cosmo.cosmo.astropy)
        rendering_centerx, rendering_centery = realization_aligned.rendering_center
        dist = self.realization.lens_cosmo.cosmo.D_C_transverse
        z_values = [0.3, 1.5, 2.5, 4.0]
        distances = [dist(zi) for zi in z_values]
        ra, dec = rendering_centerx(distances), rendering_centery(distances)
        npt.assert_almost_equal(ra, [ 0.22846704,  0.12402074, -0.07408837,  0.2000198 ])
        npt.assert_almost_equal(dec, [-1.77161805e-01, -1.68083117e-01, -5.02271413e-04,  1.60442397e-05])


if __name__ == "__main__":
    pytest.main()

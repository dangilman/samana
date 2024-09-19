import numpy as np
import numpy.testing as npt
import pytest
from samana.analysis_util import quick_setup
from lenstronomy.LensModel.lens_model import LensModel

class TestSolutionLensEquation(object):
    """
    Test that the initial kwargs_lens approximately satisifes the lens equation for the image positions
    """
    def setup_method(self):
        pass

    def _test_solution(self, lens_ID):
        print(lens_ID)
        data, model = quick_setup(lens_ID)
        data_class = data()
        model_class = model(data_class)
        lens_model_list_macro, redshift_list_macro, _, lens_model_params = model_class.setup_lens_model()
        kwargs_lens_init = lens_model_params[0]
        lens_model = LensModel(lens_model_list_macro, lens_redshift_list=list(redshift_list_macro),
                               multi_plane=True, z_source=data_class.z_source)
        source_x, source_y = lens_model.ray_shooting(data_class.x_image, data_class.y_image, kwargs_lens_init)
        dx = np.sum((source_x[0] - source_x[1]) ** 2 + \
                    (source_x[0] - source_x[2]) ** 2 + \
                    (source_x[0] - source_x[3]) ** 2 + \
                    (source_x[1] - source_x[2]) ** 2 + \
                    (source_x[1] - source_x[3]) ** 2 + \
                    (source_x[2] - source_x[3]) ** 2)
        dy = np.sum((source_y[0] - source_y[1]) ** 2 + \
                    (source_y[0] - source_y[2]) ** 2 + \
                    (source_y[0] - source_y[3]) ** 2 + \
                    (source_y[1] - source_y[2]) ** 2 + \
                    (source_y[1] - source_y[3]) ** 2 + \
                    (source_y[2] - source_y[3]) ** 2)
        dr = np.sqrt(dx + dy)
        npt.assert_almost_equal(dr, 0.0, 5)

    def test_h1413(self):

        lens_ID = 'H1413'
        self._test_solution(lens_ID)

    def test_0607(self):

        lens_ID = 'J0607'
        self._test_solution(lens_ID)

    def test_0659(self):

        lens_ID = 'J0659'
        self._test_solution(lens_ID)

    def test_he0435(self):

        lens_ID = 'HE0435'
        self._test_solution(lens_ID)

    def test_j0248(self):

        lens_ID = 'J0248'
        self._test_solution(lens_ID)

    def test_j0259(self):

        lens_ID = 'J0259_HST_475X'
        self._test_solution(lens_ID)

    def test_0405(self):

        lens_ID = 'J0405'
        self._test_solution(lens_ID)

    def test_0405(self):

        lens_ID = 'WGD2038'
        self._test_solution(lens_ID)
#
# t = TestSolutionLensEquation()
# t.setup_method()
# t.test_0607()
# t.test_0659()
# t.test_h1413()
# t.test_he0435()
# t.test_j0248()
# t.test_j0259()

if __name__ == "__main__":
    pytest.main()

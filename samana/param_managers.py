from lenstronomy.Util.param_util import shear_cartesian2polar, shear_polar2cartesian
from lenstronomy.Util.param_util import ellipticity2phi_q, phi_q2_ellipticity
import numpy as np
from copy import deepcopy


class PowerLawParamManager(object):

    """
    Base class for handling the translation between key word arguments and parameter arrays for
    EPL mass models. This class is intended for use in modeling galaxy-scale lenses
    """

    def __init__(self, kwargs_lens_init):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        """

        self.kwargs_lens = kwargs_lens_init

    def axis_ratio_penalty(self, args, q_min=0.3):
        """
        Penalize unphysical axis ratios < 0.3
        :param args: vector of numbers corresponding to kwargs_lens
        :param q_min: minimum allowed axis ratio
        :return: penalty term if q drops below q_min
        """
        e1 = args[3]
        e2 = args[4]
        c = np.sqrt(e1 ** 2 + e2 ** 2)
        c = np.minimum(c, 0.9999)
        q = (1 - c) / (1 + c)
        if q < q_min:
            return 1e9
        else:
            return 0.

    @property
    def to_vary_index(self):

        """
        The number of lens models being varied in this routine. This is set to 2 because the first three lens models
        are EPL and SHEAR, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 2

    def bounds(self, re_optimize, scale=1.):

        """
        Sets the low/high parameter bounds for the particle swarm optimization

        NOTE: The low/high values specified here are intended for galaxy-scale lenses. If you want to use this
        for a different size system you should create a new ParamClass with different settings

        :param re_optimize: keep a narrow window around each parameter
        :param scale: scales the size of the uncertainty window
        :return:
        """
        args = self.kwargs_to_args(self.kwargs_lens)
        if re_optimize:
            thetaE_shift = 0.005
            center_shift = 0.01
            e_shift = 0.05
            g_shift = 0.01

        else:
            thetaE_shift = 0.25
            center_shift = 0.2
            e_shift = 0.2
            g_shift = 0.05

        shifts = np.array([thetaE_shift, center_shift, center_shift, e_shift, e_shift, g_shift, g_shift])
        low = np.array(args) - shifts * scale
        high = np.array(args) + shifts * scale
        return low, high

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """
        thetaE = kwargs[0]['theta_E']
        center_x = kwargs[0]['center_x']
        center_y = kwargs[0]['center_y']
        e1 = kwargs[0]['e1']
        e2 = kwargs[0]['e2']
        g1 = kwargs[1]['gamma1']
        g2 = kwargs[1]['gamma2']
        args = (thetaE, center_x, center_y, e1, e2, g1, g2)
        return args

class EPLMultipole134(PowerLawParamManager):

    def __init__(self, kwargs_lens_init, a1a_init, a3a_init, a4a_init,
                 delta_phi_m1, delta_phi_m3, delta_phi_m4, q=None, gamma_ext=None):
        """

        :param kwargs_lens_init:
        :param a1a_init:
        :param a3a_init:
        :param a4a_init:
        :param delta_phi_m1:
        :param delta_phi_m3:
        :param delta_phi_m4:
        :param q:
        :param gamma_ext:
        """

        self._a1a_init = a1a_init
        self._a4a_init = a4a_init
        self._a3a_init = a3a_init
        self._delta_phi_m1 = delta_phi_m1
        self._delta_phi_m3 = delta_phi_m3
        self._delta_phi_m4 = delta_phi_m4
        self._q = q
        self._gamma_ext = gamma_ext
        super(EPLMultipole134, self).__init__(kwargs_lens_init)

    def param_chi_square_penalty(self, args, q_min=0.1):
        """

        :param args:
        :param q_min:
        :return:
        """
        return self.axis_ratio_penalty(args, q_min)

    def args_to_kwargs(self, args):
        (thetaE, center_x, center_y, _e1, _e2, _g1, _g2) = args
        if self._q is None:
            e1, e2 = _e1, _e2
        else:
            # enforce fixed q while sampling phi_q
            phi_q, _ = ellipticity2phi_q(_e1, _e2)
            e1, e2 = phi_q2_ellipticity(phi_q, self._q)
        if self._gamma_ext is None:
            g1, g2 = _g1, _g2
        else:
            phi_gamma, _ = shear_cartesian2polar(_g1, _g2)
            g1, g2 = shear_polar2cartesian(phi_gamma, self._gamma_ext)
        gamma = self.kwargs_lens[0]['gamma']
        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}
        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[0]['a1_a'] = self._a1a_init
        self.kwargs_lens[0]['a4_a'] = self._a4a_init
        self.kwargs_lens[0]['a3_a'] = self._a3a_init
        self.kwargs_lens[0]['delta_phi_m1'] = self._delta_phi_m1
        self.kwargs_lens[0]['delta_phi_m3'] = self._delta_phi_m3
        self.kwargs_lens[0]['delta_phi_m4'] = self._delta_phi_m4
        kwargs_shear = {'gamma1': g1, 'gamma2': g2}
        self.kwargs_lens[1] = kwargs_shear
        return self.kwargs_lens

class EPLMultipole134LensMassPrior(EPLMultipole134):

    def __init__(self, kwargs_lens_init, a1a_init, a3a_init, a4a_init,
                 delta_phi_m1, delta_phi_m3, delta_phi_m4, center_x, center_y, sigma_xy,
                 q=None, gamma_ext=None):
        """
        Attempts to solve the lens equation with a punishing term on the deflector mass centroid deviating from
        (center_x, center_y) by more than sigma_xy

        :param kwargs_lens_init:
        :param a1a_init:
        :param a3a_init:
        :param a4a_init:
        :param delta_phi_m1:
        :param delta_phi_m3:
        :param delta_phi_m4:
        :param center_x:
        :param center_y:
        :param sigma_xy:
        :param q:
        :param gamma_ext:
        """
        self._center_x = center_x
        self._center_y = center_y
        self._sigmaxy = sigma_xy
        self._q = q
        self._gamma_ext = gamma_ext
        super(EPLMultipole134LensMassPrior, self).__init__(kwargs_lens_init, a1a_init, a4a_init, a3a_init,
                 delta_phi_m1, delta_phi_m3, delta_phi_m4, q, gamma_ext)

    def param_chi_square_penalty(self, args, q_min=0.1):
        """

        :param args:
        :param q_min:
        :return:
        """
        return self.axis_ratio_penalty(args, q_min) + self.mass_centroid_penalty(args)

    def mass_centroid_penalty(self, args):
        """
        Penalizes mass centroids far away from a fixed coordinate
        :param args: vector of lens model parameters corresponding to kwargs_lens
        :return: penalty term
        """
        delta_center_x = args[1] - self._center_x
        delta_center_y = args[2] - self._center_y
        dr = np.hypot(delta_center_x, delta_center_y)
        if dr > 5 * self._sigmaxy:
            return 1e9
        else:
            return np.exp(-0.5 * dr ** 2 / self._sigmaxy ** 2)

def auto_param_class(lens_model_list_macro, kwargs_lens_init, macromodel_samples_fixed_dict):
    """

    :param lens_model_list_macro:
    :param kwargs_lens_init:
    :param macromodel_samples_fixed_dict:
    :return:
    """
    macromodel_samples_fixed_param_names = macromodel_samples_fixed_dict.keys()
    assert lens_model_list_macro[0] in ['EPL_MULTIPOLE_M1M3M4_ELL', 'EPL_MULTIPOLE_M1M3M4']
    assert lens_model_list_macro[1] == 'SHEAR'
    if 'gamma_ext' in macromodel_samples_fixed_param_names:
        fixed_gamma_ext = macromodel_samples_fixed_dict['gamma_ext']
    else:
        fixed_gamma_ext = None
    if 'q' in macromodel_samples_fixed_param_names:
        fixed_q = macromodel_samples_fixed_dict['q']
    else:
        fixed_q = None
    if 'mass_centroid_x' in macromodel_samples_fixed_param_names:
        assert 'mass_centroid_y' in macromodel_samples_fixed_param_names
        assert 'sigma_xy_mass_centroid' in macromodel_samples_fixed_param_names
        mass_centroid_x = macromodel_samples_fixed_dict['mass_centroid_x']
        mass_centroid_y = macromodel_samples_fixed_dict['mass_centroid_y']
        sigma_xy_mass_centroid = macromodel_samples_fixed_dict['sigma_xy_mass_centroid']
        param_class = EPLMultipole134LensMassPrior(
            kwargs_lens_init,
            macromodel_samples_fixed_dict['a1_a'],
            macromodel_samples_fixed_dict['a3_a'],
            macromodel_samples_fixed_dict['a4_a'],
            macromodel_samples_fixed_dict['delta_phi_m1'],
            macromodel_samples_fixed_dict['delta_phi_m3'],
            macromodel_samples_fixed_dict['delta_phi_m4'],
            mass_centroid_x,
            mass_centroid_y,
            sigma_xy_mass_centroid,
            fixed_q,
            fixed_gamma_ext
        )
    else:
        param_class = EPLMultipole134(
            kwargs_lens_init,
            macromodel_samples_fixed_dict['a1_a'],
            macromodel_samples_fixed_dict['a3_a'],
            macromodel_samples_fixed_dict['a4_a'],
            macromodel_samples_fixed_dict['delta_phi_m1'],
            macromodel_samples_fixed_dict['delta_phi_m3'],
            macromodel_samples_fixed_dict['delta_phi_m4'],
            fixed_q,
            fixed_gamma_ext
        )
    return param_class


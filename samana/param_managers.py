from lenstronomy.Util.param_util import shear_cartesian2polar, shear_polar2cartesian
from lenstronomy.Util.param_util import ellipticity2phi_q
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

    @staticmethod
    def am_measured_to_physical(a_m_measured, theta_E, q):
        """
        Computes the scaling factor for the multipole moments to implement the "shape distortion" convention.

        Priors are specified on a3_a and a4_a, the MEASURED shapes of ellipitical galaxies. These quantities are
        transformed in the PHYSICAL amplitudes of the multipole moments using the relation:

        a4_a_measured = a4_a_physical * lambda

        where lambda = theta_E / sqrt(q).
        """
        lam = theta_E / np.sqrt(q)
        a_m_phys = a_m_measured * lam
        # note that the multipole classes in lenstronomy do not apply this scaling, but the multipole
        # terms in EPL_MULTIPOLE_M3M4, EPL_BOXY_DISKY, and EPL_MULTIPOLE_M3M4_ELL do automatically
        # apply the scaling (so this method isn't used for those classes)
        return a_m_phys

    def param_chi_square_penalty(self, args, q_min=0.19):
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
            g_shift = 0.005

        else:
            thetaE_shift = 0.25
            center_shift = 0.2
            e_shift = 0.2
            g_shift = 0.01

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

class PowerLawFixedShear(PowerLawParamManager):

    """
    This class implements a fit of EPL + external shear with every parameter except the power law slope AND the
    shear strength allowed to vary. The user should specify shear_strengh in the args_param_class keyword when
    creating the Optimizer class
    """

    def __init__(self, kwargs_lens_init, shear_strength):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        :param shear_strength: the strenght of the external shear to be kept fixed
        """
        self._shear_strength = shear_strength

        super(PowerLawFixedShear, self).__init__(kwargs_lens_init)

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters with fixed shear = shear_strength
        """

        (thetaE, center_x, center_y, e1, e2, g1, g2) = args
        gamma = self.kwargs_lens[0]['gamma']

        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}

        phi, _ = shear_cartesian2polar(g1, g2)
        gamma1, gamma2 = shear_polar2cartesian(phi, self._shear_strength)
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}

        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[1] = kwargs_shear

        return self.kwargs_lens


class PowerLawFreeShearMultipole(PowerLawParamManager):

    """
    This class implements a fit of EPL + external shear + a multipole term with every parameter except the
    power law slope and multipole moment free to vary. The mass centroid and orientation of the multipole term are
    fixed to that of the EPL profile

    """

    def __init__(self, kwargs_lens_init, a4a_init):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        """

        self._a4a_init = a4a_init
        super(PowerLawFreeShearMultipole, self).__init__(kwargs_lens_init)

    @property
    def to_vary_index(self):

        """
        The number of lens models being varied in this routine. This is set to 3 because the first three lens models
        are EPL, SHEAR, and MULTIPOLE, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 3

    def args_to_kwargs(self, args):

        (thetaE, center_x, center_y, e1, e2, g1, g2) = args

        gamma = self.kwargs_lens[0]['gamma']

        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}
        kwargs_shear = {'gamma1': g1, 'gamma2': g2}

        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[1] = kwargs_shear

        self.kwargs_lens[2]['center_x'] = center_x
        self.kwargs_lens[2]['center_y'] = center_y
        phi, q = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[2]['a_m'] = self.am_measured_to_physical(self._a4a_init, thetaE, q)
        self.kwargs_lens[2]['phi_m'] = phi
        return self.kwargs_lens


class PowerLawFixedShearMultipole(PowerLawFixedShear):
    """
    This class implements a fit of EPL + external shear + a multipole term with every parameter except the
    power law slope, shear strength, and multipole moment free to vary. The mass centroid and orientation of the
    multipole term are fixed to that of the EPL profile
    """

    def __init__(self, kwargs_lens_init, shear_strength, a4a_init):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        """

        self._a4a_init = a4a_init
        super(PowerLawFixedShearMultipole, self).__init__(kwargs_lens_init, shear_strength)

    @property
    def to_vary_index(self):

        """
        The number of lens models being varied in this routine. This is set to 3 because the first three lens models
        are EPL, SHEAR, and MULTIPOLE, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 3

    def args_to_kwargs(self, args):

        (thetaE, center_x, center_y, e1, e2, g1, g2) = args
        gamma = self.kwargs_lens[0]['gamma']

        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}

        phi, _ = shear_cartesian2polar(g1, g2)
        gamma1, gamma2 = shear_polar2cartesian(phi, self._shear_strength)
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}

        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[1] = kwargs_shear

        self.kwargs_lens[2]['center_x'] = center_x
        self.kwargs_lens[2]['center_y'] = center_y
        phi, q = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[2]['phi_m'] = phi
        self.kwargs_lens[2]['a_m'] = self.am_measured_to_physical(self._a4a_init, thetaE, q)
        return self.kwargs_lens

class PowerLawFreeShear(PowerLawParamManager):

    """
    This class implements a fit of EPL + external shear with every parameter except the power law slope allowed to vary
    """
    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        gamma = self.kwargs_lens[0]['gamma']
        kwargs_epl = {'theta_E': args[0], 'center_x': args[1], 'center_y': args[2],
                      'e1': args[3], 'e2': args[4], 'gamma': gamma}

        kwargs_shear = {'gamma1': args[5], 'gamma2': args[6]}

        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[1] = kwargs_shear

        return self.kwargs_lens

class PowerLawFixedShearMultipole_34(PowerLawFixedShear):
    """
    This class implements a fit of EPL + external shear + two multipole terms (m=3 and m=4) with every parameter except the
    power law slope, shear strength, and multipole moments free to vary. The mass centroid and orientation of the
    m=4 multipole term are fixed to that of the EPL profile, and the orientation of m=3 multiple term is
    fixed to some user-defined angle.
    """

    def __init__(self, kwargs_lens_init, shear_strength, a4a_init, a3a_init, delta_phi_m3):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        :param shear_strength: the strenght of the external shear to be kept fixed
        :param delta_phi_m3: the orientation of the m=3 multipole relative to the EPL position angle
        """

        self._a4a_init = a4a_init
        self._a3a_init = a3a_init
        self._delta_phi_m3 = delta_phi_m3
        if kwargs_lens_init[2]['m'] == 3:
            self._idx_m3 = 2
            self._idx_m4 = 3
        else:
            self._idx_m3 = 3
            self._idx_m4 = 2
        super(PowerLawFixedShearMultipole_34, self).__init__(kwargs_lens_init, shear_strength)

    @property
    def to_vary_index(self):

        """
        The number of lens models being varied in this routine. This is set to 4 because the first four lens models
        are EPL, SHEAR, MULTIPOLE, and MULTIPOLE, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 4

    def args_to_kwargs(self, args):

        (thetaE, center_x, center_y, e1, e2, g1, g2) = args
        gamma = self.kwargs_lens[0]['gamma']

        # handle the EPL profile
        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}
        self.kwargs_lens[0] = kwargs_epl

        # determine the orientation of external shear
        phi, _ = shear_cartesian2polar(g1, g2)
        gamma1, gamma2 = shear_polar2cartesian(phi, self._shear_strength)
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}
        self.kwargs_lens[1] = kwargs_shear

        # fix m=4 multipole centroid to EPL centroid
        self.kwargs_lens[self._idx_m4]['center_x'] = center_x
        self.kwargs_lens[self._idx_m4]['center_y'] = center_y
        # fix m=4 multipole orientation to EPL orientation
        phi, q = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[self._idx_m4]['phi_m'] = phi
        self.kwargs_lens[self._idx_m4]['a_m'] = self.am_measured_to_physical(self._a4a_init, thetaE, q)

        # fix m=3 multipole centroid to EPL centroid
        self.kwargs_lens[self._idx_m3]['center_x'] = center_x
        self.kwargs_lens[self._idx_m3]['center_y'] = center_y

        phi, _ = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[self._idx_m3]['phi_m'] = phi + self._delta_phi_m3
        self.kwargs_lens[self._idx_m3]['a_m'] = self.am_measured_to_physical(self._a3a_init, thetaE, q)

        return self.kwargs_lens

class PowerLawFreeShearMultipole_34(PowerLawParamManager):

    """
    This class implements a fit of EPL + external shear + a multipole term with every parameter except the
    power law slope and multipole moment free to vary. The mass centroid and orientation of the multipole term are
    fixed to that of the EPL profile

    """

    def __init__(self, kwargs_lens_init, a4a_init, a3a_init, delta_phi_m3):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        :param shear_strength: the strenght of the external shear to be kept fixed
        :param the orientation of the m=3 multipole relative to the EPL position angle
        """

        self._a4a_init = a4a_init
        self._a3a_init = a3a_init
        self._delta_phi_m3 = delta_phi_m3
        if kwargs_lens_init[2]['m'] == 3:
            self._idx_m3 = 2
            self._idx_m4 = 3
        else:
            self._idx_m3 = 3
            self._idx_m4 = 2
        super(PowerLawFreeShearMultipole_34, self).__init__(kwargs_lens_init)

    @property
    def to_vary_index(self):

        """
        The number of lens models being varied in this routine. This is set to 3 because the first three lens models
        are EPL, SHEAR, and MULTIPOLE, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 4

    def args_to_kwargs(self, args):

        (thetaE, center_x, center_y, e1, e2, g1, g2) = args
        gamma = self.kwargs_lens[0]['gamma']

        # handle the EPL profile
        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}
        self.kwargs_lens[0] = kwargs_epl

        kwargs_shear = {'gamma1': g1, 'gamma2': g2}
        self.kwargs_lens[1] = kwargs_shear

        # fix m=4 multipole centroid to EPL centroid
        self.kwargs_lens[self._idx_m4]['center_x'] = center_x
        self.kwargs_lens[self._idx_m4]['center_y'] = center_y
        # fix m=4 multipole orientation to EPL orientation
        phi, q = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[self._idx_m4]['phi_m'] = phi
        self.kwargs_lens[self._idx_m4]['a_m'] = self.am_measured_to_physical(self._a4a_init, thetaE, q)

        # fix m=3 multipole centroid to EPL centroid
        self.kwargs_lens[self._idx_m3]['center_x'] = center_x
        self.kwargs_lens[self._idx_m3]['center_y'] = center_y

        phi, _ = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[self._idx_m3]['phi_m'] = phi + self._delta_phi_m3
        self.kwargs_lens[self._idx_m3]['a_m'] = self.am_measured_to_physical(self._a3a_init, thetaE, q)

        return self.kwargs_lens

class EPLMultipole34FixedShear(PowerLawFixedShear):

    def __init__(self, kwargs_lens_init, shear_strength, a4a_init, a3a_init, delta_phi_m3, delta_phi_m4):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        :param shear_strength: the strenght of the external shear to be kept fixed
        :param delta_phi_m3: the orientation of the m=3 multipole relative to the EPL position angle
        """

        self._a4a_init = a4a_init
        self._a3a_init = a3a_init
        self._delta_phi_m3 = delta_phi_m3
        self._delta_phi_m4 = delta_phi_m4
        self._idx_m4 = 0
        self._idx_m3 = 0
        super(EPLMultipole34FixedShear, self).__init__(kwargs_lens_init, shear_strength)

    def args_to_kwargs(self, args):

        (thetaE, center_x, center_y, e1, e2, g1, g2) = args
        gamma = self.kwargs_lens[0]['gamma']
        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}
        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[0]['a4_a'] = self._a4a_init
        self.kwargs_lens[0]['a3_a'] = self._a3a_init
        self.kwargs_lens[0]['delta_phi_m3'] = self._delta_phi_m3
        self.kwargs_lens[0]['delta_phi_m4'] = self._delta_phi_m4
        kwargs_shear = {'gamma1': g1, 'gamma2': g2}
        self.kwargs_lens[1] = kwargs_shear
        return self.kwargs_lens

class EPLMultipole34FreeShear(PowerLawParamManager):

    def __init__(self, kwargs_lens_init, a4a_init, a3a_init, delta_phi_m3, delta_phi_m4):
        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        :param shear_strength: the strenght of the external shear to be kept fixed
        :param delta_phi_m3: the orientation of the m=3 multipole relative to the EPL position angle
        """

        self._a4a_init = a4a_init
        self._a3a_init = a3a_init
        self._delta_phi_m3 = delta_phi_m3
        self._delta_phi_m4 = delta_phi_m4
        self._idx_m4 = 0
        self._idx_m3 = 0
        super(EPLMultipole34FreeShear, self).__init__(kwargs_lens_init)

    def args_to_kwargs(self, args):
        (thetaE, center_x, center_y, e1, e2, g1, g2) = args
        gamma = self.kwargs_lens[0]['gamma']
        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}
        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[0]['a4_a'] = self._a4a_init
        self.kwargs_lens[0]['a3_a'] = self._a3a_init
        self.kwargs_lens[0]['delta_phi_m3'] = self._delta_phi_m3
        self.kwargs_lens[0]['delta_phi_m4'] = self._delta_phi_m4
        kwargs_shear = {'gamma1': g1, 'gamma2': g2}
        self.kwargs_lens[1] = kwargs_shear
        return self.kwargs_lens


def auto_param_class(lens_model_list_macro, kwargs_lens_init, macromodel_samples_fixed_dict):

    macromodel_samples_fixed_param_names = macromodel_samples_fixed_dict.keys()
    condition_1 = lens_model_list_macro[0] == 'EPL_MULTIPOLE_M3M4_ELL' and lens_model_list_macro[1] == 'SHEAR'
    condition_2 = lens_model_list_macro[0] == 'EPL_MULTIPOLE_M3M4' and lens_model_list_macro[1] == 'SHEAR'
    if condition_1 or condition_2:
        if 'a4_a' not in macromodel_samples_fixed_param_names:
            raise Exception(
                'when specifying a param class with an m=4 multi-pole moment, the amplitude of the multi-moment '
                'must be sampled from a prior specified with argument a4_a')
        if 'a3_a' not in macromodel_samples_fixed_param_names:
            raise Exception(
                'when specifying a param class with an m=3 multi-pole moment, the amplitude of the multi-moment '
                'must be sampled from a prior specified with argument a3_a')
        if 'delta_phi_m3' not in macromodel_samples_fixed_param_names:
            raise Exception(
                'when specifying a param class with an m=3 multi-pole moment, the orientation of the multi-moment '
                'must be sampled from a prior specified with keyword arguement delta_phi_m3')
        if 'delta_phi_m4' not in macromodel_samples_fixed_param_names:
            raise Exception(
                'when specifying a param class with an m=4 multi-pole moment, the orientation of the multi-moment '
                'must be sampled from a prior specified with keyword arguement delta_phi_m4')
        if 'gamma_ext' in macromodel_samples_fixed_param_names:
            param_class = EPLMultipole34FixedShear(kwargs_lens_init,
                                                   macromodel_samples_fixed_dict['gamma_ext'],
                                                   macromodel_samples_fixed_dict['a4_a'],
                                                   macromodel_samples_fixed_dict['a3_a'],
                                                   macromodel_samples_fixed_dict['delta_phi_m3'],
                                                   macromodel_samples_fixed_dict['delta_phi_m4'])
        else:
            param_class = EPLMultipole34FreeShear(kwargs_lens_init,
                                                  macromodel_samples_fixed_dict['a4_a'],
                                                  macromodel_samples_fixed_dict['a3_a'],
                                                  macromodel_samples_fixed_dict['delta_phi_m3'],
                                                  macromodel_samples_fixed_dict['delta_phi_m4'])

    else:
        raise Exception('this functionality only implemented for EPL_MULTIPOLE_M3M4 plus shear model')

    if param_class is None:
        raise Exception(
            'param class could not be identified from the specified lens model list and sampled macromodel parameters'
        )
    return param_class


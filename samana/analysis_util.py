import numpy as np
from copy import deepcopy
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from samana.image_magnification_util import perturbed_fluxes_from_fluxes, perturbed_flux_ratios_from_flux_ratios
from samana.output_storage import Output
import matplotlib.pyplot as plt
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot

__all__ = ['nmax_bic_minimize',
           'cut_on_data',
           'simulation_output_to_density',
           'quick_setup',
           'numerics_setup']


def default_rendering_area(lens_ID=None,
                           data_class=None,
                           model_class=None,
                           opening_angle_factor=6.0):
    """

    :param lens_ID:
    :param data_class:
    :param model_class:
    :param opening_angle_factor:
    :return:
    """
    if data_class is None or model_class is None:
        _data_class, _model_class = quick_setup(lens_ID)
        model = _model_class(_data_class())
    else:
        # note that data class must be instantiated when passed in
        model = model_class(data_class)
    thetaE = model.setup_lens_model()[-1][0][0]['theta_E']
    return opening_angle_factor * thetaE

def gamma_macro_priors(lens_ID):

    if lens_ID == 'B1422':
        gamma_macro_prior = None
    elif lens_ID == 'WFI2026':
        gamma_macro_prior = None
    elif lens_ID == 'B2045':
        raise Exception('not yet implemented')
    elif lens_ID == 'HE0435':
        gamma_macro_prior = {'gamma': ['UNIFORM', 2.0, 2.4]}
    elif lens_ID == 'J0248':
        gamma_macro_prior = {'gamma': ['UNIFORM', 2.0, 2.4]}
    elif lens_ID == 'J0248_HST':
        gamma_macro_prior = None
    elif lens_ID in ['J0259', 'J0259_HST_475X']:
        gamma_macro_prior = {'gamma': ['UNIFORM', 1.8, 2.4]}
    elif lens_ID == 'J0405':
        gamma_macro_prior = {'gamma': ['UNIFORM', 1.8, 2.4]}
    elif lens_ID == 'J0607':
        gamma_macro_prior = {'gamma': ['UNIFORM', 1.8, 2.4]}
    elif lens_ID == 'J0608':
        gamma_macro_prior = None
    elif lens_ID == 'J0659':
        gamma_macro_prior = {'gamma': ['UNIFORM', 2.0, 2.4]}
    elif lens_ID == 'J0803':
        gamma_macro_prior = None
    elif lens_ID == 'J0924':
        gamma_macro_prior = {'gamma': ['UNIFORM', 1.8, 2.4]}
    elif lens_ID in ['J1042', 'J1042_814W']:
        gamma_macro_prior = None
    elif lens_ID == 'J1131':
        gamma_macro_prior = {'gamma': ['UNIFORM', 1.8, 2.4]}
    elif lens_ID == 'J1251':
        gamma_macro_prior = None
    elif lens_ID == 'J1537':
        gamma_macro_prior = {'gamma': ['UNIFORM', 1.8, 2.3]}
    elif lens_ID == 'J2026':
        gamma_macro_prior = None
    elif lens_ID == 'J2205_MIRI':
        gamma_macro_prior = None
    elif lens_ID == 'J2205':
        gamma_macro_prior = None
    elif lens_ID == 'J2344':
        gamma_macro_prior = {'gamma': ['UNIFORM', 1.8, 2.4]}
    elif lens_ID == 'MG0414':
        gamma_macro_prior = {'gamma': ['UNIFORM', 1.7, 2.2]}
    elif lens_ID in ['PG1115', 'PG1115_NIRCAM']:
        gamma_macro_prior = {'gamma': ['UNIFORM', 1.8, 2.4]}
    elif lens_ID == 'PSJ0147':
        gamma_macro_prior = None
    elif lens_ID == 'PSJ1606':
        gamma_macro_prior = None
    elif lens_ID == 'RXJ0911':
        gamma_macro_prior = None
    elif lens_ID == 'RXJ1131':
        gamma_macro_prior = None
    elif lens_ID == 'WFI2033':
        gamma_macro_prior = {'gamma': ['UNIFORM', 2.0, 2.25]}
    elif lens_ID == 'WGD2038':
        gamma_macro_prior = {'gamma': ['UNIFORM', 2.0, 2.5]}
    elif lens_ID in ['M1134', 'M1134_MIRI']:
        gamma_macro_prior = None
    elif lens_ID == 'H1413':
        gamma_macro_prior = None
    elif lens_ID == 'J2017':
        gamma_macro_prior = None
    elif lens_ID == 'J2145':
        gamma_macro_prior = None
    else:
        raise Exception('lens ID '+str(lens_ID)+' not recognized!')
    return gamma_macro_prior

def satellite_galaxy_priors(lens_ID):

    if lens_ID == 'H1413':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.4, 0.2],
            'satellite_1_x': ['GAUSSIAN', 1.715, 0.05],
            'satellite_1_y': ['GAUSSIAN', 3.650, 0.05]
        }
    elif lens_ID == 'HE0435':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.37, 0.05],
            'satellite_1_x': ['GAUSSIAN', -0.0830, 0.05],
            'satellite_1_y': ['GAUSSIAN', 3.8549, 0.05]
        }
    elif lens_ID == 'J0607':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.1, 0.1],
            'satellite_1_x': ['GAUSSIAN', 1.22, 0.05],
            'satellite_1_y': ['GAUSSIAN', 0.24, 0.05]
        }
    elif lens_ID == 'J0659':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.25, 0.2],
            'satellite_1_x': ['GAUSSIAN', 0.35, 0.05],
            'satellite_1_y': ['GAUSSIAN', 1.55, 0.05]
        }
    elif lens_ID == 'J1042':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.1, 0.1],
            'satellite_1_x': ['GAUSSIAN', 1.782, 0.05],
            'satellite_1_y': ['GAUSSIAN', -0.317, 0.05]
        }
    elif lens_ID == 'MG0414':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.1, 0.1],
            'satellite_1_x': ['GAUSSIAN', -0.61, 0.05],
            'satellite_1_y': ['GAUSSIAN', 1.325, 0.05]
        }
    elif lens_ID == 'PSJ1606':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.15, 0.1],
            'satellite_1_x': ['GAUSSIAN', -0.28, 0.05],
            'satellite_1_y': ['GAUSSIAN', -1.24, 0.05]
        }
    elif lens_ID == 'RXJ0911':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.25, 0.15],
            'satellite_1_x': ['GAUSSIAN', -0.767, 0.05],
            'satellite_1_y': ['GAUSSIAN', 0.657, 0.05]
        }
    elif lens_ID == 'RXJ1131':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.3, 0.2],
            'satellite_1_x': ['GAUSSIAN', -0.328, 0.05],
            'satellite_1_y': ['GAUSSIAN', 0.700, 0.05]
        }
    elif lens_ID == 'WFI2033':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.05, 0.1],
            'satellite_1_x': ['GAUSSIAN', 0.273217, 0.05],
            'satellite_1_y': ['GAUSSIAN', 2.00444, 0.05],
            'satellite_2_theta_E': ['GAUSSIAN', 0.9, 0.1],
                'satellite_2_x': ['GAUSSIAN', -3.52, 0.1],
                'satellite_2_y': ['GAUSSIAN', 0.033, 0.1]
        }
    elif lens_ID == 'M1134':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.4, 0.2],
            'satellite_1_x': ['GAUSSIAN', 3.127565, 0.05],
            'satellite_1_y': ['GAUSSIAN', -3.93903, 0.05]
        }
    elif lens_ID == 'J0248':
        satellite_prior = {
            'satellite_1_theta_E': ['GAUSSIAN', 0.08, 0.04],
            'satellite_1_x': ['GAUSSIAN', 0.99, 0.05],
            'satellite_1_y': ['GAUSSIAN', -1.46, 0.05]
        }
    else:
        satellite_prior = {}
    return satellite_prior

def numerics_setup(lens_ID):
    """
    Return the recommended factors by which to rescale ray tracing grids for magnification
    calculations
    :param lens_ID: a string that identifies a lens
    :return:
    """
    if lens_ID == 'B1422':
        rescale_grid_size = 1.4
        rescale_grid_res = 2.
    elif lens_ID == 'WFI2026':
        rescale_grid_size = 1.2
        rescale_grid_res = 2.
    elif lens_ID == 'B2045':
        raise Exception('not yet implemented')
    elif lens_ID == 'HE0435':
        rescale_grid_size = 1.
        rescale_grid_res = 2.
    elif lens_ID == 'J0248':
        rescale_grid_size = 2.5
        rescale_grid_res = 2.
    elif lens_ID == 'J0248_HST':
        rescale_grid_size = 1.0
        rescale_grid_res = 2.
    elif lens_ID in ['J0259', 'J0259_HST_475X']:
        rescale_grid_size = 1.4
        rescale_grid_res = 2.
    elif lens_ID == 'J0607':
        rescale_grid_res = 2.
        rescale_grid_size = 5.0
    elif lens_ID == 'J0608':
        rescale_grid_size = 2.0
        rescale_grid_res = 2.
    elif lens_ID == 'J0659':
        rescale_grid_size = 2.0
        rescale_grid_res = 2.
    elif lens_ID == 'J0803':
        rescale_grid_size = 2.5
        rescale_grid_res = 2.
    elif lens_ID == 'J0924':
        rescale_grid_size = 3.0
        rescale_grid_res = 2.
    elif lens_ID in ['J1042', 'J1042_814W']:
        rescale_grid_size = 5.0
        rescale_grid_res = 2.
    elif lens_ID == 'J1131':
        rescale_grid_res = 2.
        rescale_grid_size = 2.5
    elif lens_ID == 'J1251':
        rescale_grid_res = 2.
        rescale_grid_size = 2.0
    elif lens_ID == 'J1537':
        rescale_grid_res = 2.
        rescale_grid_size = 1.0
    elif lens_ID == 'J2026':
        rescale_grid_res = 2.
        rescale_grid_size = 1.2
    elif lens_ID == 'J2205_MIRI':
        rescale_grid_res = 2.
        rescale_grid_size = 1.5
    elif lens_ID == 'J2205':
        rescale_grid_res = 2.
        rescale_grid_size = 1.5
    elif lens_ID == 'J2344':
        rescale_grid_res = 2.
        rescale_grid_size = 4.0
    elif lens_ID in ['PG1115', 'PG1115_NIRCAM']:
        rescale_grid_res = 2.
        rescale_grid_size = 2.5
    elif lens_ID == 'PSJ0147':
        rescale_grid_res = 2.0
        rescale_grid_size = 2.5
    elif lens_ID == 'PSJ1606':
        rescale_grid_res = 2.
        rescale_grid_size = 1.0
    elif lens_ID == 'RXJ0911':
        rescale_grid_res = 2.
        rescale_grid_size = 1.0
    elif lens_ID == 'RXJ1131':
        rescale_grid_res = 2.
        rescale_grid_size = 2.0
    elif lens_ID == 'WFI2033':
        rescale_grid_res = 2.
        rescale_grid_size = 1.0
    elif lens_ID == 'WGD2038':
        rescale_grid_res = 2.
        rescale_grid_size = 1.
    elif lens_ID == 'J0405':
        rescale_grid_res = 2.
        rescale_grid_size = 2.0
    elif lens_ID == 'MG0414':
        rescale_grid_res = 2.0
        rescale_grid_size = 2.0
    elif lens_ID in ['M1134', 'M1134_MIRI']:
        rescale_grid_res = 2.
        rescale_grid_size = 1.0
    elif lens_ID == 'H1413':
        rescale_grid_res = 2.
        rescale_grid_size = 1.0
    elif lens_ID == 'J2017':
        rescale_grid_res = 2.
        rescale_grid_size = 2.0
    elif lens_ID == 'J2145':
        rescale_grid_res = 2.
        rescale_grid_size = 2.
    elif lens_ID == 'H1113':
        rescale_grid_res = 2.
        rescale_grid_size = 1.
    else:
        raise Exception('lens ID '+str(lens_ID)+' not recognized!')
    return rescale_grid_size, rescale_grid_res

def quick_setup(lens_ID):

    if lens_ID == 'B1422':
        from samana.Data.b1422 import B1422_HST as data_class
        from samana.Model.b1422_model import B1422ModelEPLM3M4Shear as model_class
    elif lens_ID == 'WFI2026':
        from samana.Data.j2026 import J2026 as data_class
        from samana.Model.j2026_model import J2026ModelEPLM3M4Shear as model_class
    elif lens_ID == 'B2045':
        from samana.Data.b2045 import B2045_MIRI as data_class
        from samana.Model.b2045_model import B2045ModelEPLM3M4Shear as model_class
    elif lens_ID == 'HE0435':
        from samana.Data.he0435 import HE0435_NIRCAM as data_class
        from samana.Model.he0435_model_nircam import HE0435ModelNircamEPLM1M3M4Shear as model_class
    elif lens_ID == 'J0248':
        from samana.Data.j0248 import J0248_MIRI as data_class
        from samana.Model.j0248_model import J0248ModelEPLM3M4ShearSatellite as model_class
    elif lens_ID == 'J0248_HST':
        from samana.Data.j0248 import J0248_HST as data_class
        from samana.Model.j0248_model import J0248ModelEPLM3M4ShearSatellite as model_class
    elif lens_ID == 'J0259':
        from samana.Data.j0259 import J0259_HST_F814W as data_class
        from samana.Model.j0259_model import J0259ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J0607':
        from samana.Data.j0607 import J0607_MIRI as data_class
        from samana.Model.j0607_model import J0607ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J0608':
        from samana.Data.j0608 import J0608_MIRI as data_class
        from samana.Model.j0608_model import J0608ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J0659':
        from samana.Data.j0659 import J0659_MIRI as data_class
        from samana.Model.j0659_model import J0659ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J1042':
        from samana.Data.j1042 import J1042_HST_160W as data_class
        from samana.Model.j1042_model import J1042ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J1042_814W':
        from samana.Data.j1042 import J1042_HST_814W as data_class
        from samana.Model.j1042_model import J1042ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J1131':
        from samana.Data.j1131 import J1131_HST as data_class
        from samana.Model.j1131_model import J1131ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J1251':
        from samana.Data.j1251 import J1251_HST as data_class
        from samana.Model.j1251_model import J1251ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J1537':
        from samana.Data.j1537 import J1537_HST as data_class
        from samana.Model.j1537_model import J1537ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J2205_MIRI':
        from samana.Data.j2205 import J2205_MIRI as data_class
        from samana.Model.j2205_model import J2205ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J2205':
        from samana.Data.j2205 import J2205_HST as data_class
        from samana.Model.j2205_model import J2205ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J2344':
        from samana.Data.j2344 import J2344_MIRI as data_class
        from samana.Model.j2344_model import J2344ModelEPLM3M4Shear as model_class
    elif lens_ID == 'MG0414':
        from samana.Data.mg0414 import MG014_MIRI as data_class
        from samana.Model.mg0414_model import MG0414ModelEPLM3M4Shear as model_class
    elif lens_ID == 'PG1115':
        from samana.Data.pg1115 import PG1115_HST as data_class
        from samana.Model.pg1115_model import PG1115ModelEPLM1M3M4Shear as model_class
    elif lens_ID == 'PG1115_NIRCAM':
        from samana.Data.pg1115 import PG1115_NIRCAM as data_class
        from samana.Model.pg1115_model import PG1115ModelEPLM1M3M4Shear as model_class
    elif lens_ID == 'PSJ0147':
        from samana.Data.j0147 import J0147_MIRI as data_class
        from samana.Model.j0147_model import J0147ModelEPLM3M4Shear as model_class
    elif lens_ID == 'PSJ1606':
        from samana.Data.psj1606 import PSJ1606_HST as data_class
        from samana.Model.psj1606_model import PSJ1606ModelEPLM3M4Shear as model_class
    elif lens_ID == 'RXJ0911':
        from samana.Data.rxj0911 import RXJ0911_HST as data_class
        from samana.Model.rxj0911_model import RXJ0911ModelEPLM3M4Shear as model_class
    elif lens_ID == 'RXJ1131':
        from samana.Data.rxj1131 import RXJ1131_HST as data_class
        from samana.Model.rxj1131_model import RXJ1131ModelEPLM3M4Shear as model_class
    elif lens_ID == 'WFI2033':
        from samana.Data.wfi2033 import WFI2033_NIRCAM as data_class
        from samana.Model.wfi2033_model_nircam import WFI2033NircamModelEPLM3M4Shear as model_class
    elif lens_ID == 'WFI2033_FIXEDQ':
        from samana.Data.wfi2033 import WFI2033_NIRCAM as data_class
        from samana.Model.wfi2033_model_nircam import WFI2033NircamModelEPLM3M4ShearFixedQ as model_class
    elif lens_ID == 'WGD2038':
        from samana.Data.wgd2038 import WGD2038_HST as data_class
        from samana.Model.wgd2038_model import WGD2038ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J0405':
        from samana.Data.j0405 import J0405_HST as data_class
        from samana.Model.j0405_model import J0405ModelEPLM3M4Shear as model_class
    elif lens_ID == 'MG0414':
        from samana.Data.mg0414 import MG014_MIRI as data_class
        from samana.Model.mg0414_model import MG0414ModelEPLM3M4Shear as model_class
    elif lens_ID == 'M1134':
        from samana.Data.m1134 import M1134_HST as data_class
        from samana.Model.m1134_model import M1134ModelEPLM3M4ShearSatellite as model_class
    elif lens_ID == 'M1134_MIRI':
        from samana.Data.m1134 import M1134_MIRI as data_class
        from samana.Model.m1134_model import M1134ModelEPLM3M4ShearSatellite as model_class
    elif lens_ID == 'J0924':
        from samana.Data.j0924 import J0924_MIRI as data_class
        from samana.Model.j0924_model import J0924ModelEPLM1M3M4Shear as model_class
    elif lens_ID == 'H1413':
        from samana.Data.h1413 import H1413_MIRI as data_class
        from samana.Model.h1413_model import H1413ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J2017':
        from samana.Data.j2017 import J2017_MIRI as data_class
        from samana.Model.j2017_model import J2017ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J0803':
        from samana.Data.j0803 import J0803_MIRI as data_class
        from samana.Model.j0803_model import J0803ModelEPLM3M4Shear as model_class
    elif lens_ID == 'J2145':
        from samana.Data.j2145 import J2145_MIRI as data_class
        from samana.Model.j2145_model import J2145ModelEPLM3M4Shear as model_class
    elif lens_ID == 'H1113':
        from samana.Data.he1113 import HE1113_MIRI as data_class
        from samana.Model.he1113_model import HE1113ModelEPLM3M4Shear as model_class
    else:
        raise Exception('lens ID '+str(lens_ID)+' not recognized!')
    return data_class, model_class

def nmax_bic_minimize(data_class, model_class, fitting_kwargs_list, n_max_list,
                      verbose=True, make_plots=False, shapelets_scale_factor=1):
    """

    :param data:
    :param model:
    :param fitting_kwargs_list:
    :param n_max_list:
    :param verbose:
    :return:
    """
    bic_list = []

    for idx, n_max in enumerate(n_max_list):
        if n_max == 0:
            model = model_class(data_class, shapelets_order=None)
        else:
            model = model_class(data_class, n_max, shapelets_scale_factor)

        kwargs_params = model.kwargs_params()
        kwargs_model, lens_model_init, kwargs_lens_init, index_lens_split = model.setup_kwargs_model()
        kwargs_constraints = model.kwargs_constraints
        kwargs_likelihood = model.kwargs_likelihood
        fitting_sequence = FittingSequence(data_class.kwargs_data_joint,
                                           kwargs_model,
                                           kwargs_constraints,
                                           kwargs_likelihood,
                                           kwargs_params,
                                           mpi=False,
                                           verbose=verbose)
        chain_list = fitting_sequence.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_sequence.best_fit()

        kwargs_likelihood_bic = deepcopy(kwargs_likelihood)
        kwargs_likelihood_bic['image_likelihood_mask_list'] = [data_class.likelihood_mask_imaging_weights]
        fitting_sequence_bic = FittingSequence(data_class.kwargs_data_joint,
                                           kwargs_model,
                                           kwargs_constraints,
                                           kwargs_likelihood_bic,
                                           kwargs_params,
                                           mpi=False,
                                           verbose=verbose)

        num_data = fitting_sequence_bic.likelihoodModule.num_data
        num_param_nonlinear = fitting_sequence_bic.param_class.num_param()[0]
        num_param_linear = fitting_sequence_bic.param_class.num_param_linear()
        num_param = num_param_nonlinear + num_param_linear
        param_class = fitting_sequence.param_class
        logL = fitting_sequence_bic.likelihoodModule.logL(
            param_class.kwargs2args(**kwargs_result), verbose=verbose
        )
        bic = -2 * logL + (np.log(num_data) * num_param)
        bic_list.append(bic)

        if make_plots:

            multi_band_list = data_class.kwargs_data_joint['multi_band_list']
            modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02,
                                  cmap_string="gist_heat",
                                  fast_caustic=True,
                                  image_likelihood_mask_list=[data_class.likelihood_mask])
            for i in range(len(chain_list)):
                chain_plot.plot_chain_list(chain_list, i)
            f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
            modelPlot.data_plot(ax=axes[0, 0])
            modelPlot.model_plot(ax=axes[0, 1])
            modelPlot.normalized_residual_plot(ax=axes[0, 2], v_min=-6, v_max=6)
            modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100)
            modelPlot.convergence_plot(ax=axes[1, 1], v_max=1)
            modelPlot.magnification_plot(ax=axes[1, 2])
            f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
            modelPlot.decomposition_plot(ax=axes[0, 0], text='Lens light', lens_light_add=True, unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 0], text='Lens light convolved', lens_light_add=True)
            modelPlot.decomposition_plot(ax=axes[0, 1], text='Source light', source_add=True, unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 1], text='Source light convolved', source_add=True)
            modelPlot.decomposition_plot(ax=axes[0, 2], text='All components', source_add=True, lens_light_add=True,
                                         unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1, 2], text='All components convolved', source_add=True,
                                         lens_light_add=True, point_source_add=True)
            f.tight_layout()
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
            plt.show()

            fig = plt.figure()
            fig.set_size_inches(6, 6)
            ax = plt.subplot(111)
            kwargs_plot = {'ax': ax,
                           'index_macromodel': [0, 1],
                           'with_critical_curves': True,
                           'v_min': -0.2, 'v_max': 0.2}
            modelPlot.substructure_plot(band_index=0, **kwargs_plot)
            print(kwargs_result)
            print(kwargs_result['kwargs_lens'])
            #a = input('continue')
        print(kwargs_result['kwargs_lens'])
        print('bic: ', bic)
        print('bic list: ', bic_list)

    return bic_list

def cut_on_data(output, data,
                ABC_flux_ratio_likelihood=True,
                flux_uncertainty_percentage=None,
                flux_ratio_uncertainty_percentage=None,
                uncertainty_in_flux_ratios=True,
                imaging_data_likelihood=True,
                imaging_data_hard_cut=False,
                percentile_cut_image_data=None,
                n_keep_S_statistic=None,
                S_statistic_tolerance=None,
                perturb_measurements=True,
                perturb_model=True,
                imaging_data_likelihood_scale=20,
                cut_image_data_first=True,
                verbose=False):
    """

    :param output:
    :param data:
    :param ABC_flux_ratio_likelihood:
    :param flux_uncertainty_percentage:
    :param flux_ratio_uncertainty_percentage:
    :param imaging_data_likelihood:
    :param imaging_data_hard_cut:
    :param percentile_cut_image_data:
    :param n_keep_S_statistic:
    :param S_statistic_tolerance:
    :param perturb_measurements:
    :return:
    """
    data_class = deepcopy(data)
    __out = deepcopy(output)

    if imaging_data_hard_cut is False:
        percentile_cut_image_data = 100.0 # keep everything
    else:
        assert percentile_cut_image_data is not None

    if uncertainty_in_flux_ratios:
        mags_measured = data_class.magnifications
        _flux_ratios_measured = mags_measured[1:] / mags_measured[0]
        if flux_ratio_uncertainty_percentage is None:
            flux_ratios_measured = data_class.magnifications[1:] / data_class.magnifications[0]
        elif perturb_measurements:
            delta_f = np.array(flux_ratio_uncertainty_percentage) * np.array(_flux_ratios_measured)
            flux_ratios_measured = [np.random.normal(_flux_ratios_measured[i], delta_f[i]) for i in range(0, 3)]
        else:
            flux_ratios_measured = _flux_ratios_measured
        if ABC_flux_ratio_likelihood:
            if perturb_model:
                if flux_ratio_uncertainty_percentage is None:
                    model_flux_ratios = __out.flux_ratios
                else:
                    model_flux_ratios = perturbed_flux_ratios_from_flux_ratios(__out.flux_ratios,
                                                                           flux_ratio_uncertainty_percentage)
            else:
                model_flux_ratios = __out.flux_ratios
            __out.set_flux_ratio_summary_statistic(None,
                                                   None,
                                                   measured_flux_ratios=flux_ratios_measured,
                                                   modeled_flux_ratios=model_flux_ratios,
                                                   verbose=verbose)

        else:
            model_flux_ratios = __out.flux_ratios
            __out.set_flux_ratio_likelihood(None,
                                            None,
                                            flux_ratio_uncertainty_percentage,
                                            measured_flux_ratios=flux_ratios_measured,
                                            modeled_flux_ratios=model_flux_ratios,
                                            verbose=verbose)

    else:
        if flux_uncertainty_percentage is None:
            model_image_magnifications = __out.image_magnifications
        else:
            if perturb_measurements:
                data_class.perturb_flux_measurements(flux_uncertainty_percentage)
            model_image_magnifications = perturbed_fluxes_from_fluxes(__out.image_magnifications,
                                                                      flux_uncertainty_percentage)

        observed_image_magnifications = data_class.magnifications
        if ABC_flux_ratio_likelihood:
            __out.set_flux_ratio_summary_statistic(observed_image_magnifications,
                                                 model_image_magnifications)
        else:
            __out.set_flux_ratio_likelihood(observed_image_magnifications,
                                          model_image_magnifications,
                                          flux_ratio_uncertainty_percentage)

    if cut_image_data_first:
        _out = __out.cut_on_image_data(percentile_cut=percentile_cut_image_data)
        if ABC_flux_ratio_likelihood:
            # now cut on flux ratios
            if S_statistic_tolerance is not None:
                assert n_keep_S_statistic is None
                n_keep_S_statistic = np.sum(_out.flux_ratio_summary_statistic < S_statistic_tolerance)
            weights_flux_ratios = 1.0
            out_cut_S = _out.cut_on_S_statistic(keep_best_N=n_keep_S_statistic)
        else:
            n_keep_S_statistic = -1
            out_cut_S = _out.cut_on_S_statistic(keep_best_N=n_keep_S_statistic)
            weights_flux_ratios = out_cut_S.flux_ratio_likelihood

        if imaging_data_likelihood:
            assert imaging_data_hard_cut is False
            relative_log_likelihoods = out_cut_S.image_data_logL - np.max(out_cut_S.image_data_logL)
            rescale_log_like = 1.0
            weights_imaging_data = np.exp(relative_log_likelihoods / rescale_log_like)
            effective_sample_size = np.sum(weights_imaging_data)
            target_sample_size = len(weights_imaging_data) / imaging_data_likelihood_scale
            while effective_sample_size < target_sample_size:
                rescale_log_like += 1
                weights_imaging_data = np.exp(relative_log_likelihoods / rescale_log_like)
                effective_sample_size = np.sum(weights_imaging_data)
                target_sample_size = len(weights_imaging_data) / imaging_data_likelihood_scale
            #print('rescaled relative log-likelihoods by '+str(rescale_log_like))
        else:
            weights_imaging_data = np.ones(out_cut_S.parameters.shape[0])
    else:
        if ABC_flux_ratio_likelihood:
            # now cut on flux ratios
            if S_statistic_tolerance is not None:
                assert n_keep_S_statistic is None
                n_keep_S_statistic = np.sum(__out.flux_ratio_summary_statistic < S_statistic_tolerance)
            weights_flux_ratios = 1.0
            _out = __out.cut_on_S_statistic(keep_best_N=n_keep_S_statistic)
        else:
            n_keep_S_statistic = -1
            _out = __out.cut_on_S_statistic(keep_best_N=n_keep_S_statistic)
            weights_flux_ratios = _out.flux_ratio_likelihood

        if imaging_data_likelihood:
            assert imaging_data_hard_cut is False
            relative_log_likelihoods = _out.image_data_logL - np.max(_out.image_data_logL)
            rescale_log_like = 1.0
            weights_imaging_data = np.exp(relative_log_likelihoods / rescale_log_like)
            effective_sample_size = np.sum(weights_imaging_data)
            target_sample_size = len(weights_imaging_data) / imaging_data_likelihood_scale
            while effective_sample_size < target_sample_size:
                rescale_log_like += 1
                weights_imaging_data = np.exp(relative_log_likelihoods / rescale_log_like)
                effective_sample_size = np.sum(weights_imaging_data)
                target_sample_size = len(weights_imaging_data) / imaging_data_likelihood_scale
        else:
            weights_imaging_data = np.ones(_out.parameters.shape[0])
        out_cut_S = _out.cut_on_image_data(percentile_cut=percentile_cut_image_data)

    return out_cut_S, weights_imaging_data * weights_flux_ratios

def simulation_output_to_density(output, data, param_names_plot, kwargs_cut_on_data, kwargs_density,
                                 param_names_macro_plot=None, n_resample=0, custom_weights=None, apply_cuts=True):

    if apply_cuts:
        out, weights = cut_on_data(output, data, **kwargs_cut_on_data)
        if custom_weights is not None:
            for single_weights in custom_weights:
                (param, mean, sigma) = single_weights
                weights *= np.exp(-0.5 * (out.param_dict[param] - mean) **2 / sigma**2)
        for i in range(0, n_resample):
            _out, _weights = cut_on_data(output, data, **kwargs_cut_on_data)
            out = Output.join(out, _out)
            if custom_weights is not None:
                for single_weights in custom_weights:
                    (param, mean, sigma) = single_weights
                    _weights *= np.exp(-0.5 * (_out.param_dict[param] - mean) ** 2 / sigma ** 2)
            weights = np.append(weights, _weights)
        weights = [weights]
    else:
        out = output
        weights = None
    samples = None
    if len(param_names_plot) > 0:
        samples = out.parameter_array(param_names_plot)
    if param_names_macro_plot is not None:
        samples_macro = out.macromodel_parameter_array(param_names_macro_plot)
        if samples is None:
            samples = samples_macro
        else:
            samples = np.hstack((samples, samples_macro))
        param_names = param_names_plot + param_names_macro_plot
    else:
        param_names = param_names_plot
    from trikde.pdfs import DensitySamples
    density = DensitySamples(samples, param_names, weights, **kwargs_density)
    return density, out, weights

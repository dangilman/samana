import numpy as np
from copy import deepcopy

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
        rescale_grid_res = 2.0
        rescale_grid_size = 2.5
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

def compute_fluxratio_summarystat(f, measured_flux_ratios, measurement_uncertainties,
                                  uncertainty_on_ratios, keep_index_list):

    perturbed_flux_ratio = np.empty((f.shape[0], len(keep_index_list)))
    sigmas = []
    if measurement_uncertainties is not None:
        if uncertainty_on_ratios:
            for i in keep_index_list:
                if measurement_uncertainties[i] == -1:
                    perturbed_flux_ratio[:, i] = 0.0
                    sigmas.append(-1)
                else:
                    sigmas.append(1.0)
                    perturbed_flux_ratio[:, i] = np.random.normal(f[:, i],
                                                                  measurement_uncertainties[i])
        else:
            perturbed_fluxes = np.random.normal(f,
                                                measurement_uncertainties)
            perturbed_flux_ratio = perturbed_fluxes[:, 1:] / perturbed_fluxes[:, 0, np.newaxis]
            perturbed_flux_ratio = perturbed_flux_ratio[:, keep_index_list]
            sigmas = [1.0] * perturbed_flux_ratio.shape[1]
    else:
        if uncertainty_on_ratios:
            perturbed_flux_ratio = f
        else:
            perturbed_flux_ratio = f[:, 1:] / f[:, 0, np.newaxis]
        sigmas = [1.0] * np.sum(keep_index_list)
    stat = np.sqrt(-2 * compute_fluxratio_logL(perturbed_flux_ratio, measured_flux_ratios[keep_index_list], sigmas)[0])
    return stat

def compute_logfluxratio_summarystat(flux_ratios, measured_flux_ratios, measurement_uncertainties):

    perturbed_flux_ratio = np.empty_like(flux_ratios)
    for i in range(0, 3):
        if measurement_uncertainties[i] == 10 or measurement_uncertainties[i]==-1:
            perturbed_flux_ratio[:, i] = measured_flux_ratios[i]
        else:
            perturbed_flux_ratio[:, i] = np.random.normal(flux_ratios[:, i],
                                                          measurement_uncertainties[i])
    df = np.log(perturbed_flux_ratio) - np.log(measured_flux_ratios)
    stat = np.sqrt(np.sum(df**2, axis=1))
    return stat

def compute_fluxratio_logL(flux_ratios, measured_flux_ratios, measurement_uncertainties):
    fr_logL = 0
    S = 0
    for i in range(0, len(measured_flux_ratios)):
        if measurement_uncertainties[i] == -1:
            continue
        S += (flux_ratios[:, i] - measured_flux_ratios[i])**2
        fr_logL += -0.5 * (flux_ratios[:, i] - measured_flux_ratios[i]) ** 2 / measurement_uncertainties[i] ** 2
    return fr_logL, np.sqrt(S)

def calculate_flux_ratio_likelihood(params, flux_ratios, measured_flux_ratios,
                                    measurement_uncertainties):
    assert measurement_uncertainties is not None, ('when computing a flux ratio likelihood, must '
                                                   'specify measurement uncertainties on each flux ratio')
    params_out = deepcopy(params)
    flux_ratio_logL, S_statistic = compute_fluxratio_logL(flux_ratios, measured_flux_ratios, measurement_uncertainties)
    importance_weights = np.exp(flux_ratio_logL)
    normalized_weights = importance_weights / np.max(importance_weights)
    return params_out, normalized_weights, S_statistic

def downselect_fluxratio_summary_stats(params, flux_ratios, measured_flux_ratios,
                                       measurement_uncertainties, n_keep,
                                       uncertainty_on_ratios,
                                       keep_index_list,
                                       fluxes=None,
                                       S_statistic_tolerance=None):

    kept_index_list = []
    params_out = deepcopy(params)
    normalized_weights = np.zeros(flux_ratios.shape[0])
    if uncertainty_on_ratios:
        fluxratio_summary_statistic = compute_fluxratio_summarystat(flux_ratios,
                                                                    measured_flux_ratios,
                                                                    measurement_uncertainties,
                                                                    uncertainty_on_ratios,
                                                                    keep_index_list)
    else:
        fluxratio_summary_statistic = compute_fluxratio_summarystat(fluxes,
                                                                    measured_flux_ratios,
                                                                    measurement_uncertainties,
                                                                    uncertainty_on_ratios,
                                                                    keep_index_list)
    if S_statistic_tolerance is not None:
        best_inds = np.where(fluxratio_summary_statistic < S_statistic_tolerance)[0]
        print(str(len(best_inds)) +' samples with S-statistic below threshold: ', S_statistic_tolerance)
    else:
        best_inds = np.argsort(fluxratio_summary_statistic)[0:n_keep]
    normalized_weights[best_inds] = 1.0
    kept_index_list += list(best_inds)
    normalized_weights /= np.max(normalized_weights)
    return params_out, normalized_weights, fluxratio_summary_statistic[best_inds]

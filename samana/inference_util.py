import numpy as np
from trikde.pdfs import IndependentLikelihoods, DensitySamples
from scipy.stats import multivariate_normal
from samana.output_storage import Output
from copy import deepcopy

def compute_fluxratio_summarystat(f, measured_flux_ratios, measurement_uncertainties,
                                  uncertainty_on_ratios, keep_index_list):

    perturbed_flux_ratio = np.empty((f.shape[0], len(keep_index_list)))
    sigmas = []

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
    stat = np.sqrt(-2 * compute_fluxratio_logL(perturbed_flux_ratio, measured_flux_ratios[keep_index_list], sigmas, None)[0])
    return stat / max(measured_flux_ratios)

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

def compute_fluxratio_logL(flux_ratios, measured_flux_ratios, measurement_uncertainties, keep_index_list):

    fr_logL = 0
    S = 0
    if keep_index_list is None:
        keep_index_list = list(np.arange(0, flux_ratios.shape[1]))
    for i in keep_index_list:
        S += (flux_ratios[:, i] - measured_flux_ratios[i]) ** 2
        fr_logL += -0.5 * (flux_ratios[:, i] - measured_flux_ratios[i]) ** 2 / measurement_uncertainties[i] ** 2
    return fr_logL, np.sqrt(S) / max(measured_flux_ratios)

def compute_fluxratio_logL_cov(flux_ratios, measured_flux_ratios, measurement_uncertainties, keep_index_list):

    S = 0
    df = flux_ratios - measured_flux_ratios
    df = df[:, keep_index_list]
    inverse_cov = np.linalg.inv(measurement_uncertainties)
    fr_logL = np.sum(-0.5 * np.matmul(df, inverse_cov) ** 2, axis=-1)
    for ind in keep_index_list:
            S += (flux_ratios[:, ind] - measured_flux_ratios[ind])**2
    return fr_logL, np.sqrt(S) / max(measured_flux_ratios)

def calculate_flux_ratio_likelihood(params, flux_ratios, measured_flux_ratios,
                                    measurement_uncertainties, keep_index_list):

    params_out = deepcopy(params)
    if measurement_uncertainties.ndim == 1:
        flux_ratio_logL, S_statistic = compute_fluxratio_logL(flux_ratios, measured_flux_ratios, measurement_uncertainties, keep_index_list)
    if measurement_uncertainties.ndim == 2:
        flux_ratio_logL, S_statistic = compute_fluxratio_logL_cov(flux_ratios, measured_flux_ratios, measurement_uncertainties, keep_index_list)
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
    else:
        best_inds = np.argsort(fluxratio_summary_statistic)[0:n_keep]
    normalized_weights[best_inds] = 1.0
    kept_index_list += list(best_inds)
    normalized_weights /= np.max(normalized_weights)
    return params_out, normalized_weights, fluxratio_summary_statistic[best_inds]

def compute_likelihoods(output_class,
                        image_data_logL_sigma,
                        measured_flux_ratios,
                        measurement_uncertainties,
                        uncertainty_on_ratios,
                        param_names,
                        param_ranges_dict,
                        keep_index_list,
                        percentile_cut_image_data=None,
                        n_keep=None,
                        n_bootstraps=0,
                        dm_param_names=None,
                        S_statistic_tolerance=None,
                        kwargs_kde={},
                        kwargs_kde_image_data=None
                        ):
    """

    :param output_class:
    :param image_data_logL_sigma:
    :param measured_flux_ratios:
    :param measurement_uncertainties:
    :param uncertainty_on_ratios:
    :param param_names:
    :param param_ranges_dict:
    :param keep_index_list:
    :param percentile_cut_image_data:
    :param n_keep:
    :param n_bootstraps:
    :param dm_param_names:
    :param S_statistic_tolerance:
    :param kwargs_kde:
    :return:
    """
    if dm_param_names is None:
        dm_param_names = ['log10_sigma_sub', 'log_mc',
                          'LOS_normalization', 'shmf_log_slope', 'z_lens', 'log_m_host', 'source_size_pc',
                          'summary_statistic', 'f_low', 'f_high', 'x_core_halo',
                          'log10_sigma_eff_mlow', 'log10_sigma_eff_8_mh', 'log10_subhalo_time_s',
                          ]
    if n_keep is None and n_bootstraps > 0:
        raise ValueError('when using a flux ratio likelihood specified '
                         'through n_keep=None, n_bootstraps must be set to 0')
    if kwargs_kde_image_data is None:
        kwargs_kde_image_data = deepcopy(kwargs_kde)
    params_out = None
    joint_weights = None
    param_ranges_dm = [param_ranges_dict[param_name] for param_name in param_names]
    for bootstrap_index in range(0, n_bootstraps+1):

        sim = deepcopy(output_class)
        # first down-select on imaging data likelihood
        logL_image_data = output_class.param_dict['logL_image_data']
        if bootstrap_index == 0:
            params = np.empty((sim.parameters.shape[0], len(param_names)))
            for i, parameter_name in enumerate(param_names):
                if parameter_name in dm_param_names:
                    params[:, i] = np.squeeze(sim.parameter_array([parameter_name]))
                else:
                    params[:, i] = np.squeeze(sim.macromodel_parameter_array([parameter_name]))
            print('total samples: ', logL_image_data.shape[0])
            no_image_data = False
            if image_data_logL_sigma is not None:
                assert percentile_cut_image_data is None, ('image_data_logL_sigma and percentile_cut_image_data should not '
                                                           'both be specified')
                max_logL = np.max(logL_image_data)
                logL_normalized_diff = (logL_image_data - max_logL) / image_data_logL_sigma
                weights_image_data = np.exp(-0.5 * logL_normalized_diff ** 2)
                weights_image_data = weights_image_data / np.max(weights_image_data)
            elif percentile_cut_image_data is not None:
                weights_image_data = np.zeros_like(logL_image_data)
                inds_sorted = np.argsort(logL_image_data)
                idx_cut = int(len(logL_image_data) * (100 - percentile_cut_image_data) / 100)
                logL_cut = logL_image_data[inds_sorted][idx_cut]
                weights_image_data[np.where(logL_image_data > logL_cut)] = 1
            else:
                no_image_data = True
                weights_image_data = np.ones_like(logL_image_data)
            if image_data_logL_sigma is not None:
                print('effective sample size after imaging data likelihood: ',
                      np.sum(weights_image_data))
            # now we compute the imaging data likelihood only
            if no_image_data:
                pdf_imgdata = None
            else:
                pdf_imgdata = DensitySamples(params,
                                         param_names=param_names,
                                         weights=weights_image_data,
                                         param_ranges=param_ranges_dm,
                                         **kwargs_kde_image_data)

        # now compute the flux ratio likelihood
        flux_ratios = sim.flux_ratios
        if n_keep is None:
            if uncertainty_on_ratios is False:
                raise Exception('cannot use a flux ratio likelihood with uncertainties on fluxes')
            _params_out, flux_ratio_likelihood_weights, _S_statistic = calculate_flux_ratio_likelihood(params,
                                                                             flux_ratios,
                                                                             measured_flux_ratios,
                                                                             measurement_uncertainties,
                                                                             keep_index_list)
            print('effective sample size from flux ratio likelihood: ', np.sum(flux_ratio_likelihood_weights))

        else:
            if uncertainty_on_ratios is False:
                m = sim.image_magnifications
                norm = np.max(m, axis=1)
                fluxes = m / norm[:, np.newaxis]
            else:
                fluxes = None
            _params_out, flux_ratio_likelihood_weights, _S_statistic = downselect_fluxratio_summary_stats(params,
                                                                            flux_ratios,
                                                                            measured_flux_ratios,
                                                                            measurement_uncertainties,
                                                                            n_keep,
                                                                            uncertainty_on_ratios,
                                                                            keep_index_list,
                                                                            fluxes=fluxes,
                                                                            S_statistic_tolerance=S_statistic_tolerance)

        _joint_weights = flux_ratio_likelihood_weights * weights_image_data
        if params_out is None:
            params_out = _params_out
            joint_weights = _joint_weights
            S_statistic = _S_statistic
        else:
            params_out = np.vstack((params_out, _params_out))
            joint_weights = np.append(joint_weights, _joint_weights)
            S_statistic = np.append(S_statistic, _S_statistic)

    # now compute the final pdf
    normalized_joint_weights = joint_weights / np.max(joint_weights)
    pdf_imgdata_fr = DensitySamples(params_out,
                                    param_names=param_names,
                                    weights=normalized_joint_weights,
                                    param_ranges=param_ranges_dm,
                                    **kwargs_kde
                                    )
    imaging_data_fluxratio_likelihood = IndependentLikelihoods([pdf_imgdata_fr])
    if no_image_data:
        imaging_data_likelihood = None
    else:
        imaging_data_likelihood = IndependentLikelihoods([pdf_imgdata])
    if n_keep is not None:
        print('median/worst of S_statistic distribution: ', np.median(S_statistic), np.max(S_statistic))
    if image_data_logL_sigma is not None:
        print('effective sample size using imaging+flux ratio likelihood: ', np.sum(normalized_joint_weights)/(n_bootstraps+1))
    else:
        if n_keep is not None:
            print('effective sample size flux ratios: ',
              np.sum(normalized_joint_weights > 0))
        else:
            print('effective sample size flux ratios: ',
              np.sum(normalized_joint_weights))
    return imaging_data_likelihood, imaging_data_fluxratio_likelihood, (params_out, normalized_joint_weights)

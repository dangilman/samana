import numpy as np
from trikde.pdfs import IndependentLikelihoods, DensitySamples
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
    stat = np.sqrt(-2 * compute_fluxratio_logL(perturbed_flux_ratio, measured_flux_ratios[keep_index_list], sigmas))
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
    for i in range(0, len(measured_flux_ratios)):
        if measurement_uncertainties[i] == -1:
            continue
        fr_logL += -0.5 * (flux_ratios[:, i] - measured_flux_ratios[i]) ** 2 / measurement_uncertainties[i] ** 2
    return fr_logL

def calculate_flux_ratio_likelihood(params, flux_ratios, measured_flux_ratios,
                                    measurement_uncertainties):

    params_out = deepcopy(params)
    flux_ratio_logL = compute_fluxratio_logL(flux_ratios, measured_flux_ratios, measurement_uncertainties)
    ndof = 0
    for sigma_i in measurement_uncertainties:
        if sigma_i > 0:
            ndof += 1
    reduced_chi2 = -2 * flux_ratio_logL / ndof
    importance_weights = np.exp(flux_ratio_logL)
    normalized_weights = importance_weights / np.max(importance_weights)
    print('effective sample size: ', np.sum(normalized_weights))
    print('number of good fits (reduced chi^2 < 1): ', np.sum(reduced_chi2 < 1))
    return params_out, normalized_weights

def downselect_fluxratio_summary_stats(params, flux_ratios, measured_flux_ratios,
                                       measurement_uncertainties, n_keep,
                                       n_bootstraps, uncertainty_on_ratios,
                                       keep_index_list, fluxes=None):

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
    best_inds = np.argsort(fluxratio_summary_statistic)[0:n_keep]
    normalized_weights[best_inds] = 1.0
    kept_index_list += list(best_inds)
    for n in range(0, n_bootstraps):
        _normalized_weights = np.zeros(flux_ratios.shape[0])
        if uncertainty_on_ratios:
            _fluxratio_summary_statistic = compute_fluxratio_summarystat(flux_ratios,
                                                                         measured_flux_ratios,
                                                                         measurement_uncertainties,
                                                                         uncertainty_on_ratios,
                                                                         keep_index_list)
        else:
            _fluxratio_summary_statistic = compute_fluxratio_summarystat(fluxes,
                                                                         measured_flux_ratios,
                                                                         measurement_uncertainties,
                                                                         uncertainty_on_ratios,
                                                                         keep_index_list)
        _best_inds = np.argsort(_fluxratio_summary_statistic)[0:n_keep]
        _normalized_weights[_best_inds] = 1.0
        params_out = np.vstack((params_out, deepcopy(params)))
        normalized_weights = np.append(normalized_weights, _normalized_weights)
        kept_index_list += list(_best_inds)
    normalized_weights /= np.max(normalized_weights)
    flux_ratio_norm = np.max(np.median(flux_ratios, axis=0))
    unique_values, unique_indexes = np.unique(kept_index_list, return_index=True)
    print('after '+str(n_bootstraps)+' bootstraps the number of repeated '
                                     'indexes is: ' +str(len(kept_index_list)-len(unique_values))+
          ' out of '+str(len(kept_index_list))+' samples')
    print('median/worst summary statistic value: ',
          np.median(fluxratio_summary_statistic[best_inds])/flux_ratio_norm, fluxratio_summary_statistic[best_inds[-1]]/flux_ratio_norm)
    return params_out, normalized_weights

def compute_likelihoods(output_class,
                        image_data_logL_sigma,
                        measured_flux_ratios,
                        measurement_uncertainties,
                        param_names,
                        param_ranges_dict,
                        keep_index_list,
                        use_kde=False,
                        nbins=5,
                        n_keep=None,
                        n_bootstraps=0,
                        bandwidth_scale=0.75,
                        dm_param_names=None,
                        uncertainty_on_ratios=False):

    if n_keep is None and n_bootstraps > 0:
        raise ValueError('when using a flux ratio likelihood specified '
                         'through n_keep=None, n_bootstraps must be set to 0')

    param_ranges_dm = [param_ranges_dict[param_name] for param_name in param_names]
    # first down-select on imaging data likelihood
    logL_image_data = output_class.image_data_logL
    weights_image_data = np.array([])
    for index_bootstrap in range(0, n_bootstraps+1):

        if image_data_logL_sigma is not None:
            max_logL = np.max(logL_image_data)
            logL_normalized_diff = (logL_image_data - max_logL) / image_data_logL_sigma
            image_data_weight = np.exp(-0.5 * logL_normalized_diff**2)
        else:
            image_data_weight = np.ones_like(logL_image_data)
        weights_image_data = np.append(weights_image_data, image_data_weight)

    print('total samples: ', weights_image_data.shape[0])
    print('effective sample size after imaging data likelihood: ', np.sum(weights_image_data))
    sim = deepcopy(output_class)
    params = np.empty((sim.parameters.shape[0], len(param_names)))
    if dm_param_names is None:
        dm_param_names = ['log10_sigma_sub', 'log_mc',
                          'LOS_normalization', 'shmf_log_slope','z_lens','log_m_host','source_size_pc']
    for i, parameter_name in enumerate(param_names):
        if parameter_name in dm_param_names:
            params[:, i] = np.squeeze(sim.parameter_array([parameter_name]))
        else:
            params[: ,i] = np.squeeze(sim.macromodel_parameter_array([parameter_name]))
    # now we compute the imaging data likelihood only
    pdf_imgdata = DensitySamples(params,
                                 param_names=param_names,
                                 weights=image_data_weight,
                                 param_ranges=param_ranges_dm,
                                 use_kde=use_kde,
                                 nbins=nbins,
                                 bandwidth_scale=bandwidth_scale)

    # now compute the flux ratio likelihood
    flux_ratios = sim.flux_ratios
    if n_keep is None:
        if uncertainty_on_ratios is False:
            raise Exception('cannot use a flux ratio likelihood with uncertainties on fluxes')
        params_out, flux_ratio_likelihood_weights = calculate_flux_ratio_likelihood(params,
                                                                         flux_ratios,
                                                                         measured_flux_ratios,
                                                                         measurement_uncertainties)

    else:
        if uncertainty_on_ratios is False:
            m = sim.image_magnifications
            norm = np.max(m, axis=1)
            fluxes = m / norm[:, np.newaxis]
        else:
            fluxes = None
        params_out, flux_ratio_likelihood_weights = downselect_fluxratio_summary_stats(params,
                                                                        flux_ratios,
                                                                        measured_flux_ratios,
                                                                        measurement_uncertainties,
                                                                        n_keep,
                                                                        n_bootstraps,
                                                                        uncertainty_on_ratios,
                                                                         keep_index_list,
                                                                            fluxes=fluxes)
    joint_weights = flux_ratio_likelihood_weights * weights_image_data
    joint_weights = joint_weights / np.max(joint_weights)
    print('effective sample from only flux ratio likelihood: ', np.sum(flux_ratio_likelihood_weights))
    print('effective sample size after flux ratio likelihood: ', np.sum(joint_weights))
    pdf_imgdata_fr = DensitySamples(params_out,
                                    param_names=param_names,
                                    weights=flux_ratio_likelihood_weights * weights_image_data,
                                    param_ranges=param_ranges_dm,
                                    use_kde=use_kde,
                                    nbins=nbins,
                                    bandwidth_scale=bandwidth_scale)
    # now compute the final pdf
    imaging_data_likelihood = IndependentLikelihoods([pdf_imgdata])
    imaging_data_fluxratio_likelihood = IndependentLikelihoods([pdf_imgdata_fr])
    likelihood_joint = imaging_data_fluxratio_likelihood/imaging_data_likelihood

    return imaging_data_likelihood, imaging_data_fluxratio_likelihood, likelihood_joint, (params_out, joint_weights)

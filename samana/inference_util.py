import numpy as np
from trikde.pdfs import IndependentLikelihoods, DensitySamples
from samana.output_storage import Output
from copy import deepcopy

def compute_fluxratio_summarystat(flux_ratios, measured_flux_ratios, measurement_uncertainties):

    perturbed_flux_ratio = np.empty_like(flux_ratios)
    sigmas = []
    for i in range(0, 3):
        if measurement_uncertainties[i] == -1:
            perturbed_flux_ratio[:, i] = 0.0
            sigmas.append(-1)
        else:
            sigmas.append(1.0)
            perturbed_flux_ratio[:, i] = np.random.normal(flux_ratios[:, i],
                                                          measurement_uncertainties[i])
    stat = np.sqrt(-2 * compute_fluxratio_logL(perturbed_flux_ratio, measured_flux_ratios, sigmas))
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

def downselect_fluxratio_likelihood(params, flux_ratios, measured_flux_ratios,
                                    measurement_uncertainties, w_custom=1.0):

    params_out = deepcopy(params)
    flux_ratio_logL = compute_fluxratio_logL(flux_ratios, measured_flux_ratios, measurement_uncertainties)
    ndof = 0
    for sigma_i in measurement_uncertainties:
        if sigma_i > 0:
            ndof += 1
    reduced_chi2 = -2 * flux_ratio_logL / ndof
    importance_weights = np.exp(flux_ratio_logL) * w_custom
    normalized_weights = importance_weights / np.max(importance_weights)
    print('effective sample size: ', np.sum(normalized_weights))
    print('number of good fits (reduced chi^2 < 1): ', np.sum(reduced_chi2 < 1))
    return params_out, normalized_weights

def downselect_fluxratio_likelihood_summary(params, flux_ratios, measured_flux_ratios,
                                    measurement_uncertainties, n_keep, w_custom=1.0):

    params_out = deepcopy(params)
    flux_ratio_logL = compute_fluxratio_logL(flux_ratios, measured_flux_ratios, measurement_uncertainties)
    ndof = 0
    for sigma_i in measurement_uncertainties:
        if sigma_i > 0:
            ndof += 1
    reduced_chi2 = -2 * flux_ratio_logL / ndof
    best_inds = np.argsort(reduced_chi2)[0:n_keep]
    importance_weights = np.zeros_like(reduced_chi2)
    importance_weights[best_inds] = 1.0
    importance_weights *= w_custom
    normalized_weights = importance_weights / np.max(importance_weights)
    print('effective sample size: ', np.sum(normalized_weights))
    print('number of good fits (reduced chi^2 < 1): ', np.sum(reduced_chi2 < 1))
    return params_out, normalized_weights

def downselect_logfluxratio_likelihood_summary(params, flux_ratios, measured_flux_ratios,
                                    measurement_uncertainties, n_keep, w_custom=1.0):

    params_out = deepcopy(params)
    stat = np.log(flux_ratios) - np.log(measured_flux_ratios)
    stat = np.sqrt(np.sum(stat ** 2, axis=1))
    best_inds = np.argsort(stat)[0:n_keep]
    importance_weights = np.zeros_like(stat)
    importance_weights[best_inds] = 1.0
    importance_weights *= w_custom
    normalized_weights = importance_weights / np.max(importance_weights)
    return params_out, normalized_weights

def downselect_fluxratio_summary_stats(params, flux_ratios, measured_flux_ratios,
                                       measurement_uncertainties, n_keep,
                                       n_bootstraps=0, w_custom=1.0,
                                       use_log_statistic=True):

    kept_index_list = []
    params_out = deepcopy(params)
    normalized_weights = np.zeros(flux_ratios.shape[0])
    if use_log_statistic:
        fluxratio_summary_statistic = compute_logfluxratio_summarystat(flux_ratios, measured_flux_ratios,
                                                                    measurement_uncertainties)
    else:
        fluxratio_summary_statistic = compute_fluxratio_summarystat(flux_ratios, measured_flux_ratios,
                                                                measurement_uncertainties)
    best_inds = np.argsort(fluxratio_summary_statistic)[0:n_keep]
    normalized_weights[best_inds] = 1.0
    normalized_weights *= w_custom
    kept_index_list += list(best_inds)
    for n in range(0, n_bootstraps):
        _normalized_weights = np.zeros(flux_ratios.shape[0])
        if use_log_statistic:
            _fluxratio_summary_statistic = compute_logfluxratio_summarystat(flux_ratios, measured_flux_ratios,
                                                                     measurement_uncertainties)
        else:
            _fluxratio_summary_statistic = compute_fluxratio_summarystat(flux_ratios, measured_flux_ratios,
                                                                     measurement_uncertainties)
        _best_inds = np.argsort(_fluxratio_summary_statistic)[0:n_keep]
        _normalized_weights[_best_inds] = 1.0
        _normalized_weights *= w_custom
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
                        percentile_cut_image_data,
                        measured_flux_ratios,
                        measurement_uncertainties,
                        dm_param_names,
                        param_ranges_dm_dict,
                        use_kde=False,
                        nbins=5,
                        n_keep=None,
                        n_bootstraps=0,
                        macro_param_weights=None,
                        dm_param_weights=None,
                        bandwidth_scale=0.75,
                        flux_ratio_likelihood_summary=False,
                        log_flux_ratio_likelihood_summary=False,
                        use_log_statistic=True):

    param_ranges_dm = [param_ranges_dm_dict[param_name] for param_name in dm_param_names]
    # first down-select on imaging data likelihood
    if isinstance(output_class, list):
        parameters = None
        image_magnifications = None
        macromodel_samples = None
        fitting_kwargs_list = None
        param_names = None
        macromodel_sample_names = None
        for simulation in output_class:
            _sim = simulation.cut_on_image_data(percentile_cut_image_data)
            if parameters is None:
                parameters = _sim.parameters
                image_magnifications = _sim.image_magnifications
                macromodel_samples = _sim.macromodel_samples
                param_names = _sim._param_names
                macromodel_sample_names = _sim._macromodel_sample_names
            else:
                parameters = np.vstack((parameters, _sim.parameters))
                image_magnifications = np.vstack((image_magnifications, _sim.image_magnifications))
                macromodel_samples = np.vstack((macromodel_samples, _sim.macromodel_samples))
            sim = Output(parameters, image_magnifications, macromodel_samples,
                                    fitting_kwargs_list, param_names, macromodel_sample_names,
                                    )
    else:
        sim = output_class.cut_on_image_data(percentile_cut_image_data)
    # get the dark matter parameters of interest
    params = sim.parameter_array(dm_param_names)
    # now we compute the imaging data likelihood only
    pdf_imgdata = DensitySamples(params,
                                 param_names=dm_param_names,
                                 weights=None,
                                 param_ranges=param_ranges_dm,
                                 use_kde=use_kde,
                                 nbins=nbins,
                                 bandwidth_scale=bandwidth_scale)

    # now compute the flux ratio likelihood
    w_custom = 1.0
    if dm_param_weights is not None:
        for dm_weight in dm_param_weights:
            (param, mean, sigma) = dm_weight
            w_custom *= np.exp(-0.5 * (np.squeeze(sim.parameter_array([param])) - mean) ** 2 / sigma ** 2)
    if macro_param_weights is not None:
        for macro_weight in macro_param_weights:
            (param, mean, sigma) = macro_weight
            w_custom *= np.exp(-0.5 * (np.squeeze(sim.macromodel_parameter_array([param])) - mean) ** 2 / sigma ** 2)
    flux_ratios = sim.flux_ratios
    if n_keep is None:
        params_out, normalized_weights = downselect_fluxratio_likelihood(params,
                                                                         flux_ratios,
                                                                         measured_flux_ratios,
                                                                         measurement_uncertainties,
                                                                         w_custom)

    else:
        if log_flux_ratio_likelihood_summary is True and flux_ratio_likelihood_summary is True:
            raise Exception('log_flux_ratio_likelihood_summary and flux_ratio_likelihood_summary cannot be both True')

        if log_flux_ratio_likelihood_summary:

            params_out, normalized_weights = downselect_logfluxratio_likelihood_summary(params,
                                                                                     flux_ratios,
                                                                                     measured_flux_ratios,
                                                                                     measurement_uncertainties,
                                                                                     n_keep,
                                                                                     w_custom)
        elif flux_ratio_likelihood_summary:

            params_out, normalized_weights = downselect_fluxratio_likelihood_summary(params,
                                                                                     flux_ratios,
                                                                                     measured_flux_ratios,
                                                                                     measurement_uncertainties,
                                                                                     n_keep,
                                                                                     w_custom)

        else:
            print('using log-likelihood as flux ratio summary statistic')
            params_out, normalized_weights = downselect_fluxratio_summary_stats(params,
                                                                            flux_ratios,
                                                                            measured_flux_ratios,
                                                                            measurement_uncertainties,
                                                                            n_keep,
                                                                            n_bootstraps,
                                                                            w_custom,
                                                                            use_log_statistic=use_log_statistic)

    pdf_imgdata_fr = DensitySamples(params_out,
                                    param_names=dm_param_names,
                                    weights=normalized_weights,
                                    param_ranges=param_ranges_dm,
                                    use_kde=use_kde,
                                    nbins=nbins,
                                    bandwidth_scale=bandwidth_scale)
    # now compute the final pdf
    pdf = deepcopy(pdf_imgdata_fr)
    pdf.density /= pdf_imgdata.density
    imaging_data_likelihood = IndependentLikelihoods([pdf_imgdata])
    imaging_data_fluxratio_likelihood = IndependentLikelihoods([pdf_imgdata_fr])
    likelihood_joint = IndependentLikelihoods([pdf])
    return imaging_data_likelihood, imaging_data_fluxratio_likelihood, likelihood_joint


def compute_macromodel_likelihood(output_class,
                                  percentile_cut_image_data,
                                  measured_flux_ratios,
                                  measurement_uncertainties,
                                  param_names_macro,
                                  use_kde=False,
                                  nbins=5,
                                  n_keep=None,
                                  n_bootstraps=0,
                                  param_ranges_macro=None,
                                  bandwidth_scale=0.75):

    # first down-select on imaging data likelihood
    sim = output_class.cut_on_image_data(percentile_cut_image_data)
    # get the dark matter parameters of interest
    params = sim.macromodel_parameter_array(param_names_macro)
    # now we compute the imaging data likelihood only
    pdf_imgdata = DensitySamples(params,
                                 param_names=param_names_macro,
                                 weights=None,
                                 param_ranges=param_ranges_macro,
                                 use_kde=use_kde,
                                 nbins=nbins,
                                 bandwidth_scale=bandwidth_scale)
    param_ranges_macro = pdf_imgdata.param_ranges

    # now compute the flux ratio likelihood from the down-selected samples
    flux_ratios = sim.flux_ratios
    if n_keep is None:
        params_out, normalized_weights = downselect_fluxratio_likelihood(params,
                                                                         flux_ratios,
                                                                         measured_flux_ratios,
                                                                         measurement_uncertainties,
                                                                         w_custom=1.0)
    else:
        params_out, normalized_weights = downselect_fluxratio_summary_stats(params,
                                                                            flux_ratios,
                                                                            measured_flux_ratios,
                                                                            measurement_uncertainties,
                                                                            n_keep,
                                                                            n_bootstraps=n_bootstraps,
                                                                            w_custom=1.0)
    # and the imaging data + flux ratio likelihood
    pdf_imgdata_fr = DensitySamples(params_out,
                                    param_names=param_names_macro,
                                    weights=normalized_weights,
                                    param_ranges=param_ranges_macro,
                                    use_kde=use_kde,
                                    nbins=nbins,
                                    bandwidth_scale=bandwidth_scale)

    # now only using the flux ratios
    flux_ratios = output_class.flux_ratios
    params_full = output_class.macromodel_parameter_array(param_names_macro)
    if n_keep is None:
        params_out, normalized_weights = downselect_fluxratio_likelihood(params_full,
                                                                         flux_ratios,
                                                                         measured_flux_ratios,
                                                                         measurement_uncertainties,
                                                                         w_custom=1.0)
    else:
        params_out, normalized_weights = downselect_fluxratio_summary_stats(params_full,
                                                                            flux_ratios,
                                                                            measured_flux_ratios,
                                                                            measurement_uncertainties,
                                                                            n_keep,
                                                                            n_bootstraps=0,
                                                                            w_custom=1.0)
    pdf_fr = DensitySamples(params_out,
                            param_names=param_names_macro,
                            weights=normalized_weights,
                            param_ranges=param_ranges_macro,
                            use_kde=use_kde,
                            nbins=nbins,
                            bandwidth_scale=bandwidth_scale)
    # now compute the final pdf
    pdf = deepcopy(pdf_imgdata_fr)
    inds_nonzero = np.where(pdf.density > 0)
    pdf.density[inds_nonzero] /= pdf_imgdata.density[inds_nonzero]
    imaging_data_likelihood = IndependentLikelihoods([pdf_imgdata])
    imaging_data_fluxratio_likelihood = IndependentLikelihoods([pdf_imgdata_fr])
    fr_likelihood = IndependentLikelihoods([pdf_fr])
    likelihood_joint = IndependentLikelihoods([pdf])
    return imaging_data_likelihood, fr_likelihood, imaging_data_fluxratio_likelihood, likelihood_joint

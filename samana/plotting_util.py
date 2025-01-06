import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from samana.analysis_util import simulation_output_to_density
from trikde.pdfs import IndependentLikelihoods

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.major.width'] = 3.5
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.major.width'] = 3.5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20

def setup_macromodel_inference_plot(output, mock_data_class,
                                    param_names_plot_macro=None,
                                    param_names_plot=None,
                                    flux_ratio_uncertainty=0.001,
                                    imaging_data_hard_cut=True,
                                    imaging_data_likelihood=False,
                                    percentile_cut_image_data=10,
                                    n_keep=2000,
                                    S_statistic_tolerance=None,
                                    imaging_data_likelihood_scale=20,
                                    nbins=10,
                                    use_kde=False,
                                    param_ranges=None,
                                    n_resample=0):

    if flux_ratio_uncertainty <= 0.001:
        n_resample = 0
    if param_names_plot is None:
        param_names_plot = []
    if param_names_plot_macro is None:
        param_names_plot_macro = ['theta_E', 'q', 'gamma_ext', 'gamma', 'a4_a_cos', 'a3_a_cos']
    kwargs_pdf = {'ABC_flux_ratio_likelihood': True,
                  'flux_ratio_uncertainty_percentage': [flux_ratio_uncertainty] * 3,
                  'uncertainty_in_flux_ratios': True,
                  'imaging_data_likelihood': imaging_data_likelihood,
                  'imaging_data_hard_cut': imaging_data_hard_cut,
                  'percentile_cut_image_data': percentile_cut_image_data,
                  'n_keep_S_statistic': n_keep,
                  'S_statistic_tolerance': S_statistic_tolerance,
                  'perturb_measurements': True,
                  'imaging_data_likelihood_scale': imaging_data_likelihood_scale
                  }
    kwargs_density = {'nbins': nbins,
                      'use_kde': use_kde,
                      'param_ranges': param_ranges}
    density, _, _ = simulation_output_to_density(deepcopy(output),
                                                 deepcopy(mock_data_class),
                                                 param_names_plot,
                                                 kwargs_pdf,
                                                 kwargs_density,
                                                 param_names_plot_macro,
                                                 n_resample=n_resample)
    like = IndepdendentLikelihoods([density])
    param_ranges_density = density.param_ranges
    return like, param_ranges_density

def mock_lens_data_plot(image_sim, window_size, vminmax=1.5, cmap='gist_heat', label='', zd='', zs='',
                        save_fig=False, filename=None, x_image=None, y_image=None):
    fig = plt.figure(1)
    fig.set_size_inches(8, 8)
    ax = plt.subplot(111)
    extent = [-window_size / 2, window_size / 2, -window_size / 2, window_size / 2]
    image = deepcopy(image_sim)
    image[np.where(image < 10 ** -vminmax)] = 10 ** -vminmax
    im = ax.imshow(np.log10(image), origin='lower', alpha=0.75,
                   vmin=-vminmax, vmax=vminmax, cmap=cmap, extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    xlow = -window_size / 2 + 0.2
    xhigh = xlow + 1.0
    y = -window_size / 2 + 0.3
    ax.plot([xlow, xhigh], [y, y], color='w', lw=4)
    ax.annotate('1 arcsec', xy=(xlow + 0.165, y - 0.21), fontsize=18, color='w')
    ax.annotate(label + '\n' + r'$z_{\rm{d}} = $' + str(zd) + '\n' + r'$z_{\rm{s}} = $' + str(zs), xy=(0.035, 0.78),
                xycoords='axes fraction', fontsize=24, color='w')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.01)
    cbar.set_label(r'$\log_{10} \ \rm{flux}$', fontsize=25)
    image_labels = ['A', 'B', 'C', 'D']
    if x_image is not None:
        for i in range(0, 4):
            ax.annotate(image_labels[i], xy=(x_image[i]+0.08, y_image[i]+0.08), color='w', fontsize=22)
    plt.tight_layout()
    if save_fig:
        plt.savefig(filename)
    plt.show()

def mock_substructure_plot(x_image, y_image, window_size, n_pixels, lens_model, lens_model_macro, kwargs_lens,
                           kwargs_lens_macro, label='', save_fig=False, filename=None, include_cbar=True):

    _r = np.linspace(-window_size / 2, window_size / 2, n_pixels)
    _xx, _yy = np.meshgrid(_r, _r)
    shape0 = _xx.shape
    kappa_macro = lens_model_macro.kappa(_xx.ravel(), _yy.ravel(), kwargs_lens_macro).reshape(shape0)
    kappa = lens_model.kappa(_xx.ravel(), _yy.ravel(), kwargs_lens).reshape(shape0)
    residual_kappa = kappa - kappa_macro
    mean = np.mean(residual_kappa[np.where(np.hypot(_xx, _yy) > 0.2)])

    fig = plt.figure(1)
    fig.set_size_inches(8, 8)
    ax = plt.subplot(111)
    extent = [-window_size / 2, window_size / 2, -window_size / 2, window_size / 2]

    im = ax.imshow(residual_kappa - mean, origin='lower', vmin=-0.1, vmax=0.1, cmap='seismic', alpha=1.,
                   extent=extent)
    ax.scatter(x_image, y_image, color='k', marker='x', s=200, alpha=0.8)
    ax.scatter(x_image, y_image, color='k', marker='+', s=250, alpha=0.8)

    ext = LensModelExtensions(lens_model)
    ra_crit_list, dec_crit_list, _, _ = ext.critical_curve_caustics(kwargs_lens, compute_window=window_size,
                                                                    grid_scale=0.01)
    for (racrit, deccrit) in zip(ra_crit_list[0:1], dec_crit_list[0:1]):
        ax.plot(racrit, deccrit, color='g', linestyle='--', lw=4)

    ax.set_xticks([])
    ax.set_yticks([])
    xlow = -window_size / 2 + 0.2
    xhigh = xlow + 1.0
    y = -window_size / 2 + 0.3
    ax.plot([xlow, xhigh], [y, y], color='k', lw=4)
    ax.annotate('1 arcsec', xy=(xlow + 0.165, y - 0.21), fontsize=18, color='k')
    if label is not None:
        ax.annotate(label, xy=(0.035, 0.9),
                    xycoords='axes fraction', fontsize=24, color='k',
                    bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.9, ec="k", lw=2))
    if include_cbar:
        cbar = plt.colorbar(im, fraction=0.046, pad=0.01, ticks=[-0.1, -0.05, 0.0, 0.05, 0.1])
        cbar.set_label(r'$\kappa - \kappa_{\rm{macro}}$', fontsize=25, labelpad=-2.5)
    # image_labels = ['A', 'B', 'C', 'D']
    # for i in range(0, 4):
    #     ax.annotate(image_labels[i], xy=(x_image[i]+0.05, y_image[i]+0.05), color='k', fontsize=15)

    plt.tight_layout()
    if save_fig:
        plt.savefig(filename)
        plt.show()
    return ax

def macromodel_plot(index_run, nbins):
    import numpy as np
    from samana.plotting_util import setup_macromodel_inference_plot
    import pickle
    from trikde.triangleplot import TrianglePlot
    import matplotlib.pyplot as plt
    from trikde.pdfs import IndepdendentLikelihoods, DensitySamples, InterpolatedLikelihood, CustomPriorHyperCube
    import os
    from samana.output_storage import Output
    from copy import deepcopy
    from samana.Data.Mocks.true_macromodel_params import get_true_params
    from samana.Data.Mocks.mock_1_data import Mock1Data
    from samana.Data.Mocks.mock_2_data import Mock2Data
    from samana.Data.Mocks.mock_3_data import Mock3Data
    from samana.Data.Mocks.mock_4_data import Mock4Data
    from samana.Data.Mocks.mock_5_data import Mock5Data
    from samana.Data.Mocks.mock_6_data import Mock6Data
    from samana.Data.Mocks.mock_7_data import Mock7Data
    from samana.Data.Mocks.mock_8_data import Mock8Data
    from samana.Data.Mocks.mock_9_data import Mock9Data
    from samana.Data.Mocks.mock_10_data import Mock10Data
    from samana.Data.Mocks.mock_11_data import Mock11Data
    from samana.Data.Mocks.mock_12_data import Mock12Data
    from samana.Data.Mocks.mock_13_data import Mock13Data
    from samana.Data.Mocks.mock_14_data import Mock14Data
    from samana.Data.Mocks.mock_15_data import Mock15Data
    from samana.Data.Mocks.mock_16_data import Mock16Data
    from samana.Data.Mocks.mock_17_data import Mock17Data
    from samana.Data.Mocks.mock_18_data import Mock18Data
    from samana.Data.Mocks.mock_19_data import Mock19Data
    from samana.Data.Mocks.mock_20_data import Mock20Data
    from samana.Data.Mocks.mock_21_data import Mock21Data
    from samana.Data.Mocks.mock_22_data import Mock22Data
    from samana.Data.Mocks.mock_23_data import Mock23Data
    from samana.Data.Mocks.mock_24_data import Mock24Data
    from samana.Data.Mocks.mock_25_data import Mock25Data
    from samana.Model.Mocks.mock_1_model import Mock1Model
    from samana.Model.Mocks.mock_2_model import Mock2Model
    from samana.Model.Mocks.mock_3_model import Mock3Model
    from samana.Model.Mocks.mock_4_model import Mock4Model
    from samana.Model.Mocks.mock_5_model import Mock5Model
    from samana.Model.Mocks.mock_6_model import Mock6Model
    from samana.Model.Mocks.mock_7_model import Mock7Model
    from samana.Model.Mocks.mock_8_model import Mock8Model
    from samana.Model.Mocks.mock_9_model import Mock9Model
    from samana.Model.Mocks.mock_10_model import Mock10Model
    from samana.Model.Mocks.mock_11_model import Mock11Model
    from samana.Model.Mocks.mock_12_model import Mock12Model
    from samana.Model.Mocks.mock_13_model import Mock13Model
    from samana.Model.Mocks.mock_14_model import Mock14Model
    from samana.Model.Mocks.mock_15_model import Mock15Model
    from samana.Model.Mocks.mock_16_model import Mock16Model
    from samana.Model.Mocks.mock_17_model import Mock17Model
    from samana.Model.Mocks.mock_18_model import Mock18Model
    from samana.Model.Mocks.mock_19_model import Mock19Model
    from samana.Model.Mocks.mock_20_model import Mock20Model
    from samana.Model.Mocks.mock_21_model import Mock21Model
    from samana.Model.Mocks.mock_22_model import Mock22Model
    from samana.Model.Mocks.mock_23_model import Mock23Model
    from samana.Model.Mocks.mock_24_model import Mock24Model
    from samana.Model.Mocks.mock_25_model import Mock25Model
    from samana.image_magnification_util import perturbed_fluxes_from_fluxes
    from samana.analysis_util import cut_on_data, simulation_output_to_density, inference
    cosmos_source = True
    mock_lens_data_list = [Mock1Data(cosmos_source=cosmos_source),
                           Mock2Data(cosmos_source=cosmos_source),
                           Mock3Data(cosmos_source=cosmos_source),
                           Mock4Data(cosmos_source=cosmos_source),
                           Mock5Data(cosmos_source=cosmos_source),
                           Mock6Data(cosmos_source=cosmos_source),
                           Mock7Data(cosmos_source=cosmos_source),
                           Mock8Data(cosmos_source=cosmos_source),
                           Mock9Data(cosmos_source=cosmos_source),
                           Mock10Data(cosmos_source=cosmos_source),
                           Mock11Data(cosmos_source=cosmos_source),
                           Mock12Data(cosmos_source=cosmos_source),
                           Mock13Data(cosmos_source=cosmos_source),
                           Mock14Data(cosmos_source=cosmos_source),
                           Mock15Data(cosmos_source=cosmos_source),
                           Mock16Data(cosmos_source=cosmos_source),
                           Mock17Data(cosmos_source=cosmos_source),
                           Mock18Data(cosmos_source=cosmos_source),
                           Mock19Data(cosmos_source=cosmos_source),
                           Mock20Data(cosmos_source=cosmos_source),
                           Mock21Data(cosmos_source=cosmos_source),
                           Mock22Data(cosmos_source=cosmos_source),
                           Mock23Data(cosmos_source=cosmos_source),
                           Mock24Data(cosmos_source=cosmos_source),
                           Mock25Data(cosmos_source=cosmos_source)]
    mock_lens_model_list = [Mock1Model,
                            Mock2Model,
                            Mock3Model,
                            Mock4Model,
                            Mock5Model,
                            Mock6Model,
                            Mock7Model,
                            Mock8Model,
                            Mock9Model,
                            Mock10Model,
                            Mock11Model,
                            Mock12Model,
                            Mock13Model,
                            Mock14Model,
                            Mock15Model,
                            Mock16Model,
                            Mock17Model,
                            Mock18Model,
                            Mock19Model,
                            Mock20Model,
                            Mock21Model,
                            Mock22Model,
                            Mock23Model,
                            Mock24Model,
                            Mock25Model]
    param_names_plot_macro = ['theta_E',
                              'q',
                              'phi_q',
                              'gamma_ext',
                              'phi_gamma',
                              'gamma',
                              # 'a3_a_cos',
                              # 'a4_a_cos'
                              'a3_a',
                              'a4_a'
                              ]
    path_to_data = os.getenv('HOME') + '/Code/samana/turbocharging_figures/put_data_here/output_classes/'

    def get_samples(output, data, kwargs_cut_on_data, param_names,
                    n_bootstrap):

        out, weights = cut_on_data(output, data, **kwargs_cut_on_data)
        for i in range(0, n_bootstrap):
            _out, _weights = cut_on_data(output, data, **kwargs_cut_on_data)
            out = Output.join(out, _out)
            weights = np.append(weights, _weights)
        weights = [weights]
        samples = out.macromodel_parameter_array(param_names)
        return samples, out, weights

    def get_samples_normalize_image_data(output, data, kwargs_cut_on_data, param_names,
                                         n_bootstrap, kwargs_density):

        kwargs_density_DM = {'nbins': 20, 'use_kde': False, 'param_ranges': [[-2.5, -1.0], [4, 10]]}
        # first, we cut only on the image data
        kwargs_cut_image_data = deepcopy(kwargs_cut_on_data)
        kwargs_cut_image_data['n_keep_S_statistic'] = -1
        out_cut_image_data, weights_image_data = cut_on_data(output, data, **kwargs_cut_image_data)
        samples_image_data_cut = out_cut_image_data.parameter_array(['log10_sigma_sub', 'log_mc'])

        # now we want to compute a function that takes as input (log10_sigma_sub, log10_mhm) and returns a weight
        density_image_data = DensitySamples(samples_image_data_cut,
                                            param_names=['log10_sigma_sub', 'log_mc'],
                                            weight_list=None, **kwargs_density_DM)

        like_image_data = IndepdendentLikelihoods([density_image_data])
        image_data_likelihood_interp = InterpolatedLikelihood(like_image_data,
                                                              ['log10_sigma_sub', 'log_mc'],
                                                              [[-2.5, -1.0], [4, 10]])

        out, _ = cut_on_data(output, data, **kwargs_cut_on_data)
        for i in range(0, n_bootstrap):
            _out, _ = cut_on_data(output, data, **kwargs_cut_on_data)
            out = Output.join(out, _out)

        samples_DM = out.parameter_array(['log10_sigma_sub', 'log_mc'])
        samples = out.macromodel_parameter_array(param_names)
        prob = [image_data_likelihood_interp(tuple(samples_DM[i, :])) for i in range(0, samples_DM.shape[0])]
        weights = [1 / np.array(prob)]
        return samples, out, weights

    def compute_likelihoods(output,
                            output_no_image_data,
                            mock_data_class,
                            kwargs_density_input,
                            n_keep=1000,
                            flux_ratio_uncertainty=0.03,
                            percentile_cut_image_data=5,
                            param_ranges_macro=None,
                            nbins=8,
                            n_resample=10,
                            imaging_data_hard_cut=False,
                            imaging_data_likelihood=True,
                            imaging_data_likelihood_scale=20,
                            ):

        kwargs_inference = {'ABC_flux_ratio_likelihood': True,
                            'flux_ratio_uncertainty_percentage': [flux_ratio_uncertainty] * 3,
                            'uncertainty_in_flux_ratios': True,
                            'imaging_data_likelihood': False,
                            'imaging_data_hard_cut': True,
                            'imaging_data_likelihood_scale': None,
                            'percentile_cut_image_data': 100,
                            'n_keep_S_statistic': -1,
                            'perturb_measurements': True
                            }
        kwargs_density = deepcopy(kwargs_density_input)
        kwargs_density['param_ranges'] = param_ranges_macro
        samples_imgpos, _, _ = get_samples(output_no_image_data, mock_data_class, kwargs_inference,
                                           param_names_plot_macro, n_resample)
        pdf = DensitySamples(samples_imgpos, param_names_plot_macro, None, **kwargs_density)
        like_imgpos = IndepdendentLikelihoods([pdf])
        param_ranges = like_imgpos.param_ranges
        print(param_ranges)
        print('done.')
        kwargs_density['param_ranges'] = param_ranges

        kwargs_inference = {'ABC_flux_ratio_likelihood': True,
                            'flux_ratio_uncertainty_percentage': [flux_ratio_uncertainty] * 3,
                            'uncertainty_in_flux_ratios': True,
                            'imaging_data_likelihood': False,
                            'imaging_data_hard_cut': True,
                            'imaging_data_likelihood_scale': None,
                            'percentile_cut_image_data': 100,
                            'n_keep_S_statistic': n_keep,
                            'perturb_measurements': True
                            }
        samples_FR, _, _ = get_samples(output_no_image_data, mock_data_class, kwargs_inference, param_names_plot_macro,
                                       n_resample)
        pdf = DensitySamples(samples_FR, param_names_plot_macro, None, **kwargs_density)
        like_FR = IndepdendentLikelihoods([pdf])
        print('done.')

        kwargs_inference = {'ABC_flux_ratio_likelihood': True,
                            'flux_ratio_uncertainty_percentage': [flux_ratio_uncertainty] * 3,
                            'uncertainty_in_flux_ratios': True,
                            'imaging_data_likelihood': imaging_data_likelihood,
                            'imaging_data_hard_cut': imaging_data_hard_cut,
                            'imaging_data_likelihood_scale': imaging_data_likelihood_scale,
                            'percentile_cut_image_data': percentile_cut_image_data,
                            'n_keep_S_statistic': n_keep,
                            'perturb_measurements': True
                            }
        samples_IM_FR, _, weights = get_samples_normalize_image_data(output, mock_data_class,
                                                                     kwargs_inference, param_names_plot_macro,
                                                                     n_resample, kwargs_density)
        pdf = DensitySamples(samples_IM_FR, param_names_plot_macro, weights, **kwargs_density)
        like_IM_FR = IndepdendentLikelihoods([pdf])
        print('done.')

        kwargs_inference = {'ABC_flux_ratio_likelihood': True,
                            'flux_ratio_uncertainty_percentage': [flux_ratio_uncertainty] * 3,
                            'uncertainty_in_flux_ratios': True,
                            'imaging_data_likelihood': imaging_data_likelihood,
                            'imaging_data_hard_cut': imaging_data_hard_cut,
                            'imaging_data_likelihood_scale': imaging_data_likelihood_scale,
                            'percentile_cut_image_data': percentile_cut_image_data,
                            'n_keep_S_statistic': -1,
                            'perturb_measurements': True
                            }
        samples_IM, _, weights = get_samples_normalize_image_data(output, mock_data_class,
                                                                  kwargs_inference, param_names_plot_macro, n_resample,
                                                                  kwargs_density)
        pdf = DensitySamples(samples_IM, param_names_plot_macro, weights, **kwargs_density)
        like_IM = IndepdendentLikelihoods([pdf])
        print('done.')

        return like_imgpos, like_FR, like_IM_FR, like_IM

    def rename_axes(axes, ticks, param_names):

        ax_index, label = 8, r'$q$'
        axes[ax_index].set_ylabel(label)
        axes[ax_index].set_yticks(ticks[label])
        axes[ax_index].set_yticklabels(ticks[label])

        ax_index, label = 16, r'$\phi_{\rm{q}}$'
        axes[ax_index].set_ylabel(label)
        axes[ax_index].set_yticks(ticks[label])
        axes[ax_index].set_yticklabels(ticks[label])

        ax_index, label = 24, r'$\gamma_{\rm{ext}}$'
        axes[ax_index].set_ylabel(label)
        axes[ax_index].set_yticks(ticks[label])
        axes[ax_index].set_yticklabels(ticks[label])

        ax_index, label = 32, r'$\phi_{\gamma \rm{ext}}$'
        axes[ax_index].set_ylabel(label)
        axes[ax_index].set_yticks(ticks[label])
        axes[ax_index].set_yticklabels(ticks[label])

        ax_index, label = 40, r'$\gamma$'
        axes[ax_index].set_ylabel(label)
        axes[ax_index].set_yticks(ticks[label])
        axes[ax_index].set_yticklabels(ticks[label])

        ax_index, label = 48, r'$a_3$'
        axes[ax_index].set_ylabel(label)
        axes[ax_index].set_yticks(ticks[label])
        axes[ax_index].set_yticklabels(ticks[label])

        ax_index, label = 56, r'$a_4$'
        axes[ax_index].set_ylabel(label)
        axes[ax_index].set_yticks(ticks[label])
        axes[ax_index].set_yticklabels(ticks[label])

        ax_index, label = 56, r'$\theta_{\rm{E}}$'
        axes[ax_index].set_xlabel(label)
        axes[ax_index].set_xticks(ticks[label])
        axes[ax_index].set_xticklabels(ticks[label])

        ax_index, label = 57, r'$q$'
        axes[ax_index].set_xlabel(label)
        axes[ax_index].set_xticks(ticks[label])
        axes[ax_index].set_xticklabels(ticks[label])

        ax_index, label = 58, r'$\phi_{\rm{q}}$'
        axes[ax_index].set_xlabel(label)
        axes[ax_index].set_xticks(ticks[label])
        axes[ax_index].set_xticklabels(ticks[label])

        ax_index, label = 59, r'$\gamma_{\rm{ext}}$'
        axes[ax_index].set_xlabel(label)
        axes[ax_index].set_xticks(ticks[label])
        axes[ax_index].set_xticklabels(ticks[label])

        ax_index, label = 60, r'$\phi_{\gamma \rm{ext}}$'
        axes[ax_index].set_xlabel(label)
        axes[ax_index].set_xticks(ticks[label])
        axes[ax_index].set_xticklabels(ticks[label])

        ax_index, label = 61, r'$\gamma$'
        axes[ax_index].set_xlabel(label)
        axes[ax_index].set_xticks(ticks[label])
        axes[ax_index].set_xticklabels(ticks[label])

        ax_index, label = 62, r'$a_3$'
        axes[ax_index].set_xlabel(label)
        axes[ax_index].set_xticks(ticks[label])
        axes[ax_index].set_xticklabels(ticks[label])

        ax_index, label = 63, r'$a_4$'
        axes[ax_index].set_xlabel(label)
        axes[ax_index].set_xticks(ticks[label])
        axes[ax_index].set_xticklabels(ticks[label])

    def auto_param_ticks_ranges(lens_index, param_names):
        print(lens_index)
        if lens_index == 4:
            ranges = [[0.98, 1.02], [0.5, 0.8], [0.275, 0.525], [0.05, 0.12], [0.6, 1.0], [1.8, 2.4], [-0.0125, 0.0125],
                      [-0.02, 0.02]]
            ticks = {r'$\theta_{\rm{E}}$': [0.99, 1.0, 1.01],
                     r'$q$': [0.55, 0.65, 0.75],
                     r'$\phi_{\rm{q}}$': [0.3, 0.4, 0.5],
                     r'$\gamma_{\rm{ext}}$': [0.06, 0.085, 0.11],
                     r'$\phi_{\gamma \rm{ext}}$': [0.65, 0.8, 0.95],
                     r'$\gamma$': [1.9, 2.1, 2.3],
                     r'$a_3$': [-0.01, 0.0, 0.01],
                     r'$a_4$': [-0.015, 0.0, 0.015]}

        elif lens_index == 2:
            ranges = [[0.94, 1.03], [0.4, 0.9], [-1.5, -1.0], [0.03, 0.12], [0.6, 1.8], [1.7, 2.2], [-0.02, 0.02],
                      [-0.03, 0.03]]
            ticks = {r'$\theta_{\rm{E}}$': [0.95, 0.985, 1.02],
                     r'$q$': [0.45, 0.65, 0.85],
                     r'$\phi_{\rm{q}}$': [-1.4, -1.25, -1.1],
                     r'$\gamma_{\rm{ext}}$': [0.04, 0.075, 0.11],
                     r'$\phi_{\gamma \rm{ext}}$': [0.7, 1.2, 1.7],
                     r'$\gamma$': [1.8, 1.95, 2.1],
                     r'$a_3$': [-0.015, 0.0, 0.015],
                     r'$a_4$': [-0.025, 0.0, 0.025]}

        elif lens_index == 6:

            ranges = [[0.97, 1.03], [0.45, 0.85], [0.5, 0.75], [0.0125, 0.09], [0.625, 1.425], [1.6, 2.2],
                      [-0.02, 0.02], [-0.025, 0.025]]

            ticks = {r'$\theta_{\rm{E}}$': [0.98, 1.00, 1.02],

                     r'$q$': [0.5, 0.65, 0.8],

                     r'$\phi_{\rm{q}}$': [0.55, 0.625, 0.7],

                     r'$\gamma_{\rm{ext}}$': [0.02, 0.05, 0.08],

                     r'$\phi_{\gamma \rm{ext}}$': [0.7, 1.0, 1.3],

                     r'$\gamma$': [1.7, 1.95, 2.1],

                     r'$a_3$': [-0.015, 0.0, 0.015],

                     r'$a_4$': [-0.02, 0.0, 0.02]}

        elif lens_index == 9:
            ranges = [[0.96, 1.02], [0.4, 0.8], [1.25, 1.5], [0.00, 0.1], [0.5, 1.7], [1.6, 2.4], [-0.02, 0.02],
                      [-0.03, 0.03]]
            ticks = {r'$\theta_{\rm{E}}$': [0.97, 0.99, 1.01],
                     r'$q$': [0.45, 0.6, 0.75],
                     r'$\phi_{\rm{q}}$': [1.3, 1.375, 1.45],
                     r'$\gamma_{\rm{ext}}$': [0.02, 0.05, 0.08],
                     r'$\phi_{\gamma \rm{ext}}$': [0.6, 1.1, 1.6],
                     r'$\gamma$': [1.7, 2.0, 2.3],
                     r'$a_3$': [-0.015, 0.0, 0.015],
                     r'$a_4$': [-0.025, 0.0, 0.025]}

        elif lens_index == 11:

            ranges = [[0.98, 1.02], [0.575, 0.85], [0.85, 1.175], [0.0425, 0.115], [-1.5, -0.95], [1.7, 2.15],
                      [-0.0175, 0.0175], [-0.03, 0.015]]

            ticks = {r'$\theta_{\rm{E}}$': [0.985, 1.0, 1.015],

                     r'$q$': [0.6, 0.71, 0.825],

                     r'$\phi_{\rm{q}}$': [0.875, 1.025, 1.15],

                     r'$\gamma_{\rm{ext}}$': [0.045, 0.075, 0.105],

                     r'$\phi_{\gamma \rm{ext}}$': [-1.4, -1.2, -1.0],

                     r'$\gamma$': [1.75, 1.925, 2.1],

                     r'$a_3$': [-0.015, 0.0, 0.015],

                     r'$a_4$': [-0.025, -0.0075, 0.01]}

        elif lens_index == 23:

            ranges = [[0.98, 1.03], [0.45, 0.8], [-0.005, 0.3], [0.015, 0.105], [0.4, 1.4], [1.7, 2.4], [-0.02, 0.02],
                      [-0.03, 0.03]]

            ticks = {r'$\theta_{\rm{E}}$': [0.985, 1.005, 1.025],

                     r'$q$': [0.5, 0.625, 0.75],

                     r'$\phi_{\rm{q}}$': [0.0, 0.125, 0.25],

                     r'$\gamma_{\rm{ext}}$': [0.02, 0.06, 0.1],

                     r'$\phi_{\gamma \rm{ext}}$': [0.5, 0.9, 1.3],

                     r'$\gamma$': [1.8, 2.05, 2.3],

                     r'$a_3$': [-0.015, 0.0, 0.015],

                     r'$a_4$': [-0.025, 0.0, 0.025]}

            return ticks, ranges


        else:

            ticks, ranges = None, None

            return ticks, ranges

        return ticks, ranges

    compute_lens_indexes = [index_run]
    kwargs_density = {'nbins': int(nbins), 'use_kde': False, 'param_ranges': None}
    for lens_index in compute_lens_indexes:

        truths = get_true_params(lens_index, param_names_plot_macro)

        with open(path_to_data + 'mock_' + str(lens_index) + '_cosmos_source_free_multipole_nmax10_output', 'rb') as f:
            output = pickle.load(f)
        f.close()
        with open(path_to_data + 'mock_' + str(lens_index) + '_free_multipole_no_image_data_output', 'rb') as f:
            output_no_image_data = pickle.load(f)
        f.close()

        param_ticks, param_ranges_macro = auto_param_ticks_ranges(lens_index, param_names_plot_macro)
        like_imgpos, like_FR, like_IM_FFR, like_IM = compute_likelihoods(output,
                                                                         output_no_image_data,
                                                                         mock_lens_data_list[lens_index - 1],
                                                                         kwargs_density,
                                                                         n_keep=300,
                                                                         flux_ratio_uncertainty=0.03,
                                                                         percentile_cut_image_data=2.5,
                                                                         param_ranges_macro=param_ranges_macro,
                                                                         nbins=nbins,
                                                                         n_resample=10,
                                                                         imaging_data_hard_cut=True,
                                                                         imaging_data_likelihood=False,
                                                                         imaging_data_likelihood_scale=None,
                                                                         )

        triplot = TrianglePlot([like_FR, like_IM, like_IM_FFR])
        axes = triplot.make_triplot(filled_contours=True,
                                    truths=truths,
                                    show_intervals=False,
                                    tick_label_font=14,
                                    xtick_label_rotate=40)
        axes[11].annotate('MOCK #' + str(lens_index), xy=(0.5, 1.6), xycoords='axes fraction', fontsize=25, color='0.3')
        axes[11].annotate('IMAGE POSITIONS & FLUX RATIOS', xy=(0.05, 1.2), xycoords='axes fraction', fontsize=20,
                          color='k')
        axes[11].annotate('IMAGE POSITIONS & IMAGING DATA', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=20,
                          color='b')
        axes[11].annotate('IMAGE POSITIONS, FLUX RATIOS & IMAGING DATA', xy=(0.05, 0.6), xycoords='axes fraction',
                          fontsize=20, color='m')
        if param_ticks is not None: rename_axes(axes, param_ticks, param_names_plot_macro)
        plt.savefig(os.getenv('HOME') + '/Code/samana/turbocharging_figures/macromodel_figures/mock_' + str(lens_index) + '_macro_inference_free_multipole.pdf')
        #plt.show()
        # plt.show()
# import sys
# macromodel_plot(int(sys.argv[1]), int(sys.argv[2]))

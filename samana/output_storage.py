import numpy as np
from lenstronomy.Util.param_util import shear_cartesian2polar
from lenstronomy.Util.param_util import ellipticity2phi_q
from copy import deepcopy

class Output(object):

    def __init__(self, parameters, image_magnifications,
                 macromodel_samples, fitting_kwargs_list=None,
                 param_names=None, macromodel_sample_names=None, flux_ratio_summary_statistic=None,
                 flux_ratio_likelihood=None):
        """

        :param param_names:
        :param parameters:
        :param image_magnifications:
        :param macromodel_sample_names:
        :param macromodel_samples:
        :param fitting_kwargs_list:
        """
        self.parameters = parameters
        self.image_magnifications = image_magnifications
        self.macromodel_samples = macromodel_samples
        self.fitting_kwargs_list = fitting_kwargs_list
        if parameters is not None:
            self.seed = parameters[:, -1]
            self.image_data_logL = parameters[:, -2]
            if flux_ratio_likelihood is None:
                self._flux_ratio_likelihood = deepcopy(parameters[:, -3])
            else:
                self._flux_ratio_likelihood = flux_ratio_likelihood
            if flux_ratio_summary_statistic is None:
                self._flux_ratio_stat = deepcopy(parameters[:, -4])
            else:
                self._flux_ratio_stat = flux_ratio_summary_statistic
        else:
            self.seed = None
            self.image_data_logL = None
            self._flux_ratio_likelihood = flux_ratio_likelihood
            self._flux_ratio_stat = flux_ratio_summary_statistic
        self._param_dict = None
        self._param_names = param_names
        self._macromodel_sample_names = macromodel_sample_names
        if param_names is not None:
            if parameters is not None:
                assert len(param_names) == parameters.shape[1]
                self._param_dict = {}
                for i, name in enumerate(param_names):
                    self._param_dict[name] = parameters[:, i]
            else:
                self._param_dict = {}
        self._macromodel_samples_dict = None
        if macromodel_sample_names is not None:
            if macromodel_samples is not None:
                assert len(macromodel_sample_names) == macromodel_samples.shape[1]
                self._macromodel_samples_dict = {}
                for i, name in enumerate(macromodel_sample_names):
                    self._macromodel_samples_dict[name] = macromodel_samples[:, i]
            else:
                self._macromodel_samples_dict = None

    def clean(self):
        """
        This method removes elements corresponding to np.nan flux ratios, and elements that correspond to a repeated
        random seed
        :return:
        """
        return self.clean_nans().clean_repeated_seeds()

    def clean_nans(self):

        flux_ratios_summed = np.sum(self.flux_ratios, axis=1)
        inds_keep = np.where(np.isfinite(flux_ratios_summed))[0]
        return self.down_select(inds_keep)

    def clean_repeated_seeds(self):

        _, index = np.unique(self.seed, return_index=True)
        return self.down_select(index)

    @classmethod
    def from_raw_output(cls, output_path, job_index_min, job_index_max, fitting_kwargs_list=None,
                        macromodel_sample_names=None, print_missing_files=False):

        param_names = None
        init = True
        for i in range(job_index_min, job_index_max + 1):

            folder = output_path + '/job_' + str(i) + '/'
            try:
                params = np.loadtxt(folder + 'parameters.txt', skiprows=1)
            except:
                if print_missing_files:
                    print('params file ' + folder + 'parameters.txt not found... ')
                continue
            try:
                fluxes = np.loadtxt(folder + 'fluxes.txt')
            except:
                if print_missing_files:
                    print('fluxes file ' + folder + 'fluxes.txt not found... ')
                continue
            try:
                macrosamples = np.loadtxt(folder + 'macromodel_samples.txt', skiprows=1)
            except:
                if print_missing_files:
                    print('macromodel samples file ' + folder + 'macromodel_samples.txt not found... ')
                continue
            # check the arrays are all the same length
            size_params = params.shape[0]
            size_fluxes = fluxes.shape[0]
            size_macro = macrosamples.shape[0]
            if size_params != size_fluxes:
                print('parameters and fluxes have different shape for ' + folder)
                continue
            if size_params != size_macro:
                print('parameters and macromodel samples have different shape for ' + folder)
                continue
            if param_names is None:
                with open(folder + 'parameters.txt', 'r') as f:
                    param_names = f.readlines(1)[0].split()
                f.close()
            if macromodel_sample_names is None:
                with open(folder + 'macromodel_samples.txt', 'r') as f:
                    macromodel_sample_names = f.readlines(1)[0].split()
                f.close()
            if init:
                parameters = params
                magnifications = fluxes
                macromodel_samples = macrosamples
                init = False
            else:
                parameters = np.vstack((parameters, params))
                magnifications = np.vstack((magnifications, fluxes))
                macromodel_samples = np.vstack((macromodel_samples, macrosamples))
        print('compiled ' + str(parameters.shape[0]) + ' realizations.')
        return Output(parameters, magnifications, macromodel_samples, fitting_kwargs_list,
                      param_names, macromodel_sample_names)

    @classmethod
    def join(self, output1, output2):

        params = np.vstack((output1.parameters, output2.parameters))
        mags = np.vstack((output1.image_magnifications, output2.image_magnifications))
        macro_samples = np.vstack((output1.macromodel_samples, output2.macromodel_samples))
        param_names = output1._param_names
        macromodel_sample_names = output1._macromodel_sample_names
        flux_ratio_summary_statistic = np.append(output1.flux_ratio_summary_statistic,
                                                 output2.flux_ratio_summary_statistic)
        flux_ratio_likelihood = np.append(output1.flux_ratio_likelihood,
                                          output2.flux_ratio_likelihood)
        return Output(params, mags, macro_samples, None, param_names, macromodel_sample_names,
                      flux_ratio_summary_statistic, flux_ratio_likelihood)

    @property
    def flux_ratio_likelihood(self):
        """

        :return:
        """
        if self._flux_ratio_likelihood is None:
            print('flux ratio likelihood not set!')
            return None
        else:
            return self._flux_ratio_likelihood

    @property
    def flux_ratio_summary_statistic(self):
        """

        :return:
        """
        if self._flux_ratio_stat is None:
            print('flux ratio summary statistic not set!')
            return None
        else:
            return self._flux_ratio_stat

    @property
    def imaging_data_relative_likelihood(self):
        imaging_data_weights = np.exp(self.image_data_logL - np.max(self.image_data_logL))
        return imaging_data_weights

    def set_flux_ratio_likelihood(self, measured_magnifications=None,
                                  modeled_magnifications=None,
                                  measurement_uncertainties=None,
                                  measured_flux_ratios=None,
                                  modeled_flux_ratios=None,
                                  verbose=False):

        if measured_flux_ratios is None:
            measured_flux_ratios = measured_magnifications[1:] / measured_magnifications[0]
        if modeled_flux_ratios is None:
            modeled_flux_ratios = modeled_magnifications[:,1:] / modeled_magnifications[:,0,np.newaxis]
        like = 0
        for i in range(0, 3):
            like += (measured_flux_ratios[i] - modeled_flux_ratios[:, i]) ** 2 / measurement_uncertainties[i] ** 2
        flux_ratio_likelihood = np.exp(-0.5 * like)
        norm = np.max(flux_ratio_likelihood)
        self._flux_ratio_likelihood = flux_ratio_likelihood / norm
        if verbose:
            print('flux ratio likelihood effective sample size: ', np.sum(self._flux_ratio_likelihood))

    def set_flux_ratio_summary_statistic(self, measured_magnifications, modeled_magnifications,
                                         measured_flux_ratios=None, modeled_flux_ratios=None, verbose=False):

        if measured_flux_ratios is None:
            measured_flux_ratios = measured_magnifications[1:] / measured_magnifications[0]
        if modeled_flux_ratios is None:
            modeled_flux_ratios = modeled_magnifications[:,1:] / modeled_magnifications[:,0,np.newaxis]
        stat = 0
        for i in range(0, 3):
            stat += (measured_flux_ratios[i] - modeled_flux_ratios[:,i])**2
        self._flux_ratio_stat = np.sqrt(stat)
        if verbose:
            print('SUMMARY STATISTIC THRESHOLDS: ')
            print('S = 0.02: ', np.sum(self._flux_ratio_stat < 0.02))
            print('S = 0.05: ', np.sum(self._flux_ratio_stat < 0.05))
            print('S = 0.075: ', np.sum(self._flux_ratio_stat < 0.075))
            print('S = 0.1: ', np.sum(self._flux_ratio_stat < 0.1))

    @property
    def flux_ratios(self):

        if not hasattr(self, '_flux_ratios'):
            self._flux_ratios = self.image_magnifications[:, 1:] / self.image_magnifications[:, 0, np.newaxis]
        return self._flux_ratios

    def parameter_array(self, param_names):

        samples = np.empty((self.parameters.shape[0], len(param_names)))
        for i, param_name in enumerate(param_names):
            if param_name == 'f2/f1':
                samples[:, i] = self.flux_ratios[:, 0]
            elif param_name == 'f3/f1':
                samples[:, i] = self.flux_ratios[:, 1]
            elif param_name == 'f4/f1':
                samples[:, i] = self.flux_ratios[:, 2]
            else:
                samples[:, i] = self.param_dict[param_name]
        return samples

    def macromodel_parameter_array(self, param_names):

        phi_q, q = ellipticity2phi_q(self.macromodel_samples_dict['e1'], self.macromodel_samples_dict['e2'])
        phi_gamma, gamma_ext = shear_cartesian2polar(self.macromodel_samples_dict['gamma1'],
                                                         self.macromodel_samples_dict['gamma2'])
        samples = np.empty((self.macromodel_samples.shape[0], len(param_names)))

        for i, param_name in enumerate(param_names):
            if param_name == 'q':
                samples[:, i] = q
            elif param_name == 'phi_q':
                samples[:, i] = phi_q
            elif param_name == 'phi_q_angle':
                samples[:, i] = phi_q * 180/np.pi
            elif param_name == 'gamma_ext':
                samples[:, i] = gamma_ext
            elif param_name == 'phi_gamma':
                samples[:, i] = phi_gamma
            elif param_name == 'phi_gamma_angle':
                samples[:, i] = phi_gamma * 180/np.pi
            elif param_name == 'gamma_cos_phi_gamma':
                samples[:, i] = gamma_ext * np.cos(2*phi_gamma)
            elif param_name == 'q_cos_phi':
                samples[:, i] = q * np.cos(phi_q)
            elif param_name == 'a3_a_phys':
                rescale = self.macromodel_samples_dict['theta_E'] / np.sqrt(q)
                samples[:, i] = self.macromodel_samples_dict['a3_a'] * rescale
            elif param_name == 'a4_a_phys':
                rescale = self.macromodel_samples_dict['theta_E'] / np.sqrt(q)
                samples[:, i] = self.macromodel_samples_dict['a4_a'] * rescale
            elif param_name == 'a3_a_cos':
                samples[:, i] = self.macromodel_samples_dict['a3_a'] * \
                                np.cos(3 * (phi_q + self.macromodel_samples_dict['delta_phi_m3']))
            elif param_name == 'a4_a_cos':
                samples[:, i] = self.macromodel_samples_dict['a4_a'] * \
                                np.cos(4 * (phi_q + self.macromodel_samples_dict['delta_phi_m4']))
            elif param_name == 'a4':
                #rescale = self.macromodel_samples_dict['theta_E'] / np.sqrt(q)
                a4a = self.macromodel_samples_dict['a4_a']
                theta = self.macromodel_samples_dict['delta_phi_m4'] + phi_q
                samples[:,i] = a4a * np.cos(theta)
            elif param_name == 'b4':
                #rescale = self.macromodel_samples_dict['theta_E'] / np.sqrt(q)
                a4a = self.macromodel_samples_dict['a4_a']
                theta = self.macromodel_samples_dict['delta_phi_m4'] + phi_q
                samples[:,i] = a4a * np.sin(theta)
            elif param_name == 'a3':
                #rescale = self.macromodel_samples_dict['theta_E'] / np.sqrt(q)
                a3a = self.macromodel_samples_dict['a3_a']
                theta = self.macromodel_samples_dict['delta_phi_m3'] + phi_q
                samples[:,i] = a3a * np.cos(theta)
            elif param_name == 'b3':
                #rescale = self.macromodel_samples_dict['theta_E'] / np.sqrt(q)
                a3a = self.macromodel_samples_dict['a3_a']
                theta = self.macromodel_samples_dict['delta_phi_m3'] + phi_q
                samples[:,i] = a3a * np.sin(theta)
            elif param_name == 'phi_m3':
                samples[:, i] = self.macromodel_samples_dict['delta_phi_m3'] + phi_q
            elif param_name == 'phi_m4':
                samples[:, i] = self.macromodel_samples_dict['delta_phi_m4'] + phi_q
            elif param_name == 'phi_m1':
                samples[:, i] = self.macromodel_samples_dict['delta_phi_m1'] + phi_q
            elif param_name == 'f2/f1':
                samples[:, i] = self.flux_ratios[:, 0]
            elif param_name == 'f3/f1':
                samples[:, i] = self.flux_ratios[:, 1]
            elif param_name == 'f4/f1':
                samples[:, i] = self.flux_ratios[:, 2]
            else:
                samples[:, i] = self.macromodel_samples_dict[param_name]

        return samples

    @property
    def param_dict(self):

        if self._param_dict is None:
            if self._param_names is not None:
                assert len(self._param_names) == self.parameters.shape[1]
                self._param_dict = {}
                for i, name in enumerate(self._param_names):
                    self._param_dict[name] = self.parameters[:, i]
            else:
                print('parameter names need to be specified to create a param dictionary')
                return None
        else:
            return self._param_dict

    @property
    def macromodel_samples_dict(self):

        if self._macromodel_samples_dict is None:
            if self._macromodel_sample_names is not None:
                assert len(self._macromodel_sample_names) == self.macromodel_samples.shape[1]
                self._macromodel_samples_dict = {}
                for i, name in enumerate(self._macromodel_sample_names):
                    self._macromodel_samples_dict[name] = self.macromodel_samples[:, i]
            else:
                print('parameter names need to be specified to create a param dictionary')
                return None
        else:
            return self._macromodel_samples_dict

    def down_select(self, inds_keep):
        """
        :param inds_keep:
        :return:
        """
        parameters = self.parameters[inds_keep, :]
        image_magnifications = self.image_magnifications[inds_keep, :]
        macromodel_samples = self.macromodel_samples[inds_keep, :]
        flux_ratio_summary_statistic = self.flux_ratio_summary_statistic[inds_keep]
        flux_ratio_likelihood = self.flux_ratio_likelihood[inds_keep]
        return Output(parameters, image_magnifications, macromodel_samples,
                      fitting_kwargs_list=self.fitting_kwargs_list,
                      param_names=self._param_names,
                      macromodel_sample_names=self._macromodel_sample_names,
                      flux_ratio_summary_statistic=flux_ratio_summary_statistic,
                      flux_ratio_likelihood=flux_ratio_likelihood)

    def cut_on_image_data(self, percentile_cut, logL_threshold=None, select_worst=False, undo_prior=False):
        """

        :param percentile_cut:
        :return:
        """
        if undo_prior:
            a3 = self._macromodel_samples_dict['a3_a']
            a4 = self._macromodel_samples_dict['a4_a']
            w = np.exp(-0.5 * a3 ** 2 / 0.005 ** 2) * np.exp(-0.5 * a4 ** 2 / 0.01 ** 2)
            logL_a3a4 = np.log(w)
            image_data_logL = self.image_data_logL - logL_a3a4
        else:
            image_data_logL = self.image_data_logL

        if logL_threshold is None:
            inds_sorted = np.argsort(image_data_logL)
            if select_worst:
                idx_cut = int(percentile_cut / 100 * len(image_data_logL))
                logL_cut = image_data_logL[inds_sorted[idx_cut]]
                inds_keep = np.where(image_data_logL <= logL_cut)[0]
            else:
                idx_cut = int((100 - percentile_cut) / 100 * len(image_data_logL))
                logL_cut = image_data_logL[inds_sorted[idx_cut]]
                inds_keep = np.where(image_data_logL >= logL_cut)[0]
        else:
            if select_worst:
                inds_keep = np.where(image_data_logL <= logL_threshold)[0]
            else:
                inds_keep = np.where(image_data_logL >= logL_threshold)[0]
        return self.down_select(inds_keep)

    def cut_on_S_statistic(self, keep_best_N=None, S_statistic_cut=None, select_worst=False):
        """

        :param percentile_cut:
        :return:
        """
        sorted_inds = np.argsort(self.flux_ratio_summary_statistic)
        L = len(sorted_inds)
        if keep_best_N is not None:
            assert S_statistic_cut is None
            if select_worst:
                idx = L - keep_best_N
                inds_keep = sorted_inds[idx:]
            else:
                inds_keep = sorted_inds[0:keep_best_N]
        elif S_statistic_cut is not None:
            if select_worst:
                inds_keep = np.where(self.flux_ratio_summary_statistic >= S_statistic_cut)[0]
            else:
                inds_keep = np.where(self.flux_ratio_summary_statistic <= S_statistic_cut)[0]
        else:
            raise Exception('must specify keep_best_N, percentile_cut, or S_statistic_cut')
        return self.down_select(inds_keep)

    def cut_on_flux_ratio_likelihood(self, keep_best_N=None, percentile_cut=None, likelihood_cut=None):
        """

        :param percentile_cut:
        :return:
        """
        sorted_inds = np.argsort(self.flux_ratio_likelihood)
        if keep_best_N is not None:
            assert percentile_cut is None and likelihood_cut is None
            inds_keep = sorted_inds[0:keep_best_N]
        elif percentile_cut is not None:
            assert likelihood_cut is None
            idxcut = int(self.parameters.shape[0] * percentile_cut/100)
            inds_keep = sorted_inds[0:idxcut]
        elif likelihood_cut is not None:
            inds_keep = np.where(self.flux_ratio_likelihood <= likelihood_cut)[0]
        else:
            raise Exception('must specify keep_best_N, percentile_cut, or likelihood_cut')
        return self.down_select(inds_keep)

    def plot_joint_statistics(self, ax=None, s_max=0.1, logLlower=None, **kwargs_plot):

        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure(1)
            fig.set_size_inches(6, 6)
            ax = plt.subplot(111)
        s_stat = self.flux_ratio_summary_statistic
        image_data_logL = self.image_data_logL
        ax.scatter(s_stat, image_data_logL, **kwargs_plot)
        smin, smax = np.min(s_stat), s_max
        logLupper = np.max(image_data_logL)+10
        if logLlower is None:
            logLlower = logLupper - 2000
        ax.set_xlim(smin, smax)
        ax.set_ylim(logLupper, logLlower)
        return ax

def compile_flux_ratios(output_path, job_index_min, job_index_max,
                        measured_flux_ratios, flux_ratio_uncertainties,
                        print_missing_files=False, index_reorder=None):

    init = True
    random_seeds = None
    magnifications = None
    if index_reorder is None:
        index_reorder = [0,1,2,3]
    for i in range(job_index_min, job_index_max + 1):

        folder = output_path + '/job_' + str(i) + '/'
        try:
            params = np.loadtxt(folder + 'parameters.txt', skiprows=1)
        except:
            if print_missing_files:
                print('params file ' + folder + 'parameters.txt not found... ')
            continue
        try:
            fluxes = np.loadtxt(folder + 'fluxes.txt')
        except:
            if print_missing_files:
                print('fluxes file ' + folder + 'fluxes.txt not found... ')
            continue
        try:
            macrosamples = np.loadtxt(folder + 'macromodel_samples.txt', skiprows=1)
        except:
            if print_missing_files:
                print('macromodel samples file ' + folder + 'macromodel_samples.txt not found... ')
            continue
        # check the arrays are all the same length
        size_params = params.shape[0]
        size_fluxes = fluxes.shape[0]
        size_macro = macrosamples.shape[0]
        if size_params != size_fluxes:
            print('parameters and fluxes have different shape for ' + folder)
            continue
        if size_params != size_macro:
            print('parameters and macromodel samples have different shape for ' + folder)
            continue
        if init:
            magnifications = fluxes[:,index_reorder]
            random_seeds = params[:, -1]
            init = False
        else:
            random_seeds = np.append(random_seeds, params[:, -1])
            magnifications = np.vstack((magnifications, fluxes[:,index_reorder]))

    flux_ratios = magnifications[:,1:] / magnifications[:,0,np.newaxis]
    fr_chi2 = 0
    for i in range(0, len(measured_flux_ratios)):
        if flux_ratio_uncertainties[i] == -1:
            continue
        fr_chi2 += (flux_ratios[:, i] - measured_flux_ratios[i]) ** 2 / flux_ratio_uncertainties[i] ** 2
    logL = -0.5 * fr_chi2
    y = np.empty((len(random_seeds), 2))
    y[:, 0] = logL
    y[:, 1] = random_seeds
    return y


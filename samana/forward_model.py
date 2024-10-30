from pyHalo.preset_models import preset_model_from_name
from samana.forward_model_util import filenames, sample_prior, align_realization, \
    flux_ratio_summary_statistic, flux_ratio_likelihood, split_kwargs_params
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_resolution, auto_raytracing_grid_size
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Util.class_creator import create_im_sim
from lenstronomy.LensModel.QuadOptimizer.optimizer import Optimizer
from samana.image_magnification_util import setup_gaussian_source
from samana.light_fitting import FixedLensModelNew, setup_params_light_fitting
from samana.param_managers import auto_param_class
from copy import deepcopy
from multiprocessing.pool import Pool
import os
import subprocess
import numpy as np
from time import time


def forward_model(output_path, job_index, n_keep, data_class, model, preset_model_name,
                  kwargs_sample_realization, kwargs_sample_source, kwargs_sample_fixed_macromodel,
                  tolerance, log_mlow_mass_sheets=6.0, kwargs_model_class={},
                  rescale_grid_size=1.5, rescale_grid_resolution=2.0, readout_macromodel_samples=True,
                  verbose=False, random_seed_init=None, readout_steps=2, write_sampling_rate=True,
                  n_pso_particles=10, n_pso_iterations=50, num_threads=1, astrometric_uncertainty=True,
                  image_data_grid_resolution_rescale=1.0,
                  use_imaging_data=True, fitting_sequence_kwargs=None, test_mode=False,
                  use_decoupled_multiplane_approximation=True, fixed_realization_list=None,
                  macromodel_readout_function=None,
                  kappa_scale_subhalos=1.0,
                  log10_bound_mass_cut=None,
                  parallelize=False,
                  elliptical_ray_tracing_grid=True,
                  split_image_data_reconstruction=False):
    """

    :param output_path:
    :param job_index:
    :param n_keep:
    :param data_class:
    :param model:
    :param preset_model_name:
    :param kwargs_sample_realization:
    :param kwargs_sample_source:
    :param kwargs_sample_fixed_macromodel:
    :param tolerance:
    :param log_mlow_mass_sheets:
    :param kwargs_model_class:
    :param rescale_grid_size:
    :param rescale_grid_resolution:
    :param readout_macromodel_samples:
    :param verbose:
    :param random_seed_init:
    :param readout_steps:
    :param write_sampling_rate:
    :param n_pso_particles:
    :param n_pso_iterations:
    :param num_threads:
    :param astrometric_uncertainty:
    :param image_data_grid_resolution_rescale:
    :param use_imaging_data:
    :param fitting_sequence_kwargs:
    :param test_mode:
    :param fixed_realization_list:
    :param macromodel_readout_function:
    :param kappa_scale_subhalos:
    :param log10_bound_mass_cut:
    :param parallelize:
    :param elliptical_ray_tracing_grid:
    :return:
    """

    filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio, \
    filename_macromodel_samples = filenames(output_path, job_index)
    # if the required directories do not exist, create them
    if os.path.exists(output_path) is False:
        proc = subprocess.Popen(['mkdir', output_path])
        proc.wait()
    if os.path.exists(output_path + 'job_' + str(job_index)) is False:
        proc = subprocess.Popen(['mkdir', output_path + 'job_' + str(job_index)])
        proc.wait()

    if verbose:
        print('reading output to files: ')
        print(filename_parameters)
        print(filename_mags)
    # You can restart inferences from previous runs by simply running the function again. In the following lines, the
    # code looks for existing output files, and determines how many samples to add based on how much output already
    # exists.
    if os.path.exists(filename_mags):
        _m = np.loadtxt(filename_mags)
        try:
            n_kept = _m.shape[0]
        except:
            n_kept = 1
        write_param_names = False
        write_param_names_macromodel_samples = False
    else:
        n_kept = 0
        _m = None
        write_param_names = True
        write_param_names_macromodel_samples = True
    if fixed_realization_list is not None:
        if verbose:
            if len(fixed_realization_list) != n_keep:
                print('you specified n_keep = '+str(n_keep)+' but also gave a list of precomputed substructure '
                                                            'realizations. The code will run once per realization. '
                                                            'New n_keep = '+str(len(fixed_realization_list)))
        n_keep = len(fixed_realization_list)
    if n_kept >= n_keep:
        print('\nSIMULATION ALREADY FINISHED.')
        return
    # Initialize stuff for the inference
    parameter_array = None
    mags_out = None
    macromodel_samples_array = None
    readout = False
    break_loop = False
    accepted_realizations_counter = 0
    acceptance_rate_counter = 0
    iteration_counter = 0
    # estimate the sampling rate (CPU minutes per realization) after this many iterations
    readout_sampling_rate_index = 10
    if n_keep < readout_sampling_rate_index:
        readout_sampling_rate_index = deepcopy(n_keep)
    acceptance_ratio = np.nan
    sampling_rate = np.nan
    t0 = time()

    if random_seed_init is None:
        # pick a random integer from which to generate random seeds
        random_seed_init = np.random.randint(0, 4294967295)

    if verbose:
        print('starting with ' + str(n_kept) + ' samples accepted, ' + str(n_keep - n_kept) + ' remain')
        print('existing magnifications: ', _m)
        print('samples remaining: ', n_keep - n_kept)
        print('running simulation with a summary statistic tolerance of: ', tolerance)
    # start the simulation, the while loop will execute until one has obtained n_keep samples from the posterior
    seed_counter = 0 + n_kept
    while True:

        # the random seed in numpy maxes out at 4294967295
        random_seed = random_seed_init + seed_counter
        if random_seed > 4294967295:
            random_seed = random_seed - 4294967296
        if fixed_realization_list is not None:
            fixed_realization = fixed_realization_list[acceptance_rate_counter]
        else:
            fixed_realization = None

        if parallelize:
            args = []
            for cpu_index in range(0, num_threads):
                args.append((data_class, model, preset_model_name,
                             kwargs_sample_realization,
                             kwargs_sample_source,
                             kwargs_sample_fixed_macromodel,
                             log_mlow_mass_sheets,
                             rescale_grid_size, rescale_grid_resolution,
                             image_data_grid_resolution_rescale,
                             verbose, random_seed + cpu_index, n_pso_particles,
                             n_pso_iterations, 1,
                             kwargs_model_class, astrometric_uncertainty,
                             use_imaging_data, fitting_sequence_kwargs,
                             test_mode,
                             use_decoupled_multiplane_approximation,
                             fixed_realization,
                             macromodel_readout_function,
                             kappa_scale_subhalos,
                             log10_bound_mass_cut,
                             elliptical_ray_tracing_grid,
                             split_image_data_reconstruction))

            pool = Pool(num_threads)
            output = pool.starmap(forward_model_single_iteration, args)
            pool.close()
            for _, result in enumerate(output):
                (magnifications, images, realization_samples, source_samples, macromodel_samples,
                macromodel_samples_fixed, \
                logL_imaging_data, fitting_sequence, stat, log_flux_ratio_likelihood, bic, param_names_realization,
                param_names_source, param_names_macro, \
                param_names_macro_fixed, _, _, _) = result
                acceptance_rate_counter += 1
                seed_counter += 1
                # Once we have computed a couple realizations, keep a log of the time it takes to run per realization
                if acceptance_rate_counter == readout_sampling_rate_index:
                    time_elapsed = time() - t0
                    time_elapsed_minutes = time_elapsed / 60
                    sampling_rate = time_elapsed_minutes / acceptance_rate_counter
                    readout_sampling_rate = True
                else:
                    readout_sampling_rate = False
                # this keeps track of how many realizations were analyzed, and resets after each readout (set by readout_steps)
                # The purpose of this counter is to keep track of the acceptance rate
                iteration_counter += 1
                if stat < tolerance:
                    # If the statistic is less than the tolerance threshold, we keep the parameters
                    accepted_realizations_counter += 1
                    n_kept += 1
                    params = np.append(realization_samples, source_samples)
                    params = np.append(params, bic)
                    params = np.append(params, stat)
                    params = np.append(params, log_flux_ratio_likelihood)
                    params = np.append(params, logL_imaging_data)
                    params = np.append(params, random_seed + seed_counter)
                    param_names = param_names_realization + param_names_source + ['bic', 'summary_statistic',
                                                                                  'flux_ratio_log_likelihood',
                                                                                  'logL_image_data', 'seed']
                    acceptance_ratio = accepted_realizations_counter / iteration_counter

                    if parameter_array is None:
                        parameter_array = params
                    else:
                        parameter_array = np.vstack((parameter_array, params))
                    if mags_out is None:
                        mags_out = magnifications
                    else:
                        mags_out = np.vstack((mags_out, magnifications))
                    if macromodel_samples_array is None:
                        macromodel_samples_array = np.array(macromodel_samples)
                    else:
                        macromodel_samples_array = np.vstack((macromodel_samples_array, macromodel_samples))
                    if verbose:
                        print('N_kept: ', n_kept)
                        print('N remaining: ', n_keep - n_kept)

        else:
            magnifications, images, realization_samples, source_samples, macromodel_samples, macromodel_samples_fixed, \
            logL_imaging_data, fitting_sequence, stat, log_flux_ratio_likelihood, bic, param_names_realization, param_names_source, param_names_macro, \
            param_names_macro_fixed, _, _, _ = forward_model_single_iteration(data_class, model, preset_model_name, kwargs_sample_realization,
                                                kwargs_sample_source, kwargs_sample_fixed_macromodel, log_mlow_mass_sheets,
                                                rescale_grid_size, rescale_grid_resolution, image_data_grid_resolution_rescale,
                                                verbose, random_seed, n_pso_particles, n_pso_iterations, num_threads,
                                                kwargs_model_class, astrometric_uncertainty,
                                                use_imaging_data, fitting_sequence_kwargs, test_mode,
                                                use_decoupled_multiplane_approximation, fixed_realization,
                                                macromodel_readout_function,
                                                kappa_scale_subhalos, log10_bound_mass_cut,
                                                elliptical_ray_tracing_grid,
                                                split_image_data_reconstruction)

            seed_counter += 1
            acceptance_rate_counter += 1
            # Once we have computed a couple realizations, keep a log of the time it takes to run per realization
            if acceptance_rate_counter == readout_sampling_rate_index:
                time_elapsed = time() - t0
                time_elapsed_minutes = time_elapsed / 60
                sampling_rate = time_elapsed_minutes / acceptance_rate_counter
                readout_sampling_rate = True
            else:
                readout_sampling_rate = False

            # this keeps track of how many realizations were analyzed, and resets after each readout (set by readout_steps)
            # The purpose of this counter is to keep track of the acceptance rate
            iteration_counter += 1
            if stat < tolerance:
                # If the statistic is less than the tolerance threshold, we keep the parameters
                accepted_realizations_counter += 1
                n_kept += 1
                params = np.append(realization_samples, source_samples)
                params = np.append(params, bic)
                params = np.append(params, stat)
                params = np.append(params, log_flux_ratio_likelihood)
                params = np.append(params, logL_imaging_data)
                params = np.append(params, random_seed)
                param_names = param_names_realization + param_names_source + ['bic', 'summary_statistic', 'flux_ratio_log_likelihood',
                                                                              'logL_image_data', 'seed']
                acceptance_ratio = accepted_realizations_counter / iteration_counter

                if parameter_array is None:
                    parameter_array = params
                else:
                    parameter_array = np.vstack((parameter_array, params))
                if mags_out is None:
                    mags_out = magnifications
                else:
                    mags_out = np.vstack((mags_out, magnifications))
                if macromodel_samples_array is None:
                    macromodel_samples_array = np.array(macromodel_samples)
                else:
                    macromodel_samples_array = np.vstack((macromodel_samples_array, macromodel_samples))
                if verbose:
                    print('N_kept: ', n_kept)
                    print('N remaining: ', n_keep - n_kept)

        if verbose:
            print('accepted realizations counter: ', acceptance_rate_counter)
        # readout if either of these conditions are met
        if accepted_realizations_counter >= readout_steps:
            readout = True
            if verbose:
                print('reading out data on this iteration.')
            accepted_realizations_counter = 0
            iteration_counter = 0
        # break loop if we have collected n_keep samples
        if n_kept >= n_keep:
            readout = True
            break_loop = True
            if verbose:
                print('final data readout...')
        if readout_sampling_rate and write_sampling_rate:
            with open(filename_sampling_rate, 'w') as f:
                f.write(str(np.round(sampling_rate, 2)) + ' ')
                f.write('\n')

        if readout:
            # Now write stuff to file
            readout = False
            with open(filename_acceptance_ratio, 'a') as f:
                f.write(str(np.round(acceptance_ratio, 8)) + ' ')
                f.write('\n')
            if verbose:
                print('writing parameter output to ' + filename_parameters)
            with open(filename_parameters, 'a') as f:
                if write_param_names:
                    param_name_string = ''
                    for name in param_names:
                        param_name_string += name + ' '
                    f.write(param_name_string + '\n')
                    write_param_names = False

                nrows, ncols = int(parameter_array.shape[0]), int(parameter_array.shape[1])
                for row in range(0, nrows):
                    for col in range(0, ncols):
                        f.write(str(np.round(parameter_array[row, col], 7)) + ' ')
                    f.write('\n')
            if verbose:
                print('writing flux ratio output to ' + filename_mags)
            with open(filename_mags, 'a') as f:
                nrows, ncols = int(mags_out.shape[0]), int(mags_out.shape[1])
                for row in range(0, nrows):
                    for col in range(0, ncols):
                        f.write(str(np.round(mags_out[row, col], 5)) + ' ')
                    f.write('\n')

            if readout_macromodel_samples:
                if verbose:
                    print('writing macromodel samples to ' + filename_macromodel_samples)
                nrows, ncols = int(macromodel_samples_array.shape[0]), int(macromodel_samples_array.shape[1])
                with open(filename_macromodel_samples, 'a') as f:
                    if write_param_names_macromodel_samples:
                        param_name_string = ''
                        for name in param_names_macro:
                            param_name_string += name + ' '
                        f.write(param_name_string + '\n')
                        write_param_names_macromodel_samples = False
                    for row in range(0, nrows):
                        for col in range(0, ncols):
                            f.write(str(np.round(macromodel_samples_array[row, col], 5)) + ' ')
                        f.write('\n')

            parameter_array = None
            mags_out = None
            macromodel_samples_array = None

        if break_loop:
            print('\nSIMULATION FINISHED')
            break

def forward_model_single_iteration(data_class, model, preset_model_name, kwargs_sample_realization,
                            kwargs_sample_source, kwargs_sample_macro_fixed, log_mlow_mass_sheets=6.0, rescale_grid_size=1.0,
                            rescale_grid_resolution=2.0, image_data_grid_resolution_rescale=1.0, verbose=False, seed=None,
                                   n_pso_particles=10, n_pso_iterations=50, num_threads=1,
                                   kwargs_model_class={}, astrometric_uncertainty=True,
                                   use_imaging_data=True,
                                   fitting_kwargs_list=None,
                                   test_mode=False,
                                   use_decoupled_multiplane_approximation=True,
                                   fixed_realization=None,
                                   macromodel_readout_function=None,
                                   kappa_scale_subhalos=1.0,
                                   log10_bound_mass_cut=None,
                                   elliptical_ray_tracing_grid=True,
                                   split_image_data_reconstruction=False):

    # set the random seed for reproducibility
    np.random.seed(seed)

    if astrometric_uncertainty:
        delta_x_image, delta_y_image = data_class.perturb_image_positions()
    else:
        delta_x_image, delta_y_image = np.zeros(len(data_class.x_image)), np.zeros(len(data_class.y_image))

    if data_class.redshift_sampling:
        z_lens = data_class.sample_z_lens()
        if verbose: print('deflector redshift: ', z_lens)
    else:
        z_lens = data_class.z_lens

    model_class = model(data_class, **kwargs_model_class)
    realization_dict, realization_samples, realization_param_names = sample_prior(kwargs_sample_realization)
    source_dict, source_samples, source_param_names = sample_prior(kwargs_sample_source)
    macromodel_samples_fixed_dict, samples_macromodel_fixed, param_names_macro_fixed = sample_prior(kwargs_sample_macro_fixed)

    if fixed_realization is not None:
        if verbose: print('using a precomputed dark matter realization')
        realization_init = fixed_realization
        preset_realization = True
    else:
        present_model_function = preset_model_from_name(preset_model_name)
        if 'cone_opening_angle_arcsec' not in realization_dict.keys():
            theta_E = model_class.setup_lens_model()[-1][0][0]['theta_E']
            realization_dict['cone_opening_angle_arcsec'] = max(6.0 * theta_E, 6.0)
        realization_init = present_model_function(z_lens, data_class.z_source, **realization_dict)
        preset_realization = False
    if verbose:
        print('random seed: ', seed)
        print('SOURCE PARAMETERS: ')
        print(source_dict)
        print('REALIZATION PARAMETERS: ')
        print(realization_dict)
        print('FIXED MACROMODEL SAMPLES: ')
        print(macromodel_samples_fixed_dict)

    kwargs_lens_macro_init = None
    astropy_cosmo = realization_init.lens_cosmo.cosmo.astropy
    kwargs_model_align, _, _, _ = model_class.setup_kwargs_model(
        decoupled_multiplane=False,
        kwargs_lens_macro_init=kwargs_lens_macro_init,
        macromodel_samples_fixed=macromodel_samples_fixed_dict,
        astropy_cosmo=astropy_cosmo)
    kwargs_params = model_class.kwargs_params(kwargs_lens_macro_init=kwargs_lens_macro_init,
                                              delta_x_image=-delta_x_image,
                                              delta_y_image=-delta_y_image,
                                              macromodel_samples_fixed=macromodel_samples_fixed_dict)
    pixel_size = data_class.coordinate_properties[0] / data_class.kwargs_numerics['supersampling_factor']
    kwargs_lens_align = kwargs_params['lens_model'][0]
    if preset_realization:
        realization = realization_init
    else:
        if verbose:
            print('realization has ' + str(len(realization_init.halos)) + ' halos')
        if log10_bound_mass_cut is not None:
            realization_init = realization_init.filter_bound_mass(10 ** log10_bound_mass_cut)
            if verbose:
                print('realization has ' + str(len(realization_init.halos)) + ' halos after cut on '
                             'bound mass above 10^'+str(log10_bound_mass_cut))
        realization, _, _, lens_model_align, _ = align_realization(realization_init, kwargs_model_align['lens_model_list'],
                                    kwargs_model_align['lens_redshift_list'],
                                    kwargs_lens_align,
                                    data_class.x_image,
                                    data_class.y_image,
                                    astropy_cosmo)
    lens_model_list_halos, redshift_list_halos, kwargs_halos, _ = realization.lensing_quantities(
        kwargs_mass_sheet={'log_mlow_sheets': log_mlow_mass_sheets, 'kappa_scale_subhalos': kappa_scale_subhalos})
    grid_resolution_image_data = pixel_size * image_data_grid_resolution_rescale
    astropy_cosmo = realization.lens_cosmo.cosmo.astropy
    kwargs_model, lens_model_init, kwargs_lens_init, index_lens_split = model_class.setup_kwargs_model(
        decoupled_multiplane=use_decoupled_multiplane_approximation,
        lens_model_list_halos=lens_model_list_halos,
        kwargs_lens_macro_init=kwargs_lens_macro_init,
        grid_resolution=grid_resolution_image_data,
        redshift_list_halos=list(redshift_list_halos),
        kwargs_halos=kwargs_halos,
        verbose=verbose,
        macromodel_samples_fixed=macromodel_samples_fixed_dict,
        astropy_cosmo=astropy_cosmo)
    kwargs_constraints = model_class.kwargs_constraints
    kwargs_likelihood = model_class.kwargs_likelihood
    kwargs_params = split_kwargs_params(kwargs_params, index_lens_split)

    if astrometric_uncertainty:
        kwargs_constraints['point_source_offset'] = True
    else:
        kwargs_constraints['point_source_offset'] = False

    if use_imaging_data and split_image_data_reconstruction is False:
        if verbose:
            print('running fitting sequence...')
            t0 = time()
        #if index_lens_split has a different length from lens_model_list,
        #it means we have a macromodel with deflectors at different redshifts
        if len(index_lens_split) != len(kwargs_model['lens_model_list']):
            raise Exception('image data reconstruction with decoupled deflectors at multiple '
                            'lens planes is not yet implemented!')
        fitting_sequence = FittingSequence(data_class.kwargs_data_joint,
                                           kwargs_model,
                                           kwargs_constraints,
                                           kwargs_likelihood,
                                           kwargs_params,
                                           mpi=False, verbose=verbose)
        if fitting_kwargs_list is None:
            fitting_kwargs_list = [
                ['PSO', {'sigma_scale': 1., 'n_particles': n_pso_particles, 'n_iterations': n_pso_iterations,
                         'threadCount': num_threads}]
            ]
        chain_list = fitting_sequence.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_sequence.best_fit()
        if verbose:
            print('done in ' + str(time() - t0) + ' seconds')
            likelihood_module = fitting_sequence.likelihoodModule
            print(likelihood_module.log_likelihood(kwargs_result, verbose=True))
        kwargs_solution = kwargs_result['kwargs_lens']
        kwargs_multiplane_model = kwargs_model['kwargs_multiplane_model']

    else:
        param_class_4pointsolver = model_class.param_class_4pointsolver(lens_model_init.lens_model_list,
                                                                        kwargs_lens_init,
                                                                        macromodel_samples_fixed_dict)
        if use_decoupled_multiplane_approximation:
            if param_class_4pointsolver is None:
                param_class = auto_param_class(lens_model_init.lens_model_list,
                                           kwargs_lens_align,
                                           macromodel_samples_fixed_dict)
            else:
                param_class = param_class_4pointsolver
            kwargs_lens_init = kwargs_lens_align + kwargs_lens_init[len(kwargs_lens_align):]
            opt = Optimizer.decoupled_multiplane(data_class.x_image,
                                                 data_class.y_image,
                                                 lens_model_init,
                                                 kwargs_lens_init,
                                                 index_lens_split,
                                                 param_class,
                                                 tol_simplex_func=1e-5,
                                                 simplex_n_iterations=500
                                                 )
            kwargs_solution, _ = opt.optimize(50, 50, verbose=verbose, seed=seed)
            kwargs_multiplane_model = opt.kwargs_multiplane_model
        else:

            kwargs_lens_init = kwargs_lens_align + kwargs_halos
            if param_class_4pointsolver is None:
                param_class = auto_param_class(lens_model_init.lens_model_list,
                                               kwargs_lens_align,
                                               macromodel_samples_fixed_dict)
            else:
                param_class = param_class_4pointsolver
            opt = Optimizer.full_raytracing(data_class.x_image,
                                            data_class.y_image,
                                            lens_model_init.lens_model_list,
                                            lens_model_init.redshift_list,
                                            z_lens,
                                            data_class.z_source,
                                            param_class,
                                            tol_simplex_func=1e-5,
                                            simplex_n_iterations=500,
                                            particle_swarm=True)
            kwargs_solution, _ = opt.optimize(50, 50, verbose=verbose, seed=seed)
            kwargs_multiplane_model = opt.kwargs_multiplane_model

    if use_decoupled_multiplane_approximation:
        lens_model = LensModel(lens_model_list=kwargs_model['lens_model_list'],
                               lens_redshift_list=kwargs_model['lens_redshift_list'],
                               multi_plane=kwargs_model['multi_plane'],
                               decouple_multi_plane=kwargs_model['decouple_multi_plane'],
                               kwargs_multiplane_model=kwargs_multiplane_model,
                               z_source=kwargs_model['z_source'],
                               cosmo=astropy_cosmo)
    else:
        lens_model = LensModel(lens_model_list=lens_model_init.lens_model_list,
                               lens_redshift_list=lens_model_init.redshift_list,
                               multi_plane=True,
                               cosmo=astropy_cosmo,
                               z_source=kwargs_model['z_source'])

    source_x, source_y = lens_model.ray_shooting(data_class.x_image, data_class.y_image,
                                                 kwargs_solution)
    if verbose:
        print('\n')
        print('kwargs solution: ', kwargs_solution)
        print('\n')
        print('computing image magnifications...')
    t0 = time()
    if verbose and use_imaging_data:
        print('recovered source position: ', source_x, source_y)
    source_model_quasar, kwargs_source = setup_gaussian_source(source_dict['source_size_pc'],
                                                               np.mean(source_x), np.mean(source_y),
                                                               astropy_cosmo, data_class.z_source)
    grid_size = rescale_grid_size * auto_raytracing_grid_size(source_dict['source_size_pc'])
    grid_resolution = rescale_grid_resolution * auto_raytracing_grid_resolution(source_dict['source_size_pc'])
    magnifications, images = model_class.image_magnification_gaussian(source_model_quasar,
                                                                      kwargs_source,
                                                                      lens_model_init,
                                                                      kwargs_lens_init,
                                                                      kwargs_solution,
                                                                      grid_size,
                                                                      grid_resolution,
                                                                      lens_model,
                                                                      elliptical_ray_tracing_grid)
    tend = time()
    if verbose:
        print('computed magnifications in '+str(np.round(tend - t0, 1))+' seconds')
        print('magnifications: ', magnifications)
        print(kwargs_solution)
        print('\n')
        print(macromodel_samples_fixed_dict)
    if macromodel_readout_function is not None:
        samples_macromodel, param_names_macro = macromodel_readout_function(kwargs_solution,
                                                                            macromodel_samples_fixed_dict)
    else:
        param_names_macro = []
        samples_macromodel = []
        if use_decoupled_multiplane_approximation:
            j_max = len(index_lens_split)
        else:
            j_max = len(kwargs_model_align)
        for lm in kwargs_solution[0:j_max]:
            for key in lm.keys():
                samples_macromodel.append(lm[key])
                param_names_macro.append(key)
        if use_decoupled_multiplane_approximation:
            for fixed_param in ['satellite_1_theta_E', 'satellite_1_x', 'satellite_1_y',
                                'satellite_2_theta_E', 'satellite_2_x', 'satellite_2_y']:
                if fixed_param in macromodel_samples_fixed_dict.keys():
                    samples_macromodel.append(macromodel_samples_fixed_dict[fixed_param])
                    param_names_macro.append(fixed_param)
        samples_macromodel = np.array(samples_macromodel)

    if use_imaging_data:
        bic = fitting_sequence.bic
        image_model = create_im_sim(data_class.kwargs_data_joint['multi_band_list'],
                                    data_class.kwargs_data_joint['multi_band_type'],
                                    kwargs_model,
                                    bands_compute=None,
                                    image_likelihood_mask_list=[data_class.likelihood_mask_imaging_weights],
                                    band_index=0,
                                    kwargs_pixelbased=None,
                                    linear_solver=True)
        logL_imaging_data = image_model.likelihood_data_given_model(kwargs_result['kwargs_lens'],
                kwargs_result['kwargs_source'],
                kwargs_result['kwargs_lens_light'],
                kwargs_result['kwargs_ps'],
                kwargs_extinction=kwargs_result['kwargs_extinction'],
                kwargs_special=kwargs_result['kwargs_special'],
                source_marg=False,
                linear_prior=None,
                check_positive_flux=False)[0]
        if verbose:
            logL_imaging_data_no_custom_mask = fitting_sequence.likelihoodModule.image_likelihood.logL(**kwargs_result)[
                0]
            print('imaging data likelihood (without custom mask): ', logL_imaging_data_no_custom_mask)
            print('imaging data likelihood (with custom mask): ', logL_imaging_data)
    else:

        if use_decoupled_multiplane_approximation:
            lens_model = LensModel(lens_model_list=kwargs_model['lens_model_list'],
                                   lens_redshift_list=kwargs_model['lens_redshift_list'],
                                   multi_plane=kwargs_model['multi_plane'],
                                   cosmo=astropy_cosmo,
                                   decouple_multi_plane=kwargs_model['decouple_multi_plane'],
                                   kwargs_multiplane_model=kwargs_model['kwargs_multiplane_model'],
                                   z_source=kwargs_model['z_source'])
        else:
            lens_model = LensModel(lens_model_list=lens_model_init.lens_model_list,
                                   lens_redshift_list=lens_model_init.redshift_list,
                                   multi_plane=True,
                                   cosmo=astropy_cosmo,
                                   z_source=kwargs_model['z_source'])

        if split_image_data_reconstruction:
            tabulated_lens_model = FixedLensModelNew(data_class, lens_model, kwargs_solution,
                           image_data_grid_resolution_rescale / data_class.kwargs_numerics['supersampling_factor'])
            kwargs_model_lightfit = model_class.setup_kwargs_model(decoupled_multiplane=False)[0]
            kwargs_model_lightfit['lens_model_list'] = ['TABULATED_DEFLECTIONS']
            kwargs_model_lightfit['multi_plane'] = False
            kwargs_constraints_light_fit = {'num_point_source_list': [len(data_class.x_image)],
                                  'point_source_offset': True,
                                  #'joint_source_with_point_source': [[0, 0]]
                                  }
            kwargs_likelihood_lightfit = deepcopy(kwargs_likelihood)
            kwargs_likelihood_lightfit['prior_lens'] = None
            kwargs_likelihood_lightfit['custom_logL_addition'] = None
            kwargs_model_lightfit['tabulated_deflection_angles'] = tabulated_lens_model
            kwargs_model_lightfit['point_source_model_list'] = ['UNLENSED']
            kwargs_params_lightfit = setup_params_light_fitting(kwargs_params, np.mean(source_x), np.mean(source_y))
            fitting_sequence = FittingSequence(data_class.kwargs_data_joint,
                                               kwargs_model_lightfit,
                                               kwargs_constraints_light_fit,
                                               kwargs_likelihood_lightfit,
                                               kwargs_params_lightfit,
                                               mpi=False,
                                               verbose=verbose)
            if fitting_kwargs_list is None:
                fitting_kwargs_list = [
                    ['PSO', {'sigma_scale': 1., 'n_particles': 10, 'n_iterations': 100,
                             'threadCount': num_threads}]
                ]
            chain_list = fitting_sequence.fit_sequence(fitting_kwargs_list)
            kwargs_result = fitting_sequence.best_fit()
            if verbose:
                print('result of light fitting: ', kwargs_result)
            bic = fitting_sequence.bic
            image_model = create_im_sim(data_class.kwargs_data_joint['multi_band_list'],
                                        data_class.kwargs_data_joint['multi_band_type'],
                                        kwargs_model_lightfit,
                                        bands_compute=None,
                                        image_likelihood_mask_list=[data_class.likelihood_mask_imaging_weights],
                                        band_index=0,
                                        kwargs_pixelbased=None,
                                        linear_solver=True)

            logL_imaging_data = image_model.likelihood_data_given_model(kwargs_result['kwargs_lens'],
                                                                        kwargs_result['kwargs_source'],
                                                                        kwargs_result['kwargs_lens_light'],
                                                                        kwargs_result['kwargs_ps'],
                                                                        kwargs_extinction=kwargs_result[
                                                                            'kwargs_extinction'],
                                                                        kwargs_special=kwargs_result['kwargs_special'],
                                                                        source_marg=False,
                                                                        linear_prior=None,
                                                                        check_positive_flux=False)[0]
            if verbose:
                logL_imaging_data_no_custom_mask = \
                fitting_sequence.likelihoodModule.image_likelihood.logL(**kwargs_result)[
                    0]
                print('imaging data likelihood (without custom mask): ', logL_imaging_data_no_custom_mask)
                print('imaging data likelihood (with custom mask): ', logL_imaging_data)
        else:
            bic = -1000
            logL_imaging_data = -1000

    stat, flux_ratios, flux_ratios_data = flux_ratio_summary_statistic(data_class.magnifications,
                                                                       magnifications,
                                                                       data_class.flux_uncertainty,
                                                                       data_class.keep_flux_ratio_index,
                                                                       data_class.uncertainty_in_fluxes)

    _flux_ratio_likelihood_weight = flux_ratio_likelihood(data_class.magnifications, magnifications,
                                                         data_class.flux_uncertainty, data_class.uncertainty_in_fluxes,
                                                         data_class.keep_flux_ratio_index)
    logl_flux = -np.log(_flux_ratio_likelihood_weight)
    log_flux_ratio_likelihood = min(abs(logl_flux), -100)

    if verbose:
        print('flux ratios data: ', flux_ratios_data)
        print('flux ratios model: ', flux_ratios)
        print('statistic: ', stat)
        print('log flux_ratio likelihood: ', log_flux_ratio_likelihood)
        if use_imaging_data:
            print('BIC: ', bic)

    if use_imaging_data:
        kwargs_model_plot = {'multi_band_list': data_class.kwargs_data_joint['multi_band_list'],
                         'kwargs_model': kwargs_model,
                         'kwargs_params': kwargs_result}
    else:
        fitting_sequence = FittingSequence(data_class.kwargs_data_joint,
                                           kwargs_model,
                                           kwargs_constraints,
                                           kwargs_likelihood,
                                           kwargs_params,
                                           mpi=False, verbose=verbose)
        kwargs_result = fitting_sequence.best_fit()
        kwargs_result['kwargs_lens'] = kwargs_solution
        kwargs_model_plot = {'multi_band_list': data_class.kwargs_data_joint['multi_band_list'],
                             'kwargs_model': kwargs_model,
                             'kwargs_params': kwargs_result}

    if test_mode:

        from lenstronomy.Plots.model_plot import ModelPlot
        from lenstronomy.Plots import chain_plot
        import matplotlib.pyplot as plt
        fig = plt.figure(1)
        fig.set_size_inches(16,8)
        ax1 = plt.subplot(141)
        ax2 = plt.subplot(142)
        ax3 = plt.subplot(143)
        ax4 = plt.subplot(144)
        axes_list = [ax1, ax2, ax3, ax4]
        for mag, ax, image in zip(magnifications, axes_list, images):
            ax.imshow(image, origin='lower')
            ax.annotate('magnification: '+str(np.round(mag,2)), xy=(0.3,0.9),
                        xycoords='axes fraction',color='w',fontsize=12)
        plt.show()

        modelPlot = ModelPlot(data_class.kwargs_data_joint['multi_band_list'],
                              kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",
                              fast_caustic=True,
                              image_likelihood_mask_list=[data_class.likelihood_mask_imaging_weights])
        if use_imaging_data:
            chain_plot.plot_chain_list(chain_list, 0)
            print('num degrees of freedom: ', fitting_sequence.likelihoodModule.effective_num_data_points(**kwargs_result))

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
                       'index_macromodel': list(np.arange(0, len(kwargs_result['kwargs_lens']))),
                       'with_critical_curves': True,
                       'v_min': -0.1, 'v_max': 0.1,
                       'super_sample_factor': 5}
        modelPlot.substructure_plot(band_index=0, **kwargs_plot)
        plt.show()
        a=input('continue?')

    if data_class.redshift_sampling:
        realization_samples = np.append(realization_samples, z_lens)
        realization_param_names += ['z_lens']

    return magnifications, images, realization_samples, source_samples, samples_macromodel, samples_macromodel_fixed, \
           logL_imaging_data, fitting_sequence, \
           stat, log_flux_ratio_likelihood, bic, realization_param_names, \
           source_param_names, param_names_macro, \
           param_names_macro_fixed, kwargs_model_plot, lens_model, kwargs_solution

from pyHalo.preset_models import preset_model_from_name
from samana.forward_model_util import filenames, sample_prior, align_realization, \
    flux_ratio_summary_statistic, split_kwargs_params, check_lens_equation_solution
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_resolution, auto_raytracing_grid_size
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Util.class_creator import create_im_sim
from lenstronomy.LensModel.QuadOptimizer.optimizer import Optimizer
from samana.image_magnification_util import setup_gaussian_source
from samana.param_managers import auto_param_class
from scipy.stats import multivariate_normal
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
                  use_imaging_data=True,
                  fitting_sequence_kwargs=None,
                  test_mode=False,
                  use_decoupled_multiplane_approximation=True,
                  fixed_realization_list=None,
                  kappa_scale_subhalos=1.0,
                  log10_bound_mass_cut=None,
                  parallelize=False,
                  filter_subhalo_kwargs=None,
                  custom_preset_model_function=None,
                  run_initial_PSO=False,
                  scipy_minimize_method='COBYQA_import',
                  use_JAXstronomy=False,
                  split_image_data_reconstruction=False,
                  magnification_method='CIRCULAR_APERTURE',
                  tolerance_source_reconstruction=None,
                  fr_logL_source_reconstruction=None,
                  return_astrometric_rejections=False,
                  background_shifting=True):
    """
    Top-level function for forward modeling strong lenses with substructure. This function makes repeated calls to
    the forward_model_single_iteration routine below, and outputs the results to text files. Lens modeling and dark matter
    calculations take place inside the forward_model_single_iteration function

    :param output_path: directory where output files will be saved, for example ~/OUTPUT_FILES/LENS_NUMBER_1/
    :param job_index: the unique integer identifying output from each CPU (if running inside a job array). Output is
    written to ~/OUTPUT_FILES/LENS_NUMBER_1/job_I where I = job_index
    :param n_keep: the number of samples to produce on each CPU; the code will stop once n_keep lenses are simulated. To
    increase the number of realizations, perform calculations across more CPUs or increase n_keep, both are equivalent
    :param data_class: an instance of a Data class with image positions, imaging data, lens/source redshifts, etc. See
    class in the Data module
    :param model:  an instance of a Model class; see classes in the Model module
    :param preset_model_name: a string identifying a preset model pyHalo
    :param kwargs_sample_realization: a dictionary specifying priors on DM parameters, for example:
    - kwargs_sample_realization = {'sigma_sub': ['UNIFORM', 0.0, 0.1]} is a uniform prior on sigma_sub between 0 and 0.1.
    - kwargs_sample_realization = {'sigma_sub': ['GAUSSIAN', 0.0, 0.1]} is a Gaussian prior on sigma_sub with mean 0 and
    standard deviation 0.1.
    - kwargs_sample_realization = {'sigma_sub': ['FIXED', 0.1]} fixes the value of sigma_sub = 0.1
    :param kwargs_sample_source: a dictionary specifying priors on source parameters, same syntax as kwargs_sample_realization.
    :param kwargs_sample_fixed_macromodel: a dictionary specifying priors on macromodel parameters, same syntax as kwargs_sample_realization
    :param tolerance: tolerance for accepting/rejecting flux ratio summary statistic; np.inf will accept everything
    :param log_mlow_mass_sheets: minimum halo mass used to calculate the amount of convergence to subtract at each lens plane,
    should be the minimum halo mass rendered along the line of sight
    :param kwargs_model_class: keyword arguments to be passed to the Model class; see classes in the Model module
    :param rescale_grid_size: rescales the size of the ray tracing grid for computing image magnifications
    :param rescale_grid_resolution: rescales the resolution of the ray tracing grid for computing image magnifications
    :param readout_macromodel_samples: bool; if True, will create text files with macromodel samples saved
    :param verbose: bool; make print statements while running
    :param random_seed_init: a random seed for reproducibility
    :param readout_steps: determines how frequently to write output to files; recommended to use a number > 2. For example,
    readout_steps=2 will append output to files after every 2 realizations
    :param write_sampling_rate: bool; create text files with the sampling rate (minutes per realization)
    :param n_pso_particles: number of PSO particles to use when modeling imaging data
    :param n_pso_iterations: number of PSO iteraations to use when modeling imaging data
    :param num_threads: number of threads to parallization; WARNING, this appears to mess with the reproducibility/
    random_seed functionality
    :param astrometric_uncertainty: bool; add astrometric uncertainty to image positions before lens modeling
    :param image_data_grid_resolution_rescale: rescales the resoltuion of the image grid for imaging data calculations
    :param use_imaging_data: bool; reconstruct imaging data
    :param fitting_sequence_kwargs: a list specifying the fitting sequence operations done by lenstronomy
    :param test_mode: bool; prints output and makes plots of convergence, image magnifications, reconstructed light, etc
    :param use_decoupled_multiplane_approximation: bool; use the decoupled multiplane approximation as detailed in Gilman et al. 2024
    :param fixed_realization_list: a list of fixed pyHalo realizations to use in the calculations
    :param kappa_scale_subhalos: rescales the negative convergence sheet associated with subhalos; should be approx. equal
    to the amplitude of the bound mass function relative to the infall mass function
    :param log10_bound_mass_cut: remove subhalos with bound masses below 10^log10_bound_mass_cut
    :param parallelize: bool; use parallelization
    :param filter_subhalo_kwargs: keyword arguments passed to pyHalo to remove low-mass subhalos that are far away from images
    :param custom_preset_model_function: a custom preset_model function that can be passed to pyHalo; only used when
    preset_model_name='CUSTOM'
    :param run_initial_PSO: bool; run initial particle swarm optimization when NOT using imaging data
    :param scipy_minimize_method: string that specifies the minimize method used by scipy when solving for point source positions
     without imaging data
    :param use_JAXstronomy: bool; use JAXstronomy deflector profiles where available
    :param split_image_data_reconstruction: bool; if True, reconstructs the imaging data only for systems for which
    the flux ratios match the data better than a specified tolerance threshold
    :param magnification_method: the algorithm for computing the image magnification, options include:
     - CIRCULAR_APERTURE: start with a circular aperture centered at image position, gradually increase size until
     magnification converges to 0.1%
     - ELLIPTICAL_APERTURE: start with an elliptical aperture centered at image position, gradually increase size until
     magnification converges to 0.1%. The orientation and size of ellipse is estimated from the hessian eigenvectors at
     the image position
     - ADAPTIVE: estimates the shape of the image from a low-resolution calculation, then performs a high-resolution
     ray-tracing calculation around pixels identified in the low-resolution calculation
    :param tolerance_source_reconstruction: the tolerance on the summary statistic that triggers the reconstruction of
    the source and lens light when split_image_data_reconstruction=True
    :param fr_logL_source_reconstruction: the same functionality as tolerance source_reconstruction, but triggers
    the reconstruction of the imaging data based on the flux ratio likelihood. If specified, fr_logL_source_reconstruction
    should be abs(log_likelihood), which triggers the source light modeling if abs(logL) < fr_logL_source_reconstruction
    :param return_astrometric_rejections: if True, will return the macromodel parameters that produced a lens model that
    doesn't fit the image positions; if False, these solutions will be rejected and not saved as output
    :param background_shifting: toggles the shifting of background halos to align with the direction to the source
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
    acceptance_ratio = np.nan
    sampling_rate = np.nan
    t0 = time()

    if random_seed_init is None:
        # pick a random integer from which to generate random seeds
        random_seed_init = np.random.randint(0, 4294967295)
    elif isinstance(random_seed_init, list) or isinstance(random_seed_init, np.ndarray):
        if n_keep != len(random_seed_init):
            print('setting n_keep = '+str(len(random_seed_init)))
        n_keep = len(random_seed_init)
    if n_keep < readout_sampling_rate_index:
        readout_sampling_rate_index = deepcopy(n_keep)
    if verbose:
        print('starting with ' + str(n_kept) + ' samples accepted, ' + str(n_keep - n_kept) + ' remain')
        print('existing magnifications: ', _m)
        print('samples remaining: ', n_keep - n_kept)
        print('running simulation with a summary statistic tolerance of: ', tolerance)
    # start the simulation, the while loop will execute until one has obtained n_keep samples from the posterior
    seed_counter = 0 + n_kept
    macromodel_readout_function = None
    return_realization = False
    while True:

        if isinstance(random_seed_init, list) or isinstance(random_seed_init, np.ndarray):
            if verbose:
                print('seed counter: ', seed_counter)
                print('random seed array: ', random_seed_init)
                print('running with random seed: ', int(random_seed_init[seed_counter]))
            random_seed = int(random_seed_init[seed_counter])
        else:
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
                scale_window_size_decoupled_multiplane = 1
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
                             kappa_scale_subhalos,
                             log10_bound_mass_cut,
                             filter_subhalo_kwargs,
                             macromodel_readout_function,
                             return_realization,
                             custom_preset_model_function,
                             run_initial_PSO,
                             scipy_minimize_method,
                             use_JAXstronomy,
                             split_image_data_reconstruction,
                             magnification_method,
                             tolerance_source_reconstruction,
                             fr_logL_source_reconstruction,
                             scale_window_size_decoupled_multiplane,
                             return_astrometric_rejections,
                             background_shifting))

            pool = Pool(num_threads)
            output = pool.starmap(forward_model_single_iteration, args)
            pool.close()
            for _, result in enumerate(output):
                (magnifications, images, realization_samples, source_samples, macromodel_samples,
                macromodel_samples_fixed, \
                logL_imaging_data, fitting_sequence, stat, bic, param_names_realization,
                param_names_source, param_names_macro, \
                param_names_macro_fixed, _, _, _, source_plane_image_solution) = result
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
                if magnifications is not None and stat < tolerance:
                    # If the statistic is less than the tolerance threshold, we keep the parameters
                    accepted_realizations_counter += 1
                    n_kept += 1
                    params = np.append(realization_samples, source_samples)
                    params = np.append(params, bic)
                    params = np.append(params, stat)
                    params = np.append(params, logL_imaging_data)
                    params = np.append(params, source_plane_image_solution)
                    params = np.append(params, random_seed + seed_counter)
                    param_names = param_names_realization + param_names_source + ['bic', 'summary_statistic',
                                                                                  'logL_image_data', 'source_plane_sol', 'seed']
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
            scale_window_size_decoupled_multiplane = 1
            magnifications, images, realization_samples, source_samples, macromodel_samples, macromodel_samples_fixed, \
            logL_imaging_data, fitting_sequence, stat, bic, param_names_realization, param_names_source, param_names_macro, \
            param_names_macro_fixed, _, _, _, source_plane_image_solution = forward_model_single_iteration(data_class, model, preset_model_name, kwargs_sample_realization,
                                                kwargs_sample_source, kwargs_sample_fixed_macromodel, log_mlow_mass_sheets,
                                                rescale_grid_size, rescale_grid_resolution, image_data_grid_resolution_rescale,
                                                verbose, random_seed, n_pso_particles, n_pso_iterations, num_threads,
                                                kwargs_model_class, astrometric_uncertainty,
                                                use_imaging_data, fitting_sequence_kwargs, test_mode,
                                                use_decoupled_multiplane_approximation, fixed_realization,
                                                kappa_scale_subhalos, log10_bound_mass_cut,
                                                filter_subhalo_kwargs,
                                                macromodel_readout_function,
                                                return_realization,
                                                custom_preset_model_function,
                                                run_initial_PSO,
                                                scipy_minimize_method,
                                                use_JAXstronomy,
                                                split_image_data_reconstruction,
                                                magnification_method,
                                                tolerance_source_reconstruction,
                                                fr_logL_source_reconstruction,
                                                scale_window_size_decoupled_multiplane,
                                                return_astrometric_rejections,
                                                background_shifting)

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
            if magnifications is not None and stat < tolerance:
                # If the statistic is less than the tolerance threshold, we keep the parameters
                accepted_realizations_counter += 1
                n_kept += 1
                params = np.append(realization_samples, source_samples)
                params = np.append(params, bic)
                params = np.append(params, stat)
                params = np.append(params, logL_imaging_data)
                params = np.append(params, source_plane_image_solution)
                params = np.append(params, random_seed)
                param_names = param_names_realization + param_names_source + ['bic', 'summary_statistic',
                                                                              'logL_image_data', 'source_plane_sol','seed']
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
            print('accepted realizations counter: ', accepted_realizations_counter)
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
                           kappa_scale_subhalos=1.0,
                           log10_bound_mass_cut=None,
                           filter_subhalo_kwargs=None,
                           macromodel_readout_function=None,
                           return_realization=False,
                           custom_preset_model_function=None,
                           run_initial_PSO=True,
                           minimize_method='Nelder-Mead',
                           use_JAXstronomy=False,
                           split_image_data_reconstruction=False,
                           magnification_method=None,
                           tolerance_source_reconstruction=None,
                           fr_logL_source_reconstruction=None,
                           scale_window_size_decoupled_multiplane=1.0,
                           return_astrometric_rejections=False,
                           background_shifting=True,
                           log_mhigh_mass_sheets=10.7):
    """

    :param data_class:
    :param model:
    :param preset_model_name:
    :param kwargs_sample_realization:
    :param kwargs_sample_source:
    :param kwargs_sample_macro_fixed:
    :param log_mlow_mass_sheets:
    :param rescale_grid_size:
    :param rescale_grid_resolution:
    :param image_data_grid_resolution_rescale:
    :param verbose:
    :param seed:
    :param n_pso_particles:
    :param n_pso_iterations:
    :param num_threads:
    :param kwargs_model_class:
    :param astrometric_uncertainty:
    :param use_imaging_data:
    :param fitting_kwargs_list:
    :param test_mode:
    :param use_decoupled_multiplane_approximation:
    :param fixed_realization:
    :param kappa_scale_subhalos:
    :param log10_bound_mass_cut:
    :param filter_subhalo_kwargs:
    :param macromodel_readout_function:
    :param return_realization:
    :param custom_preset_model_function:
    :param run_initial_PSO:
    :param minimize_method:
    :param use_JAXstronomy:
    :param split_image_data_reconstruction:
    :param magnification_method:
    :param tolerance_source_reconstruction:
    :param fr_logL_source_reconstruction:
    :param scale_window_size_decoupled_multiplane:
    :param return_astrometric_rejections:
    :return:
    """
    # set the random seed for reproducibility
    np.random.seed(seed)
    if split_image_data_reconstruction and use_imaging_data:
        raise Exception('cannot use split_image_data_reconstruction=True when use_imaging_data=True. The methodology '
                        'triggered by split_image_data_reconstruction=True reconstructs the source for lens models '
                        'that fit the flux ratios, but the initial lens modeling should be done without imaging data'
                        '(use_imaging_data = False)')
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
        present_model_function = preset_model_from_name(preset_model_name,
                                                        custom_function=custom_preset_model_function)
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

    astropy_cosmo = realization_init.lens_cosmo.cosmo.astropy
    # generate a macromodel that satisfies the lens equation for the perturbed image positions
    kwargs_model_align, _, kwargs_lens_macro_init, _, _ = model_class.setup_kwargs_model(
        decoupled_multiplane=False,
        kwargs_lens_macro_init=None,
        macromodel_samples_fixed=macromodel_samples_fixed_dict,
        astropy_cosmo=astropy_cosmo,
        x_image=data_class.x_image,
        y_image=data_class.y_image,
        verbose=verbose)
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
            print('realization has ' + str(len(realization_init.halos)) + ' halos...')
        if background_shifting:
            # shift halos such that they are symmetric around the center of the lensing volume
            realization, ray_align_x, ray_align_y, _, _ = align_realization(realization_init,
                                                        kwargs_model_align['lens_model_list'],
                                                        kwargs_model_align['lens_redshift_list'],
                                                        kwargs_lens_align,
                                                        data_class.x_image,
                                                        data_class.y_image,
                                                        astropy_cosmo)
        else:
            realization = realization_init
        if filter_subhalo_kwargs is not None:
            realization = realization.filter_subhalos(**filter_subhalo_kwargs)
            if verbose:
                print('realization has ' + str(len(realization.halos)) + ' halos after '
                            'downselecting on subhalo mass/position...')
        if log10_bound_mass_cut is not None:
            realization = realization.filter_bound_mass(10 ** log10_bound_mass_cut)
            if verbose:
                print('realization has ' + str(len(realization.halos)) + ' halos after cut on '
                             'bound mass above 10^'+str(log10_bound_mass_cut)+'... ')
    if return_realization:
        return realization
    lens_model_list_halos, redshift_list_halos, kwargs_halos, _ = realization.lensing_quantities(
        kwargs_mass_sheet={'log_mlow_sheets': log_mlow_mass_sheets,
                           'log_mhigh_sheets': log_mhigh_mass_sheets,
                           'kappa_scale_subhalos': kappa_scale_subhalos})
    astropy_cosmo = realization.lens_cosmo.cosmo.astropy
    grid_resolution_image_data = pixel_size / image_data_grid_resolution_rescale
    if use_imaging_data:
        decoupled_multiplane_grid_type = 'GRID'
    else:
        decoupled_multiplane_grid_type = 'POINT'
    kwargs_model, lens_model_init, kwargs_lens_init, index_lens_split, setup_decoupled_multiplane_lens_model_output = (
        model_class.setup_kwargs_model(
            decoupled_multiplane=use_decoupled_multiplane_approximation,
            lens_model_list_halos=lens_model_list_halos,
            kwargs_lens_macro_init=kwargs_lens_macro_init,
            grid_resolution=grid_resolution_image_data,
            redshift_list_halos=list(redshift_list_halos),
            kwargs_halos=kwargs_halos,
            verbose=verbose,
            macromodel_samples_fixed=macromodel_samples_fixed_dict,
            astropy_cosmo=astropy_cosmo,
            use_JAXstronomy=use_JAXstronomy,
            decoupled_multiplane_grid_type=decoupled_multiplane_grid_type,
            scale_window_size=scale_window_size_decoupled_multiplane
        ))
    if 'q' in param_names_macro_fixed and use_imaging_data:
        model_class.set_fixed_q(macromodel_samples_fixed_dict['q'])
    kwargs_constraints = model_class.kwargs_constraints
    kwargs_likelihood = model_class.kwargs_likelihood
    kwargs_params = split_kwargs_params(kwargs_params, index_lens_split)
    if astrometric_uncertainty:
        kwargs_constraints['point_source_offset'] = True
    else:
        kwargs_constraints['point_source_offset'] = False

    if use_imaging_data:
        image_data_grids_computed = True
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
        image_data_grids_computed = False
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
            # we use the macromodel parameters that satisfy the lens equation to set up the decopuled multiplane approx.
            # inside the class
            kwargs_lens_init = kwargs_lens_align + kwargs_lens_init[len(kwargs_lens_align):]
            opt = Optimizer.decoupled_multiplane(data_class.x_image,
                                                 data_class.y_image,
                                                 lens_model_init,
                                                 kwargs_lens_init,
                                                 index_lens_split,
                                                 param_class,
                                                 particle_swarm=run_initial_PSO,
                                                 tol_simplex_func=1e-9,
                                                 simplex_n_iterations=800,
                                                 tol_source=1e-7,
                                                 )
            if minimize_method == 'FSOLVE':
                if verbose:
                    print('using root finding with FSOLVE... ')
                from lenstronomy.LensModel.Solver.solver import Solver4Point
                from samana.param_managers import FixedAxisRatioSolver
                param_class_fixed_q = FixedAxisRatioSolver(macromodel_samples_fixed_dict['q'])
                solver = Solver4Point(opt.ray_shooting_class,
                                      solver_type="CUSTOM",
                                      parameter_module=param_class_fixed_q)
                kwargs_solution, _ = solver.constraint_lensmodel(data_class.x_image,
                                                              data_class.y_image,
                                                              opt._param_class.kwargs_lens)

            else:
                if minimize_method == 'COBYQA_import':
                    method_name = 'COBYQA'
                    try:
                        from cobyqa import minimize as minimize_method
                    except:
                        raise Exception('cobyqa not installed, install cobyqa first: "pip install cobyqa"')
                else:
                    method_name = minimize_method
                if verbose:
                    print('using optimization routine '+str(method_name))
                kwargs_solution, _ = opt.optimize(50, 50, verbose=verbose, seed=seed,
                                                      minimize_method=minimize_method)
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
                                            tol_simplex_func=1e-6,
                                            simplex_n_iterations=600,
                                            particle_swarm=run_initial_PSO)
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

    if macromodel_readout_function is None:
        macromodel_readout_function = model_class.macromodel_readout_function
    samples_macromodel, param_names_macro = macromodel_readout_function(kwargs_solution,
                                       macromodel_samples_fixed_dict)
    source_x, source_y = lens_model.ray_shooting(data_class.x_image,
                                                 data_class.y_image,
                                                 kwargs_solution)
    if verbose:
        print('\n')
        print('kwargs solution: ', kwargs_solution)
    t0 = time()
    if verbose and use_imaging_data:
        print('recovered source position: ', source_x, source_y)
    # verify that the lens equation is satisfied to high precision
    source_plane_image_solution = check_lens_equation_solution(source_x, source_y, tolerance=0.0001)
    output_vector_none = [None] * 18
    return_sampling_distribution = False
    if return_astrometric_rejections or return_sampling_distribution:
        if source_plane_image_solution > 1 or return_sampling_distribution:
            magnifications = np.array([1.0] * 4)
            images = None
            fitting_sequence = None
            logL_imaging_data = 1
            stat = -1
            bic = 1
            kwargs_model_plot = {}
            output_vector = (magnifications, images, realization_samples, source_samples, samples_macromodel,
                             samples_macromodel_fixed, \
                             logL_imaging_data, fitting_sequence, \
                             stat, bic, realization_param_names, \
                             source_param_names, param_names_macro, \
                             param_names_macro_fixed, kwargs_model_plot, lens_model, kwargs_solution,
                             source_plane_image_solution)
            return output_vector
        else:
            return output_vector_none
    if source_plane_image_solution > 1:
        # reject this lens model on the basis of not satisfying lens equation
        if verbose:
            print('rejecting lens model on the basis of not satisfying the lens equation')
        return output_vector_none
    else:
        if verbose:
            print('computing image magnifications...')
        source_model_quasar, kwargs_source = setup_gaussian_source(source_dict['source_size_pc'],
                                                                   np.mean(source_x), np.mean(source_y),
                                                                   astropy_cosmo, data_class.z_source)
        grid_size_base = auto_raytracing_grid_size(source_dict['source_size_pc'])
        grid_resolution = rescale_grid_resolution * auto_raytracing_grid_resolution(source_dict['source_size_pc'])
        if isinstance(rescale_grid_size, list) or isinstance(rescale_grid_size, np.ndarray):
            assert len(rescale_grid_size) == len(data_class.x_image)
            grid_size_list = []
            for rescale_size in rescale_grid_size:
                grid_size_list.append(rescale_size * grid_size_base)
        else:
            grid_size_list = [rescale_grid_size * grid_size_base] * len(data_class.x_image)
        # we pass in setup_decoupled_multiplane_lens_model_output, the decoupled multiplane parameters
        # computed for the proposed macromodel in setup_kwargs_model
        magnifications, images = model_class.image_magnification_gaussian(source_model_quasar,
                                                                              kwargs_source,
                                                                              lens_model_init,
                                                                              kwargs_lens_init,
                                                                              kwargs_solution,
                                                                              grid_size_list,
                                                                              grid_resolution,
                                                                              lens_model,
                                                                              setup_decoupled_multiplane_lens_model_output,
                                                                              magnification_method=magnification_method)
        flux_uncertainty = None
        stat, flux_ratios, flux_ratios_data = flux_ratio_summary_statistic(data_class.magnifications,
                                                                               magnifications,
                                                                                flux_uncertainty,
                                                                               data_class.keep_flux_ratio_index,
                                                                               data_class.uncertainty_in_fluxes)

    tend = time()
    if verbose:
        print('computed magnifications in '+str(np.round(tend - t0, 1))+' seconds')
        print('magnifications: ', magnifications)
        print('flux ratios data: ', np.array(data_class.magnifications)[1:] / data_class.magnifications[0])
        print('flux ratios model: ', magnifications[1:] / magnifications[0])
        print('statistic: ', stat)
        print(kwargs_solution)
        print('\n')
        print(macromodel_samples_fixed_dict)

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
        if tolerance_source_reconstruction is not None:
            assert fr_logL_source_reconstruction is None, ('If tolerance_source_reconstruction is specified, '
                                                        'then fr_logL_source_reconstruction must not also be'
                                                           'specified')
            if verbose and split_image_data_reconstruction: print('triggering image data modeling with a flux ratio summary statistic tolerance of '
                              +str(tolerance_source_reconstruction))
            if stat < tolerance_source_reconstruction:
                reconstruct_image_data = True
            else:
                reconstruct_image_data = False
        elif fr_logL_source_reconstruction is not None and fr_logL_source_reconstruction > 0:
            if verbose: print('triggering image data modeling with a flux ratio log-likelihood tolerance of '
                              +str(fr_logL_source_reconstruction))
            _flux_ratio_logL = multivariate_normal.logpdf(np.array(flux_ratios),
                                   mean=np.array(flux_ratios_data),
                                   cov=data_class.flux_ratio_covariance_matrix)
            _flux_ratio_logL_norm = multivariate_normal.logpdf(np.array(flux_ratios_data),
                                   mean=np.array(flux_ratios_data),
                                   cov=data_class.flux_ratio_covariance_matrix)
            flux_ratio_logL = _flux_ratio_logL - _flux_ratio_logL_norm
            if verbose: print('flux ratio logL: ', flux_ratio_logL)
            if flux_ratio_logL > -1*fr_logL_source_reconstruction:
                reconstruct_image_data = True
            else:
                reconstruct_image_data = False
        else:
            reconstruct_image_data = False
        if split_image_data_reconstruction and reconstruct_image_data:
            image_data_grids_computed = True
            kwargs_model, lens_model_init, kwargs_lens_init, index_lens_split, setup_decoupled_multiplane_lens_model_output = (
                model_class.setup_kwargs_model(
                    decoupled_multiplane=use_decoupled_multiplane_approximation,
                    lens_model_list_halos=lens_model_list_halos,
                    kwargs_lens_macro_init=kwargs_lens_macro_init,
                    grid_resolution=grid_resolution_image_data,
                    redshift_list_halos=list(redshift_list_halos),
                    kwargs_halos=kwargs_halos,
                    verbose=verbose,
                    macromodel_samples_fixed=macromodel_samples_fixed_dict,
                    astropy_cosmo=astropy_cosmo,
                    x_image=data_class.x_image,
                    y_image=data_class.y_image,
                    use_JAXstronomy=use_JAXstronomy,
                    decoupled_multiplane_grid_type='GRID',
                    scale_window_size=scale_window_size_decoupled_multiplane
                ))
            kwargs_params = model_class.kwargs_params(kwargs_lens_macro_init=kwargs_solution,
                                                      delta_x_image=-delta_x_image,
                                                      delta_y_image=-delta_y_image,
                                                      macromodel_samples_fixed=macromodel_samples_fixed_dict,
                                                      fixed_lens_model=True,
                                                      kwargs_lens_fixed=kwargs_solution)

            kwargs_likelihood['prior_lens'] = None
            kwargs_likelihood['custom_logL_addition'] = None
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
                                                                        kwargs_special=kwargs_result['kwargs_special'],
                                                                        source_marg=False,
                                                                        linear_prior=None,
                                                                        check_positive_flux=False)[0]
            if verbose:
                print('result of light fitting: ', kwargs_result)
                print('logL image data: ', logL_imaging_data)

        else:
            bic = 1
            logL_imaging_data = 1

    if verbose:
        if use_imaging_data or split_image_data_reconstruction:
            print('BIC: ', bic)

    if use_imaging_data:
        kwargs_model_plot = {'multi_band_list': data_class.kwargs_data_joint['multi_band_list'],
                             'kwargs_model': kwargs_model,
                             'kwargs_params': kwargs_result}
    elif split_image_data_reconstruction and reconstruct_image_data:
        kwargs_model_plot = {'multi_band_list': data_class.kwargs_data_joint['multi_band_list'],
                             'kwargs_model': kwargs_model,
                             'kwargs_params': kwargs_result}
    else:
        if image_data_grids_computed is False and test_mode:
            kwargs_model, lens_model_init, kwargs_lens_init, index_lens_split, setup_decoupled_multiplane_lens_model_output = (
                model_class.setup_kwargs_model(
                    decoupled_multiplane=use_decoupled_multiplane_approximation,
                    lens_model_list_halos=lens_model_list_halos,
                    kwargs_lens_macro_init=kwargs_lens_macro_init,
                    grid_resolution=grid_resolution_image_data,
                    redshift_list_halos=list(redshift_list_halos),
                    kwargs_halos=kwargs_halos,
                    verbose=verbose,
                    macromodel_samples_fixed=macromodel_samples_fixed_dict,
                    astropy_cosmo=astropy_cosmo,
                    x_image=data_class.x_image,
                    y_image=data_class.y_image,
                    use_JAXstronomy=use_JAXstronomy,
                    decoupled_multiplane_grid_type='GRID'
                ))
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
        try:
            modelPlot.decomposition_plot(ax=axes[1, 2], text='All components convolved', source_add=True,
                                     lens_light_add=True, point_source_add=True)
        except:
            print('failed to create decomposition plot')
        f.tight_layout()
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        plt.show()

        fig = plt.figure()
        fig.set_size_inches(6, 6)
        ax = plt.subplot(111)
        kwargs_plot = {'ax': ax,
                       'index_macromodel': list(np.arange(0, len(kwargs_result['kwargs_lens']))),
                       'with_critical_curves': True,
                       'v_min': -0.075, 'v_max': 0.075,
                       'super_sample_factor': 5,
                       'subtract_mean': True}
        modelPlot.substructure_plot(band_index=0, **kwargs_plot)
        plt.show()

        fig = plt.figure()
        fig.set_size_inches(12, 12)
        ax = plt.axes(projection='3d')
        if background_shifting:
            realization.plot(ax,
                             ray_interp_x_list=ray_align_x,
                             ray_interp_y_list=ray_align_y)
        else:
            realization.plot(ax)
        plt.show()
        a=input('continue?')

    if data_class.redshift_sampling:
        realization_samples = np.append(realization_samples, z_lens)
        realization_param_names += ['z_lens']

    if 'HIERARCHICAL_MULTIPOLE_PRIOR' in list(kwargs_sample_macro_fixed.keys()):
        for index, name in enumerate(param_names_macro_fixed):
            if name == 'scale_multipole':
                break
        else:
            raise Exception('you specified HIERARCHICAL_MULTIPOLE_PRIOR but the sampled macrmodel arguments dont contain'
                            'the required keywords')
        realization_samples = np.append(realization_samples,samples_macromodel_fixed[index])
        realization_param_names += ['scale_multipole']
        if verbose:
            print('hierachical multipole scaling: ', realization_samples[-1])
    output_vector = (magnifications, images, realization_samples, source_samples, samples_macromodel, samples_macromodel_fixed, \
           logL_imaging_data, fitting_sequence, \
           stat, bic, realization_param_names, \
           source_param_names, param_names_macro, \
           param_names_macro_fixed, kwargs_model_plot, lens_model, kwargs_solution, source_plane_image_solution)
    return output_vector

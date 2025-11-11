import numpy as np
from samana.forward_model import forward_model
from samana.analysis_util import quick_setup, numerics_setup, default_rendering_area
import os
import sys


"""
This script provides an example for how to model the strong lens system HE0435 with dark matter substructure
"""
def run():
    ######################## LENS SELECTION ########################
    lens_ID = 'HE0435'
    data_class, model_class = quick_setup(lens_ID)
    data_class = data_class()
    rescale_grid_size, rescale_grid_res = numerics_setup(lens_ID)
    cone_opening_angle_arcsec = default_rendering_area(data_class=data_class,
                                                       model_class=model_class)
    fluxes = np.array([1.0, 1.009, 0.928, 0.585])
    keep_flux_ratio_index = [0,1,2]
    data_class.magnifications = fluxes
    data_class.flux_uncertainty = None # keep this as None
    data_class.flux_ratio_covariance_matrix = np.array([[0.0005161032050153507, 1.4092644323619424e-05, 1.5474966187747706e-05],
                                                        [1.4092644323619424e-05, 0.00043293962174831426, 1.243753314694643e-05],
                                                        [1.5474966187747706e-05, 1.243753314694643e-05, 0.00018241299265942914]])
    data_class.keep_flux_ratio_index = keep_flux_ratio_index
    ######################## DARK MATTER PARAMETER PRIOR DEFINITIONS ########################
    import realization_priors_wdm as realization_priors
    preset_model_name = 'WDM' # warm dark matter
    kwargs_globular_cluster = {'log10_mgc_mean': 5.3, # median log10 globular cluster mass
                               'log10_mgc_sigma': 0.6, # standard deviation of log-normal mass function
                               'rendering_radius_arcsec': 0.2, # place GCs around images inside this radius
                               'gc_surface_mass_density': 10 ** 5.6, # surface mass density in solar mass per square kpc
                               'gc_density_profile': 'PTMASS' # model them as point masses
                               }
    kwargs_globular_cluster['center_x'] = data_class.x_image
    kwargs_globular_cluster['center_y'] = data_class.y_image
    kwargs_sample_realization = {'log10_sigma_sub': ['UNIFORM', -2.2, 0.2], # subhalo mass function normalization
                                 'log_mlow': ['FIXED', 6.0], # minimum halo mass
                                 'log_mhigh': ['FIXED', 10.7], # largest halo mass
                                 'log_mc': ['FIXED', 4, 10], # half-mode mass
                                 'LOS_normalization': ['UNIFORM', 0.9, 1.1], # amplitude of LOS mass function relative to Sheth-Tormen
                                 'shmf_log_slope': ['UNIFORM', -1.95, -1.85], # slope of subhalo mass function
                                 'log_m_host': ['GAUSSIAN', 13.3, 0.3], # host halo mass
                                 'truncation_model_subhalos': ['FIXED', 'TRUNCATION_GALACTICUS'], # tidal evolution by Du et al. (2025)
                                 'add_globular_clusters': ['FIXED', True], # include GCs
                                 'subhalo_spatial_distribution': ['FIXED', 'UNIFORM'] # uniform spatial distribution for subhalos
                                 }
    kwargs_sample_realization['kwargs_globular_clusters'] = ['FIXED', kwargs_globular_cluster]
    kwargs_sample_realization['cone_opening_angle_arcsec'] = ['FIXED', cone_opening_angle_arcsec]
    filter_subhalo_kwargs = {'aperture_radius_arcsec': 0.25, # only place low-mass subhalos within this distance of image
                             'log10_mass_minimum': 7.0} # don't render subhalos less massive than this everywhere
    filter_subhalo_kwargs['x_coords'] = data_class.x_image
    filter_subhalo_kwargs['y_coords'] = data_class.y_image
    kappa_scale_subhalos = 0.05 # scale the negative convergence sheet correction for subhalos according to median bound mass fraction 0.05
    log10_bound_mass_cut = 4.0 # remove subhalos with bound masses less than this
    ######################## MACROMODEL PRIOR ########################
    kwargs_sample_macro_fixed = {
        'OPTICAL_MULTIPOLE_PRIOR_M1': [], # include m=1, 3, 4 multipoles
        'gamma': ['GAUSSIAN', 2.1, 0.1], # logarithmic profile slope of main deflector
        'satellite_1_theta_E': ['GAUSSIAN', 0.35, 0.05], # mass of companion galaxy
        'satellite_1_x': ['GAUSSIAN', -1.82, 0.1], # position of companion galaxy
        'satellite_1_y': ['GAUSSIAN', -3.09, 0.1], # position of companion galaxy
        'q': ['TRUNC-HALF-GAUSS', 0.88, 0.2,0.73, 0.999] # mass axis ratio of main deflector
    }
    ######################## BACKGROUND SOURCE PRIORS ########################
    import background_source_priors
    kwargs_sample_source = {'source_size_pc': ['UNIFORM', 1, 10]} # uniform sampling 1-10pc
    kwargs_model_class = {'shapelets_order': 28} # n_max 28 for modeling lensed arcs
    ######################## SETUP OUTPUT DIRECTORIES ########################
    if os.getenv('HOME') == '/Users/danielgilman': # can replace with your home directory
        test_mode = True
        verbose = True
        base_path = os.getcwd()
        n_particles = 10
        n_iterations = 20
        job_index = 1
        parallel = False
        num_threads = 1
    else:
        test_mode = False
        verbose = False
        base_path = os.getenv('SCRATCH') # replace with where you want output files to appear
        n_particles = 10
        n_iterations = 80
        job_index = int(sys.argv[1]) # have a cluster pass in a unique task_ID through job array
        parallel = False
        num_threads = 1
        if base_path == '/scratch/midway3':
            base_path = '/scratch/midway3/gilmanda'
    random_seed_init = None # can set a random seed for reproducibility
    readout_steps = 2  # readout to output files every 2 realizations
    ############################## OPTIONS FOR MODELING IMAGE DATA ################################
    use_imaging_data = False # If True, will reconstruct the lens mass and source light for every realizations (very slow)
    split_image_data_reconstruction = True # If this is True, will reconstruct the lensed arcs separately from the lens mass model
    tolerance_source_reconstruction = float(sys.argv[3]) # -log-likelihood of flux ratios that triggers source reconstruction when split_image_data_reconstruction=True
    n_keep = int(sys.argv[2]) # number of realizations to keep per task_ID/CPU core
    output_path = base_path + '/' + lens_ID + '_example/'
    tolerance = np.inf  # accept every lens model, regardless of what the flux ratio is
    fitting_sequence_kwargs = [
        ['PSO', {'sigma_scale': 1., 'n_particles': n_particles,
                 'n_iterations': n_iterations,
                 'threadCount': num_threads}] # used by lenstronomy to reconstruct imaging data
    ]
    ############################## RUN SIMULATION ################################
    forward_model(output_path,
                  job_index,
                  n_keep,
                  data_class,
                  model_class,
                  preset_model_name,
                  kwargs_sample_realization,
                  kwargs_sample_source,
                  kwargs_sample_macro_fixed,
                  tolerance,
                  random_seed_init=random_seed_init,
                  readout_steps=readout_steps,
                  rescale_grid_resolution=rescale_grid_res,
                  rescale_grid_size=rescale_grid_size,
                  kwargs_model_class=kwargs_model_class,
                  verbose=verbose, n_pso_particles=n_particles,
                  n_pso_iterations=n_iterations,
                  num_threads=num_threads,
                  test_mode=test_mode,
                  use_imaging_data=use_imaging_data,
                  parallelize=parallel,
                  kappa_scale_subhalos=kappa_scale_subhalos,
                  log10_bound_mass_cut=log10_bound_mass_cut,
                  split_image_data_reconstruction=split_image_data_reconstruction,
                  filter_subhalo_kwargs=filter_subhalo_kwargs,
                  scipy_minimize_method='COBYQA_import',
                  fr_logL_source_reconstruction=tolerance_source_reconstruction,
                  fitting_sequence_kwargs=fitting_sequence_kwargs)

if __name__ == '__main__':
    run()

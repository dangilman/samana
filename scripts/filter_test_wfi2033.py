from samana.forward_model import forward_model
import sys
import numpy as np
from samana.analysis_util import (raytracing_grid_orientation, quick_setup,
                                  numerics_setup, default_rendering_area)

import os
from copy import deepcopy

# set the job index for the run
try:
    out_path = os.getenv('SCRATCH') + '/'  # replace this with whatever directory you want to output files
    job_index = int(sys.argv[1])
    test_mode = False
except:
    out_path = os.getcwd()
    job_index = 1
    test_mode = True

lens_ID = 'WFI2033'
data_class, model_class = quick_setup(lens_ID)
data_class = data_class()
rescale_grid_size, rescale_grid_res = numerics_setup(lens_ID)
cone_opening_angle_arcsec = default_rendering_area(data_class=data_class,
                                                   model_class=model_class)
rotation_angle_list, hessian_eigenvalue_list = raytracing_grid_orientation(lens_ID)
import flux_ratio_measurements
fluxes = flux_ratio_measurements.measurements[lens_ID]['measured_fluxes']
keep_flux_ratio_index = flux_ratio_measurements.measurements[lens_ID]['keep_flux_ratio_index']
data_class.magnifications = fluxes
data_class.flux_uncertainty = None
data_class.flux_ratio_covariance_matrix = flux_ratio_measurements.measurements[lens_ID][
    'flux_ratio_covariance_matrix']
data_class.keep_flux_ratio_index = keep_flux_ratio_index
preset_model_name_cdm = 'CDM'
kwargs_sample_realization_cdm = {'log10_sigma_sub': ['FIXED', -1.0],
                                     'log_mlow': ['FIXED', 6.],
                                     'log_mhigh': ['FIXED', 10.7],
                                     'LOS_normalization': ['FIXED', 1.0],
                                     'shmf_log_slope': ['FIXED', -1.925],
                                     'log_m_host': ['FIXED', 13.3],
                                     'truncation_model_subhalos': ['FIXED', 'TRUNCATION_GALACTICUS'],
                                     'add_globular_clusters': ['FIXED', False]
                                     }
filter_subhalo_kwargs = {'aperture_radius_arcsec': 0.25,
                             'log10_mass_minimum': 7.0}
filter_subhalo_kwargs['x_coords'] = data_class.x_image
filter_subhalo_kwargs['y_coords'] = data_class.y_image
kwargs_sample_source = {'source_size_pc': ['FIXED', 5.0]}
kwargs_sample_macro_fixed = {
        'OPTICAL_MULTIPOLE_PRIOR_M1': [],
        'gamma': ['GAUSSIAN', 1.9, 0.1],
        'satellite_1_theta_E': ['GAUSSIAN', 0.05, 0.05],
        'satellite_1_x': ['GAUSSIAN', 0.273217, 0.05],
        'satellite_1_y': ['GAUSSIAN', 2.00444, 0.05],
        'satellite_2_theta_E': ['GAUSSIAN', 0.7, 0.1],
        'satellite_2_x': ['GAUSSIAN', -3.52, 0.1],
        'satellite_2_y': ['GAUSSIAN', 0.033, 0.1],
        'q': ['TRUNC-HALF-GAUSS', 0.80, 0.2,0.65, 0.999]
    }
use_imaging_data = False
n_keep = 20
tolerance = np.inf
verbose = True
random_seed_init = None
n_pso_particles = 10
n_pso_iterations = 5
log10_bound_mass_cut = 4.7
kwargs_model_class = {'shapelets_order': 10} # source complexity
num_threads = 1
magnification_method = 'ELLIPTICAL_APERTURE'
if out_path is None:
    out_path = os.getcwd()

output_path = out_path + '/wfi2033_nofiltering/'
downselect_halo_mass = None
forward_model(output_path, job_index, n_keep, data_class, model_class, preset_model_name_cdm,
                  kwargs_sample_realization_cdm, kwargs_sample_source, kwargs_sample_macro_fixed,
               tolerance, random_seed_init=random_seed_init,
              rescale_grid_resolution=1.2,
              rescale_grid_size=1.5,
              kwargs_model_class=kwargs_model_class,
              verbose=verbose, n_pso_particles=n_pso_particles,
              n_pso_iterations=n_pso_iterations, num_threads=num_threads,
              test_mode=test_mode, use_imaging_data=use_imaging_data,
              log10_bound_mass_cut=log10_bound_mass_cut,
                  filter_subhalo_kwargs=filter_subhalo_kwargs,
              fr_logL_source_reconstruction=0.0,
              downselect_halo_mass=downselect_halo_mass,
              hessian_eigenvalue_list=hessian_eigenvalue_list,
              magnification_method=magnification_method,
              rotation_angle_list=rotation_angle_list,
              log_mhigh_mass_sheets=10.7
              )

output_path = out_path + '/wfi2033_filtering_1/'
downselect_halo_mass = {'aperture_radius': 0.2,
                            'log10_mass_allowed_global': 6.7,
                            'aperture_units': 'ANGLES',
                            'geometric_weighting': True}
forward_model(output_path, job_index, n_keep, data_class, model_class, preset_model_name_cdm,
                  kwargs_sample_realization_cdm, kwargs_sample_source, kwargs_sample_macro_fixed,
               tolerance, random_seed_init=random_seed_init,
              rescale_grid_resolution=1.2,
              rescale_grid_size=1.5,
              kwargs_model_class=kwargs_model_class,
              verbose=verbose, n_pso_particles=n_pso_particles,
              n_pso_iterations=n_pso_iterations, num_threads=num_threads,
              test_mode=test_mode, use_imaging_data=use_imaging_data,
              log10_bound_mass_cut=log10_bound_mass_cut,
                  filter_subhalo_kwargs=filter_subhalo_kwargs,
              fr_logL_source_reconstruction=0.0,
              downselect_halo_mass=downselect_halo_mass,
              hessian_eigenvalue_list=hessian_eigenvalue_list,
              magnification_method=magnification_method,
              rotation_angle_list=rotation_angle_list,
                log_mhigh_mass_sheets=10.7
              )

output_path = out_path + '/wfi2033_filtering_2/'
downselect_halo_mass = {'aperture_radius': 0.25,
                            'log10_mass_allowed_global': 7.0,
                            'aperture_units': 'ANGLES',
                            'geometric_weighting': True}
forward_model(output_path, job_index, n_keep, data_class, model_class, preset_model_name_cdm,
                  kwargs_sample_realization_cdm, kwargs_sample_source, kwargs_sample_macro_fixed,
               tolerance, random_seed_init=random_seed_init,
              rescale_grid_resolution=1.2,
              rescale_grid_size=1.5,
              kwargs_model_class=kwargs_model_class,
              verbose=verbose, n_pso_particles=n_pso_particles,
              n_pso_iterations=n_pso_iterations, num_threads=num_threads,
              test_mode=test_mode, use_imaging_data=use_imaging_data,
              log10_bound_mass_cut=log10_bound_mass_cut,
                  filter_subhalo_kwargs=filter_subhalo_kwargs,
              fr_logL_source_reconstruction=0.0,
              downselect_halo_mass=downselect_halo_mass,
              hessian_eigenvalue_list=hessian_eigenvalue_list,
              magnification_method=magnification_method,
              rotation_angle_list=rotation_angle_list,
                log_mhigh_mass_sheets=10.7
              )

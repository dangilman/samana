from samana.forward_model import forward_model
from samana.Data.Mocks.baseline_smooth_mock import BaselineSmoothMockMultipole3, BaselineSmoothMockModel
import os
import numpy as np
import sys

def main():
    # set the job index for the run

    output_path_base = os.getenv('SCRATCH') + '/'
    job_index = int(sys.argv[1])
    data_class = BaselineSmoothMockMultipole3()
    model_class = BaselineSmoothMockModel
    rescale_grid_res = 1.0
    rescale_grid_size = 1.0
    n_particles = None
    n_iterations = None
    kwargs_model_class = {}
    parallelize = False
    num_threads = 1

    n_keep = 2000
    readout_steps = 100
    tolerance = np.inf
    test_mode = False
    verbose = False
    use_imaging_data = False

    kwargs_sample_source = {'source_size_pc': ['FIXED', 5]}
    preset_model_name = 'CDM'
    kwargs_sample_realization = {'sigma_sub': ['FIXED', 0.00],
                                'LOS_normalization': ['FIXED',  0.0]}
    kwargs_sample_macro = {
            'gamma': ['GAUSSIAN', 2.08, 0.1],
            'a4_a': ['GAUSSIAN', 0.0, 0.01],
            'a3_a': ['GAUSSIAN', 0.0, 0.005],
            #'delta_phi_m3': ['FIXED', 0.0],
            # 'delta_phi_m4': ['FIXED', 0.0],
        'delta_phi_m3': ['UNIFORM', -np.pi/6, np.pi/6],
            'delta_phi_m4': ['UNIFORM', -np.pi/8, np.pi/8]
    }
    output_path = output_path_base + '/multipole_test_mock_3/'
    forward_model(output_path, job_index, n_keep, data_class, model_class,
                  preset_model_name, kwargs_sample_realization, kwargs_sample_source,
                  kwargs_sample_macro, tolerance, random_seed_init=None,
                  readout_steps=readout_steps, rescale_grid_resolution=rescale_grid_res,
                  rescale_grid_size=rescale_grid_size, kwargs_model_class=kwargs_model_class,
                  verbose=verbose, n_pso_particles=n_particles, n_pso_iterations=n_iterations,
                  num_threads=num_threads, test_mode=test_mode, use_imaging_data=use_imaging_data,
                  parallelize=parallelize)

if __name__ == '__main__':
    main()

"""
Tools for saving the lens model produced by a single forward-model iteration to disk, and for
re-loading it later for analysis.

The philosophy here is that we do NOT pickle live lenstronomy classes. Instead we write out a
lightweight, human-readable specification (gzipped JSON) containing the profile names, redshifts,
keyword arguments and numerical settings that define the lens model. Everything else -- the
LensModel instances, the decoupled multi-plane approximation, the effective convergence maps and
the magnification surfaces around each image -- is rebuilt on demand by the LensSimulationAnalysis
class. This keeps the files small and portable across lenstronomy/pyHalo versions.

Typical usage inside a forward model run:

    forward_model(..., save_for_analysis=True, save_for_analysis_path='./saved_models/')

which will prompt the user with a yes/no question after each realization is computed.

Typical usage when analyzing the output:

    from samana.model_storage import LensSimulationAnalysis
    sim = LensSimulationAnalysis.from_file('./saved_models/lens_model_seed12345.json.gz')

    # effective multi-plane convergence across the whole image plane, macromodel subtracted
    kappa, extent = sim.effective_convergence(npix=500)

    # ... or zoomed in on one image
    kappa, extent = sim.convergence_around_image(0, grid_size=0.5)

    # ray-traced surface brightness (magnification) surface around each image, or just one image
    mags, surfaces, extents = sim.magnification_surfaces()
    mag, surface, extent = sim.magnification_surface(2)

    # the same images computed with and without the near/far splitting approximation
    results = sim.compare_magnification_methods(['CIRCULAR_APERTURE', 'NEAR_FAR_SPLITTING'])
    fig, axes, results = sim.plot_method_comparison()

    # critical curves computed from the LensModel class (no lenstronomy plotting routines), either
    # for the smooth macromodel or for the full deflector including every halo
    ra_crit, dec_crit, ra_caustic, dec_caustic = sim.critical_curves(include_substructure=True)
    sim.plot_convergence(include_substructure_critical_curves=True)
"""

import gzip
import json
import os
from copy import deepcopy

import numpy as np

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

__all__ = ['LensSimulationAnalysis', 'save_lens_model', 'prompt_save_lens_model',
           'lens_model_spec_from_forward_model']

# bump this if the layout of the saved dictionary changes in a backwards-incompatible way
FILE_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------------------------
# serialization helpers
# ---------------------------------------------------------------------------------------------

def _to_native(obj):
    """
    Recursively convert numpy types/arrays into native python types so that the object can be
    written to JSON.

    :param obj: any object
    :return: the same object with numpy types replaced by python types
    """
    if isinstance(obj, dict):
        return {str(key): _to_native(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        # json writes these as Infinity/NaN, which json.load reads back correctly, but we make the
        # conversion explicit here so that the behavior is obvious
        return float(obj)
    return obj


def _restore_arrays(obj):
    """
    Inverse of _to_native for the pieces that lenstronomy expects to be numpy arrays. Any list whose
    entries are all numbers is converted into a numpy array; dictionaries and lists of dictionaries
    are traversed recursively. Lists of strings (e.g. lens_model_list) are left alone.

    :param obj: an object loaded from JSON
    :return: the object with numeric lists converted to numpy arrays
    """
    if isinstance(obj, dict):
        return {key: _restore_arrays(value) for key, value in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 0 and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                                for value in obj):
            return np.array(obj)
        return [_restore_arrays(value) for value in obj]
    return obj


def _cosmology_to_dict(astropy_cosmo):
    """
    Serialize an astropy cosmology instance.

    :param astropy_cosmo: an instance of an astropy cosmology, or None
    :return: a dictionary representation, or None
    """
    if astropy_cosmo is None:
        return None
    try:
        mapping = astropy_cosmo.to_format('mapping')
        out = {}
        for key, value in mapping.items():
            if key == 'cosmology':
                # astropy stores the class itself here; store the class name instead
                out['cosmology'] = getattr(value, '__name__', str(value))
            elif hasattr(value, 'value'):  # astropy Quantity
                out[key] = _to_native(value.value)
            else:
                out[key] = _to_native(value)
        return out
    except Exception:
        # fall back on the minimal set of parameters needed to rebuild a flat LCDM cosmology
        return {'cosmology': 'FlatLambdaCDM',
                'H0': float(astropy_cosmo.H0.value),
                'Om0': float(astropy_cosmo.Om0),
                'Ob0': None if astropy_cosmo.Ob0 is None else float(astropy_cosmo.Ob0)}


def _cosmology_from_dict(cosmology_dict):
    """
    Rebuild an astropy cosmology from the output of _cosmology_to_dict.

    :param cosmology_dict: a dictionary, or None
    :return: an astropy cosmology instance, or None
    """
    if cosmology_dict is None:
        return None
    import astropy.cosmology as astropy_cosmology
    kwargs = deepcopy(cosmology_dict)
    class_name = kwargs.pop('cosmology', 'FlatLambdaCDM')
    cosmology_class = getattr(astropy_cosmology, class_name, astropy_cosmology.FlatLambdaCDM)
    kwargs.pop('meta', None)
    # only pass through arguments the class actually accepts
    import inspect
    allowed = set(inspect.signature(cosmology_class.__init__).parameters.keys())
    kwargs = {key: value for key, value in kwargs.items() if key in allowed and value is not None}
    return cosmology_class(**kwargs)


def _open(filename, mode):
    """Open a plain or gzipped text file depending on the extension."""
    if filename.endswith('.gz'):
        return gzip.open(filename, mode + 't')
    return open(filename, mode)


# ---------------------------------------------------------------------------------------------
# building and writing the specification
# ---------------------------------------------------------------------------------------------

def lens_model_spec_from_forward_model(lens_model_init,
                                       kwargs_lens_init,
                                       kwargs_solution,
                                       index_lens_split,
                                       astropy_cosmo,
                                       z_lens,
                                       z_source,
                                       x_image,
                                       y_image,
                                       source_x=None,
                                       source_y=None,
                                       source_size_pc=None,
                                       magnification_method='CIRCULAR_APERTURE',
                                       grid_size_list=None,
                                       grid_resolution_list=None,
                                       rotation_angle_list=None,
                                       hessian_eigenvalue_list=None,
                                       halo_masses=None,
                                       magnifications=None,
                                       flux_ratios=None,
                                       flux_ratios_data=None,
                                       summary_statistic=None,
                                       logL_imaging_data=None,
                                       bic=None,
                                       samples=None,
                                       macromodel_samples_fixed=None,
                                       window_size=None,
                                       pixel_size=None,
                                       decoupled_multiplane=True,
                                       grid_resolution_image_data=None,
                                       scale_window_size=1.0,
                                       kwargs_light=None,
                                       seed=None,
                                       lens_id=None,
                                       data_class_name=None,
                                       model_class_name=None,
                                       notes=None):
    """
    Assemble the dictionary that fully specifies a lens model from a single forward model iteration.

    All of the lens model information is taken from lens_model_init, which is the full multi-plane
    lens model containing the macromodel plus every halo in the realization. The macromodel
    components are identified by index_lens_split.

    :param lens_model_init: the full (macromodel + halos) multi-plane LensModel instance
    :param kwargs_lens_init: keyword arguments for lens_model_init; the macromodel entries are the
     initial guess used to set up the decoupled multi-plane approximation
    :param kwargs_solution: keyword arguments for the macromodel that solve the lens equation. May
     be either only the macromodel components, or the full list (macromodel + halos)
    :param index_lens_split: indexes of lens_model_init corresponding to the macromodel
    :param astropy_cosmo: astropy cosmology instance
    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param x_image: image positions used in the lens model (i.e. including astrometric perturbations)
    :param y_image: image positions used in the lens model
    :param source_x: source position inferred from the lens model
    :param source_y: source position inferred from the lens model
    :param source_size_pc: FWHM of the Gaussian source in pc
    :param magnification_method: the algorithm used to compute image magnifications
    :param grid_size_list: size of the ray tracing grid used for each image (arcsec)
    :param grid_resolution_list: resolution of the ray tracing grid used for each image (arcsec/pixel)
    :param rotation_angle_list: rotation angles used by ELLIPTICAL_APERTURE/ADAPTIVE methods
    :param hessian_eigenvalue_list: hessian eigenvalue ratios used by ELLIPTICAL_APERTURE/ADAPTIVE
    :param halo_masses: one mass per entry in the halo lens model list (bound mass for subhalos,
     infall mass otherwise, NaN for the negative convergence sheets). Required to reproduce the
     NEAR_FAR_SPLITTING magnification calculation, which sets the near-field radius from the halo mass
    :param magnifications: image magnifications computed during the run
    :param flux_ratios: model flux ratios
    :param flux_ratios_data: measured flux ratios
    :param summary_statistic: the flux ratio summary statistic
    :param logL_imaging_data: log-likelihood of the imaging data
    :param bic: bayesian information criterion from the imaging data fit
    :param samples: a dictionary of {parameter name: value} for the sampled parameters
    :param macromodel_samples_fixed: dictionary of fixed macromodel parameters
    :param window_size: size of the imaging data window (arcsec); used as the default size of
     convergence maps
    :param pixel_size: pixel size of the imaging data (arcsec)
    :param decoupled_multiplane: bool; whether the run used the decoupled multi-plane approximation
    :param grid_resolution_image_data: resolution of the grid used for the decoupled multi-plane
     interpolation
    :param scale_window_size: rescaling of the window size used for the decoupled multi-plane setup
    :param kwargs_light: dictionary with the source/lens light/point source keyword arguments
    :param seed: the random seed of the realization
    :param lens_id: a string identifying the lens
    :param data_class_name: name of the Data class used in the run
    :param model_class_name: name of the Model class used in the run
    :param notes: an optional free-form string
    :return: a dictionary
    """
    lens_model_list = list(lens_model_init.lens_model_list)
    redshift_list = list(lens_model_init.redshift_list)
    index_lens_split = [int(index) for index in index_lens_split]
    n_macro = len(index_lens_split)

    if len(kwargs_solution) == len(lens_model_list):
        # the full ray-tracing calculation was used; the macromodel is the first n_macro entries
        kwargs_macro_solution = [kwargs_solution[index] for index in index_lens_split]
    elif len(kwargs_solution) == n_macro:
        kwargs_macro_solution = list(kwargs_solution)
    else:
        raise Exception('kwargs_solution has length ' + str(len(kwargs_solution)) + ', which matches '
                        'neither the number of macromodel components (' + str(n_macro) + ') nor the '
                        'full lens model list (' + str(len(lens_model_list)) + ')')

    try:
        profile_kwargs_list = _to_native(lens_model_init.profile_kwargs_list)
    except Exception:
        profile_kwargs_list = None

    spec = {
        'file_format_version': FILE_FORMAT_VERSION,
        'lens_id': lens_id,
        'data_class_name': data_class_name,
        'model_class_name': model_class_name,
        'notes': notes,
        'seed': seed,
        # ---- geometry and cosmology ----
        'z_lens': z_lens,
        'z_source': z_source,
        'cosmology': _cosmology_to_dict(astropy_cosmo),
        # ---- the lens model ----
        'lens_model_list': lens_model_list,
        'redshift_list': redshift_list,
        'profile_kwargs_list': profile_kwargs_list,
        'kwargs_lens_init': kwargs_lens_init,
        'kwargs_macro_solution': kwargs_macro_solution,
        'index_lens_split': index_lens_split,
        'macromodel_samples_fixed': macromodel_samples_fixed,
        # ---- images and source ----
        'x_image': x_image,
        'y_image': y_image,
        'source_x': source_x,
        'source_y': source_y,
        'source_size_pc': source_size_pc,
        # ---- numerics used for the magnification calculation ----
        'magnification_method': magnification_method,
        'grid_size_list': grid_size_list,
        'grid_resolution_list': grid_resolution_list,
        'rotation_angle_list': rotation_angle_list,
        'hessian_eigenvalue_list': hessian_eigenvalue_list,
        'halo_masses': halo_masses,
        # ---- numerics used for the imaging data / decoupled multiplane setup ----
        'decoupled_multiplane': decoupled_multiplane,
        'grid_resolution_image_data': grid_resolution_image_data,
        'scale_window_size': scale_window_size,
        'window_size': window_size,
        'pixel_size': pixel_size,
        # ---- results ----
        'magnifications': magnifications,
        'flux_ratios': flux_ratios,
        'flux_ratios_data': flux_ratios_data,
        'summary_statistic': summary_statistic,
        'logL_imaging_data': logL_imaging_data,
        'bic': bic,
        'samples': samples,
        'kwargs_light': kwargs_light,
    }
    return _to_native(spec)


def save_lens_model(filename, spec, output_dir='./', verbose=True):
    """
    Write a lens model specification to disk as (gzipped) JSON.

    :param filename: name of the output file. If it ends in .gz the file is gzipped
    :param spec: the dictionary produced by lens_model_spec_from_forward_model
    :param output_dir: directory in which to write the file
    :param verbose: bool; print the path of the file that was written
    :return: the full path of the file that was written
    """
    if output_dir not in ['', None] and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename) if output_dir else filename
    with _open(path, 'w') as f:
        json.dump(_to_native(spec), f)
    if verbose:
        size_mb = os.path.getsize(path) / 1024 ** 2
        print('wrote lens model to ' + path + ' (' + str(np.round(size_mb, 2)) + ' MB)')
    return path


def prompt_save_lens_model(spec, output_dir='./', filename=None, verbose=True,
                           show_figures=True, magnifications=None, surfaces=None,
                           kwargs_inspection_figure=None):
    """
    Show an inspection figure for a lens model, print a short summary of it, and ask the user
    whether to save it. Intended to be called from inside a forward model run with
    save_for_analysis=True.

    :param spec: the dictionary produced by lens_model_spec_from_forward_model
    :param output_dir: directory in which to write the file
    :param filename: default filename; if None, one is generated from the lens ID and random seed
    :param verbose: bool; print a summary of the model before prompting
    :param show_figures: bool; display an inspection figure before asking whether to save
    :param magnifications: image magnifications already computed during the run. Passing these
     avoids recomputing the ray tracing just to make the figure
    :param surfaces: magnification surfaces already computed during the run (the 'images' returned
     by the magnification class)
    :param kwargs_inspection_figure: keyword arguments passed to
     LensSimulationAnalysis.inspection_figure
    :return: the path of the file that was written, or None if the user declined
    """
    if filename is None:
        label = spec.get('lens_id', None) or 'lens'
        seed = spec.get('seed', None)
        filename = str(label) + '_seed' + str(seed) + '.json.gz'

    if show_figures:
        try:
            import matplotlib.pyplot as plt
            simulation = LensSimulationAnalysis(spec)
            simulation.inspection_figure(magnifications=magnifications, surfaces=surfaces,
                                         **(kwargs_inspection_figure or {}))
            plt.show()
        except Exception as exception:
            print('failed to make the inspection figure: ' + str(exception))

    if verbose:
        print('\n---------------- candidate model for analysis ----------------')
        print('lens ID:              ', spec.get('lens_id', None))
        print('random seed:          ', spec.get('seed', None))
        print('number of deflectors: ', len(spec['lens_model_list']))
        print('magnifications:       ', np.round(np.atleast_1d(spec['magnifications']), 4)
              if spec.get('magnifications', None) is not None else None)
        print('summary statistic:    ', spec.get('summary_statistic', None))
        print('logL imaging data:    ', spec.get('logL_imaging_data', None))
        print('-------------------------------------------------------------')

    while True:
        answer = input('save this lens model for later analysis? [y/n]: ').strip().lower()
        if answer in ['y', 'yes']:
            break
        if answer in ['n', 'no', '']:
            print('not saving this lens model')
            return None
        print("please enter 'y' or 'n'")

    answer = input('filename [' + filename + ']: ').strip()
    if answer != '':
        filename = answer
        if not (filename.endswith('.json') or filename.endswith('.json.gz')):
            filename += '.json.gz'
    return save_lens_model(filename, spec, output_dir=output_dir, verbose=True)


# ---------------------------------------------------------------------------------------------
# the analysis class
# ---------------------------------------------------------------------------------------------

class LensSimulationAnalysis(object):
    """
    Rebuilds the lenstronomy classes for a lens model saved by a forward model run, and computes
    the quantities that are useful for inspecting it: the effective multi-plane convergence, the
    magnification (ray-traced surface brightness) surface around each image, and the critical
    curves. None of the methods in this class call lenstronomy plotting routines.
    """

    def __init__(self, spec):
        """
        :param spec: the dictionary written by save_lens_model / produced by
         lens_model_spec_from_forward_model
        """
        self.spec = spec
        self._cache = {}

    # -------------------------------------------------------------------- constructors / io

    @classmethod
    def from_file(cls, filename):
        """
        Load a saved lens model.

        :param filename: path to a .json or .json.gz file written by save_lens_model
        :return: an instance of LensSimulationAnalysis
        """
        with _open(filename, 'r') as f:
            spec = json.load(f)
        version = spec.get('file_format_version', None)
        if version is not None and version > FILE_FORMAT_VERSION:
            print('warning: file was written with format version ' + str(version) + ' but this '
                  'version of samana understands version ' + str(FILE_FORMAT_VERSION))
        return cls(spec)

    def save(self, filename, output_dir='./', verbose=True):
        """Write this model back out to disk."""
        return save_lens_model(filename, self.spec, output_dir=output_dir, verbose=verbose)

    # -------------------------------------------------------------------- simple accessors

    @property
    def lens_id(self):
        return self.spec.get('lens_id', None)

    @property
    def seed(self):
        return self.spec.get('seed', None)

    @property
    def z_lens(self):
        return self.spec['z_lens']

    @property
    def z_source(self):
        return self.spec['z_source']

    @property
    def x_image(self):
        return np.array(self.spec['x_image'])

    @property
    def y_image(self):
        return np.array(self.spec['y_image'])

    @property
    def source_x(self):
        return self.spec.get('source_x', None)

    @property
    def source_y(self):
        return self.spec.get('source_y', None)

    @property
    def magnifications(self):
        """Image magnifications computed during the forward model run."""
        mags = self.spec.get('magnifications', None)
        return None if mags is None else np.array(mags)

    @property
    def flux_ratios(self):
        ratios = self.spec.get('flux_ratios', None)
        return None if ratios is None else np.array(ratios)

    @property
    def summary_statistic(self):
        return self.spec.get('summary_statistic', None)

    @property
    def samples(self):
        """Dictionary of the parameters sampled for this realization."""
        return self.spec.get('samples', {}) or {}

    @property
    def astropy_cosmo(self):
        """The astropy cosmology used in the run."""
        if 'astropy_cosmo' not in self._cache:
            self._cache['astropy_cosmo'] = _cosmology_from_dict(self.spec.get('cosmology', None))
        return self._cache['astropy_cosmo']

    @property
    def index_lens_split(self):
        """Indexes of the lens model list corresponding to the macromodel."""
        return list(self.spec['index_lens_split'])

    @property
    def lens_model_list(self):
        return list(self.spec['lens_model_list'])

    @property
    def redshift_list(self):
        return list(self.spec['redshift_list'])

    # -------------------------------------------------------------------- lens model rebuilds

    @property
    def kwargs_lens_init(self):
        """Keyword arguments for the full lens model with the *initial* macromodel parameters."""
        if 'kwargs_lens_init' not in self._cache:
            self._cache['kwargs_lens_init'] = _restore_arrays(deepcopy(self.spec['kwargs_lens_init']))
        return self._cache['kwargs_lens_init']

    @property
    def kwargs_macro(self):
        """Keyword arguments for the macromodel that solve the lens equation."""
        if 'kwargs_macro' not in self._cache:
            self._cache['kwargs_macro'] = _restore_arrays(deepcopy(self.spec['kwargs_macro_solution']))
        return self._cache['kwargs_macro']

    @property
    def kwargs_lens(self):
        """
        Keyword arguments for the full lens model (macromodel + halos) with the macromodel
        parameters set to the solution found during the lens modeling.
        """
        if 'kwargs_lens' not in self._cache:
            kwargs = deepcopy(self.kwargs_lens_init)
            for counter, index in enumerate(self.index_lens_split):
                kwargs[index] = self.kwargs_macro[counter]
            self._cache['kwargs_lens'] = kwargs
        return self._cache['kwargs_lens']

    @property
    def profile_kwargs_list(self):
        profile_kwargs_list = self.spec.get('profile_kwargs_list', None)
        if profile_kwargs_list is None:
            return None
        return _restore_arrays(deepcopy(profile_kwargs_list))

    @property
    def halo_masses(self):
        """
        One mass per entry in the halo lens model list, used by the NEAR_FAR_SPLITTING magnification
        method to set the near-field radius around each halo. Returns None if it was not saved.
        """
        halo_masses = self.spec.get('halo_masses', None)
        return None if halo_masses is None else np.array(halo_masses, dtype=float)

    @property
    def lens_model(self):
        """
        The full multi-plane LensModel instance containing the macromodel and every halo in the
        realization. This performs exact multi-plane ray tracing (no decoupled approximation).
        """
        if 'lens_model' not in self._cache:
            self._cache['lens_model'] = LensModel(
                lens_model_list=self.lens_model_list,
                lens_redshift_list=self.redshift_list,
                multi_plane=True,
                z_source=self.z_source,
                cosmo=self.astropy_cosmo,
                profile_kwargs_list=self.profile_kwargs_list)
        return self._cache['lens_model']

    @property
    def lens_model_macro(self):
        """A LensModel instance containing only the smooth macromodel components."""
        if 'lens_model_macro' not in self._cache:
            profile_kwargs_list = self.profile_kwargs_list
            index_lens_split = self.index_lens_split
            self._cache['lens_model_macro'] = LensModel(
                lens_model_list=[self.lens_model_list[i] for i in index_lens_split],
                lens_redshift_list=[self.redshift_list[i] for i in index_lens_split],
                multi_plane=True,
                z_source=self.z_source,
                cosmo=self.astropy_cosmo,
                profile_kwargs_list=None if profile_kwargs_list is None else
                [profile_kwargs_list[i] for i in index_lens_split])
        return self._cache['lens_model_macro']

    def lens_model_decoupled(self, grid_size=None, grid_resolution=None,
                             coordinate_type='GRID', use_solution=True):
        """
        Rebuild the decoupled multi-plane approximation to the lens model, as used during the lens
        modeling. This returns a LensModel instance whose free parameters are the macromodel
        components, with the deflection field of the halos absorbed into interpolated foreground and
        background deflection fields.

        :param grid_size: size of the interpolation grid in arcsec. Defaults to the window size of
         the imaging data times the scale_window_size used in the run
        :param grid_resolution: resolution of the interpolation grid in arcsec/pixel. Defaults to
         the value used in the run
        :param coordinate_type: 'GRID' interpolates on a regular grid, 'POINT' sets up the
         approximation at a single point (much faster, but only valid near the images)
        :param use_solution: bool; if True, the approximation is set up around the macromodel that
         solves the lens equation. If False, it is set up around the initial macromodel guess, which
         is what the forward model actually did
        :return: (LensModel instance, keyword arguments for the macromodel components)
        """
        from lenstronomy.LensModel.Util.decouple_multi_plane_util import (
            setup_lens_model, setup_grids, coordinates_and_deflections,
            decoupled_multiplane_class_setup)

        if grid_size is None:
            grid_size = (self.spec.get('window_size', None) or 4.0) * \
                        (self.spec.get('scale_window_size', 1.0) or 1.0)
        if grid_resolution is None:
            grid_resolution = self.spec.get('grid_resolution_image_data', None) or \
                              self.spec.get('pixel_size', None) or 0.05

        kwargs_lens_full = self.kwargs_lens if use_solution else self.kwargs_lens_init
        (lens_model_fixed, lens_model_free, kwargs_lens_fixed, kwargs_lens_free,
         z_source, z_split, cosmo_bkg) = setup_lens_model(self.lens_model, kwargs_lens_full,
                                                          self.index_lens_split)
        if coordinate_type == 'GRID':
            x_grid, y_grid, interp_points, _ = setup_grids(grid_size, grid_resolution)
        elif coordinate_type == 'POINT':
            x_grid, y_grid = 0.0, 0.0
            interp_points = (0.0, 0.0)
        else:
            raise ValueError('coordinate_type must be GRID or POINT, got ' + str(coordinate_type))

        xD, yD, alpha_x_foreground, alpha_y_foreground, alpha_beta_subx, alpha_beta_suby = \
            coordinates_and_deflections(lens_model_fixed, lens_model_free, kwargs_lens_fixed,
                                        kwargs_lens_free, x_grid, y_grid, z_split, z_source,
                                        cosmo_bkg)
        kwargs_class_setup = decoupled_multiplane_class_setup(
            lens_model_free, xD, yD, alpha_x_foreground, alpha_y_foreground,
            alpha_beta_subx, alpha_beta_suby, z_split,
            coordinate_type=coordinate_type, interp_points=interp_points)
        lens_model_decoupled = LensModel(**kwargs_class_setup)
        return lens_model_decoupled, kwargs_lens_free

    # -------------------------------------------------------------------- convergence

    def coordinate_grid(self, grid_size=None, npix=500, center_x=0.0, center_y=0.0):
        """
        Build a square coordinate grid.

        :param grid_size: side length of the grid in arcsec; defaults to the imaging data window
        :param npix: number of pixels per side
        :param center_x: center of the grid in arcsec
        :param center_y: center of the grid in arcsec
        :return: 2d x coordinates, 2d y coordinates, extent = [xmin, xmax, ymin, ymax]
        """
        if grid_size is None:
            grid_size = self.spec.get('window_size', None) or 4.0
        x = np.linspace(-grid_size / 2, grid_size / 2, npix) + center_x
        y = np.linspace(-grid_size / 2, grid_size / 2, npix) + center_y
        xx, yy = np.meshgrid(x, y)
        extent = [x[0], x[-1], y[0], y[-1]]
        return xx, yy, extent

    def effective_convergence(self, grid_size=None, npix=500, center_x=0.0, center_y=0.0,
                              subtract_macromodel=True, subtract_mean=False, decoupled=False):
        """
        Compute the effective multi-plane convergence, optionally with the smooth macromodel
        subtracted. This is the quantity shown by lenstronomy's substructure_plot, computed here
        directly from the LensModel class.

        The default grid covers the whole image plane (the imaging data window). To zoom in on a
        single image, pass center_x/center_y (or use convergence_around_image).

        :param grid_size: side length of the map in arcsec; defaults to the imaging data window
        :param npix: number of pixels per side
        :param center_x: center of the map in arcsec
        :param center_y: center of the map in arcsec
        :param subtract_macromodel: bool; subtract the convergence of the smooth macromodel, leaving
         only the contribution from substructure and line-of-sight halos
        :param subtract_mean: bool; subtract the mean of the resulting map
        :param decoupled: bool; use the decoupled multi-plane approximation instead of exact ray
         tracing. Slower to set up but reproduces exactly what the lens modeling used
        :return: the convergence map (npix x npix), extent = [xmin, xmax, ymin, ymax]
        """
        xx, yy, extent = self.coordinate_grid(grid_size, npix, center_x, center_y)
        shape = xx.shape
        if decoupled:
            lens_model, kwargs_lens = self.lens_model_decoupled(
                grid_size=grid_size if grid_size is not None else None)
        else:
            lens_model, kwargs_lens = self.lens_model, self.kwargs_lens
        kappa = lens_model.kappa(xx.ravel(), yy.ravel(), kwargs_lens).reshape(shape)
        if subtract_macromodel:
            kappa_macro = self.lens_model_macro.kappa(xx.ravel(), yy.ravel(),
                                                      self.kwargs_macro).reshape(shape)
            kappa = kappa - kappa_macro
        if subtract_mean:
            kappa = kappa - np.mean(kappa)
        return kappa, extent

    def convergence_around_image(self, image_index, grid_size=0.5, npix=200, **kwargs):
        """
        Effective convergence in a small region centered on one of the lensed images.

        :param image_index: index of the image (0, 1, 2, ...)
        :param grid_size: side length of the map in arcsec
        :param npix: number of pixels per side
        :param kwargs: passed to effective_convergence
        :return: the convergence map (npix x npix), extent = [xmin, xmax, ymin, ymax]
        """
        return self.effective_convergence(grid_size=grid_size, npix=npix,
                                          center_x=self.x_image[image_index],
                                          center_y=self.y_image[image_index], **kwargs)

    def critical_curves(self, include_substructure=False, compute_window=None, grid_scale=0.01):
        """
        Compute critical curves and caustics from the LensModel class (no plotting routines).

        :param include_substructure: bool; if True the critical curves are computed for the full
         deflector, i.e. the macromodel plus every halo in the realization, which produces the
         corrugated critical curve and the small curves around individual halos. If False only the
         smooth macromodel is used, which gives the single tangential/radial critical curve
        :param compute_window: size of the region in which to search, in arcsec. Defaults to the
         imaging data window, i.e. the whole image plane
        :param grid_scale: resolution of the grid used to find the critical curves. Substructure
         requires a finer grid than the macromodel; 0.005 or smaller is typical
        :return: ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list
        """
        if compute_window is None:
            compute_window = self.spec.get('window_size', None) or 4.0
        if include_substructure:
            lens_model, kwargs_lens = self.lens_model, self.kwargs_lens
        else:
            lens_model, kwargs_lens = self.lens_model_macro, self.kwargs_macro
        extension = LensModelExtensions(lens_model)
        return extension.critical_curve_caustics(kwargs_lens,
                                                 compute_window=compute_window,
                                                 grid_scale=grid_scale)

    # -------------------------------------------------------------------- magnification surfaces

    _BATCH_PROFILES = {'CORE_COLLAPSED_HALO', 'TNFW', 'TNFWC', 'NFW', 'KING'}

    def _batched_lens_model(self):
        """
        A version of the full lens model in which runs of identical halo profiles at the same
        redshift are collapsed into MULTI_HALO_BATCH entries. This is only a speed optimization; the
        deflection field is identical. Falls back on the unbatched model if the macromodel would be
        swept into a batch.

        :return: (LensModel instance, keyword arguments)
        """
        if 'batched_lens_model' not in self._cache:
            from samana.forward_model_util import batch_lens_profiles
            n_macro = len(self.index_lens_split)
            names, redshifts, kwargs = batch_lens_profiles(
                list(self.lens_model_list), list(self.redshift_list), self.kwargs_lens_init,
                profiles_to_batch=self._BATCH_PROFILES, min_group=4)
            if list(names[:n_macro]) != list(self.lens_model_list[:n_macro]):
                self._cache['batched_lens_model'] = (self.lens_model, self.kwargs_lens_init)
            else:
                lens_model = LensModel(names, lens_redshift_list=list(redshifts),
                                       cosmo=self.astropy_cosmo, z_source=self.z_source,
                                       multi_plane=True)
                self._cache['batched_lens_model'] = (lens_model, kwargs)
        return self._cache['batched_lens_model']

    def _decoupled_setup_output(self, batched=False):
        """
        The tuple returned by lenstronomy's setup_lens_model: the fixed (halo) lens model whose
        deflections get interpolated, the free (macromodel) lens model, and the associated keyword
        arguments and distances. This is what the magnification routines consume.

        :param batched: bool; collapse repeated halo profiles into MULTI_HALO_BATCH entries
        :return: (lens_model_fixed, lens_model_free, kwargs_fixed, kwargs_free, z_source, z_split,
         cosmo_bkg)
        """
        from lenstronomy.LensModel.Util.decouple_multi_plane_util import setup_lens_model
        key = 'decoupled_setup_batched' if batched else 'decoupled_setup'
        if key not in self._cache:
            output = setup_lens_model(self.lens_model, self.kwargs_lens_init, self.index_lens_split)
            if batched:
                from samana.forward_model_util import batch_lens_profiles
                (lens_model_fixed, lens_model_free, kwargs_fixed, kwargs_free,
                 z_source, z_split, cosmo_bkg) = output
                names, redshifts, kwargs = batch_lens_profiles(
                    lens_model_fixed.lens_model_list, list(lens_model_fixed.redshift_list),
                    kwargs_fixed, profiles_to_batch=self._BATCH_PROFILES, min_group=4)
                lens_model_fixed_batched = LensModel(names, lens_redshift_list=list(redshifts),
                                                     cosmo=self.astropy_cosmo,
                                                     z_source=self.z_source, multi_plane=True)
                output = (lens_model_fixed_batched, lens_model_free, kwargs, kwargs_free,
                          z_source, z_split, cosmo_bkg)
            self._cache[key] = output
        return self._cache[key]

    def grid_orientation_from_hessian(self, inflation_factor=2.0):
        """
        Estimate the orientation and axis ratio of each lensed image from the hessian of the
        macromodel at the image position. Used as a fallback for the ELLIPTICAL_APERTURE, ADAPTIVE
        and NEAR_FAR_SPLITTING_ADAPTIVE methods when the values used in the run were not saved.

        The image is elongated along the eigenvector of the magnification tensor with the smallest
        absolute eigenvalue, with an axis ratio equal to the ratio of the eigenvalues. An aperture
        with exactly that axis ratio clips the wings of the image and loses of order 10% of the flux,
        so the ratio is inflated by inflation_factor to give a more generous aperture.

        :param inflation_factor: multiplies the axis ratio; the result is capped at 1 (a circle).
         The default of 2 reproduces the circular aperture magnification to better than 0.1% in
         testing, while still being much cheaper than a circular aperture
        :return: rotation angles (radians), axis ratios
        """
        rotation_angle_list, axis_ratio_list = [], []
        for x, y in zip(self.x_image, self.y_image):
            f_xx, f_xy, f_yx, f_yy = self.lens_model_macro.hessian(x, y, self.kwargs_macro)
            off_diagonal = -0.5 * (float(f_xy) + float(f_yx))
            tensor = np.array([[1 - float(f_xx), off_diagonal],
                               [off_diagonal, 1 - float(f_yy)]])
            eigenvalues, eigenvectors = np.linalg.eigh(tensor)
            order = np.argsort(np.abs(eigenvalues))
            # the image is stretched along the eigenvector with the smallest |eigenvalue|
            major_axis = eigenvectors[:, order[0]]
            rotation_angle_list.append(float(np.arctan2(major_axis[1], major_axis[0])))
            axis_ratio = np.abs(eigenvalues[order[0]]) / np.abs(eigenvalues[order[1]])
            axis_ratio_list.append(float(min(1.0, inflation_factor * axis_ratio)))
        return np.array(rotation_angle_list), np.array(axis_ratio_list)

    def magnification_surfaces(self, source_size_pc=None, magnification_method=None,
                               image_index=None, grid_size_list=None, grid_resolution_list=None,
                               rescale_grid_size=1.0, rescale_grid_resolution=1.0,
                               source_x=None, source_y=None, use_batched_lens_model=True,
                               rotation_angle_list=None, hessian_eigenvalue_list=None,
                               halo_masses=None, fallback='ELLIPTICAL_APERTURE',
                               mu_tolerance=0.05, verbose=False):
        """
        Ray trace a zoomed region around each lensed image and return the resulting surface
        brightness (magnification) surfaces. Integrating a surface over the grid and dividing by the
        total flux of the unlensed source gives the magnification of that image.

        By default this reproduces the calculation performed during the forward model run, using the
        same source size, grid sizes, grid resolutions and magnification method. Passing a different
        magnification_method recomputes the same images with a different algorithm, which is the
        intended way to check the near/far splitting approximation against exact ray tracing.

        :param source_size_pc: FWHM of the Gaussian source in pc; defaults to the value used in the run
        :param magnification_method: one of
         - CIRCULAR_APERTURE: exact ray tracing on a circular aperture that grows until convergence
         - ELLIPTICAL_APERTURE: exact ray tracing on an elliptical aperture
         - ADAPTIVE: exact ray tracing on an adaptively refined quadtree
         - NEAR_FAR_SPLITTING: near/far splitting approximation on a regular grid
         - NEAR_FAR_SPLITTING_ADAPTIVE: near/far splitting on an adaptively refined quadtree
         defaults to the method used in the run
        :param image_index: compute only this image (0, 1, 2, ...). If None, all images are computed
        :param grid_size_list: size of the grid around each image in arcsec
        :param grid_resolution_list: resolution of the grid around each image in arcsec/pixel
        :param rescale_grid_size: multiplies the default grid sizes; useful for zooming out
        :param rescale_grid_resolution: multiplies the default grid resolutions; values < 1 give a
         finer grid
        :param source_x: source position; defaults to the value saved from the run
        :param source_y: source position; defaults to the value saved from the run
        :param use_batched_lens_model: bool; collapse repeated halo profiles into batched profiles.
         This does not change the result, only the speed
        :param rotation_angle_list: override the grid orientations used by the elliptical/adaptive
         methods; defaults to the values used in the run, or an estimate from the macromodel hessian
        :param hessian_eigenvalue_list: override the grid axis ratios; same defaults as above
        :param halo_masses: one mass per halo lens model, used by the near/far splitting methods to
         set the near-field radius; defaults to the values saved from the run
        :param fallback: the method used by the near/far splitting routines when the approximation
         is flagged as inaccurate for an image
        :param mu_tolerance: point-source magnification discrepancy above which the near/far
         splitting result is discarded in favor of the fallback method
        :param verbose: bool; print output from the magnification calculation
        :return: magnifications, list of surfaces (one per image), list of extents (one per image).
         For the adaptive methods each surface is [flux_array, tiling] rather than a 2d array
        """
        from samana.image_magnification_util import (setup_gaussian_source,
                                                     magnification_finite_decoupled)

        if source_size_pc is None:
            source_size_pc = self.spec['source_size_pc']
        if source_size_pc is None:
            raise Exception('no source size was saved with this model, pass source_size_pc explicitly')
        if magnification_method is None:
            magnification_method = self.spec.get('magnification_method', 'CIRCULAR_APERTURE')
        if source_x is None:
            source_x = self.source_x
        if source_y is None:
            source_y = self.source_y
        if source_x is None or source_y is None:
            source_x, source_y = self.lens_model.ray_shooting(self.x_image, self.y_image,
                                                              self.kwargs_lens)

        n_image = len(self.x_image)
        if grid_size_list is None:
            grid_size_list = self.spec.get('grid_size_list', None)
            if grid_size_list is None:
                from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size
                grid_size_list = [auto_raytracing_grid_size(source_size_pc)] * n_image
        if grid_resolution_list is None:
            grid_resolution_list = self.spec.get('grid_resolution_list', None)
            if grid_resolution_list is None:
                from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_resolution
                grid_resolution_list = [auto_raytracing_grid_resolution(source_size_pc)] * n_image
        grid_size_list = list(np.atleast_1d(grid_size_list).astype(float) * rescale_grid_size)
        grid_resolution_list = list(np.atleast_1d(grid_resolution_list).astype(float) *
                                    rescale_grid_resolution)

        if rotation_angle_list is None:
            rotation_angle_list = self.spec.get('rotation_angle_list', None)
        if hessian_eigenvalue_list is None:
            hessian_eigenvalue_list = self.spec.get('hessian_eigenvalue_list', None)
        needs_orientation = magnification_method in ['ELLIPTICAL_APERTURE', 'ADAPTIVE',
                                                     'NEAR_FAR_SPLITTING_ADAPTIVE'] or \
            (magnification_method == 'NEAR_FAR_SPLITTING' and fallback in ['ELLIPTICAL_APERTURE',
                                                                          'ADAPTIVE'])
        if needs_orientation and (rotation_angle_list is None or hessian_eigenvalue_list is None):
            if verbose:
                print('no grid orientation was saved with this model, estimating it from the '
                      'macromodel hessian at the image positions')
            rotation_angle_list, hessian_eigenvalue_list = self.grid_orientation_from_hessian()
        rotation_angle_list = None if rotation_angle_list is None else list(np.atleast_1d(rotation_angle_list))
        hessian_eigenvalue_list = None if hessian_eigenvalue_list is None else \
            list(np.atleast_1d(hessian_eigenvalue_list))

        if halo_masses is None:
            halo_masses = self.halo_masses
        if magnification_method in ['NEAR_FAR_SPLITTING', 'NEAR_FAR_SPLITTING_ADAPTIVE'] and \
                halo_masses is None and verbose:
            print('no halo masses were saved with this model; the near/far splitting calculation '
                  'will use a fixed near-field radius of 0.4 arcsec for every halo')

        # select the images to compute
        x_image, y_image = self.x_image, self.y_image
        if image_index is not None:
            indexes = [int(image_index)]
        else:
            indexes = list(range(0, n_image))
        x_image = np.array([x_image[i] for i in indexes])
        y_image = np.array([y_image[i] for i in indexes])
        grid_size_list = [grid_size_list[i] for i in indexes]
        grid_resolution_list = [grid_resolution_list[i] for i in indexes]
        if rotation_angle_list is not None:
            rotation_angle_list = [rotation_angle_list[i] for i in indexes]
        if hessian_eigenvalue_list is not None:
            hessian_eigenvalue_list = [hessian_eigenvalue_list[i] for i in indexes]

        source_model, kwargs_source = setup_gaussian_source(source_size_pc,
                                                            np.mean(source_x), np.mean(source_y),
                                                            self.astropy_cosmo, self.z_source)
        if use_batched_lens_model:
            lens_model_batch, kwargs_lens_batch = self._batched_lens_model()
        else:
            lens_model_batch, kwargs_lens_batch = self.lens_model, self.kwargs_lens_init

        magnifications, surfaces = magnification_finite_decoupled(
            source_model, kwargs_source, x_image, y_image,
            self.lens_model, self.kwargs_lens_init, self.kwargs_macro, self.index_lens_split,
            grid_size_list, grid_resolution_list,
            setup_decoupled_multiplane_lens_model_output=self._decoupled_setup_output(False),
            setup_decoupled_multiplane_lens_model_output_batch=self._decoupled_setup_output(
                use_batched_lens_model),
            magnification_method=magnification_method,
            rotation_angle_list=rotation_angle_list,
            hessian_eigenvalue_list=hessian_eigenvalue_list,
            lens_model_batch=lens_model_batch,
            kwargs_lens_batch=kwargs_lens_batch,
            halo_masses=halo_masses,
            fallback=fallback,
            MU_TOLERANCE=mu_tolerance,
            verbose=verbose)
        extents = []
        for counter in range(0, len(indexes)):
            half = grid_size_list[counter] / 2
            extents.append([x_image[counter] - half, x_image[counter] + half,
                            y_image[counter] - half, y_image[counter] + half])
        return np.array(magnifications), surfaces, extents

    def image_extents(self, grid_size_list=None):
        """
        The [xmin, xmax, ymin, ymax] bounding box of the ray tracing grid around each image, for use
        with imshow. Useful when plotting magnification surfaces that were computed elsewhere, for
        example the ones returned by the forward model itself.

        :param grid_size_list: size of the grid around each image in arcsec; defaults to the values
         used in the run
        :return: a list of extents, one per image
        """
        if grid_size_list is None:
            grid_size_list = self.spec.get('grid_size_list', None)
        if grid_size_list is None:
            source_size_pc = self.spec.get('source_size_pc', None)
            if source_size_pc is None:
                return [None] * len(self.x_image)
            from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size
            grid_size_list = [auto_raytracing_grid_size(source_size_pc)] * len(self.x_image)
        grid_size_list = np.atleast_1d(grid_size_list).astype(float)
        extents = []
        for index in range(0, len(self.x_image)):
            half = grid_size_list[index % len(grid_size_list)] / 2
            extents.append([self.x_image[index] - half, self.x_image[index] + half,
                            self.y_image[index] - half, self.y_image[index] + half])
        return extents

    def magnification_surface(self, image_index, **kwargs):
        """
        Ray trace a single lensed image.

        :param image_index: index of the image (0, 1, 2, ...)
        :param kwargs: passed to magnification_surfaces
        :return: magnification, surface, extent = [xmin, xmax, ymin, ymax]
        """
        magnifications, surfaces, extents = self.magnification_surfaces(image_index=image_index,
                                                                        **kwargs)
        return magnifications[0], surfaces[0], extents[0]

    def compare_magnification_methods(self, methods=('CIRCULAR_APERTURE', 'NEAR_FAR_SPLITTING'),
                                      **kwargs):
        """
        Compute the magnification surfaces with several algorithms and return them side by side.
        The intended use is checking the near/far splitting approximation against exact ray tracing:

            results = sim.compare_magnification_methods(['CIRCULAR_APERTURE', 'NEAR_FAR_SPLITTING'])
            exact = results['CIRCULAR_APERTURE']['magnifications']
            approximate = results['NEAR_FAR_SPLITTING']['magnifications']

        :param methods: an iterable of magnification method names
        :param kwargs: passed to magnification_surfaces
        :return: a dictionary keyed by method name, each entry a dictionary with keys
         'magnifications', 'surfaces', 'extents', 'flux_ratios' and 'runtime_seconds'
        """
        from time import time
        results = {}
        for method in methods:
            t0 = time()
            magnifications, surfaces, extents = self.magnification_surfaces(
                magnification_method=method, **kwargs)
            results[method] = {'magnifications': magnifications,
                               'surfaces': surfaces,
                               'extents': extents,
                               'flux_ratios': magnifications[1:] / magnifications[0]
                               if len(magnifications) > 1 else None,
                               'runtime_seconds': time() - t0}
        return results

    # -------------------------------------------------------------------- convenience

    def summary(self):
        """Print a short summary of the saved model."""
        print('lens ID:                 ', self.lens_id)
        print('random seed:             ', self.seed)
        print('z_lens / z_source:       ', self.z_lens, '/', self.z_source)
        print('number of deflectors:    ', len(self.lens_model_list))
        print('macromodel components:   ', [self.lens_model_list[i] for i in self.index_lens_split])
        print('image positions (x):     ', np.round(self.x_image, 5))
        print('image positions (y):     ', np.round(self.y_image, 5))
        if self.magnifications is not None:
            print('magnifications:          ', np.round(self.magnifications, 4))
        if self.flux_ratios is not None:
            print('flux ratios (model):     ', np.round(self.flux_ratios, 4))
        if self.spec.get('flux_ratios_data', None) is not None:
            print('flux ratios (data):      ', np.round(np.array(self.spec['flux_ratios_data']), 4))
        print('summary statistic:       ', self.summary_statistic)
        print('logL imaging data:       ', self.spec.get('logL_imaging_data', None))
        print('magnification method:    ', self.spec.get('magnification_method', None))
        print('source size (pc, FWHM):  ', self.spec.get('source_size_pc', None))
        if len(self.samples) > 0:
            print('sampled parameters:')
            for name, value in self.samples.items():
                print('    ' + str(name) + ': ' + str(value))

    @staticmethod
    def _plot_surface(ax, surface, extent, cmap='inferno'):
        """Draw a single magnification surface onto an axis, handling the adaptive tiling case."""
        if isinstance(surface, (list, tuple)):
            # the adaptive methods return [flux_array, tiling]
            from samana.image_magnification_util import plot_tiled_image
            plot_tiled_image(*surface, ax=ax, cmap=cmap)
        else:
            ax.imshow(surface, origin='lower', extent=extent, cmap=cmap)

    def plot_convergence(self, ax=None, grid_size=None, npix=300, center_x=0.0, center_y=0.0,
                         vmin=-0.075, vmax=0.075, cmap='seismic', with_critical_curves=True,
                         include_substructure_critical_curves=False, grid_scale=0.01,
                         critical_curve_color='k', show_images=True, **kwargs_convergence):
        """
        Plot the effective convergence, by default across the whole image plane, optionally with the
        critical curve of the full deflector overlaid. Uses matplotlib directly.

        :param ax: an existing matplotlib axis; a new figure is created if None
        :param grid_size: side length of the map in arcsec; defaults to the whole image plane
        :param npix: number of pixels per side
        :param center_x: center of the map in arcsec
        :param center_y: center of the map in arcsec
        :param vmin: lower limit of the color scale
        :param vmax: upper limit of the color scale
        :param cmap: matplotlib colormap
        :param with_critical_curves: bool; overlay the critical curves
        :param include_substructure_critical_curves: bool; compute the critical curves for the full
         deflector including every halo, rather than the smooth macromodel alone
        :param grid_scale: resolution of the grid used to find the critical curves
        :param critical_curve_color: color of the critical curves
        :param show_images: bool; mark the lensed image positions
        :param kwargs_convergence: passed to effective_convergence (subtract_macromodel,
         subtract_mean, decoupled)
        :return: the matplotlib axis
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 7))
        kappa, extent = self.effective_convergence(grid_size=grid_size, npix=npix,
                                                    center_x=center_x, center_y=center_y,
                                                    **kwargs_convergence)
        ax.imshow(kappa, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        if with_critical_curves:
            compute_window = extent[1] - extent[0]
            ra_crit_list, dec_crit_list, _, _ = self.critical_curves(
                include_substructure=include_substructure_critical_curves,
                compute_window=compute_window, grid_scale=grid_scale)
            for ra_crit, dec_crit in zip(ra_crit_list, dec_crit_list):
                ax.plot(ra_crit, dec_crit, color=critical_curve_color, lw=1.5, ls='--')
        if show_images:
            ax.scatter(self.x_image, self.y_image, color='k', marker='+', s=120)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        return ax

    def plot_flux_ratios(self, ax=None, magnifications=None):
        """
        Compare the model flux ratios to the measured ones as a bar chart.

        :param ax: an existing matplotlib axis; a new figure is created if None
        :param magnifications: image magnifications to use; defaults to the values saved in the run
        :return: the matplotlib axis
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4))
        if magnifications is not None and len(np.atleast_1d(magnifications)) > 1:
            magnifications = np.atleast_1d(magnifications)
            flux_ratios_model = magnifications[1:] / magnifications[0]
        else:
            flux_ratios_model = self.flux_ratios
        flux_ratios_data = self.spec.get('flux_ratios_data', None)
        if flux_ratios_model is None:
            ax.text(0.5, 0.5, 'no flux ratios saved', ha='center', va='center',
                    transform=ax.transAxes)
            return ax
        flux_ratios_model = np.atleast_1d(flux_ratios_model)
        positions = np.arange(0, len(flux_ratios_model))
        width = 0.4 if flux_ratios_data is not None else 0.8
        if flux_ratios_data is not None:
            flux_ratios_data = np.atleast_1d(flux_ratios_data)
            ax.bar(positions - width / 2, flux_ratios_data, width=width, color='0.5', label='data')
            ax.bar(positions + width / 2, flux_ratios_model, width=width, color='steelblue',
                   label='model')
            ax.legend(fontsize=9, frameon=False)
        else:
            ax.bar(positions, flux_ratios_model, width=width, color='steelblue', label='model')
        labels = ['B/A', 'C/A', 'D/A', 'E/A', 'F/A']
        ax.set_xticks(positions)
        ax.set_xticklabels([labels[i] if i < len(labels) else str(i) for i in positions])
        ax.set_ylabel('flux ratio', fontsize=10)
        statistic = self.summary_statistic
        if statistic is not None:
            ax.set_title('S = ' + str(np.round(np.atleast_1d(statistic)[0], 4)), fontsize=10)
        return ax

    def inspection_figure(self, magnifications=None, surfaces=None, extents=None, npix=200,
                          vmin=-0.075, vmax=0.075, zoom_grid_size=0.5, zoom_npix=100,
                          zoom_vmax=None, include_convergence_zoom=True,
                          decoupled_convergence=True, decoupled_convergence_zoom=False,
                          with_critical_curves=True, include_substructure_critical_curves=False,
                          grid_scale=0.02, figsize=None, source_size_pc=None,
                          magnification_method=None):
        """
        Build the figure used to decide whether a lens model is worth saving.

        Top row: the effective convergence across the whole image plane with the critical curve and
        the image positions, followed by the ray-traced magnification surface around each image.
        Bottom row: the model and measured flux ratios, followed by the effective convergence zoomed
        in on each image, so that the halos responsible for each image's flux can be seen.

        :param magnifications: magnifications already computed during the run. If given together
         with surfaces, no ray tracing is repeated
        :param surfaces: magnification surfaces already computed during the run
        :param extents: extents of those surfaces; derived from the saved grid sizes if None
        :param npix: resolution of the whole image plane convergence map
        :param vmin: lower limit of the convergence color scale
        :param vmax: upper limit of the convergence color scale
        :param zoom_grid_size: side length of the zoomed convergence maps in arcsec
        :param zoom_npix: resolution of the zoomed convergence maps
        :param zoom_vmax: color scale limit for the zoomed convergence maps; if None, each panel is
         scaled to its own 99th percentile so that the halos near the image are visible
        :param include_convergence_zoom: bool; include the bottom row of zoomed convergence maps
        :param decoupled_convergence: bool; compute the whole image plane convergence with the
         decoupled multi-plane approximation. This is what the lens modeling used and is far faster
         to evaluate on a grid than exact multi-plane ray tracing
        :param decoupled_convergence_zoom: bool; use the decoupled approximation for the zoomed maps
         as well. Off by default: the zoomed pixels are much smaller than the interpolation grid the
         approximation is built on, so the second derivatives pick up interpolation artifacts. The
         zoomed maps are small enough that exact ray tracing is cheap
        :param with_critical_curves: bool; overlay the critical curves
        :param include_substructure_critical_curves: bool; use the full deflector rather than the
         smooth macromodel for the critical curves
        :param grid_scale: resolution of the grid used to find the critical curves
        :param figsize: figure size; chosen automatically if None
        :param source_size_pc: passed to magnification_surfaces if the surfaces must be recomputed
        :param magnification_method: passed to magnification_surfaces if the surfaces must be
         recomputed
        :return: the matplotlib figure and the array of axes
        """
        import matplotlib.pyplot as plt

        if surfaces is None or magnifications is None:
            magnifications, surfaces, extents = self.magnification_surfaces(
                source_size_pc=source_size_pc, magnification_method=magnification_method)
        else:
            magnifications = np.atleast_1d(magnifications)
            if extents is None:
                extents = self.image_extents()
        n_image = len(surfaces)
        n_row = 2 if include_convergence_zoom else 1
        if figsize is None:
            figsize = (3.6 * (n_image + 1), 3.8 * n_row)
        fig, axes = plt.subplots(n_row, n_image + 1, figsize=figsize, squeeze=False)

        try:
            self.plot_convergence(ax=axes[0, 0], npix=npix, vmin=vmin, vmax=vmax,
                                  with_critical_curves=with_critical_curves,
                                  include_substructure_critical_curves=
                                  include_substructure_critical_curves,
                                  grid_scale=grid_scale, decoupled=decoupled_convergence)
        except Exception as exception:
            print('failed to plot the convergence with critical curves: ' + str(exception))
            self.plot_convergence(ax=axes[0, 0], npix=npix, vmin=vmin, vmax=vmax,
                                  with_critical_curves=False, decoupled=decoupled_convergence)
        axes[0, 0].set_title(r'$\kappa - \kappa_{\rm{macro}}$, full image plane', fontsize=10)

        image_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        for index in range(0, n_image):
            label = image_labels[index] if index < len(image_labels) else str(index)
            ax = axes[0, index + 1]
            self._plot_surface(ax, surfaces[index], extents[index])
            ax.set_title(label + r': $\mu = $' + str(np.round(magnifications[index], 2)),
                         fontsize=10)
            if include_convergence_zoom:
                ax = axes[1, index + 1]
                kappa, extent = self.convergence_around_image(index, grid_size=zoom_grid_size,
                                                              npix=zoom_npix,
                                                              decoupled=decoupled_convergence_zoom)
                if zoom_vmax is None:
                    # scale each panel to its own dynamic range so that nearby halos are visible
                    scale = np.percentile(np.absolute(kappa - np.median(kappa)), 99)
                    scale = max(float(scale), 1e-6)
                else:
                    scale = zoom_vmax
                ax.imshow(kappa - np.median(kappa), origin='lower', extent=extent, cmap='seismic',
                          vmin=-scale, vmax=scale)
                ax.scatter([self.x_image[index]], [self.y_image[index]], color='k', marker='+',
                           s=120)
                ax.set_title(label + r': $\kappa - \kappa_{\rm{macro}}$, $\pm$' +
                             str(float(np.format_float_scientific(scale, precision=1))),
                             fontsize=10)

        if include_convergence_zoom:
            self.plot_flux_ratios(ax=axes[1, 0], magnifications=magnifications)

        title = 'lens ' + str(self.lens_id) + ', seed ' + str(self.seed) + ', ' + \
                str(len(self.lens_model_list) - len(self.index_lens_split)) + ' halos'
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        return fig, axes

    def quick_look(self, npix=300, vmin=-0.075, vmax=0.075, figsize=(16, 4.5),
                   source_size_pc=None, magnification_method=None,
                   include_substructure_critical_curves=False, grid_scale=0.01):
        """
        Make a single figure with the effective convergence across the whole image plane (with the
        critical curve and the image positions) and the magnification surface around each image.
        Uses matplotlib directly rather than any lenstronomy plotting routine.

        :param npix: resolution of the convergence map
        :param vmin: lower limit of the convergence color scale
        :param vmax: upper limit of the convergence color scale
        :param figsize: figure size
        :param source_size_pc: passed to magnification_surfaces
        :param magnification_method: passed to magnification_surfaces
        :param include_substructure_critical_curves: bool; show the critical curve of the full
         deflector including substructure rather than the smooth macromodel alone
        :param grid_scale: resolution of the grid used to find the critical curves
        :return: the matplotlib figure and the list of axes
        """
        import matplotlib.pyplot as plt

        magnifications, surfaces, extents = self.magnification_surfaces(
            source_size_pc=source_size_pc, magnification_method=magnification_method)

        n_image = len(surfaces)
        fig, axes = plt.subplots(1, n_image + 1, figsize=figsize)
        try:
            self.plot_convergence(ax=axes[0], npix=npix, vmin=vmin, vmax=vmax,
                                  include_substructure_critical_curves=
                                  include_substructure_critical_curves, grid_scale=grid_scale)
        except Exception as exception:
            print('failed to overlay critical curves: ' + str(exception))
            self.plot_convergence(ax=axes[0], npix=npix, vmin=vmin, vmax=vmax,
                                  with_critical_curves=False)
        axes[0].set_title(r'$\kappa - \kappa_{\rm{macro}}$')

        image_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        for index, (surface, image_extent) in enumerate(zip(surfaces, extents)):
            ax = axes[index + 1]
            self._plot_surface(ax, surface, image_extent)
            label = image_labels[index] if index < len(image_labels) else str(index)
            ax.set_title(label + r': $\mu = $' + str(np.round(magnifications[index], 2)))
        plt.tight_layout()
        return fig, axes

    def plot_method_comparison(self, methods=('CIRCULAR_APERTURE', 'NEAR_FAR_SPLITTING'),
                               figsize=None, **kwargs):
        """
        Compute the magnification surfaces with several algorithms and plot them in a grid, one row
        per method, one column per image. Titles report the magnification of each image and, for
        every method after the first, the fractional difference relative to the first method.

        :param methods: an iterable of magnification method names
        :param figsize: figure size; chosen automatically if None
        :param kwargs: passed to magnification_surfaces
        :return: the matplotlib figure, the axes, and the dictionary returned by
         compare_magnification_methods
        """
        import matplotlib.pyplot as plt
        methods = list(methods)
        results = self.compare_magnification_methods(methods=methods, **kwargs)
        n_image = len(results[methods[0]]['surfaces'])
        if figsize is None:
            figsize = (4 * n_image, 4 * len(methods))
        fig, axes = plt.subplots(len(methods), n_image, figsize=figsize, squeeze=False)
        reference = results[methods[0]]['magnifications']
        image_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        for row, method in enumerate(methods):
            magnifications = results[method]['magnifications']
            for column in range(0, n_image):
                ax = axes[row, column]
                self._plot_surface(ax, results[method]['surfaces'][column],
                                   results[method]['extents'][column])
                label = image_labels[column] if column < len(image_labels) else str(column)
                title = label + r': $\mu = $' + str(np.round(magnifications[column], 3))
                if row > 0:
                    difference = (magnifications[column] - reference[column]) / reference[column]
                    title += '\n' + str(np.round(100 * difference, 2)) + '% vs ' + methods[0]
                ax.set_title(title, fontsize=10)
                if column == 0:
                    ax.set_ylabel(method, fontsize=11)
        plt.tight_layout()
        return fig, axes, results

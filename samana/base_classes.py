from copy import deepcopy

from samana.image_magnification_util import setup_gaussian_source
from samana.forward_model_util import flux_ratio_summary_statistic
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size, auto_raytracing_grid_resolution

import numpy as np


class DarkMatterBaseClass(object):

    def __init__(self, astropy_cosmo):
        """
        astropy_cosmo: an instance of Astropy cosmology
        """
        self.astropy_cosmo = astropy_cosmo

    def halo_modifications(self, realization, realization_dict):
        """
        Perform additional modifications on halo properties after process halos
        This method defaults to doing nothing, as it is often not used
        :param realization: an instance of SingleRealization
        :param realization_dict: keyword arguments for the DM model
        :return: realization
        """
        return realization

    def process_halos(self, realization):
        """
        This method takes a given realization and performs operations such as removing very low mass halos far from lensed
        images, cuts on bound mass, etc
        :param realization:
        :return: realization, kwargs_mass_sheet_correction
        """
        raise Exception("This method should be implemented for each user defined class")

    def add_globular_clusters(self, kwargs_globular_clusters):
        """
        This method adds globular clusters
        :param kwargs_globular_clusters:
        :return: return an instance of realization with GCs added
        """
        raise Exception("This method should be implemented for each user defined class")

    def __call__(self, z_lens, z_source, realization_dict):
        """

        :param z_lens: main deflector redshift
        :param z_source: source redshift
        :param realization_dict: a dictionary of keyword arguments
        :return: an instance of SingleRealization in pyHalo
        """
        raise Exception("This method should be implemented for each user defined class")

class ImageMagnificationBaseClass(object):

    def __init__(self):
        pass

    def __call__(self,
                 source_dict,
                 source_x,
                 source_y,
                 astropy_cosmo,
                 z_source):
        """

        :param source_dict: dictionary of keyword arguments for the lensed quasar source
        :param source_x: center of source x coordinate
        :param source_y: center of source y coordinate
        :param astropy_cosmo: an instance of astropy
        :param z_source: source redshift
        :return: magnifications, images, stat, flux_ratios, flux_ratios_data
        """
        raise Exception("This method should be implemented for each user defined class")

class SingleGaussianMagnification(object):
    """Compute the magnification of a lensed quasar image for a quad lens"""

    def __init__(self, astropy_cosmo,
                 rescale_grid_size,
                 rescale_grid_resolution,
                 magnification_method,
                 rotation_angle_list,
                 hessian_eigenvalue_list):
        """

        :param astropy_cosmo:
        :param rescale_grid_size:
        :param rescale_grid_resolution:
        :param magnification_method:
        :param rotation_angle_list:
        :param hessian_eigenvalue_list:
        """
        self.astropy_cosmo = astropy_cosmo
        self.rescale_grid_size = rescale_grid_size
        self.rescale_grid_resolution = rescale_grid_resolution
        self.magnification_method = magnification_method
        self.rotation_angle_list = rotation_angle_list
        self.hessian_eigenvalue_list = hessian_eigenvalue_list

    def __call__(self, source_dict, source_x, source_y, data_class, model_class,
                 lens_model_init, kwargs_lens_init, kwargs_solution, setup_decoupled_multiplane_lens_model_output):

        source_model_quasar, kwargs_source = setup_gaussian_source(source_dict['source_size_pc'],
                                                                   np.mean(source_x), np.mean(source_y),
                                                                   self.astropy_cosmo, data_class.z_source)
        grid_size_base = auto_raytracing_grid_size(source_dict['source_size_pc'])
        grid_resolution = self.rescale_grid_resolution * auto_raytracing_grid_resolution(source_dict['source_size_pc'])
        if isinstance(self.rescale_grid_size, list) or isinstance(self.rescale_grid_size, np.ndarray):
            assert len(self.rescale_grid_size) == 4
            grid_size_list = []
            for rescale_size in self.rescale_grid_size:
                grid_size_list.append(rescale_size * grid_size_base)
        else:
            grid_size_list = [self.rescale_grid_size * grid_size_base] * 4
        # we pass in setup_decoupled_multiplane_lens_model_output, the decoupled multiplane parameters
        # computed for the proposed macromodel in setup_kwargs_model
        magnifications, images = model_class.image_magnification_gaussian(source_model_quasar,
                                                                              kwargs_source,
                                                                              lens_model_init,
                                                                              kwargs_lens_init,
                                                                              kwargs_solution,
                                                                              grid_size_list,
                                                                              grid_resolution,
                                                                              setup_decoupled_multiplane_lens_model_output,
                                                                              magnification_method=self.magnification_method,
                                                                              rotation_angle_list=self.rotation_angle_list,
                                                                              hessian_eigenvalue_list=self.hessian_eigenvalue_list)
        flux_uncertainty = None
        stat, flux_ratios, flux_ratios_data = flux_ratio_summary_statistic(data_class.magnifications,
                                                                               magnifications,
                                                                                flux_uncertainty,
                                                                               data_class.keep_flux_ratio_index,
                                                                               data_class.uncertainty_in_fluxes)
        return magnifications, images, stat, flux_ratios, flux_ratios_data

class DoubleGaussianMagnification(object):
    """Compute the magnification of a lensed quasar image for a quad lens for two different
    source sizes at the same position in the source plane"""

    def __init__(self, astropy_cosmo,
                 rescale_grid_size,
                 rescale_grid_resolution,
                 magnification_method,
                 rotation_angle_list,
                 hessian_eigenvalue_list):
        """

        :param astropy_cosmo:
        :param rescale_grid_size:
        :param rescale_grid_resolution:
        :param magnification_method:
        :param rotation_angle_list:
        :param hessian_eigenvalue_list:
        """
        self.astropy_cosmo = astropy_cosmo
        self.rescale_grid_size = rescale_grid_size
        self.rescale_grid_resolution = rescale_grid_resolution
        self.magnification_method = magnification_method
        self.rotation_angle_list = rotation_angle_list
        self.hessian_eigenvalue_list = hessian_eigenvalue_list
        self._single_source_magnification = SingleGaussianMagnification(astropy_cosmo,
                 rescale_grid_size,
                 rescale_grid_resolution,
                 magnification_method,
                 rotation_angle_list,
                 hessian_eigenvalue_list)

    def __call__(self, source_dict, source_x, source_y, data_class, model_class,
                 lens_model_init, kwargs_lens_init, kwargs_solution, setup_decoupled_multiplane_lens_model_output):

        source_dict_copy_1 = deepcopy(source_dict)
        source_dict_copy_2 = deepcopy(source_dict)
        source_dict_copy_1['source_size_pc'] = source_dict['source_size_pc_1']
        source_dict_copy_2['source_size_pc'] = source_dict['source_size_pc_2']

        mags_1, images_1, stat_1, flux_ratios_1, flux_ratios_data = self._single_source_magnification(source_dict_copy_1,
                                                                                                        source_x,
                                                                                                        source_y,
                                                                                                        data_class,
                                                                                                        model_class,
                                                                                                        lens_model_init,
                                                                                                        kwargs_lens_init,
                                                                                                        kwargs_solution,
                                                                                                        setup_decoupled_multiplane_lens_model_output)
        mags_2, images_2, stat_2, flux_ratios_2, _ = self._single_source_magnification(
            source_dict_copy_2,
            source_x,
            source_y,
            data_class,
            model_class,
            lens_model_init,
            kwargs_lens_init,
            kwargs_solution,
            setup_decoupled_multiplane_lens_model_output)

        magnifications = np.append(mags_1, mags_2)
        images = images_1 + images_2
        stat = np.append(stat_1, stat_2)
        flux_ratios = np.append(flux_ratios_1, flux_ratios_2)
        return magnifications, images, stat, flux_ratios, flux_ratios_data








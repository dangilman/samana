======
samana
======

This is a code repository meant to facilitate easy access to galaxy-scale strong lens data. 

.. image:: https://github.com/dangilman/samana/blob/main/mosaic_figure.pdf
        :target: https://github.com/dangilman/samana/blob/main/mosaic_figure

Features
--------
This repository stores (in python array form) reduced data, PSF models, and lens models for 28 quadruply-imaged quasars. The data are all in the Data module, and the lens models are in the Model module. The lens modeling is intended to be performed with lenstronomy (https://github.com/lenstronomy/lenstronomy). 

Notebooks that do lens modeling for 24 systems can be found in notebooks/baseline_lensmodels directory. 

This repository also includes functions and methods used for forward modeling quad lenses with dark matter substructure. The functionality is inside the "forward_model.py" file. An example script that shows how to run a dark matter inference with the data stored in this repo can be found in /notebooks/JWST_DM_survey_IV/example_lauch_script_0435.py. 

The scripts used for the analysis presented in the JWST lensed quasar DM survey papers III and IV can also be found in the JWST_DM_survey_IV directory. 

Credits
-------
When using this repository, please cite JWST lensed quasar DM survey IV, Gilman et al. (2025)

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

* Free software: MIT license


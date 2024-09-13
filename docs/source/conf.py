# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pytRIBS'
copyright = '2024, L. Wren Raming, C. Josh Cederstrom, Enrique R. Vivoni, among others'
author = 'L. Wren Raming, C. Josh Cederstrom, Enrique R. Vivoni, among others'
release = '2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

language = 'english'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Optional: For Google and NumPy style docstrings
    'sphinx.ext.autosummary',  # Optional: For automatic summary generation
    'sphinx.ext.viewcode',  # Optional: To include links to source code
]

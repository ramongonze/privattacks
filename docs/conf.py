# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Ensure the     project is importable
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))

project = 'Privattacks'
copyright = '2025, Ramon Gonçalves Gonze'
author = 'Ramon Gonçalves Gonze'

from importlib.metadata import version as pkg_version
release = pkg_version("privattacks")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',  # For generating documentation from docstrings
    'sphinx.ext.napoleon', # For Google-style or NumPy-style docstrings
    'sphinx.ext.viewcode', # Include links to source code
    'sphinx.ext.mathjax',  # Renders LaTeX equations using MathJax
    'myst_parser' # Enables the usage of Markdown for the documentation
]
autosummary_generate = True

myst_enable_extensions = [
    "dollarmath",      # <-- enables $ and $$ math blocks
    "amsmath",         # optional, for more LaTeX environments
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# Add the RTD theme
import sphinx_rtd_theme

# Theme configuration
html_theme = 'sphinx_rtd_theme'

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Privattacks'
copyright = '2025, Ramon G. Gonze'
author = 'Ramon G. Gonze'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Ensure the     project is importable
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.autodoc',  # For generating documentation from docstrings
    'sphinx.ext.napoleon', # For Google-style or NumPy-style docstrings
    'sphinx.ext.viewcode', # Include links to source code
    'sphinx.ext.mathjax'  # Renders LaTeX equations using MathJax
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# Add the RTD theme
import sphinx_rtd_theme

# Theme configuration
html_theme = 'sphinx_rtd_theme'

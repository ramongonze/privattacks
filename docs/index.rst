.. Privattacks documentation master file, created by
   sphinx-quickstart on Tue Jan  7 20:37:42 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Privattacks documentation
=========================

Privattacks is a Python package for evaluating privacy risks in tabular datasets. It provides tools to quantify re-identification and attribute inference vulnerabilities based on combinations of quasi-identifiers an adversary knows about a target. The package supports both high-level and granular analysis, including per-record vulnerability distributions, multiprocessing for efficiency, and flexible input handling via pandas DataFrames and other sources.

Available tools
---------------
- Prior vulnerability for re-identification and attribute inference.
- Posterior vulnerability for a given combination of qids and/or sensitive attribute.
- Posterior vulnerability for a subset of all possible combinations of QIDs.
- Parellel code.
- Generate the histogram of vulnerabilities (i.e., vulnerability per record).

Installation
------------
You can install via PyPI:

.. code-block:: python

   pip install privattacks

or manuallly by copying this repository to your local machine and running:

.. code-block:: python
   
   pip install path/to/privattacks

To verify if the package was install corretly, you can run tests:

.. code-block:: python

   cd path/to/privattacks
   python -m unittest discover tests

License
-------

This project is licensed under the `MIT License <https://opensource.org/license/mit>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/attack_formalization
   source/getting_started
   source/privattacks

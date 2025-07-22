Getting Started
===============

This guide walks you through a basic usage example of the `privattacks` Python package, showing how to load data, define quasi-identifiers, and evaluate both re-identification and attribute inference vulnerabilities.

Data Preparation
----------------

We begin by creating a simple synthetic dataset using `pandas`.

.. code-block:: python

    import pandas as pd
    import numpy as np
    import privattacks

    df = pd.DataFrame({
        "age":[20,30,30,30,30,55,55,55],
        "education":["Master", "High School", "High School", "PhD", "PhD", "Bachelor", "Bachelor", "Bachelor"],
        "income":["low", "medium", "low", "medium", "medium", "high", "high", "medium"]
    })
    display(df)

.. list-table::
   :header-rows: 1
   :widths: 10 20 10

   * - age
     - education
     - income
   * - 20
     - Master
     - low
   * - 30
     - High School
     - medium
   * - 30
     - High School
     - low
   * - 30
     - PhD
     - medium
   * - 30
     - PhD
     - medium
   * - 55
     - Bachelor
     - high
   * - 55
     - Bachelor
     - high
   * - 55
     - Bachelor
     - medium

This dataset contains three columns:

- `age` and `education` are considered *quasi-identifiers (QIDs)*.
- `income` is treated as a *sensitive attribute*.

Defining QIDs and Sensitive Attribute
-------------------------------------

.. code-block:: python

    qids = ["age", "education"]
    sensitive = "income" # It's possible to run the attack for a list of sensitive attributes

These variables are passed to the `privattacks` data wrapper and attack engine.

.. code-block:: python

    data = privattacks.data.Data(dataframe=df)
    attack = privattacks.attacks.Attack(data)

.. note::
    The dataset can be read directly from a file, see :mod:`privattacks.data`.

Evaluating Prior and Posterior Vulnerabilities
----------------------------------------------

We first calculate the prior and posterior vulnerabilities for:

- Re-identification attacks
- Attribute inference attacks

.. code-block:: python

    prior_reid = attack.prior_vulnerability("reid")
    prior_ai = attack.prior_vulnerability("ai", sensitive)
    posterior_reid = attack.posterior_vulnerability("reid", qids)
    posterior_ai = attack.posterior_vulnerability("ai", qids, sensitive)

    print(f"Re-identification\n"+\
          f"Prior vulnerability; {prior_reid:.5f}\n"+\
          f"Posterior vulnerability: {posterior_reid:.5f}")

    print(f"\nAttribute inference - {sensitive}\n"+\
          f"Prior vulnerability; {prior_ai[sensitive]:.5f}\n"+\
          f"Posterior vulnerability: {posterior_ai[sensitive]:.5f}")

.. code-block:: python

    Re-identification
    Prior vulnerability; 0.12500
    Posterior vulnerability: 0.50000

    Attribute inference - income
    Prior vulnerability; 0.33333
    Posterior vulnerability: 0.75000

This provides an initial assessment of the risk posed by attackers with and without auxiliary information (quasi-identifiers).

Using the Optimized Evaluation Method
-------------------------------------

For convenience and performance, you can run both attacks in a single call:

.. code-block:: python

    posteriors = attack.posterior_vulnerability("all", qids, sensitive)

    print(f"Re-identification\n"+\
          f"Posterior vulnerability: {posteriors['reid']:.5f}")

    print(f"\nAttribute inference - {sensitive}\n"+\
          f"Posterior vulnerability: {posteriors['ai'][sensitive]:.5f}")

.. code-block::

    Re-identification
    Posterior vulnerability: 0.50000

    Attribute inference - income
    Posterior vulnerability: 0.75000

Analyzing Individual Vulnerabilities
------------------------------------

You can also inspect the distribution of vulnerabilities per record using the `distribution=True` flag.

.. code-block:: python

    posterior_reid, hist_reid = attack.posterior_vulnerability("reid", qids, distribution=True)
    print(f"Re-identification - distribution on records\n"+\
          f"{hist_reid}\nMean of the distribution: {np.mean(hist_reid)}")

    posterior_reid, hist_ai = attack.posterior_vulnerability("ai", qids, sensitive, distribution=True)
    print("\nAttribute inference - distribution on records\n"+\
          f"{sensitive}:\n{hist_ai[sensitive]}\nMean of the distribution: {np.mean(hist_ai[sensitive])}")

.. code-block:: python
    
    Re-identification - distribution on records
    [1.         0.5        0.5        0.5        0.5        0.33333333
    0.33333333 0.33333333]
    Mean of the distribution: 0.5

    Attribute inference - distribution on records
    income:
    [1.         0.5        0.5        1.         1.         0.66666667
    0.66666667 0.66666667]
    Mean of the distribution: 0.75

Optimized Method with Distributions
-----------------------------------

The optimized method also supports distributions:

.. code-block:: python

    posteriors = attack.posterior_vulnerability("all", qids, sensitive, distribution=True)
    posterior_reid, hist_reid = posteriors["reid"]
    posteriors_ai, hist_ai = posteriors["ai"]

    print("Re-identification histogram\n"+\
          f"{hist_reid}")

    print("\nAttribute inference histogram\n"+\
          f"{sensitive}:\n"+\
          f"{hist_ai[sensitive]}")

.. code-block:: python

    Re-identification histogram
    [1.         0.5        0.5        0.5        0.5        0.33333333
    0.33333333 0.33333333]

    Attribute inference histogram
    income:
    [1.         0.5        0.5        1.         1.         0.66666667
    0.66666667 0.66666667]

Evaluating Multiple Combinations of QIDs
----------------------------------------

You can evaluate the vulnerabilities for *all combinations* of the QIDs (e.g., single attributes, pairs, etc.):

.. code-block:: python

    combinations = list(range(1, len(qids)+1))  # Sizes 1 to len(qids)

    results_reid = attack.posterior_vulnerability(
        atk="reid",
        qids=qids,
        combinations=combinations,
        n_processes=2
    )
    display(results_reid)

.. list-table::
   :header-rows: 1
   :widths: 5 15 20 20

   * - 
     - n_qids
     - qids
     - posterior_reid
   * - 0
     - 1
     - age
     - 0.375000000
   * - 1
     - 1
     - education
     - 0.500000000
   * - 2
     - 2
     - age,education
     - 0.500000000

.. code-block:: python

    results_ai = attack.posterior_vulnerability(
        atk="ai",
        qids=qids,
        sensitive=sensitive,
        combinations=combinations,
        distribution=True,
        n_processes=2
    )
    display(results_ai)

.. list-table::
   :header-rows: 1
   :widths: 5 15 20 20 35

   * - 
     - n_qids
     - qids
     - posterior_income
     - posterior_income_record
   * - 0
     - 1
     - age
     - 0.750000000
     - [1.00000000, 0.50000000, 0.50000000, 1.0000000...
   * - 1
     - 1
     - education
     - 0.750000000
     - [1.00000000, 0.50000000, 0.50000000, 1.0000000...
   * - 2
     - 2
     - age,education
     - 0.750000000
     - [1.00000000, 0.50000000, 0.50000000, 1.0000000...

You can run both types of attack simultaneously for all combinations:

.. code-block:: python

    results = attack.posterior_vulnerability(
        atk="all",
        qids=qids,
        sensitive=sensitive,
        combinations=combinations,
        n_processes=2
    )
    display(results)

.. list-table::
   :header-rows: 1
   :widths: 5 15 20 20 20

   * - 
     - n_qids
     - qids
     - posterior_reid
     - posterior_income
   * - 0
     - 1
     - age
     - 0.375000000
     - 0.750000000
   * - 1
     - 1
     - education
     - 0.500000000
     - 0.750000000
   * - 2
     - 2
     - age,education
     - 0.500000000
     - 0.750000000

This approach provides a comprehensive evaluation of how different combinations of quasi-identifiers affect vulnerability.


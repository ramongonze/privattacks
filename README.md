# privattacks
Privattacks is a Python package for evaluating privacy risks in tabular datasets. It provides tools to quantify re-identification and attribute inference vulnerabilities based on combinations of quasi-identifiers an adversary knows about a target. The package supports both high-level and granular analysis, including per-record vulnerability distributions, multiprocessing for efficiency, and flexible input handling via pandas DataFrames and other sources.

## Supported attacks
- Re-identification
- Attribute Inference

### Settings
- Posterior vulnerability for a given combination of qids and/or sensitive attribute.
- Posterior vulnerability for a subset of all possible combinations of QIDs.
- Parellel code.
- Generate the histogram of vulnerabilities (i.e., vulnerability per record).

## Usage
Copy this repository to your local machine and, to install it, use
```
pip install path/to/privattacks
```

To verify if the package was install corretly, you can run tests:

```
cd path/to/privattacks
python -m unittest discover tests
```

## Documentation
The documentation is available in HTML on `docs/_build/html`.
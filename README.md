# privattacks
Privattacks is a Python package for evaluating privacy risks in tabular datasets. It provides tools to quantify re-identification and attribute inference vulnerabilities based on combinations of quasi-identifiers an adversary knows about a target. The package supports both high-level and granular analysis, including per-record vulnerability distributions, multiprocessing for efficiency, and flexible input handling via pandas DataFrames and other sources.

## Documentation
The documentation is available at [https://privattacks.readthedocs.io](https://privattacks.readthedocs.io).

## Available tools
- Prior vulnerability for re-identification and attribute inference.
- Posterior vulnerability for a given combination of qids and/or sensitive attribute.
- Posterior vulnerability for a subset of all possible combinations of QIDs.
- Parellel code.
- Generate the histogram of vulnerabilities (i.e., vulnerability per record).

## Installation
You can install via PyPI:
```
pip install privattacks
```

or manuallly by copying this repository to your local machine and running:

```
pip install path/to/privattacks
```

To verify if the package was installed correctly, you can run tests:

```
cd path/to/privattacks
python -m unittest discover tests
```

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

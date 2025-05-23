# privattacks
Python package for re-identification and attribute inference attacks.

## Supported attacks
- Probabilistic Re-identification.
- Probabilistic Attribute Inference.
- Vulnerability for a given combination of qids and/or sensitive attribute.
- Vulnerability for a subset of all possible combinations of QIDs. The parameters will include a min # qids and a max # qids.
- Parellel code.
- Generate the histogram of vulnerabilities.
- For each # qids, select the combination that produced the maximum vulnerability.

## Usage
Copy this repository to your local machine and, to install it, use
```
pip install path/to/privattacks
```

To verify if the package was install corretly, you can run tests:

```
cd path/to/privattacks
pyton -m unittest discover tests
```

## Documentation
The documentation is available in HTML on `docs/_build/html`.
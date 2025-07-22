# privattacks
Python package for re-identification and attribute inference attacks.

## Supported attacks
- Re-identification.
- Attribute Inference.


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
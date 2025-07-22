import setuptools

# Read dependencies from requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        # Filter out comments and empty lines
        return [line.strip() for line in lines if line.strip() and not line.startswith('#')]

install_requires = parse_requirements('requirements.txt')
docs_requires = [
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx_rtd_theme"
]
install_requires += docs_requires

setuptools.setup( 
    name="privattacks",
    version="1.0",
    python_requires=">=3.9",
    description="Privattacks is a Python package for evaluating privacy risks in tabular datasets. It provides tools to quantify re-identification and attribute inference vulnerabilities based on combinations of quasi-identifiers an adversary knows about a target. The package supports both high-level and granular analysis, including per-record vulnerability distributions, multiprocessing for efficiency, and flexible input handling via pandas DataFrames and other sources.",
    author="Ramon Gonçalves Gonze",
    author_email="ramongonze@gmail.com",
    url="https://github.com/ramongonze/privattacks",
    packages=setuptools.find_packages(),
    install_requires=install_requires
)

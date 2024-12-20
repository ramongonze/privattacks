import setuptools

install_requires = [
    "pandas",
    "numpy",
    "tqdm",
    "pyreadr"
]

setuptools.setup(
    name="privattacks",
    version="1.0",
    python_requires=">=3.9",
    description="Python library for privacy attacks.",
    author="Ramon Gonçalves Gonze",
    author_email="ramongonze@gmail.com",
    url="https://github.com/ramongonze/privattacks",
    packages=setuptools.find_packages(),
    install_requires=install_requires
)

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import find_packages, setup

setup(
    name="sepsis",
    version="0.0.1",
    description="Models for sepsis",
    url="https://github.com/som-shahlab/sepsis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas>=1.0.0"
    ],
)

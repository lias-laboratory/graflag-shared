"""Setup script for graflag_bond package."""

from setuptools import setup

setup(
    name="graflag_bond",
    version="1.1.1",
    description="Universal PyGOD detector wrapper for GraFlag BOND methods",
    author="GraFlag Team",
    url="https://github.com/lias-laboratory/graflag-shared",
    project_urls={
        "Documentation": "https://lias-laboratory.github.io/graflag/",
        "Source": "https://github.com/lias-laboratory/graflag-shared/tree/main/libs/graflag_bond",
    },
    packages=["graflag_bond"],
    package_dir={"graflag_bond": "."},
    install_requires=[
        # train.py does `from graflag_runner import ResultWriter, info, ...`.
        # It worked only because every bond Dockerfile happens to pip-install
        # both on one line; nothing made that a real dependency.
        "graflag-runner>=1.1.0",
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "pygod>=1.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.7",
)

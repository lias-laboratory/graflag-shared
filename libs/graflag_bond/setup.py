"""Setup script for graflag_bond package."""

from setuptools import setup

setup(
    name="graflag_bond",
    version="1.0.0",
    description="Universal PyGOD detector wrapper for GraFlag BOND methods",
    author="GraFlag Team",
    packages=["graflag_bond"],
    package_dir={"graflag_bond": "."},
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "pygod>=1.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.7",
)

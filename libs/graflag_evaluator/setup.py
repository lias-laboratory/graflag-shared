"""Setup script for graflag_evaluator package."""

from setuptools import setup

setup(
    name="graflag_evaluator",
    version="1.0.0",
    description="Evaluation framework for graph anomaly detection methods in GraFlag",
    author="GraFlag Team",
    packages=["graflag_evaluator"],
    package_dir={"graflag_evaluator": "."},
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.7",
)

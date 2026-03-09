"""Setup script for graflag_runner package."""

from setuptools import setup

setup(
    name="graflag_runner",
    version="1.0.0",
    description="Framework for executing graph anomaly detection methods with resource monitoring",
    author="GraFlag Team",
    packages=["graflag_runner"],
    package_dir={"graflag_runner": "."},
    install_requires=[
        "psutil>=5.8.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "graflag-run=graflag_runner.runner:main",
        ],
    },
)

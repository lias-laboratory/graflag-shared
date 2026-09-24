"""Setup script for graflag_runner package."""

from setuptools import setup

setup(
    name="graflag_runner",
    version="1.1.1",
    description="Framework for executing graph anomaly detection methods with resource monitoring",
    author="GraFlag Team",
    url="https://github.com/lias-laboratory/graflag-shared",
    project_urls={
        "Documentation": "https://lias-laboratory.github.io/graflag/",
        "Source": "https://github.com/lias-laboratory/graflag-shared/tree/main/libs/graflag_runner",
    },
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

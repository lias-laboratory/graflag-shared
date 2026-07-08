"""Setup script for graflag_data package."""

from setuptools import setup

setup(
    name="graflag_data",
    version="1.0.0",
    description="Dataset metadata + downloader for GraFlag benchmarks.",
    author="GraFlag Team",
    packages=["graflag_data"],
    package_dir={"graflag_data": "."},
    python_requires=">=3.7",
    extras_require={
        "gdrive": ["gdown>=5.0"],
    },
    entry_points={
        "console_scripts": [
            "graflag-data=graflag_data.cli:main",
        ],
    },
)

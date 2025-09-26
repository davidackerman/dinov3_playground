#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for dinov3_playground package
"""

from setuptools import setup, find_packages
import os


# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return (
        "DINOv3 feature extraction and fine-tuning playground for cell image analysis"
    )


# Read the license file
def read_license():
    license_path = os.path.join(os.path.dirname(__file__), "LICENSE")
    if os.path.exists(license_path):
        with open(license_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="dinov3_playground",
    version="0.1.0",
    description="DINOv3 feature extraction and fine-tuning playground for cell image analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="GitHub Copilot",
    author_email="copilot@github.com",
    url="https://github.com/davidackerman/dinov3_playground",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy",
        "pillow",
        "transformers>=4.20.0",
        "matplotlib",
        "scikit-image",
        "scipy",
        "zarr",
        "funlib.geometry",
        "cellmap_flow",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords="dinov3 computer-vision machine-learning cell-biology image-analysis",
    project_urls={
        "Homepage": "https://github.com/davidackerman/dinov3_playground",
        "Repository": "https://github.com/davidackerman/dinov3_playground",
        "Issues": "https://github.com/davidackerman/dinov3_playground/issues",
    },
    include_package_data=True,
    zip_safe=False,
)

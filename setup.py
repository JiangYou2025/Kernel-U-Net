"""
Setup configuration for Kernel U-Net package
"""

from setuptools import setup, find_packages
import os

# Read version from src/version.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'src', 'version.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('__version__ ='):
                # Extract the version string from the line
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.0.0"

# Read long description from README
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'Readme.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="kernel-unet",
    version=get_version(),
    author="Kernel U-Net Team",
    author_email="contact@kunet.ai",
    description="A flexible U-Net architecture with customizable kernels",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/kernel-unet/Kernel-U-Net",
    project_urls={
        "Bug Tracker": "https://github.com/kernel-unet/Kernel-U-Net/issues",
        "Documentation": "https://github.com/kernel-unet/Kernel-U-Net/wiki",
        "Source Code": "https://github.com/kernel-unet/Kernel-U-Net",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kunet-info=src:print_package_info",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "neural networks", "unet", "pytorch", "machine learning"],
) 
"""
Version information for Kernel U-Net package.

This module provides version information and utilities for the package.
"""

import datetime
from typing import Tuple

# Version components
MAJOR = 1
MINOR = 0
PATCH = 0
PRE_RELEASE = None  # None, 'alpha', 'beta', 'rc'
PRE_RELEASE_NUM = None  # None or int

# Build metadata
BUILD_DATE = datetime.datetime.now().strftime("%Y%m%d")
BUILD_TIME = datetime.datetime.now().strftime("%H%M%S")

def get_version_tuple() -> Tuple[int, int, int]:
    """Return version as a tuple of integers."""
    return (MAJOR, MINOR, PATCH)

def get_version_string() -> str:
    """Return version as a string."""
    version = f"{MAJOR}.{MINOR}.{PATCH}"
    
    if PRE_RELEASE:
        version += f"-{PRE_RELEASE}"
        if PRE_RELEASE_NUM is not None:
            version += f".{PRE_RELEASE_NUM}"
    
    return version

def get_full_version_string() -> str:
    """Return full version string including build metadata."""
    version = get_version_string()
    version += f"+{BUILD_DATE}.{BUILD_TIME}"
    return version

def get_version_info() -> dict:
    """Return detailed version information as a dictionary."""
    return {
        "version": get_version_string(),
        "full_version": get_full_version_string(),
        "major": MAJOR,
        "minor": MINOR,
        "patch": PATCH,
        "pre_release": PRE_RELEASE,
        "pre_release_num": PRE_RELEASE_NUM,
        "build_date": BUILD_DATE,
        "build_time": BUILD_TIME,
        "version_tuple": get_version_tuple(),
    }

def print_version_info():
    """Print detailed version information."""
    info = get_version_info()
    print(f"Kernel U-Net Version: {info['version']}")
    print(f"Full Version: {info['full_version']}")
    print(f"Version Tuple: {info['version_tuple']}")
    print(f"Build Date: {info['build_date']}")
    print(f"Build Time: {info['build_time']}")

# Export the main version string
__version__ = get_version_string()

# For compatibility
VERSION = get_version_tuple()
version = __version__ 
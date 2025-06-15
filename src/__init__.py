"""
Kernel U-Net (KUNet) - A flexible U-Net architecture with customizable kernels

This package provides a modular implementation of U-Net architecture with support for 
various kernel types including Linear, LSTM, RNN, and Transformer layers.

Author: Kernel U-Net Team
Version: 1.0.0
License: MIT
"""

# Import version information
from .version import __version__, VERSION, get_version_info, print_version_info

__author__ = "Kernel U-Net Team"
__email__ = "contact@kunet.ai"
__license__ = "MIT"

# Package metadata
__title__ = "Kernel U-Net"
__description__ = "A flexible U-Net architecture with customizable kernels"
__url__ = "https://github.com/kernel-unet/Kernel-U-Net"

# Version info tuple (for backward compatibility)
VERSION_INFO = VERSION

# Import main classes and functions
from .kunlib import (
    # Core classes
    Kernel,
    KernelWrapper,
    
    # Kernel implementations
    Linear,
    LSTM,
    RNN,
    Transformer,
    
    # U-Net components
    KUNetEncoder,
    KUNetDecoder,
    KUNetEncoderDecoder,
    KUNet,
    
    # Utility classes
    Projector,
    
    # Attention components
    MultiHeadAttention,
    PositionalEncoding,
    AttentionBlock,
)

# Define what gets imported with "from src import *"
__all__ = [
    # Core classes
    'Kernel',
    'KernelWrapper',
    
    # Kernel implementations
    'Linear',
    'LSTM', 
    'RNN',
    'Transformer',
    
    # U-Net components
    'KUNetEncoder',
    'KUNetDecoder', 
    'KUNetEncoderDecoder',
    'KUNet',
    
    # Utility classes
    'Projector',
    
    # Attention components
    'MultiHeadAttention',
    'PositionalEncoding',
    'AttentionBlock',
    
    # Version info
    '__version__',
    '__author__',
    '__license__',
    'VERSION',
    'VERSION_INFO',
    'get_version',
    'get_version_tuple',
    'get_version_info',
    'print_version_info',
    'print_package_info',
]

def get_version():
    """Return the version string."""
    return __version__

def get_version_tuple():
    """Return the version info tuple (for backward compatibility)."""
    return VERSION_INFO

def print_package_info():
    """Print package information."""
    print(f"{__title__} v{__version__}")
    print(f"Description: {__description__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print(f"URL: {__url__}") 
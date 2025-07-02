"""
Data package for market analysis.

This package contains components for data fetching, processing, and feature engineering,
organized into the following subpackages:
- fetchers: Components for retrieving data from various sources
- processors: Components for processing and transforming data
- features: Components for feature engineering and selection
"""

# Import subpackages for easier access
from . import fetchers
from . import processors
from . import features

__all__ = [
    'fetchers',
    'processors',
    'features'
]
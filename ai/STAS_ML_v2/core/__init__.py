"""
Core module for STAS_ML v2

Contains fundamental classes and configurations.
"""

from .config import Config
from .trainer import ModelTrainer
from .base import BaseModel, BasePreprocessor

__all__ = ['Config', 'ModelTrainer', 'BaseModel', 'BasePreprocessor']
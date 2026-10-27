"""Isolated direct multi-step forecasting experiments.

This package intentionally does not modify or import the executable baseline
entry point.  Training starts only through an explicit function or CLI mode.
"""

from .models import ModelParameters, build_model

__all__ = ["ModelParameters", "build_model"]

"""Simtools - A simulation toolkit for Sim Companies."""

try:
    from importlib.metadata import version

    __version__ = version("simtools")
except Exception:
    # Fallback for development or when package is not installed
    __version__ = "0.1.0"

from simtools.models.resource import Resource
from simtools.models.building import Building
from simtools.api import SimcoAPI

__all__ = ["Resource", "Building", "SimcoAPI", "__version__"]


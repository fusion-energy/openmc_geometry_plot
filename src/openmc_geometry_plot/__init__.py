from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("openmc_geometry_plot")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

__all__ = ["__version__"]

from .core import *
from .app import *

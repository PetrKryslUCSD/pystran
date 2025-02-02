"""
pystran - Python package for structural analysis with trusses and beams 

(C) 2025, Petr Krysl
"""

# Define the __all__ variable
__all__ = [
    "pystran",
    "gauss",
    "model",
    "section",
    "rotation",
    "geometry",
    "assemble",
    "truss",
    "beam",
    "plots",
]

# Import the submodules
from . import model
from . import section
from . import rotation
from . import geometry
from . import gauss
from . import assemble
from . import truss
from . import beam
from . import plots

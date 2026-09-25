"""tomogrid -- multilevel evaluation of the attenuated Radon (PET/SPECT) transform.

See docs/formulation.md for the derivation.
"""

from .axis import Axis
from .cost import estimate_work
from .hierarchy import Hierarchy, build_hierarchy, thetas_of
from .image import Image, sample_function, square_grid
from .multilevel import Options, Sinogram, forward
from .segment import Triple, merge, merge_level, quad_linear, quad_spectral

__all__ = [
    "Axis",
    "Hierarchy",
    "Image",
    "Options",
    "Sinogram",
    "Triple",
    "build_hierarchy",
    "estimate_work",
    "forward",
    "merge",
    "merge_level",
    "quad_linear",
    "quad_spectral",
    "sample_function",
    "square_grid",
    "thetas_of",
]

"""cangle - static water contact angle from side-view sessile-drop images."""
from .measure import (
    AngleResult,
    DropMeasurement,
    annotate,
    batch_measure,
    classify,
    fit_circle,
    load_image,
    measure,
    measure_file,
)
from .synthetic import cap_geometry, make_drop

__version__ = "0.1.0"
__all__ = [
    "AngleResult", "DropMeasurement", "annotate", "batch_measure", "classify",
    "fit_circle", "load_image", "measure", "measure_file", "cap_geometry",
    "make_drop", "__version__",
]

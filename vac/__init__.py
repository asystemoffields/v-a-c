"""VAC: Variable Allocation Compression.

Fisher-informed structured compression for transformer models.
Compress any HuggingFace model to ~2x smaller with minimal quality loss.
"""

__version__ = "0.1.0"

from vac.compress import compress_model, compress_sequential
from vac.modeling import FactorizedLinear, VACModel

__all__ = [
    "compress_model",
    "compress_sequential",
    "FactorizedLinear",
    "VACModel",
]

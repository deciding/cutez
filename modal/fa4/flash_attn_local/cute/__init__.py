"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import cutlass.cute as cute

if not hasattr(cute.core, "ThrMma") and hasattr(cute, "ThrMma"):
    cute.core.ThrMma = cute.ThrMma
if not hasattr(cute.core, "ThrCopy") and hasattr(cute, "ThrCopy"):
    cute.core.ThrCopy = cute.ThrCopy
if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
    cute.make_fragment = cute.make_rmem_tensor

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

from .cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]

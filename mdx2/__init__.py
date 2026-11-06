import sys as _sys
from importlib.metadata import version

__version__ = version("mdx2")

# bitshuffle (mdx2.import_data: Eiger frames decompressed outside HDF5) is built with OpenMP. cctbx's omptbx, loaded
# with dxtbx models, defines stub omp_* functions; an extension loaded after it binds to those stubs (Python loads
# extensions with RTLD_NOW) and crashes in bitshuffle's parallel regions. So bitshuffle is loaded here, before mdx2
# imports dxtbx; if cctbx came first (e.g. dxtbx imported before mdx2), mdx2 does not call bitshuffle.
BITSHUFFLE_SAFE = "bitshuffle" in _sys.modules or "omptbx_ext" not in _sys.modules
try:
    import bitshuffle as _bitshuffle  # noqa: F401
except ImportError:  # pragma: no cover
    BITSHUFFLE_SAFE = False

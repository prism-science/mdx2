"""Test session setup."""

import os

# Tests call the compiled kernels directly (not through mdx2.scale's thread setup), so select numba's workqueue
# threading layer here as mdx2.parallel does: numba's GNU OpenMP layer executed parallel-loop iterations more than once
# on some multi-core Linux nodes. Must be set before numba is imported.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

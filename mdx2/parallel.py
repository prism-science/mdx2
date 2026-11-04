"""Thread-level parallelism for mdx2's compiled (numba) kernels.

numba's threading layer is set to "workqueue" unless NUMBA_THREADING_LAYER is set in the environment: on two 96- and
104-core Linux nodes (numba 0.65.1 and 0.67.0), the GNU OpenMP layer executed parallel-loop iterations more than once,
giving wrong sums and very slow kernels. `check_count` lets kernels verify an invariant (observations counted) so such
a runtime fails loudly instead of producing wrong results.
"""

import os

import numba as nb
from loguru import logger


def available_cpus():
    """CPUs this process may use: the affinity mask and, in containers, the cgroup CPU quota (whichever is smaller).
    os.cpu_count() reports the whole machine (e.g. 104 on a pod limited to 64 cores)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        n = os.cpu_count() or 1
    for quota_file, period_file in (
        ("/sys/fs/cgroup/cpu.max", None),  # cgroup v2: "<quota> <period>" or "max <period>"
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),  # cgroup v1
    ):
        try:
            with open(quota_file) as f:
                fields = f.read().split()
            if period_file is None:
                quota, period = fields[0], fields[1]
            else:
                with open(period_file) as f:
                    quota, period = fields[0], f.read().split()[0]
            if quota not in ("max", "-1"):
                n = min(n, max(1, int(quota) // int(period)))
            break
        except (OSError, ValueError, IndexError):
            continue
    return max(1, n)


def set_threads(nproc):
    """Choose the threading layer and the number of threads for the kernels. nproc: N > 0 threads; -1 all available
    CPUs; -N all available CPUs but N - 1. Call before the first parallel kernel runs (that starts numba's threads)."""
    if "NUMBA_THREADING_LAYER" not in os.environ:
        nb.config.THREADING_LAYER = "workqueue"
    avail = available_cpus()
    n = nproc if nproc > 0 else max(1, avail + 1 + nproc)
    nb.set_num_threads(min(n, nb.config.NUMBA_NUM_THREADS))
    logger.info(
        "Using {} thread(s) ({} CPU(s) available), numba threading layer {}",
        nb.get_num_threads(),
        avail,
        nb.config.THREADING_LAYER,
    )
    return nb.get_num_threads()


def check_count(what, got, expected):
    """Raise if a parallel kernel counted a different number of observations than expected (a threading runtime that
    executes parallel-loop iterations more than once)."""
    if got != expected:
        try:
            layer = nb.threading_layer()
        except ValueError:  # not initialized yet
            layer = nb.config.THREADING_LAYER
        raise RuntimeError(
            f"{what} counted {got} observations, expected {expected}. The numba threading layer ({layer}) is not "
            f"executing parallel loops correctly on this machine; run with NUMBA_THREADING_LAYER=workqueue (mdx2's "
            f"default) or, for mdx2.scale, --engine reference."
        )

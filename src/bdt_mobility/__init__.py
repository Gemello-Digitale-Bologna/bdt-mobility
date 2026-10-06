"""bdt_mobility package.

Sets BLAS thread-count environment variables *before* numpy/scipy are
imported anywhere else in the process. This works around a known OpenBLAS
0.3.34 crash ("terminate called without an active exception" / SIGABRT)
that intermittently occurs when its internal thread pool is torn down at
process exit with multi-threaded settings — observed with the numpy/scipy
versions pulled in transitively by civic-digital-twins>=0.11. Restricting
BLAS to a single thread avoids the race; the extraction steps only run
small linear-algebra operations, so there is no meaningful performance
cost.

Respects any value the caller has already set, so this is a safe default
rather than a hard override.
"""
import os

for _env_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import logging
import time
from datetime import datetime
from contextlib import contextmanager


@contextmanager
def timed():
    """Context manager that logs elapsed wall-clock time on exit.

    Output format:
        Starting time: YYYY-MM-DD HH:MM:SS.ffffff; Ending time: YYYY-MM-DD HH:MM:SS.ffffff; Duration: N.NNs
    """
    start = time.perf_counter()
    start_dt = datetime.now()
    yield
    elapsed = time.perf_counter() - start
    logging.info(
        f"Starting time: {start_dt}; "
        f"Ending time: {datetime.now()}; "
        f"Duration: {elapsed:.2f}s"
    )

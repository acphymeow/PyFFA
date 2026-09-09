#!/usr/bin/env python3
"""Entry point for the original Gaussian tune-shift benchmark.

The complete, path-normalized benchmark implementation is retained in
``gaussian_tune_shift_baseline.py`` in this directory.  This entry keeps the
original project filename visible in the referee-response bundle without
maintaining a second copy of the same calculation.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("gaussian_tune_shift_baseline.py")),
        run_name="__main__",
    )

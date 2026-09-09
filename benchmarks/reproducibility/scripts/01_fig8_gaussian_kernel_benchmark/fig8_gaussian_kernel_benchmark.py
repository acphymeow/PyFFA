#!/usr/bin/env python3
"""Run the authoritative three-panel Gaussian-kernel Fig. 8 benchmark.

The validated calculation remains in the legacy response-bundle location so
that existing response scripts keep working.  This clean entry point runs that
same implementation while redirecting its figures and text report to this
folder's ``outputs`` directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEGACY_SCRIPT = (
    HERE.parent / "01_fig8_es" / "fig8_gaussian_kernel_test.py"
)
OUTPUT_DIR = HERE / "outputs"


def load_benchmark_module():
    if not LEGACY_SCRIPT.exists():
        raise FileNotFoundError(f"Benchmark source not found: {LEGACY_SCRIPT}")
    spec = importlib.util.spec_from_file_location(
        "fig8_gaussian_kernel_validated_source",
        LEGACY_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load benchmark source: {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark_module()
    benchmark.OUT_TXT = OUTPUT_DIR / "fig8_gaussian_kernel_test.txt"
    benchmark.OUT_FIG_KERNEL = OUTPUT_DIR / "fig8_gaussian_kernel_shape.png"
    benchmark.OUT_FIG_BENCH = OUTPUT_DIR / "fig8_gaussian_kernel_benchmark.png"
    benchmark.main()


if __name__ == "__main__":
    main()

"""Shared path resolution for the benchmark reproducibility scripts."""
from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parent


def find_repo_root():
    """Locate the PyFFA checkout without depending on the current directory."""
    for path in SCRIPT_ROOT.parents:
        if (path / "src" / "FFAG_Bunch.py").is_file() and (path / "benchmarks").is_dir():
            return path
    raise FileNotFoundError("Cannot locate the PyFFA repository root")


REPO_ROOT = find_repo_root()
PYFFA_SRC = REPO_ROOT / "src"
PYFFA_EXAMPLES = REPO_ROOT / "examples"
for import_root in (PYFFA_SRC, PYFFA_EXAMPLES, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

SHARED_LONGITUDINAL = SCRIPT_ROOT / "shared_longitudinal"
CANONICAL_FIG8 = SCRIPT_ROOT / "01_fig8_es"
CANONICAL_TUNE = SCRIPT_ROOT / "04_lsc_tune_comparison"
CANONICAL_TOY = SCRIPT_ROOT / "03_longitudinal_phase_space"
CANONICAL_LEVEL0 = SHARED_LONGITUDINAL

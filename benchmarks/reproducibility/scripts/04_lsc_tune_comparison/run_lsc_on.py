from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _run_canonical import run

run("04_lsc_tune_comparison/compare_new_lsc_on_nominal60.py")

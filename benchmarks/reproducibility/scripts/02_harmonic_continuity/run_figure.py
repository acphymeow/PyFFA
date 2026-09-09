from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _run_canonical import run

run("02_harmonic_continuity/plot_fig5_core_bin_method.py")

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _run_canonical import run

run("03_longitudinal_phase_space/plot_longitudinal_phase_space_on_off.py")

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _run_canonical import run

run("01_fig8_es/fig8_gaussian_kernel_test.py")

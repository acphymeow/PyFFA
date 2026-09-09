"""Run one script from this self-contained reproducibility tree."""
from pathlib import Path
import os
import runpy
import sys


def run(relative_path):
    here = Path(__file__).resolve()
    repo = next(parent for parent in here.parents
                if (parent / "src" / "FFAG_Bunch.py").is_file())
    script_root = here.parent
    target = script_root / relative_path
    if not target.is_file():
        raise FileNotFoundError(target)
    for import_root in (repo / "src", repo / "examples", script_root, target.parent):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    os.chdir(repo)
    runpy.run_path(str(target), run_name="__main__")

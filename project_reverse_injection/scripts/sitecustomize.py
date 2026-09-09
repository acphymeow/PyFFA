from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for search_path in (REPO_ROOT, REPO_ROOT / "examples", REPO_ROOT / "src"):
    search_text = str(search_path)
    if search_text not in sys.path:
        sys.path.insert(0, search_text)

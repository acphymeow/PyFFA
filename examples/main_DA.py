import argparse
import json
import sys
from pathlib import Path


def find_repo_root() -> Path:
    candidates = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for start in candidates:
        for path in (start, *start.parents):
            if (path / "src" / "FFAG_Field.py").is_file() or (
                path / "FFAG_Field.py"
            ).is_file():
                return path
    return Path(__file__).resolve().parents[1]


REPO_ROOT = find_repo_root()
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
for path in (THIS_DIR, REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import FFAG_DA  # noqa: E402
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Cannot import FFAG_DA.py. Put it in src/, or put FFAG_DA.py next to "
        "main_DA.py for old root-level PyFFA layouts."
    ) from exc


def load_config(json_path):
    with open(json_path, "r", encoding="utf-8") as f_in:
        return json.load(f_in)


def main(json_path):
    config_data = load_config(json_path)
    da_config = FFAG_DA.DARunConfig.from_dict(config_data["DA"])
    FFAG_DA.run_da_scan(da_config, config_data=config_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PyFFA fixed-energy DA scan.")
    parser.add_argument(
        "-j",
        "--json_path",
        default="config_DA.json",
        help="Path to config_DA.json. Default: ./config_DA.json",
    )
    args = parser.parse_args()
    main(args.json_path)

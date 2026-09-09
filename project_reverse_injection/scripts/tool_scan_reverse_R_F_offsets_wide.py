from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tool_scan_reverse_R_F_offsets_grid as grid


grid.OUTPUT_DIR = Path("./project_reverse_injection/reverse_R_F_offset_wide_output")
grid.RADIAL_OFFSETS_M = [-0.090, -0.120, -0.150, -0.180]


if __name__ == "__main__":
    grid.main()

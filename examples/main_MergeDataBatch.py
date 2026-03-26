#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path
import argparse


def run_merge_for_all_tbt(parent_folder: Path, main_merge_path: Path):
    # 搜索所有 TBT_Dump_xxx 目录
    tbt_dirs = sorted([d for d in parent_folder.glob("TBT_Dump_*") if d.is_dir()],
                      key=lambda p: int(p.name.split("_")[-1]))

    if not tbt_dirs:
        print(f"[Warning] No TBT_Dump_xxx directories found in {parent_folder}")
        return

    print(f"[Info] Found {len(tbt_dirs)} TBT_Dump_xxx directories.\n")

    for d in tbt_dirs:
        print(f"[Run] python {main_merge_path} {d}")
        subprocess.run(
            ["python", str(main_merge_path), str(d)],
            check=True
        )
        print(f"[OK] Finished merging in {d}\n")


def main():
    parser = argparse.ArgumentParser(description="Auto-run main_MergeData.py on all TBT_Dump_xxx folders.")
    parser.add_argument("folder", help="Parent folder containing TBT_Dump_xxx")
    parser.add_argument("--main", default="main_MergeData.py",
                        help="Path to main_MergeData.py (default: main_MergeData.py)")

    args = parser.parse_args()

    parent_folder = Path(args.folder).resolve()
    main_merge_path = Path(args.main).resolve()

    if not main_merge_path.exists():
        print(f"[Error] main_MergeData.py not found: {main_merge_path}")
        return

    run_merge_for_all_tbt(parent_folder, main_merge_path)


if __name__ == "__main__":
    main()

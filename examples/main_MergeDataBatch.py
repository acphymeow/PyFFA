#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

from FFAG_dump import merge_files_in_folder


def run_merge_for_all_tbt(parent_folder: Path, turn_min=None, turn_max=None):
    # 搜索所有 TBT_Dump_xxx 目录
    tbt_dirs = sorted([d for d in parent_folder.glob("TBT_Dump_*") if d.is_dir()],
                      key=lambda p: int(p.name.split("_")[-1]))
    if turn_min is not None:
        tbt_dirs = [
            d for d in tbt_dirs
            if int(d.name.split("_")[-1]) >= turn_min
        ]
    if turn_max is not None:
        tbt_dirs = [
            d for d in tbt_dirs
            if int(d.name.split("_")[-1]) <= turn_max
        ]

    if not tbt_dirs:
        print(f"[Warning] No TBT_Dump_xxx directories found in {parent_folder}")
        return

    print(f"[Info] Found {len(tbt_dirs)} TBT_Dump_xxx directories.\n")

    for d in tbt_dirs:
        print(f"[Run] Merging {d}")
        merge_files_in_folder(str(d))
        print(f"[OK] Finished merging in {d}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge rank data in selected TBT_Dump_xxx folders."
    )
    parser.add_argument("folder", help="Parent folder containing TBT_Dump_xxx")
    parser.add_argument("--turn-min", type=int, default=None,
                        help="Only merge TBT_Dump directories at or above this turn.")
    parser.add_argument("--turn-max", type=int, default=None,
                        help="Only merge TBT_Dump directories at or below this turn.")

    args = parser.parse_args()

    parent_folder = Path(args.folder).resolve()

    if not parent_folder.is_dir():
        parser.error(f"folder does not exist: {parent_folder}")

    if args.turn_min is not None and args.turn_max is not None:
        if args.turn_min > args.turn_max:
            parser.error("--turn-min cannot be greater than --turn-max")

    run_merge_for_all_tbt(
        parent_folder,
        turn_min=args.turn_min,
        turn_max=args.turn_max,
    )


if __name__ == "__main__":
    main()

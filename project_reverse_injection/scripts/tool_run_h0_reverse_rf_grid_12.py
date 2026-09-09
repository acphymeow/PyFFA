"""Run the 12 R/F-grid H0 reverse cases serially and write one summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import tool_reverse_inj_h0_one_lattice as h0


ROOT = Path(__file__).resolve().parents[2]
CASE_CSV = ROOT / "project_reverse_injection/lattice_da_200turn_RF_grid_12_order2_300MeV/rf_grid_da_summary.csv"
OUTPUT_DIR = ROOT / "project_reverse_injection/h0_reverse_rf_grid_12_output"


def load_cases() -> list[dict[str, str]]:
    with CASE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return sorted(rows, key=lambda row: int(row["slot"]))


def summary_path_for(lattice: str) -> Path:
    mirror_tag = "_mirror_x0" if h0.MIRROR_FINAL_PLOT_ABOUT_X0 else ""
    stem = (
        f"{lattice}_300MeV_H0_reverse_"
        f"ccw{h0.STRAIGHT_SECTION_OFFSET_CCW}_straight_sections{mirror_tag}"
    )
    return OUTPUT_DIR / f"{stem}_summary.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    cases = load_cases()
    h0.OUTPUT_DIR = OUTPUT_DIR

    for index, case in enumerate(cases, start=1):
        lattice = case["lattice"]
        h0.BMAP_DIR = ROOT / "Bmaps" / lattice
        print(f"[{index:02d}/{len(cases):02d}] serial H0 reverse: {lattice}", flush=True)
        h0.main()
        summary = json.loads(summary_path_for(lattice).read_text(encoding="utf-8"))
        summary["slot"] = int(case["slot"])
        summary["R_ratio"] = float(case["R_ratio"])
        summary["F_deg"] = float(case["F_deg"])
        results.append(summary)

    summary_json = OUTPUT_DIR / "h0_reverse_rf_grid_12_summary.json"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fields = [
        "slot", "R_ratio", "F_deg", "lattice", "max_order",
        "straight_section_offset_ccw", "final_plot_mirrored_about_x0",
        "foil_phi_deg", "foil_r_m", "foil_pr_rad", "valid_time_ns",
        "path_length_m", "r_min_m", "r_max_m", "end_r_m", "end_phi_deg",
        "sampled_bz_min_T", "sampled_bz_max_T", "sampled_max_abs_bz_T",
        "stop_reason",
    ]
    summary_csv = OUTPUT_DIR / "h0_reverse_rf_grid_12_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"All {len(results)} serial H0 reverse cases finished.")
    print(summary_csv)
    print(summary_json)


if __name__ == "__main__":
    main()

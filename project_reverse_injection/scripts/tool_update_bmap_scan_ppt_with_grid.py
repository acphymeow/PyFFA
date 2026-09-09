import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import tool_reverse_inj_compare_k_analytical as field_base


SOURCE_PPTX = Path("./bmap_scan_parameter_summary.pptx")
OUTPUT_PPTX = Path("./outputs/bmap_scan_parameter_summary_extended_RF_grid_file_tables.pptx")
GRID_DIR = Path("./reverse_R_F_offset_grid_output")
EXTRA_SUMMARY = Path("./reverse_extra_param_scan_output/extra_param_reverse_summary.csv")
GRID_SUMMARY = GRID_DIR / "R_F_offset_grid_summary.csv"
BEST_BY_OFFSET = GRID_DIR / "R_F_offset_grid_best_by_offset.csv"
ASSET_DIR = Path("./outputs/bmap_scan_ppt_assets_extended")


def remove_shape(shape):
    el = shape._element
    el.getparent().remove(el)


def set_cell(cell, text, size=12, bold=False, fill=None):
    cell.text = str(text)
    cell.margin_left = Inches(0.05)
    cell.margin_right = Inches(0.05)
    cell.margin_top = Inches(0.02)
    cell.margin_bottom = Inches(0.02)
    if fill:
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(*fill)
    tf = cell.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.LEFT
        for run in p.runs:
            run.font.name = "Arial"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = RGBColor(0, 0, 0)


def replace_table(slide, rows, x, y, w, h, widths, font_size=11):
    for shape in list(slide.shapes):
        if getattr(shape, "has_table", False) and shape.has_table:
            remove_shape(shape)
            break
    table = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, h).table
    for j, width in enumerate(widths):
        table.columns[j].width = width
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            set_cell(table.cell(i, j), val, size=font_size if i else font_size + 1, bold=(i == 0), fill=(232, 232, 232) if i == 0 else None)
    return table


def add_text(slide, text, x, y, w, h, size=20, bold=False, color=(0, 0, 0), fill=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.text = text
    run = p.runs[0]
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color)
    if fill:
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(*fill)
        box.line.color.rgb = RGBColor(190, 190, 190)
    return box


def add_table(slide, rows, x, y, w, h, widths, font_size=12):
    table = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for j, width in enumerate(widths):
        table.columns[j].width = Inches(width)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            set_cell(table.cell(i, j), val, size=font_size if i else font_size + 1, bold=(i == 0), fill=(232, 232, 232) if i == 0 else None)
    return table


def load_track(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    r = np.asarray(d["r"], dtype=float)
    fi = np.asarray(d["fi"], dtype=float)
    bz = np.asarray(d["bz"], dtype=float)
    ok = np.isfinite(r) & np.isfinite(fi) & np.isfinite(bz)
    r, fi, bz = r[ok], fi[ok], bz[ok]
    x, y = r * np.cos(fi), r * np.sin(fi)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    return s, r, bz


def plot_field_cell(cases, output_path):
    fig, ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
    for case in cases:
        bmap = field_base.FFAG_Bfield_analytical(str(case["bmap"] / "config_Bmap.json"), 0)
        phi = np.linspace(0.0, 2.0 * np.pi / bmap.Nsectors, 500)
        r = np.full_like(phi, 11.5)
        z = np.zeros_like(phi)
        bz = field_base.get_bfields_for_points(bmap, r, phi, z, 0)[0]
        ax.plot(np.rad2deg(phi), bz, lw=1.2, label=case["label"])
    ax.set_title("Bz at R=11.5 m", fontsize=9)
    ax.set_xlabel("cell fi [deg]", fontsize=8)
    ax.set_ylabel("Bz [T]", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.2, loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_track_r(cases, output_path):
    fig, ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
    for case in cases:
        s, r, _ = load_track(case["csv"])
        ax.plot(s, r, lw=1.2, label=case["label"])
    ax.set_title("Reverse track R(s)", fontsize=9)
    ax.set_xlabel("s [m]", fontsize=8)
    ax.set_ylabel("R [m]", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.2, loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_track_bz(cases, output_path):
    fig, ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
    for case in cases:
        s, _, bz = load_track(case["csv"])
        ax.plot(s, bz, lw=1.2, label=case["label"])
    ax.axhline(1.0, color="0.35", lw=0.8, ls="--")
    ax.set_title("Reverse track Bz(s)", fontsize=9)
    ax.set_xlabel("s [m]", fontsize=8)
    ax.set_ylabel("Bz [T]", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.2, loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def replace_slide_pictures(slide, image_paths):
    positions = []
    for shape in list(slide.shapes):
        if shape.shape_type == 13:
            positions.append((shape.left, shape.top, shape.width, shape.height))
            remove_shape(shape)
    positions = sorted(positions, key=lambda p: p[0])[:3]
    for (left, top, width, height), path in zip(positions, image_paths):
        slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def update_single_param_plots(prs):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    df_cases = [
        {"label": "R=0.265", "bmap": Path("Bmaps/Bmap_FD16_k7p63_B1p65_R265_F0_s49p97"), "csv": GRID_DIR / "k7p63_B1p65_R265_F0_s49p97_offm30mm_r_fi_bz.csv"},
        {"label": "R=0.365", "bmap": Path("Bmaps/Bmap_FD16_k7p63_B1p65_R365_F0_s49p97"), "csv": GRID_DIR / "k7p63_B1p65_R365_F0_s49p97_offm30mm_r_fi_bz.csv"},
        {"label": "R=0.465", "bmap": Path("Bmaps/Bmap_FD16_k7p63_B1p65_R465_F0_s49p97"), "csv": GRID_DIR / "k7p63_B1p65_R465_F0_s49p97_offm30mm_r_fi_bz.csv"},
        {"label": "R=0.565", "bmap": Path("Bmaps/Bmap_FD16_k7p63_B1p65_R565_F0_s49p97"), "csv": GRID_DIR / "k7p63_B1p65_R565_F0_s49p97_offm30mm_r_fi_bz.csv"},
    ]
    f_cases = [
        {"label": "F=-0.25", "bmap": Path("Bmaps/Bmap_FD16_k7p63_B1p65_R465_Fn0p25_s49p97"), "csv": Path("outputs/bmap_scan_package/r_fi_bz_files/k7p63_B1p65_R465_Fn0p25_s49p97_offm30mm_r_fi_bz.csv")},
        {"label": "F=0.00", "bmap": Path("Bmaps/Bmap_FD16_k7p63_B1p65_R465_F0_s49p97"), "csv": GRID_DIR / "k7p63_B1p65_R465_F0_s49p97_offm30mm_r_fi_bz.csv"},
        {"label": "F=0.50", "bmap": Path("Bmaps/Bmap_FD16_k7p63_B1p65_R465_Fp0p5_s49p97"), "csv": GRID_DIR / "k7p63_B1p65_R465_Fp0p5_s49p97_offm30mm_r_fi_bz.csv"},
        {"label": "F=0.75", "bmap": Path("Bmaps/Bmap_FD16_k7p63_B1p65_R465_Fp0p75_s49p97"), "csv": GRID_DIR / "k7p63_B1p65_R465_Fp0p75_s49p97_offm30mm_r_fi_bz.csv"},
    ]
    for prefix, cases, slide_index in (("df_ratio", df_cases, 4), ("f_angle", f_cases, 5)):
        paths = [ASSET_DIR / f"{prefix}_field.png", ASSET_DIR / f"{prefix}_rs.png", ASSET_DIR / f"{prefix}_bzs.png"]
        plot_field_cell(cases, paths[0])
        plot_track_r(cases, paths[1])
        plot_track_bz(cases, paths[2])
        replace_slide_pictures(prs.slides[slide_index], paths)


def clear_strip_eff_columns(prs):
    for slide in prs.slides:
        for shape in slide.shapes:
            if not (getattr(shape, "has_table", False) and shape.has_table):
                continue
            table = shape.table
            if len(table.rows) == 0:
                continue
            for col in range(len(table.columns)):
                header = table.cell(0, col).text.lower()
                if "strip" in header:
                    for row in range(1, len(table.rows)):
                        table.cell(row, col).text = ""


def load_existing_strip_eff():
    sources = [
        SOURCE_PPTX,
        Path("./outputs/bmap_scan_parameter_summary_with_efficiency.pptx"),
        Path("./outputs/bmap_scan_package/bmap_scan_parameter_summary_with_efficiency.pptx"),
    ]
    eff = {}
    case_pat = re.compile(r"(k[0-9p]+_B[0-9p]+_R[0-9]+_F(?:n|p)?[0-9p]+_s[0-9p]+(?:_offm?[0-9]+mm)?)")
    for source in sources:
        if not source.exists():
            continue
        for slide in Presentation(source).slides:
            for shape in slide.shapes:
                if not (getattr(shape, "has_table", False) and shape.has_table):
                    continue
                table = shape.table
                strip_cols = [c for c in range(len(table.columns)) if "strip" in table.cell(0, c).text.lower()]
                for strip_col in strip_cols:
                    for row_idx in range(1, len(table.rows)):
                        strip = table.cell(row_idx, strip_col).text.strip()
                        if not strip:
                            continue
                        row_text = " ".join(table.cell(row_idx, c).text for c in range(len(table.columns)))
                        for key in case_pat.findall(row_text):
                            eff.setdefault(key, strip)
                            eff.setdefault(re.sub(r"_offm?[0-9]+mm$", "", key), strip)
    return eff


def load_extra_values():
    vals = {}
    if EXTRA_SUMMARY.exists():
        with EXTRA_SUMMARY.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vals[row["folder"]] = float(row["max_abs_Bz_T"])
    if GRID_SUMMARY.exists():
        with GRID_SUMMARY.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["offset_mm"]) == -30:
                    vals[row["folder"]] = float(row["max_abs_Bz_T"])
    return vals


def load_best_rows():
    rows = []
    with BEST_BY_OFFSET.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: int(r["offset_mm"]), reverse=True)
    return rows


def update_single_param_tables(prs):
    vals = load_extra_values()
    eff = load_existing_strip_eff()
    df_rows = [
        ["parameter", "file name", "max Bz [T]", "strip eff."],
        ["DF ratio", "k7p63_B1p65_R265_F0_s49p97\nDF field ratio 0.265", f"{vals['Bmap_FD16_k7p63_B1p65_R265_F0_s49p97']:.6f}", ""],
        ["DF ratio", "k7p63_B1p65_R365_F0_s49p97\nDF field ratio 0.365", f"{vals['Bmap_FD16_k7p63_B1p65_R365_F0_s49p97']:.6f}", eff.get("k7p63_B1p65_R365_F0_s49p97", "")],
        ["DF ratio", "k7p63_B1p65_R465_F0_s49p97\nDF field ratio 0.465", f"{vals['Bmap_FD16_k7p63_B1p65_R465_F0_s49p97']:.6f}", eff.get("k7p63_B1p65_R465_F0_s49p97", "")],
        ["DF ratio", "k7p63_B1p65_R565_F0_s49p97\nDF field ratio 0.565", f"{vals['Bmap_FD16_k7p63_B1p65_R565_F0_s49p97']:.6f}", eff.get("k7p63_B1p65_R565_F0_s49p97", "")],
    ]
    f_rows = [
        ["parameter", "file name", "max Bz [T]", "strip eff."],
        ["F sector angle", "k7p63_B1p65_R465_Fn0p25_s49p97\nF -0.25 deg, D +0.25 deg", "1.119347", eff.get("k7p63_B1p65_R465_Fn0p25_s49p97", "")],
        ["F sector angle", "k7p63_B1p65_R465_F0_s49p97\nF unchanged, D unchanged", f"{vals['Bmap_FD16_k7p63_B1p65_R465_F0_s49p97']:.6f}", eff.get("k7p63_B1p65_R465_F0_s49p97", "")],
        ["F sector angle", "k7p63_B1p65_R465_Fp0p5_s49p97\nF +0.50 deg, D -0.50 deg", f"{vals['Bmap_FD16_k7p63_B1p65_R465_Fp0p5_s49p97']:.6f}", eff.get("k7p63_B1p65_R465_Fp0p5_s49p97", "")],
        ["F sector angle", "k7p63_B1p65_R465_Fp0p75_s49p97\nF +0.75 deg, D -0.75 deg", f"{vals['Bmap_FD16_k7p63_B1p65_R465_Fp0p75_s49p97']:.6f}", ""],
    ]
    widths = [Inches(1.25), Inches(4.30), Inches(1.20), Inches(1.15)]
    replace_table(prs.slides[4], df_rows, Inches(0.55), Inches(1.34), Inches(8.20), Inches(2.45), widths, font_size=10)
    replace_table(prs.slides[5], f_rows, Inches(0.55), Inches(1.34), Inches(8.20), Inches(2.45), widths, font_size=10)


def add_best_slide(prs):
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    add_text(slide, "R-F grid scan: r-fi-Bz file index", 0.55, 0.30, 11.2, 0.55, size=30, bold=True)
    add_text(slide, "Each editable table is one offset. Cells contain the r-fi-Bz CSV file basename.", 0.58, 0.88, 11.8, 0.35, size=15, color=(80, 80, 80))

    rows = []
    with GRID_SUMMARY.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["R_ratio"] = float(row["R_ratio"])
            row["F_deg"] = float(row["F_deg"])
            row["offset_mm"] = int(row["offset_mm"])
            rows.append(row)
    r_vals = sorted({row["R_ratio"] for row in rows})
    f_vals = sorted({row["F_deg"] for row in rows})
    table_positions = [
        (0, 0.22, 1.18),
        (-15, 4.58, 1.18),
        (-30, 8.94, 1.18),
        (-45, 2.40, 4.08),
        (-60, 6.76, 4.08),
    ]
    for off, x, y in table_positions:
        table_rows = [[f"offset {off:+d} mm", *[f"R={r:.3f}" for r in r_vals]]]
        for f_deg in f_vals:
            line = [f"F={f_deg:.2f}"]
            for r_ratio in r_vals:
                match = next(row for row in rows if row["offset_mm"] == off and abs(row["F_deg"] - f_deg) < 1e-12 and abs(row["R_ratio"] - r_ratio) < 1e-12)
                line.append(Path(match["r_fi_bz_csv"]).name)
            table_rows.append(line)
        add_table(slide, table_rows, x, y, 4.15, 2.45, [0.50, 0.91, 0.91, 0.91, 0.91], font_size=5.0)
    return slide


def add_heatmap_slide(prs):
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    add_text(slide, "R-F grid scan: one table per offset", 0.55, 0.30, 11.2, 0.55, size=30, bold=True)
    add_text(slide, "Each editable table reports max |Bz| [T]. Rows are F-angle increases; columns are DF ratio.", 0.58, 0.88, 11.8, 0.35, size=15, color=(80, 80, 80))

    rows = []
    with GRID_SUMMARY.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["R_ratio"] = float(row["R_ratio"])
            row["F_deg"] = float(row["F_deg"])
            row["offset_mm"] = int(row["offset_mm"])
            row["max_abs_Bz_T"] = float(row["max_abs_Bz_T"])
            rows.append(row)
    r_vals = sorted({row["R_ratio"] for row in rows})
    f_vals = sorted({row["F_deg"] for row in rows})
    offsets = sorted({row["offset_mm"] for row in rows}, reverse=True)
    table_positions = [
        (0, 0.55, 1.35),
        (-15, 4.70, 1.35),
        (-30, 8.85, 1.35),
        (-45, 2.62, 4.38),
        (-60, 6.78, 4.38),
    ]
    for off, x, y in table_positions:
        table_rows = [[f"offset {off:+d} mm", *[f"R={r:.3f}" for r in r_vals]]]
        for f_deg in f_vals:
            line = [f"F={f_deg:.2f}"]
            for r_ratio in r_vals:
                match = next(row for row in rows if row["offset_mm"] == off and abs(row["F_deg"] - f_deg) < 1e-12 and abs(row["R_ratio"] - r_ratio) < 1e-12)
                line.append(f"{match['max_abs_Bz_T']:.6f}")
            table_rows.append(line)
        add_table(slide, table_rows, x, y, 3.85, 1.80, [0.72, 0.78, 0.78, 0.78, 0.78], font_size=7)
    return slide


def main():
    prs = Presentation(SOURCE_PPTX)
    update_single_param_tables(prs)
    update_single_param_plots(prs)
    add_best_slide(prs)
    add_heatmap_slide(prs)
    OUTPUT_PPTX.parent.mkdir(exist_ok=True)
    prs.save(OUTPUT_PPTX)
    print(OUTPUT_PPTX)


if __name__ == "__main__":
    main()

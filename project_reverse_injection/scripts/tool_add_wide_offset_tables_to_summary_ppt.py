import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


SOURCE_PPTX = Path("./project_reverse_injection/bmap_scan_parameter_summarys_with_DA_H0_reverse_wide.pptx")
OUTPUT_PPTX = Path("./project_reverse_injection/bmap_scan_parameter_summarys_with_DA_H0_reverse_wide_offsets_tables.pptx")
WIDE_SUMMARY = Path("./project_reverse_injection/reverse_R_F_offset_wide_output/R_F_offset_grid_summary.csv")


def set_cell(cell, text, size=10, bold=False, fill=None, align=PP_ALIGN.CENTER):
    cell.text = str(text)
    cell.margin_left = Inches(0.03)
    cell.margin_right = Inches(0.03)
    cell.margin_top = Inches(0.02)
    cell.margin_bottom = Inches(0.02)
    if fill:
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(*fill)
    tf = cell.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for p in tf.paragraphs:
        p.alignment = align
        for run in p.runs:
            run.font.name = "Arial"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = RGBColor(0, 0, 0)


def add_text(slide, text, x, y, w, h, size=24, bold=False, color=(0, 0, 0)):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.text = text
    run = p.runs[0]
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color)
    return box


def load_grid():
    rows = []
    with WIDE_SUMMARY.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["R_ratio"] = float(row["R_ratio"])
            row["F_deg"] = float(row["F_deg"])
            row["offset_mm"] = int(row["offset_mm"])
            row["max_abs_Bz_T"] = float(row["max_abs_Bz_T"])
            rows.append(row)
    r_vals = sorted({row["R_ratio"] for row in rows})
    f_vals = sorted({row["F_deg"] for row in rows})
    offsets = [-90, -120, -150, -180]
    value = {(row["offset_mm"], row["F_deg"], row["R_ratio"]): row["max_abs_Bz_T"] for row in rows}
    return r_vals, f_vals, offsets, value


def add_rf_table(slide, title, x, y, r_vals, f_vals, value_map=None, blank_body=False):
    rows = [[title, *[f"R={r:.3f}" for r in r_vals]]]
    for f_deg in f_vals:
        line = [f"F={f_deg:.2f}"]
        for r_ratio in r_vals:
            if blank_body:
                line.append("")
            else:
                val = value_map.get((f_deg, r_ratio))
                line.append(f"{val:.6f}" if val is not None else "N/A")
        rows.append(line)

    table = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(5.95), Inches(2.15)).table
    widths = [1.00, 1.23, 1.23, 1.23, 1.23]
    for j, width in enumerate(widths):
        table.columns[j].width = Inches(width)
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            fill = (232, 232, 232) if i == 0 else None
            if text == "N/A":
                fill = (245, 220, 220)
            set_cell(table.cell(i, j), text, size=9 if i else 10, bold=(i == 0), fill=fill)
    return table


def add_wide_offset_slide(prs, r_vals, f_vals, offsets, value, blank_body=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if blank_body:
        add_text(slide, "扩展 offset 扫描：剥离率待填", 0.55, 0.28, 12.0, 0.52, size=28, bold=True)
        add_text(slide, "每个表格对应一个 offset；行是 F 铁宽度变化，列是 DF ratio。数值格留空，后续手动填写。", 0.58, 0.84, 12.0, 0.32, size=13, color=(80, 80, 80))
    else:
        add_text(slide, "扩展 offset 扫描：反推轨迹最大磁场", 0.55, 0.28, 12.0, 0.52, size=28, bold=True)
        add_text(slide, "每个表格对应一个 offset；单元格为 max |Bz| [T]。N/A 表示该点未生成有效轨迹。", 0.58, 0.84, 12.0, 0.32, size=13, color=(80, 80, 80))

    positions = {
        -90: (0.55, 1.25),
        -120: (6.85, 1.25),
        -150: (0.55, 4.08),
        -180: (6.85, 4.08),
    }
    for off in offsets:
        vals = {(f_deg, r): value.get((off, f_deg, r)) for f_deg in f_vals for r in r_vals}
        add_rf_table(slide, f"offset {off} mm", *positions[off], r_vals, f_vals, vals, blank_body=blank_body)


def main():
    prs = Presentation(SOURCE_PPTX)
    r_vals, f_vals, offsets, value = load_grid()
    add_wide_offset_slide(prs, r_vals, f_vals, offsets, value, blank_body=False)
    add_wide_offset_slide(prs, r_vals, f_vals, offsets, value, blank_body=True)
    prs.save(OUTPUT_PPTX)
    print(OUTPUT_PPTX)


if __name__ == "__main__":
    main()

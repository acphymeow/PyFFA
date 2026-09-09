#!/usr/bin/env python3
"""Insert the three-panel Fig. 8 diagnostic into Referee A, Comment 2."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.shared import Inches, Pt


HERE = Path(__file__).resolve().parent
BUNDLE_DIR = HERE.parent
DOCX_PATH = BUNDLE_DIR / (
    "审稿意见回复版_已补第一条_中英双语_审稿人关注版_参数表占位_"
    "共振线_方形结构_包络与tune_depression.docx"
)
IMAGE_PATH = (
    HERE / "01_fig8_es" / "fig8_gaussian_kernel_benchmark.png"
)
CAPTION_PREFIX = "Figure R2. Evolution of the longitudinal-field benchmark"
CAPTION_TEXT = (
    "Figure R2. Evolution of the longitudinal-field benchmark at "
    "NS=256: (a) the original uniform-disk result, (b) the "
    "Gaussian-kernel result, and (c) the Gaussian-kernel result after "
    "smoothing the sampled line charge density."
)


def paragraph_after(paragraph):
    element = OxmlElement("w:p")
    paragraph._p.addnext(element)
    return Paragraph(element, paragraph._parent)


def main():
    document = Document(DOCX_PATH)
    target = next(
        p
        for p in document.paragraphs
        if "2.86%" in p.text and "峰值区" in p.text
    )
    target.text = (
        "综合以上处理，如图 R2 所示，原图峰值区约 10% 的局部偏差已明显减小；"
        "在最终配置下，以理论峰值归一化的 rms 偏差为 2.86%，峰值区也不再表现为系统性偏低。"
    )

    if any(p.text.startswith(CAPTION_PREFIX) for p in document.paragraphs):
        document.save(DOCX_PATH)
        print(DOCX_PATH)
        return

    figure_paragraph = paragraph_after(target)
    figure_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    figure_paragraph.paragraph_format.space_before = Pt(6)
    figure_paragraph.paragraph_format.space_after = Pt(3)
    figure_paragraph.paragraph_format.keep_with_next = True
    run = figure_paragraph.add_run()
    picture = run.add_picture(str(IMAGE_PATH), width=Inches(5.75))
    doc_pr = picture._inline.docPr
    doc_pr.set("descr", "Three-panel longitudinal electric-field benchmark")

    caption = paragraph_after(figure_paragraph)
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(0)
    caption.paragraph_format.space_after = Pt(8)
    caption.paragraph_format.keep_together = True
    caption_run = caption.add_run(CAPTION_TEXT)
    caption_run.italic = True
    caption_run.font.name = "Times New Roman"
    caption_run.font.size = Pt(9)
    caption_run._element.get_or_add_rPr().rFonts.set(
        qn("w:eastAsia"), "Times New Roman"
    )

    document.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()

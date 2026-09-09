"""
Stitch the merged Fig.2+3 panel and the Fig.4 panel into one compact figure:
  (a) left  : fig23_combined_proj.png        (wide arc view)
  (b) right : figure_global2local_panelb.png (construction zoom)

No changes to the source figures; pure image composition.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
IMG_A = OUTPUT_DIR / "fig23_combined_proj.png"
IMG_B = OUTPUT_DIR / "figure_global2local_panelb.png"
OUT = OUTPUT_DIR / "fig234_ab_stitched.png"

FIG_W = 12.0          # inches
A_WIDTH_FRAC = 0.61
B_WIDTH_FRAC = 0.36
PANEL_GAP_FRAC = 0.03
MARGIN_TOP = 0.10     # inches
MARGIN_BOT = 0.08     # inches


def main(img_a=IMG_A, img_b=IMG_B, out=OUT, show=True):
    im_a = Image.open(img_a)
    im_b = Image.open(img_b)

    aspect_a = im_a.size[1] / im_a.size[0]
    aspect_b = im_b.size[1] / im_b.size[0]

    w_a = FIG_W * A_WIDTH_FRAC
    h_a = w_a * aspect_a
    w_b = FIG_W * B_WIDTH_FRAC
    h_b = w_b * aspect_b
    panel_h = max(h_a, h_b)
    fig_h = MARGIN_TOP + panel_h + MARGIN_BOT

    fig = plt.figure(figsize=(FIG_W, fig_h))

    # (a) left; vertically centered against panel (b)
    y_a = (MARGIN_BOT + 0.5 * (panel_h - h_a)) / fig_h
    ax_a = fig.add_axes([0.0, y_a, A_WIDTH_FRAC, h_a / fig_h])
    ax_a.imshow(np.asarray(im_a))
    ax_a.axis("off")

    # (b) right; vertically centered against panel (a)
    x_b = A_WIDTH_FRAC + PANEL_GAP_FRAC
    y_b = (MARGIN_BOT + 0.5 * (panel_h - h_b)) / fig_h
    ax_b = fig.add_axes([x_b, y_b, B_WIDTH_FRAC, h_b / fig_h])
    ax_b.imshow(np.asarray(im_b))
    ax_b.axis("off")

    # panel labels
    ax_a.text(
        0.090, 0.955, "(a)", transform=ax_a.transAxes,
        fontsize=24, fontweight="bold", ha="left", va="top",
    )
    ax_b.text(
        0.96, 0.96, "(b)", transform=ax_b.transAxes,
        fontsize=24, fontweight="bold", ha="right", va="top",
    )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    print("saved:", out)
    if show:
        plt.show()
    return out


if __name__ == "__main__":
    main()

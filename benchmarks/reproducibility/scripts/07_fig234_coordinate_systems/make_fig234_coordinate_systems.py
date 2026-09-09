"""Generate the proposed combined replacement for original Figs. 2--4."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_fig23_combined_proj import (
    OUTPUT_DIR,
    SEO_FILE,
    plot_two_energy_fig23,
)
from plot_fig4_panel_b import plot_projection_coordinate_conversion_final
from stitch_fig23_fig4 import main as stitch_panels


def main(show=True):
    """Generate both source panels, stitch them, and display only the final figure."""
    panel_a = plot_two_energy_fig23(
        seo_file=SEO_FILE,
        save_name=OUTPUT_DIR / "fig23_combined_proj.png",
        show=False,
    )
    panel_b = plot_projection_coordinate_conversion_final(
        seo_file=SEO_FILE,
        target_energy=600.0,
        search_range_deg=(80.7, 81.2),
        x_local=0.65,
        save_name=OUTPUT_DIR / "figure_global2local_panelb.png",
        show=False,
    )

    plt.close("all")
    return stitch_panels(
        img_a=panel_a,
        img_b=panel_b,
        out=OUTPUT_DIR / "fig234_ab_stitched.png",
        show=show,
    )


if __name__ == "__main__":
    main(show=True)

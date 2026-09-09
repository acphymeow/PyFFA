"""Plot the full-ring lattice field and the injection/extraction closed orbits."""

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as path_effects
import numpy as np


# Paper-oriented style controls
FONT_FAMILY = "DejaVu Sans"
LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 15
LINE_WIDTH = 2.2
FIGURE_SIZE = (7.4, 6.2)
DPI = 350

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from bundle_paths import REPO_ROOT
BMAP_DIR = REPO_ROOT / "Bmap_FD16"
SEO_DIR = BMAP_DIR / "resultsSEO_noError"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_NAME = "fig_lattice_layout.png"


def load_closed_orbit(energy_mev: float) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(SEO_DIR / "SEO_r.txt", skiprows=1)
    phi = table[0, 1:]
    energies = table[1:, 0]
    index = int(np.argmin(np.abs(energies - energy_mev)))
    if not np.isclose(energies[index], energy_mev):
        raise ValueError(f"Closed orbit at {energy_mev:g} MeV is unavailable.")
    return phi, table[index + 1, 1:]


def load_full_ring_field(radial_stride: int = 4, azimuthal_stride: int = 10):
    with open(BMAP_DIR / "config_Bmap.json", encoding="utf-8") as stream:
        config = json.load(stream)

    bmap = config["Bmap"]
    sectors = int(bmap["NSector"])
    r_axis = np.asarray(bmap["r_axis"], dtype=float)[::radial_stride]
    bz_cell = np.load(BMAP_DIR / "Bz_order_0.npz")["Bz_Taylor_coeff"]
    bz_cell = bz_cell[::radial_stride, ::azimuthal_stride]

    cell_angle = 2.0 * np.pi / sectors
    phi_cell = np.linspace(0.0, cell_angle, bz_cell.shape[1])
    phi_parts = []
    field_parts = []
    for sector in range(sectors):
        keep = slice(None, -1) if sector < sectors - 1 else slice(None)
        phi_parts.append(phi_cell[keep] + sector * cell_angle)
        field_parts.append(bz_cell[:, keep])

    phi = np.concatenate(phi_parts)
    bz = np.concatenate(field_parts, axis=1)
    x = r_axis[:, None] * np.cos(phi)[None, :]
    y = r_axis[:, None] * np.sin(phi)[None, :]
    return x, y, bz


def make_figure(show: bool = False) -> Path:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": TICK_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "mathtext.fontset": "dejavusans",
        }
    )

    x, y, bz = load_full_ring_field()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    norm = TwoSlopeNorm(vmin=-1.9, vcenter=0.0, vmax=1.9)
    mesh = ax.pcolormesh(
        x,
        y,
        bz,
        shading="gouraud",
        cmap="RdBu_r",
        norm=norm,
        rasterized=True,
    )

    orbit_style = [path_effects.Stroke(linewidth=4.2, foreground="white"), path_effects.Normal()]
    for energy, color, linestyle in (
        (300.0, "black", "-"),
        (600.0, "#14843b", "--"),
    ):
        phi, radius = load_closed_orbit(energy)
        line, = ax.plot(
            radius * np.cos(phi),
            radius * np.sin(phi),
            color=color,
            linestyle=linestyle,
            linewidth=LINE_WIDTH,
            label=rf"{energy:.0f} MeV closed orbit",
            zorder=5,
        )
        line.set_path_effects(orbit_style)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-12.6, 12.6)
    ax.set_ylim(-12.6, 12.6)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.tick_params(direction="in", top=True, right=True, length=5, width=1.0)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="0.75")

    colorbar = fig.colorbar(mesh, ax=ax, pad=0.025, fraction=0.050)
    colorbar.set_label(r"$B_z$ [T]", fontsize=LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_SIZE, direction="in", length=4)

    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.98)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / FIGURE_NAME
    pdf_path = OUTPUT_DIR / FIGURE_NAME.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=DPI, facecolor="white")
    fig.savefig(pdf_path, dpi=DPI, facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it.")
    args = parser.parse_args()
    output = make_figure(show=args.show)
    print(f"Saved {output}")

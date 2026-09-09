"""Plot the matched LSC-off/on 3-D PIC and fixed-toy tune footprints."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "combined_nominal60"
XRANGE = (2.90, 3.30)
YRANGE = (2.50, 2.90)
QR_BARE = 3.2331493677008223
QZ_BARE = 2.813027979088133


def references(axis, star):
    axis.axvline(QR_BARE, ls="--", color="tab:blue", lw=1.6)
    axis.axhline(QZ_BARE, ls="--", color="tab:blue", lw=1.6)
    axis.axvline(star[0], ls=":", color="tab:red", lw=1.7)
    axis.axhline(star[1], ls=":", color="tab:red", lw=1.7)
    axis.scatter(QR_BARE, QZ_BARE, marker="o", s=110, facecolor="white",
                 edgecolor="tab:blue", linewidth=2, label="Bare tune", zorder=10)
    axis.scatter(*star, marker="*", s=360, color="red", edgecolor="black",
                 linewidth=1.3, label="Common reference", zorder=11)
    axis.set_xlim(XRANGE)
    axis.set_ylim(YRANGE)
    axis.set_aspect("auto")
    axis.legend(loc="lower right", frameon=True)


OUT.mkdir(parents=True, exist_ok=True)
off = np.load(RESULTS / "new_lsc_off_nominal60" /
              "new_3d_vs_nominal60_toy_lsc_off.npz")
on = np.load(RESULTS / "new_lsc_on_nominal60" /
             "new_3d_vs_nominal60_theory_lsc_on.npz")

datasets_3d = ((off["qr_3d"], off["qz_3d"]),
               (on["new_qr_fft"], on["new_qz_fft"]))
histograms = [np.histogram2d(qr, qz, bins=256, range=(XRANGE, YRANGE))[0]
              for qr, qz in datasets_3d]
norm = LogNorm(vmin=1, vmax=max(hist.max() for hist in histograms))

fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.2), sharex=True, sharey=True,
                         constrained_layout=True)
titles = (
    "3-D PIC tracking (LSC off)",
    r"Fixed toy (LSC off, $\epsilon_{x,z}=60\pi$ mm mrad)",
    "3-D PIC tracking (LSC on)",
    r"Fixed toy (LSC on, $\epsilon_{x,z}=60\pi$ mm mrad)",
)
stars = (tuple(off["common_star"]), tuple(off["common_star"]),
         tuple(on["common_star"]), tuple(on["common_star"]))

mesh = None
for row, histogram in enumerate(histograms):
    mesh = axes[row, 0].pcolormesh(
        np.linspace(*XRANGE, 257), np.linspace(*YRANGE, 257), histogram.T,
        norm=norm, cmap="viridis", shading="auto", rasterized=True,
    )
axes[0, 1].scatter(
    off["qr_toy"], off["qz_toy"], s=15, alpha=.72,
    color="tab:purple", linewidths=0, rasterized=True,
)
axes[1, 1].scatter(
    on["theory_qr_fft"], on["theory_qz_fft"], s=15, alpha=.72,
    color="tab:purple", linewidths=0, rasterized=True,
)

for index, axis in enumerate(axes.flat):
    axis.set_title(titles[index])
    references(axis, stars[index])
    axis.text(.045, .89, f"({chr(97 + index)})", transform=axis.transAxes,
              fontsize=25, fontweight="bold")
for axis in axes[:, 0]:
    axis.set_ylabel(r"$Q_z$")
for axis in axes[1, :]:
    axis.set_xlabel(r"$Q_r$")

fig.colorbar(mesh, ax=axes[:, 0], label="Counts", pad=.045, fraction=.045)
output = OUT / "lsc_on_off_combined_common_axes_common_stars"
fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved {output.with_suffix('.png')}")
print(f"Axes: Qr={XRANGE}, Qz={YRANGE}")

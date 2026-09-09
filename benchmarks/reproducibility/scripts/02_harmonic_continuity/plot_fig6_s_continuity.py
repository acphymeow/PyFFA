"""Reproduce the revised paper Fig. 6 (h=1 longitudinal continuity).

The occupied interval in panel (c) is shown as -30 to +30 deg, matching the
displayed bunch extent and the referee-requested correction.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as ROOT

SEO_FILE = ROOT / "Bmap_FD16" / "resultsSEO_noError" / "SEO_r.txt"
OUT_PNG = HERE / "fig6_s_continuity.png"
OUT_PDF = HERE / "fig6_s_continuity.pdf"
PERIOD = 2.0 * np.pi


def continuous_phi_mod(phi_mod, period=PERIOD):
    shifts = np.array([0.0, 0.5 * period, 0.25 * period, 0.75 * period])
    shifted = (phi_mod[:, None] + shifts) % period
    best = np.argmin(np.ptp(shifted, axis=0))
    return shifted[:, best] - shifts[best]


def generate_bunch(seo_phi, seo_r, n_points=50000, seed=1234):
    rng = np.random.default_rng(seed)
    center = np.deg2rad(90.0)
    half_width = np.deg2rad(30.0)
    phi = np.clip(rng.normal(center, np.deg2rad(8.0), n_points),
                  center - half_width, center + half_width)
    r = np.interp(phi, seo_phi, seo_r) + rng.normal(0.0, 0.008, n_points)
    return phi, r


def plot_figure(phi_true, phi_wrapped, phi_cont, radius):
    phi_true = np.rad2deg(phi_true)
    phi_wrapped = np.rad2deg(phi_wrapped)
    phi_cont = np.rad2deg(phi_cont)

    rng = np.random.default_rng(2026)
    shown = rng.choice(len(radius), min(18000, len(radius)), replace=False)
    scatter = dict(s=1.2, alpha=0.22, linewidths=0, rasterized=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), sharey=True,
                             constrained_layout=True)

    axes[0].scatter(phi_true[shown], radius[shown], color="0.25", **scatter)
    axes[0].set_title("(a) Multi-turn coordinate", fontsize=13)
    axes[0].set_xlabel(r"Unwrapped azimuth $\phi$ [deg]", fontsize=13)
    axes[0].set_ylabel(r"Radius $r$ [m]", fontsize=18)

    axes[1].scatter(phi_wrapped[shown], radius[shown], color="tab:red", **scatter)
    axes[1].set_title(r"(b) After modulo $2\pi$", fontsize=13)
    axes[1].set_xlabel(r"Wrapped azimuth $\phi_{\rm mod}$ [deg]", fontsize=13)
    axes[1].set_xlim(0.0, 360.0)
    axes[1].axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.65)
    axes[1].axvline(360.0, color="k", lw=0.8, ls="--", alpha=0.65)
    axes[1].text(100.0, radius.max() - 0.02 * np.ptp(radius),
                 "periodic\ncut", fontsize=12, va="top", ha="left")

    axes[2].scatter(phi_cont[shown], radius[shown], color="tab:blue", **scatter)
    axes[2].set_title("(c) Reconstructed coordinate", fontsize=13)
    axes[2].set_xlabel(r"Continuous coordinate $\phi_c$ [deg]", fontsize=13)
    y_arrow = radius.min() + 0.08 * np.ptp(radius)
    axes[2].annotate("", xy=(-30.0, y_arrow), xytext=(30.0, y_arrow),
                     arrowprops=dict(arrowstyle="<->", lw=1.0, color="k"))
    axes[2].text(0.0, y_arrow + 0.04 * np.ptp(radius),
                 "occupied longitudinal interval", fontsize=14,
                 ha="center", va="bottom")
    axes[2].set_xlim(-34.0, 34.0)

    pad = 0.08 * np.ptp(radius)
    for ax in axes:
        ax.tick_params(labelsize=9, direction="in")
        ax.set_ylim(radius.min() - pad, radius.max() + pad)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT_PNG}")
    print(f"saved -> {OUT_PDF}")


def main():
    raw = np.loadtxt(SEO_FILE, skiprows=1)
    seo_phi = raw[0, 1:]
    seo_r = raw[2, 1:]
    phi, radius = generate_bunch(seo_phi, seo_r)
    phi_origin = phi - np.mean(phi)
    phi_origin[:10001] += 2.0 * np.pi
    phi_origin[10001:20001] += 4.0 * np.pi
    phi_origin[20001:30001] += 6.0 * np.pi
    phi_wrapped = phi_origin % PERIOD
    phi_cont = continuous_phi_mod(phi_wrapped)
    plot_figure(phi_origin, phi_wrapped, phi_cont, radius)


if __name__ == "__main__":
    main()

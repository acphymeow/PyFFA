from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_SCRIPTS = SCRIPT_DIR.parent
if str(BUNDLE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BUNDLE_SCRIPTS))

from bundle_paths import REPO_ROOT


OUTPUT_DIR = SCRIPT_DIR / "outputs"
SEO_FILE = REPO_ROOT / "Bmap_FD16" / "resultsSEO_noError" / "SEO_r.txt"


# ============================================================
# 0. Global plot style
# ============================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14


# ============================================================
# 1. Color settings
#    Modify colors here only.
# ============================================================
COLORS = {
    # Background / orbit / grid
    "orbit": "#2F2F2F",
    "label_orbit": "#2F2F2F",
    "grid": "#D0D0D0",
    "legend_edge": "#8A8A8A",
    "marker_edge": "k",

    # Particle point
    "particle": "#A64D79",
    "label_particle": "#A64D79",

    # Projection point R0(phi*) and local coordinate axes
    "projection": "#0072B2",
    "label_projection": "#0072B2",
    "local_axis": "#0072B2",

    # Same-azimuth point R0(phi_p) and cylindrical coordinate axes
    "same_phi": "#E66100",
    "label_same_phi": "#E66100",
    "label_phi": "#E66100",
    "cyl_axis": "#E66100",

    # Radial dashed lines
    "radial_line": "#B3B3B3",              # phi_p radial line
    "radial_line_phi_star": "#9E9E9E",    # phi* radial line
    "label_phi_star": "#0072B2",

    # Construction / auxiliary lines
    "projection_line": "#202020",
    "same_phi_line": "#8F8F8F",

    # Geometric annotations
    "distance_arrow": "#101010",
    "right_angle": "#101010",
}


# ============================================================
# 2. Geometry / arrow-length settings
#    Modify sizes and lengths here only.
# ============================================================
GEOM = {
    # Arrow lengths
    "local_s_length": 0.20,
    "local_x_length": 0.20,
    "cyl_length": 0.18,

    # Label scaling along arrows
    "local_s_text_scale": 1.16,
    "local_x_text_scale": 1.15,
    "cyl_text_scale": 1.20,

    # Local basis label adjustment
    "local_s_label_dx": -0.02,
    "local_s_label_dy": 0.06,
    "local_x_label_dx": 0.13,
    "local_x_label_dy": 0.08,

    # Cylindrical basis label adjustment
    # These are tuned for \hat{e}_r and \hat{e}_phi labels.
    "cyl_er_label_dx": 0.10,
    "cyl_er_label_dy": -0.04,
    "cyl_ephi_label_dx": -0.105,
    "cyl_ephi_label_dy": -0.050,

    # Right-angle marker
    "right_angle_size": 0.052,

    # Double-arrow for x
    "x_arrow_offset": 0.10,

    # Radial label positions
    "phi_frac": 0.33,
    "phi_normal_offset": 0.035,
    "phi_star_frac": 0.38,
    "phi_star_normal_offset": -0.050,
    "phi_star_extend_frac": 0.04,

    # Line widths
    "orbit_lw": 1.8,
    "same_phi_line_lw": 2.0,
    "projection_line_lw": 1.45,
    "local_axis_lw": 2.0,
    "cyl_axis_lw": 1.35,
    "distance_arrow_lw": 1.0,
    "right_angle_lw": 1.1,
    "radial_line_lw": 2.0,
    "radial_line_phi_star_lw": 2.0,

    # Arrow mutation scales
    "main_mutation_scale": 16,
    "distance_mutation_scale": 13,

    # Marker sizes
    "particle_size": 90,
    "projection_size": 95,
    "same_phi_size": 82,

    # Legend marker sizes
    "legend_particle_size": 80,
    "legend_projection_size": 85,
    "legend_same_phi_size": 80,

    # Figure size
    "figsize": (7.2, 6.4),

    # Grid
    "grid_lw": 0.45,
    "grid_alpha": 0.20,
}


# ============================================================
# 3. Basic utilities
# ============================================================
def find_energy_row(ek_list, target_energy):
    return np.argmin(np.abs(ek_list - target_energy))


def orbit_xy_from_row(r_row, phi_rad):
    return r_row * np.cos(phi_rad), r_row * np.sin(phi_rad)


def unit(vx, vy):
    norm = np.hypot(vx, vy)
    if norm == 0.0:
        return 1.0, 0.0
    return vx / norm, vy / norm


def interp_periodic(phi_query, phi_axis, values):
    twopi = 2.0 * np.pi
    phi_q = np.mod(phi_query, twopi)

    phi_ext = np.concatenate([phi_axis, [phi_axis[0] + twopi]])
    val_ext = np.concatenate([values, [values[0]]])

    return np.interp(phi_q, phi_ext, val_ext)


def local_tangent_normal(xo, yo, idx):
    i0 = max(idx - 1, 0)
    i1 = min(idx + 1, len(xo) - 1)

    tx = xo[i1] - xo[i0]
    ty = yo[i1] - yo[i0]
    tx, ty = unit(tx, ty)

    nx = -ty
    ny = tx

    # Make normal approximately outward.
    if xo[idx] * nx + yo[idx] * ny < 0:
        nx = -nx
        ny = -ny

    return tx, ty, nx, ny


def find_max_rprime_index(phi_rad, r_row, search_range_deg=(40.0, 140.0)):
    rprime = np.gradient(r_row, phi_rad)
    phi_deg = np.rad2deg(phi_rad)

    pmin, pmax = search_range_deg
    mask = (phi_deg >= pmin) & (phi_deg <= pmax)
    idx_candidates = np.where(mask)[0]

    if len(idx_candidates) == 0:
        raise ValueError("search_range_deg does not overlap with phi_rad.")

    idx = idx_candidates[np.argmax(np.abs(rprime[idx_candidates]))]
    return idx, phi_deg[idx], rprime[idx]


# ============================================================
# 4. Drawing utilities
# ============================================================
def draw_arrow(ax, x0, y0, vx, vy, label=None,
               length=0.30, color="k", lw=1.5,
               fontsize=16, text_scale=1.20,
               label_dx=0.0, label_dy=0.0,
               zorder=10, mutation_scale=16):

    ax.annotate(
        "",
        xy=(x0 + length * vx, y0 + length * vy),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color,
            mutation_scale=mutation_scale,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )

    if label is not None:
        ax.text(
            x0 + text_scale * length * vx + label_dx,
            y0 + text_scale * length * vy + label_dy,
            label,
            fontsize=fontsize,
            color=color,
            ha="center",
            va="center",
            zorder=zorder + 1,
        )


def draw_right_angle_marker(ax, x0, y0, tx, ty, nx, ny,
                            size=0.055, color="k", lw=1.0, zorder=20):
    p1x = x0 + size * tx
    p1y = y0 + size * ty

    p2x = p1x + size * nx
    p2y = p1y + size * ny

    p3x = x0 + size * nx
    p3y = y0 + size * ny

    ax.plot(
        [p1x, p2x, p3x],
        [p1y, p2y, p3y],
        color=color,
        lw=lw,
        zorder=zorder,
    )


def draw_double_arrow(ax, x1, y1, x2, y2, label,
                      color="k", lw=1.3, fontsize=18,
                      offset=0.095, zorder=25,
                      mutation_scale=13):
    dx = x2 - x1
    dy = y2 - y1
    tx, ty = unit(dx, dy)

    nx = -ty
    ny = tx

    xa1 = x1 + offset * nx
    ya1 = y1 + offset * ny
    xa2 = x2 + offset * nx
    ya2 = y2 + offset * ny

    ax.annotate(
        "",
        xy=(xa2, ya2),
        xytext=(xa1, ya1),
        arrowprops=dict(
            arrowstyle="<->",
            lw=lw,
            color=color,
            mutation_scale=mutation_scale,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )

    ax.plot([x1, xa1], [y1, ya1], color=color, lw=0.8, zorder=zorder)
    ax.plot([x2, xa2], [y2, ya2], color=color, lw=0.8, zorder=zorder)

    xm = 0.5 * (xa1 + xa2)
    ym = 0.5 * (ya1 + ya2)

    ax.text(
        xm - 0.04,
        ym + 0.04,
        label,
        fontsize=fontsize,
        color=color,
        ha="left",
        va="center",
        zorder=zorder + 1,
    )


def draw_visible_radial_line(ax, x_target, y_target, xlim, ylim,
                             color=None, lw=1.0,
                             ls="--", alpha=0.65, zorder=2,
                             extend_frac=0.0):
    """
    Draw the visible part of the radial line from origin toward target point.

    If extend_frac > 0, the radial line is extended beyond the target point
    by extend_frac * |target|.
    """
    if color is None:
        color = COLORS["radial_line"]

    x0, y0 = 0.0, 0.0

    # Extend the line beyond the target point.
    x1 = x_target * (1.0 + extend_frac)
    y1 = y_target * (1.0 + extend_frac)

    ts = []

    if abs(x1 - x0) > 1.0e-14:
        ts.append((xlim[0] - x0) / (x1 - x0))
        ts.append((xlim[1] - x0) / (x1 - x0))

    if abs(y1 - y0) > 1.0e-14:
        ts.append((ylim[0] - y0) / (y1 - y0))
        ts.append((ylim[1] - y0) / (y1 - y0))

    candidates = []
    for t in ts:
        if 0.0 <= t <= 1.0:
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)

            if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]:
                candidates.append((t, x, y))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda item: item[0])
    _, x_enter, y_enter = candidates[0]

    ax.plot(
        [x_enter, x1],
        [y_enter, y1],
        ls=ls,
        lw=lw,
        color=color,
        alpha=alpha,
        zorder=zorder,
    )

    return x_enter, y_enter


def draw_radial_label_on_line(ax, x_target, y_target, xlim, ylim,
                              label, color="k", fontsize=17,
                              frac=0.42, normal_offset=0.035,
                              line_color=None, line_lw=1.0,
                              line_alpha=0.65,
                              zorder=30,
                              manual_xy=None,
                              extend_frac=0.0):
    """
    Draw a radial dashed line and place a label on it.

    If manual_xy is not None, the label will be placed at manual_xy.
    Otherwise, the label position is computed along the visible radial line.
    """
    if line_color is None:
        line_color = color

    p_enter = draw_visible_radial_line(
        ax,
        x_target,
        y_target,
        xlim,
        ylim,
        color=line_color,
        lw=line_lw,
        ls="--",
        alpha=line_alpha,
        zorder=2,
        extend_frac=extend_frac,
    )

    if p_enter is None:
        return

    if manual_xy is not None:
        xl, yl = manual_xy
    else:
        xe, ye = p_enter
        xl = xe + frac * (x_target - xe)
        yl = ye + frac * (y_target - ye)

        vx, vy = unit(x_target, y_target)
        nx, ny = -vy, vx

        xl = xl + normal_offset * nx
        yl = yl + normal_offset * ny

    ax.text(
        xl,
        yl,
        label,
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
        zorder=zorder,
    )


# ============================================================
# 5. Main plotting function
# ============================================================
def plot_projection_coordinate_conversion_final(
        seo_file=SEO_FILE,
        target_energy=600.0,
        search_range_deg=(80.7, 81.2),
        x_local=0.55,
        save_name=OUTPUT_DIR / "figure_global2local_panelb.png",
        show=True):

    seo_data = np.loadtxt(seo_file, skiprows=1)

    phi_rad = seo_data[0, 1:]
    r_matrix = seo_data[1:, 1:]
    ek_list = seo_data[1:, 0]

    iE = find_energy_row(ek_list, target_energy)
    r_row = r_matrix[iE]
    xo, yo = orbit_xy_from_row(r_row, phi_rad)

    idx, phi_star_deg, rprime_star = find_max_rprime_index(
        phi_rad,
        r_row,
        search_range_deg=search_range_deg,
    )

    phi_star = phi_rad[idx]
    X_star, Y_star = xo[idx], yo[idx]

    shat_x, shat_y, xhat_x, xhat_y = local_tangent_normal(xo, yo, idx)

    # Construct a sample particle point along local x-hat.
    Xp = X_star + x_local * xhat_x
    Yp = Y_star + x_local * xhat_y

    phi_p = np.arctan2(Yp, Xp)
    if phi_p < 0:
        phi_p += 2.0 * np.pi

    phi_p_deg = np.rad2deg(phi_p)

    # Same-azimuth orbit point R0(phi_p).
    r_same = interp_periodic(phi_p, phi_rad, r_row)
    X_same = r_same * np.cos(phi_p)
    Y_same = r_same * np.sin(phi_p)

    # Cylindrical basis at phi_p.
    erp_x = np.cos(phi_p)
    erp_y = np.sin(phi_p)
    ephip_x = -np.sin(phi_p)
    ephip_y = np.cos(phi_p)

    print("Selected geometry:")
    print(f"  Ek              = {target_energy:.1f} MeV")
    print(f"  phi*            = {phi_star_deg:.3f} deg")
    print(f"  phi_p           = {phi_p_deg:.3f} deg")
    print(f"  phi_p - phi*    = {phi_p_deg - phi_star_deg:.3f} deg")
    print(f"  dr/dphi(phi*)   = {rprime_star:.6e} m/rad")
    print(f"  x_local         = {x_local:.3f} m")

    all_x = np.array([X_star, Xp, X_same])
    all_y = np.array([Y_star, Yp, Y_same])

    half_width = max(
        0.72,
        1.38 * max(
            np.max(all_x) - np.min(all_x),
            np.max(all_y) - np.min(all_y),
        ),
    )

    xc = 0.5 * (np.min(all_x) + np.max(all_x))
    yc = 0.5 * (np.min(all_y) + np.max(all_y))

    xlim = (xc - half_width * 0.95, xc + half_width * 1.05)
    ylim = (yc - half_width * 0.90, yc + half_width * 0.90)

    fig, ax = plt.subplots(figsize=GEOM["figsize"], constrained_layout=True)

    # --------------------------------------------------------
    # Closed orbit
    # --------------------------------------------------------
    handle_orbit, = ax.plot(
        xo,
        yo,
        color=COLORS["orbit"],
        lw=GEOM["orbit_lw"],
        zorder=3,
        label=r"$\mathbf{R}_0(\theta)$: closed orbit",
    )

    # Keep the in-figure closed-orbit label.
    ax.text(
        2.70,
        11.42,
        r"$\mathbf{R}_0(\theta)$",
        fontsize=18,
        color=COLORS["label_orbit"],
        ha="center",
        va="top",
        zorder=30,
    )

    # --------------------------------------------------------
    # Radial lines: phi_p and phi*
    # --------------------------------------------------------
    draw_radial_label_on_line(
        ax,
        X_same,
        Y_same,
        xlim,
        ylim,
        label=r"$\theta_p$",
        color=COLORS["label_phi"],
        fontsize=18,
        frac=GEOM["phi_frac"],
        normal_offset=GEOM["phi_normal_offset"],
        line_color=COLORS["radial_line"],
        line_lw=GEOM["radial_line_lw"],
        line_alpha=0.65,
        manual_xy=(2.04, 11.46),
        extend_frac=0.0,
    )

    draw_radial_label_on_line(
        ax,
        X_star,
        Y_star,
        xlim,
        ylim,
        label=r"$\theta^*$",
        color=COLORS["label_phi_star"],
        fontsize=18,
        frac=GEOM["phi_star_frac"],
        normal_offset=GEOM["phi_star_normal_offset"],
        line_color=COLORS["radial_line_phi_star"],
        line_lw=GEOM["radial_line_phi_star_lw"],
        line_alpha=0.80,
        manual_xy=(1.79, 11.46),
        extend_frac=GEOM["phi_star_extend_frac"],
    )

    # --------------------------------------------------------
    # Construction lines
    # --------------------------------------------------------
    # Same-azimuth construction line
    ax.plot(
        [X_same, Xp],
        [Y_same, Yp],
        ls="--",
        lw=GEOM["same_phi_line_lw"],
        color=COLORS["same_phi_line"],
        alpha=0.75,
        zorder=5,
    )

    # Normal projection line
    ax.plot(
        [X_star, Xp],
        [Y_star, Yp],
        ls="--",
        lw=GEOM["projection_line_lw"],
        color=COLORS["projection_line"],
        alpha=0.95,
        zorder=6,
    )

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------
    ax.scatter(
        [Xp],
        [Yp],
        s=GEOM["particle_size"],
        c=COLORS["particle"],
        edgecolors=COLORS["marker_edge"],
        linewidths=0.8,
        zorder=14,
    )

    ax.scatter(
        [X_star],
        [Y_star],
        s=GEOM["projection_size"],
        c=COLORS["projection"],
        edgecolors=COLORS["marker_edge"],
        linewidths=0.8,
        zorder=13,
    )

    ax.scatter(
        [X_same],
        [Y_same],
        s=GEOM["same_phi_size"],
        c=COLORS["same_phi"],
        edgecolors=COLORS["marker_edge"],
        linewidths=0.75,
        zorder=12,
    )

    # --------------------------------------------------------
    # Local basis at R0(phi*)
    # --------------------------------------------------------
    draw_arrow(
        ax,
        X_star,
        Y_star,
        shat_x,
        shat_y,
        r"$\hat{\mathbf{s}}$",
        length=GEOM["local_s_length"],
        color=COLORS["local_axis"],
        lw=GEOM["local_axis_lw"],
        fontsize=18,
        text_scale=GEOM["local_s_text_scale"],
        label_dx=GEOM["local_s_label_dx"],
        label_dy=GEOM["local_s_label_dy"],
        zorder=20,
        mutation_scale=GEOM["main_mutation_scale"],
    )

    draw_arrow(
        ax,
        X_star,
        Y_star,
        xhat_x,
        xhat_y,
        r"$\hat{\mathbf{x}}$",
        length=GEOM["local_x_length"],
        color=COLORS["local_axis"],
        lw=GEOM["local_axis_lw"],
        fontsize=18,
        text_scale=GEOM["local_x_text_scale"],
        label_dx=GEOM["local_x_label_dx"],
        label_dy=GEOM["local_x_label_dy"],
        zorder=20,
        mutation_scale=GEOM["main_mutation_scale"],
    )

    draw_right_angle_marker(
        ax,
        X_star,
        Y_star,
        shat_x,
        shat_y,
        xhat_x,
        xhat_y,
        size=GEOM["right_angle_size"],
        color=COLORS["right_angle"],
        lw=GEOM["right_angle_lw"],
        zorder=21,
    )

    draw_double_arrow(
        ax,
        X_star,
        Y_star,
        Xp,
        Yp,
        r"$x$",
        offset=GEOM["x_arrow_offset"],
        fontsize=18,
        color=COLORS["distance_arrow"],
        lw=GEOM["distance_arrow_lw"],
        zorder=22,
        mutation_scale=GEOM["distance_mutation_scale"],
    )

    # --------------------------------------------------------
    # Cylindrical basis at R0(phi_p)
    # --------------------------------------------------------
    draw_arrow(
        ax,
        X_same,
        Y_same,
        erp_x,
        erp_y,
        r"$\hat{\mathbf{e}}_r$",
        length=GEOM["cyl_length"],
        color=COLORS["cyl_axis"],
        lw=GEOM["cyl_axis_lw"],
        fontsize=15,
        text_scale=GEOM["cyl_text_scale"],
        label_dx=GEOM["cyl_er_label_dx"],
        label_dy=GEOM["cyl_er_label_dy"],
        zorder=21,
        mutation_scale=GEOM["main_mutation_scale"],
    )

    draw_arrow(
        ax,
        X_same,
        Y_same,
        ephip_x,
        ephip_y,
        r"$\hat{\mathbf{e}}_{\theta}$",
        length=GEOM["cyl_length"],
        color=COLORS["cyl_axis"],
        lw=GEOM["cyl_axis_lw"],
        fontsize=15,
        text_scale=GEOM["cyl_text_scale"],
        label_dx=GEOM["cyl_ephi_label_dx"],
        label_dy=GEOM["cyl_ephi_label_dy"],
        zorder=21,
        mutation_scale=GEOM["main_mutation_scale"],
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------
    handle_particle = ax.scatter(
        [],
        [],
        s=GEOM["legend_particle_size"],
        c=COLORS["particle"],
        edgecolors=COLORS["marker_edge"],
        linewidths=0.75,
        label=r"$\mathbf{R}_p$: particle position",
    )

    handle_same = ax.scatter(
        [],
        [],
        s=GEOM["legend_same_phi_size"],
        c=COLORS["same_phi"],
        edgecolors=COLORS["marker_edge"],
        linewidths=0.75,
        label=r"$\mathbf{R}_0(\theta_p)$: same-azimuth orbit point",
    )

    handle_proj = ax.scatter(
        [],
        [],
        s=GEOM["legend_projection_size"],
        c=COLORS["projection"],
        edgecolors=COLORS["marker_edge"],
        linewidths=0.8,
        label=r"$\mathbf{R}_0(\theta^*)$: normal projection point",
    )

    ax.legend(
        handles=[handle_orbit, handle_particle, handle_same, handle_proj],
        loc="upper left",
        frameon=False,
        fontsize=12.5,
        handlelength=1.4,
        handletextpad=0.6,
        labelspacing=0.5,
        borderpad=0.0,
    )

    # --------------------------------------------------------
    # Axes / grid
    # --------------------------------------------------------
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Global coordinate $X$ [m]")
    ax.set_ylabel(r"Global coordinate $Y$ [m]")

    ax.grid(
        True,
        lw=GEOM["grid_lw"],
        alpha=GEOM["grid_alpha"],
        ls="--",
        color=COLORS["grid"],
    )

    save_path = Path(save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return save_path


if __name__ == "__main__":
    plot_projection_coordinate_conversion_final(
        seo_file=SEO_FILE,
        target_energy=600.0,
        search_range_deg=(80.7, 81.2),
        x_local=0.65,
        save_name=OUTPUT_DIR / "figure_global2local_panelb.png",
        show=True,
    )

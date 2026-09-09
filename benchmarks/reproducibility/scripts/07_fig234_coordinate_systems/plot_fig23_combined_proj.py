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


# =========================================================
# 0. Matplotlib global style
# =========================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16


# =========================================================
# 1. User-adjustable colors and geometry
# =========================================================
COLORS = {
    "bunch_high": "#BFBFBF",   # main bunch, neutral gray
    "bunch_low": "#CDE5C0",    # secondary bunch, light muted green
    "orbit": "#000000",        # reference closed orbits
    "ref_point": "#9A60A8",    # purple
    "local": "#0072B2",        # blue
    "global": "#D55E00",       # orange
    "grid": "0.80",
}

GEOM = {
    # -----------------------------------------------------
    # Two energies [MeV] (nearest rows in SEO table are used)
    # -----------------------------------------------------
    "energy_low": 300.0,
    "energy_high": 600.0,

    # -----------------------------------------------------
    # Bunch generation (same style as original Fig. 3)
    # -----------------------------------------------------
    "n_particles_high": 55000,
    "n_particles_low": 45000,
    "sigma_x": 0.090,
    "phi_center_deg": 90.0,
    "phi_sigma_deg": 13.0,

    # Representative reference points (before left-right sorting)
    # on the HIGH-energy orbit
    "example_phi_deg_list": (62.0, 82.0, 101.0, 119.0),

    # Which point (after left-to-right sorting: 0,1,2,3)
    # gets the cylindrical frame
    "global_frame_index": 2,

    # -----------------------------------------------------
    # Arrow lengths
    # -----------------------------------------------------
    "main_Ls": 1.1,
    "main_Lx": 1.1,
    "main_Ler": 1.1,
    "main_Lephi": 1.1,

    # -----------------------------------------------------
    # Arrow styles
    # -----------------------------------------------------
    "arrow_lw_local_main": 1.6,
    "arrow_lw_global_main": 1.6,
    "arrow_ms_main": 13,

    # -----------------------------------------------------
    # Scatter sizes / alpha
    # -----------------------------------------------------
    "bunch_size": 1.2,
    "bunch_alpha_high": 0.45,
    "bunch_alpha_low": 0.55,
    "ref_point_size_main": 140,

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------
    "figsize": (11.0, 5.6),

    # -----------------------------------------------------
    # Small visual rotation (deg) of local frame at the
    # selected point only, to separate blue/orange arrows.
    # -----------------------------------------------------
    "selected_local_rotation_deg": -4.0,

    # -----------------------------------------------------
    # Font sizes
    # -----------------------------------------------------
    "frame_label_fontsize": 20,
    "title_label_fontsize": 18,
    "energy_label_fontsize": 17,

    # -----------------------------------------------------
    # Dashed radial line
    # -----------------------------------------------------
    "radial_dash_lw": 3.0,
    "radial_dash_alpha": 0.55,

    # -----------------------------------------------------
    # Energy labels near the right window edge
    # -----------------------------------------------------
    "energy_label_x_high": 6.20,
    "energy_label_x_low": 5.80,
    "energy_label_y_high": 10.50,
    "energy_label_y_low": 8.30,
    "energy_label_ha": "left",
}


# =========================================================
# 2. Per-point label position control (1..4 after L->R sort)
# =========================================================
LOCAL_LABEL_CFG = {
    1: {
        "s_scale": 1.12, "s_dx": 0.2, "s_dy": -0.5,
        "x_scale": 1.16, "x_dx": -0.03, "x_dy":  0.02,
    },
    2: {
        "s_scale": 1.10, "s_dx": 0.10, "s_dy": -0.40,
        "x_scale": 1.18, "x_dx": -0.20, "x_dy": -0.40,
    },
    3: {
        "s_scale": 1.12, "s_dx": 0.15, "s_dy": 0.35,
        "x_scale": 1.18, "x_dx":  0.50, "x_dy":  -0.45,
    },
    4: {
        "s_scale": 1.10, "s_dx": -0.03, "s_dy": -0.35,
        "x_scale": 1.16, "x_dx":  0.03, "x_dy":  0.01,
    },
}

# =========================================================
# 3. Global cylindrical label position control
# =========================================================
GLOBAL_LABEL_CFG = {
    "er_scale": 1.15,
    "er_dx": -0.85,
    "er_dy": -0.25,

    "ephi_scale": 1.30,
    "ephi_dx": -0.10,
    "ephi_dy": -0.45,
}


# =========================================================
# 4. Basic utilities
# =========================================================
def find_energy_row(ek_list, target_energy):
    return np.argmin(np.abs(ek_list - target_energy))


def orbit_xy_from_row(r_row, phi_rad):
    x = r_row * np.cos(phi_rad)
    y = r_row * np.sin(phi_rad)
    return x, y


def normalize(vx, vy):
    nrm = np.hypot(vx, vy)
    if nrm == 0.0:
        return 1.0, 0.0
    return vx / nrm, vy / nrm


def rotate_vec(vx, vy, angle_deg):
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    xr = c * vx - s * vy
    yr = s * vx + c * vy
    return xr, yr


def local_tangent_normal(xo, yo, idx):
    npt = len(xo)
    i0 = max(idx - 1, 0)
    i1 = min(idx + 1, npt - 1)

    tx = xo[i1] - xo[i0]
    ty = yo[i1] - yo[i0]
    tx, ty = normalize(tx, ty)

    nx = -ty
    ny = tx

    if xo[idx] * nx + yo[idx] * ny < 0:
        nx = -nx
        ny = -ny

    return tx, ty, nx, ny


def global_cylindrical_basis(x0, y0):
    erx, ery = normalize(x0, y0)
    ephix, ephiy = -ery, erx
    return erx, ery, ephix, ephiy


def generate_gaussian_arc_bunch_from_orbit(
        xo, yo, phi_rad,
        phi_center_deg=90.0,
        phi_sigma_deg=13.0,
        n_particles=50000,
        sigma_x=0.085,
        seed=2026):
    rng = np.random.default_rng(seed)

    phi_center = np.deg2rad(phi_center_deg)
    phi_sigma = np.deg2rad(phi_sigma_deg)

    phi_p = rng.normal(phi_center, phi_sigma, n_particles)
    phi_mod = np.mod(phi_p, 2.0 * np.pi)

    x0_p = np.interp(phi_mod, phi_rad, xo)
    y0_p = np.interp(phi_mod, phi_rad, yo)

    nx_p = np.empty(n_particles)
    ny_p = np.empty(n_particles)

    for i in range(n_particles):
        idx = np.argmin(np.abs(phi_rad - phi_mod[i]))
        _, _, nx, ny = local_tangent_normal(xo, yo, idx)
        nx_p[i] = nx
        ny_p[i] = ny

    x_local = rng.normal(0.0, sigma_x, n_particles)

    xb = x0_p + x_local * nx_p
    yb = y0_p + x_local * ny_p

    return xb, yb


def build_reference_points(xo, yo, phi_rad, example_phi_deg_list):
    pts = []

    for phi_deg in example_phi_deg_list:
        phi0 = np.deg2rad(phi_deg)
        idx = np.argmin(np.abs(phi_rad - phi0))

        x0 = xo[idx]
        y0 = yo[idx]

        tx, ty, nx, ny = local_tangent_normal(xo, yo, idx)
        erx, ery, ephix, ephiy = global_cylindrical_basis(x0, y0)

        pts.append({
            "idx": idx,
            "phi_deg": phi_deg,
            "x0": x0,
            "y0": y0,
            "tx": tx,
            "ty": ty,
            "nx": nx,
            "ny": ny,
            "erx": erx,
            "ery": ery,
            "ephix": ephix,
            "ephiy": ephiy,
        })

    pts = sorted(pts, key=lambda d: d["x0"])
    return pts


# =========================================================
# 5. Drawing helpers
# =========================================================
def draw_arrow(ax, x0, y0, vx, vy, length, color, lw, ms, zorder=10):
    ax.annotate(
        "",
        xy=(x0 + length * vx, y0 + length * vy),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color,
            mutation_scale=ms,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )


def add_vec_label(ax, x, y, text, color,
                  fontsize=13, ha="center", va="center", zorder=20):
    ax.text(
        x, y, text,
        color=color,
        fontsize=fontsize,
        ha=ha,
        va=va,
        zorder=zorder
    )


def draw_local_frame(
        ax, pt,
        Ls, Lx,
        lw, ms,
        point_number=None,
        show_labels=True,
        label_fs=13,
        visual_rotation_deg=0.0,
        label_cfg=None,
        zorder=12):
    x0, y0 = pt["x0"], pt["y0"]

    tx, ty = pt["tx"], pt["ty"]
    nx, ny = pt["nx"], pt["ny"]

    if abs(visual_rotation_deg) > 1e-12:
        tx, ty = rotate_vec(tx, ty, visual_rotation_deg)
        nx, ny = rotate_vec(nx, ny, visual_rotation_deg)

    draw_arrow(
        ax, x0, y0, tx, ty,
        Ls, COLORS["local"], lw, ms, zorder=zorder
    )
    draw_arrow(
        ax, x0, y0, nx, ny,
        Lx, COLORS["local"], lw, ms, zorder=zorder
    )

    if show_labels:
        if label_cfg is None:
            label_cfg = {}

        s_scale = label_cfg.get("s_scale", 1.14)
        s_dx = label_cfg.get("s_dx", 0.0)
        s_dy = label_cfg.get("s_dy", 0.0)

        x_scale = label_cfg.get("x_scale", 1.18)
        x_dx = label_cfg.get("x_dx", 0.0)
        x_dy = label_cfg.get("x_dy", 0.0)

        if point_number is None:
            s_text = r"$\hat{\mathbf{s}}$"
            x_text = r"$\hat{\mathbf{x}}$"
        else:
            s_text = rf"$\hat{{\mathbf{{s}}}}_{{{point_number}}}$"
            x_text = rf"$\hat{{\mathbf{{x}}}}_{{{point_number}}}$"

        s_lab_x = x0 + s_scale * Ls * tx + s_dx
        s_lab_y = y0 + s_scale * Ls * ty + s_dy
        add_vec_label(
            ax, s_lab_x, s_lab_y, s_text,
            COLORS["local"], fontsize=label_fs, zorder=zorder + 5
        )

        x_lab_x = x0 + x_scale * Lx * nx + x_dx
        x_lab_y = y0 + x_scale * Lx * ny + x_dy
        add_vec_label(
            ax, x_lab_x, x_lab_y, x_text,
            COLORS["local"], fontsize=label_fs, zorder=zorder + 5
        )


def draw_global_cylindrical_frame(
        ax, pt,
        Ler, Lephi,
        lw, ms,
        show_labels=True,
        label_fs=13,
        label_cfg=None,
        zorder=11):
    x0, y0 = pt["x0"], pt["y0"]

    draw_arrow(
        ax, x0, y0, pt["erx"], pt["ery"],
        Ler, COLORS["global"], lw, ms, zorder=zorder
    )
    draw_arrow(
        ax, x0, y0, pt["ephix"], pt["ephiy"],
        Lephi, COLORS["global"], lw, ms, zorder=zorder
    )

    if show_labels:
        if label_cfg is None:
            label_cfg = {}

        er_scale = label_cfg.get("er_scale", 1.15)
        er_dx = label_cfg.get("er_dx", 0.0)
        er_dy = label_cfg.get("er_dy", 0.0)

        ephi_scale = label_cfg.get("ephi_scale", 1.15)
        ephi_dx = label_cfg.get("ephi_dx", 0.0)
        ephi_dy = label_cfg.get("ephi_dy", 0.0)

        er_lab_x = x0 + er_scale * Ler * pt["erx"] + er_dx
        er_lab_y = y0 + er_scale * Ler * pt["ery"] + er_dy

        ephi_lab_x = x0 + ephi_scale * Lephi * pt["ephix"] + ephi_dx
        ephi_lab_y = y0 + ephi_scale * Lephi * pt["ephiy"] + ephi_dy

        add_vec_label(
            ax, er_lab_x, er_lab_y,
            r"$\hat{\mathbf{e}}_r$",
            COLORS["global"], fontsize=label_fs, zorder=zorder + 5
        )

        add_vec_label(
            ax, ephi_lab_x, ephi_lab_y,
            r"$\hat{\mathbf{e}}_{\theta}$",
            COLORS["global"], fontsize=label_fs, zorder=zorder + 5
        )


def draw_reference_point(ax, x0, y0, size, zorder=15):
    ax.scatter(
        [x0], [y0],
        s=size,
        c=COLORS["ref_point"],
        edgecolors="k",
        linewidths=0.85,
        alpha=0.5,
        zorder=zorder
    )


# =========================================================
# 6. Main plotting function: two energies, merged Fig. 2+3
# =========================================================
def plot_two_energy_fig23(
        seo_file=SEO_FILE,
        save_name=OUTPUT_DIR / "fig23_combined_proj.png",
        show=True):

    # -----------------------------------------------------
    # Load SEO orbit table
    # -----------------------------------------------------
    seo_data = np.loadtxt(seo_file, skiprows=1)

    phi_rad = seo_data[0, 1:]
    r_matrix = seo_data[1:, 1:]
    ek_list = seo_data[1:, 0]

    iE_high = find_energy_row(ek_list, GEOM["energy_high"])
    iE_low = find_energy_row(ek_list, GEOM["energy_low"])

    ek_high = ek_list[iE_high]
    ek_low = ek_list[iE_low]

    xo_h, yo_h = orbit_xy_from_row(r_matrix[iE_high], phi_rad)
    xo_l, yo_l = orbit_xy_from_row(r_matrix[iE_low], phi_rad)

    # -----------------------------------------------------
    # Generate bunches on both orbits
    # -----------------------------------------------------
    xb_h, yb_h = generate_gaussian_arc_bunch_from_orbit(
        xo_h, yo_h, phi_rad,
        phi_center_deg=GEOM["phi_center_deg"],
        phi_sigma_deg=GEOM["phi_sigma_deg"],
        n_particles=GEOM["n_particles_high"],
        sigma_x=GEOM["sigma_x"],
        seed=2026
    )

    xb_l, yb_l = generate_gaussian_arc_bunch_from_orbit(
        xo_l, yo_l, phi_rad,
        phi_center_deg=GEOM["phi_center_deg"],
        phi_sigma_deg=GEOM["phi_sigma_deg"],
        n_particles=GEOM["n_particles_low"],
        sigma_x=GEOM["sigma_x"],
        seed=3407
    )

    # -----------------------------------------------------
    # Reference points on the HIGH-energy orbit
    # -----------------------------------------------------
    ref_pts = build_reference_points(
        xo_h, yo_h, phi_rad,
        GEOM["example_phi_deg_list"]
    )

    global_idx = int(np.clip(GEOM["global_frame_index"], 0, len(ref_pts) - 1))
    pt_sel = ref_pts[global_idx]

    # -----------------------------------------------------
    # Visible range from both bunches
    # -----------------------------------------------------
    x_all = np.concatenate([xb_h, xb_l])
    y_all = np.concatenate([yb_h, yb_l])
    x1, x2 = np.percentile(x_all, [0.15, 99.85])
    y1, y2 = np.percentile(y_all, [0.15, 99.85])

    xlim = (x1 - 0.75, x2 + 0.75)
    ylim = (y1 - 0.75, y2 + 0.95)

    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------
    fig = plt.figure(figsize=GEOM["figsize"])
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.80])

    # -----------------------------------------------------
    # Bunches
    # -----------------------------------------------------
    ax.scatter(
        xb_l, yb_l,
        s=GEOM["bunch_size"],
        c=COLORS["bunch_low"],
        alpha=GEOM["bunch_alpha_low"],
        linewidths=0,
        rasterized=True,
        zorder=1
    )

    ax.scatter(
        xb_h, yb_h,
        s=GEOM["bunch_size"],
        c=COLORS["bunch_high"],
        alpha=GEOM["bunch_alpha_high"],
        linewidths=0,
        rasterized=True,
        zorder=2
    )

    # -----------------------------------------------------
    # Closed orbits
    # -----------------------------------------------------
    orbit_mask_h = (
        (xo_h > xlim[0] - 0.3) & (xo_h < xlim[1] + 0.3) &
        (yo_h > ylim[0] - 0.3) & (yo_h < ylim[1] + 0.3)
    )
    orbit_mask_l = (
        (xo_l > xlim[0] - 0.3) & (xo_l < xlim[1] + 0.3) &
        (yo_l > ylim[0] - 0.3) & (yo_l < ylim[1] + 0.3)
    )

    ax.plot(
        xo_l[orbit_mask_l], yo_l[orbit_mask_l],
        color=COLORS["orbit"], lw=2.0, zorder=4, alpha=0.45
    )
    ax.plot(
        xo_h[orbit_mask_h], yo_h[orbit_mask_h],
        color=COLORS["orbit"], lw=2.0, zorder=4, alpha=0.6
    )

    # -----------------------------------------------------
    # Energy labels near the right window edge
    # -----------------------------------------------------
    ax.text(
        GEOM["energy_label_x_high"], GEOM["energy_label_y_high"],
        rf"{ek_high:.0f} MeV",
        fontsize=GEOM["energy_label_fontsize"],
        ha=GEOM["energy_label_ha"], va="center", zorder=20,
    )
    ax.text(
        GEOM["energy_label_x_low"], GEOM["energy_label_y_low"],
        rf"{ek_low:.0f} MeV",
        fontsize=GEOM["energy_label_fontsize"],
        ha=GEOM["energy_label_ha"], va="center", zorder=20,
    )

    # -----------------------------------------------------
    # Dashed radial line (selected point only)
    # -----------------------------------------------------
    ax.plot(
        [0.0, pt_sel["x0"]],
        [0.0, pt_sel["y0"]],
        ls="--",
        lw=GEOM["radial_dash_lw"],
        color=COLORS["global"],
        alpha=GEOM["radial_dash_alpha"],
        zorder=3
    )

    # -----------------------------------------------------
    # Four representative points with local frames
    # -----------------------------------------------------
    for i, pt in enumerate(ref_pts):
        point_id = i + 1

        draw_reference_point(
            ax, pt["x0"], pt["y0"],
            size=GEOM["ref_point_size_main"],
            zorder=15
        )

        local_rotation = (
            GEOM["selected_local_rotation_deg"] if i == global_idx else 0.0
        )

        draw_local_frame(
            ax, pt,
            Ls=GEOM["main_Ls"],
            Lx=GEOM["main_Lx"],
            lw=GEOM["arrow_lw_local_main"],
            ms=GEOM["arrow_ms_main"],
            point_number=point_id,
            show_labels=True,
            label_fs=GEOM["frame_label_fontsize"],
            visual_rotation_deg=local_rotation,
            label_cfg=LOCAL_LABEL_CFG.get(point_id, None),
            zorder=12
        )

        if i == global_idx:
            draw_global_cylindrical_frame(
                ax, pt,
                Ler=GEOM["main_Ler"],
                Lephi=GEOM["main_Lephi"],
                lw=GEOM["arrow_lw_global_main"],
                ms=GEOM["arrow_ms_main"],
                show_labels=True,
                label_fs=GEOM["frame_label_fontsize"],
                label_cfg=GLOBAL_LABEL_CFG,
                zorder=11
            )

    # -----------------------------------------------------
    # Explanatory texts (lower-left free area)
    # -----------------------------------------------------
    ax.text(
        -6.2, ylim[0] + 0.68,
        r"Cylindrical frame $(\hat{\mathbf{e}}_r,\hat{\mathbf{e}}_{\theta})$",
        color=COLORS["global"],
        fontsize=GEOM["title_label_fontsize"],
        ha="left", va="bottom", zorder=20
    )
    ax.text(
        -6.2, ylim[0] + 0.18,
        r"Local frame $(\hat{\mathbf{x}}_i,\hat{\mathbf{s}}_i)$",
        color=COLORS["local"],
        fontsize=GEOM["title_label_fontsize"],
        ha="left", va="bottom", zorder=20
    )

    # -----------------------------------------------------
    # Axis formatting
    # -----------------------------------------------------
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"Global coordinate $X$ [m]")
    ax.set_ylabel(r"Global coordinate $Y$ [m]")
    ax.grid(True, lw=0.45, color=COLORS["grid"], alpha=0.6)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    save_path = Path(save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return save_path


# =========================================================
# 7. Run
# =========================================================
if __name__ == "__main__":
    plot_two_energy_fig23(
        seo_file=SEO_FILE,
        save_name=OUTPUT_DIR / "fig23_combined_proj.png",
        show=True,
    )

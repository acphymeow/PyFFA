import os
from itertools import groupby
from operator import itemgetter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from FFAG_Field import FFAG_Bfield_3D_opera


BMAP_PATH = "./Bmaps/OperaFieldMap"
SEO_FOLDER = "resultsSEO"
TARGET_EK_MEV = 300.0
TARGET_GAP_INDEX = 0
GAP_THRESHOLD_T = 0.08
MIN_GAP_WIDTH_POINTS = 2
MIN_STRAIGHT_GAP_WIDTH_DEG = 2.0
OUTPUT_DIR = "./opera3d_inj_gap_plots"


def load_seo_table(path):
    data = np.loadtxt(path, skiprows=1)
    phi_axis = data[0, 1:]
    ek_axis = data[1:, 0]
    values = data[1:, 1:]
    return ek_axis, phi_axis, values


def load_seo_inputs(seo_path):
    ek_r, phi_full, seo_r = load_seo_table(os.path.join(seo_path, "SEO_r.txt"))
    ek_pr, phi_pr, seo_pr = load_seo_table(os.path.join(seo_path, "SEO_pr.txt"))
    ek_bz, phi_two_cell, seo_bz_two_cell = load_seo_table(os.path.join(seo_path, "Bz.txt"))

    if not np.allclose(ek_r, ek_pr) or not np.allclose(phi_full, phi_pr):
        raise ValueError("SEO_r.txt and SEO_pr.txt axes are inconsistent.")

    return {
        "ek_axis": ek_r,
        "phi_full": phi_full,
        "seo_r": seo_r,
        "seo_pr": seo_pr,
        "bz_ek_axis": ek_bz,
        "phi_two_cell": phi_two_cell,
        "seo_bz_two_cell": seo_bz_two_cell,
    }


def select_nearest_energy(ek_axis, target_ek):
    idx = int(np.argmin(np.abs(ek_axis - target_ek)))
    return idx, float(ek_axis[idx])


def find_periodic_gaps(theta_rad, bz_array, threshold=0.08, min_gap_width=2):
    theta_rad = np.asarray(theta_rad, dtype=np.float64)
    bz_array = np.asarray(bz_array, dtype=np.float64)
    if theta_rad.ndim != 1 or bz_array.ndim != 1:
        raise ValueError("theta_rad and bz_array must be 1-D arrays.")
    if theta_rad.size != bz_array.size:
        raise ValueError("theta_rad and bz_array lengths differ.")

    gap_mask = np.abs(bz_array) < threshold
    gap_indices = np.where(gap_mask)[0]
    if gap_indices.size == 0:
        return []

    groups = []
    for _, group in groupby(enumerate(gap_indices), lambda ix: ix[0] - ix[1]):
        groups.append(list(map(itemgetter(1), group)))

    if len(groups) >= 2 and gap_mask[0] and gap_mask[-1]:
        groups = [groups[-1] + groups[0]] + groups[1:-1]

    groups = [g for g in groups if len(g) >= min_gap_width]
    if len(groups) == 0:
        return []

    dtheta = float(np.median(np.diff(theta_rad)))
    period_rad = float(theta_rad[-1] - theta_rad[0] + dtheta)
    period_deg = float(np.rad2deg(period_rad))
    theta_deg = np.rad2deg(theta_rad)

    gaps = []
    for group in groups:
        wraps = bool(np.any(np.diff(group) < 0))
        if wraps:
            deg_unwrapped = []
            previous = group[0]
            wrapped_part = False
            for idx in group:
                value = float(theta_deg[idx])
                if idx < previous:
                    wrapped_part = True
                if wrapped_part:
                    value += period_deg
                deg_unwrapped.append(value)
                previous = idx
            start_deg = float(theta_deg[group[0]])
            end_deg = float(theta_deg[group[-1]])
            width_deg = float((period_deg - start_deg) + end_deg)
            center_deg = float(deg_unwrapped[len(deg_unwrapped) // 2] % period_deg)
        else:
            start_deg = float(theta_deg[group[0]])
            end_deg = float(theta_deg[group[-1]])
            width_deg = float(end_deg - start_deg)
            center_deg = float(theta_deg[group[len(group) // 2]])

        gaps.append({
            "id": len(gaps) + 1,
            "indices": group,
            "start_deg": start_deg,
            "end_deg": end_deg,
            "center_deg": center_deg,
            "center_rad": float(np.deg2rad(center_deg)),
            "width_deg": width_deg,
            "npts": len(group),
            "wraps": wraps,
        })

    gaps.sort(key=lambda item: item["center_deg"])
    for idx, gap in enumerate(gaps):
        gap["id"] = idx + 1
    return gaps


def gap_points_from_full_ring(gaps, phi_full, r_row, pr_row, ek_mev, bmap):
    points = []
    for gap in gaps:
        phi_abs = float(gap["center_rad"] % (2.0 * np.pi))
        r_center = float(np.interp(phi_abs, phi_full, r_row))
        pr_center = float(np.interp(phi_abs, phi_full, pr_row))
        bz_center = float(bmap.GetBz(np.array([r_center]), np.array([phi_abs]), SearchMod=False, z=0.0)[0])
        points.append({
            "Ek_MeV": ek_mev,
            "gap_id_in_template": gap["id"],
            "repeat_id": 0,
            "phi_rad": phi_abs,
            "phi_deg": float(np.rad2deg(phi_abs)),
            "r_m": r_center,
            "pr": pr_center,
            "x_m": float(r_center * np.cos(phi_abs)),
            "y_m": float(r_center * np.sin(phi_abs)),
            "bz_T": bz_center,
            "gap_width_deg": gap["width_deg"],
        })
    points.sort(key=lambda item: item["phi_rad"])
    return points


def build_bz_background(bmap, r_stride=2, phi_points=1440):
    r_axis = bmap.r_axis[::r_stride]
    phi_axis = np.linspace(0.0, 2.0 * np.pi, phi_points, endpoint=False)
    rr, pp = np.meshgrid(r_axis, phi_axis, indexing="ij")
    bz = bmap.GetBz(rr, pp, SearchMod=False, z=0.0)

    r_step = float(np.mean(np.diff(r_axis)))
    r_edges = np.concatenate((
        [r_axis[0] - 0.5 * r_step],
        0.5 * (r_axis[:-1] + r_axis[1:]),
        [r_axis[-1] + 0.5 * r_step],
    ))
    r_edges = np.clip(r_edges, bmap.r_min, bmap.r_max)
    phi_edges = np.linspace(0.0, 2.0 * np.pi, phi_points + 1, endpoint=True)
    rr_edges, pp_edges = np.meshgrid(r_edges, phi_edges, indexing="ij")
    return rr_edges * np.cos(pp_edges), rr_edges * np.sin(pp_edges), bz


def make_bz_norm(data):
    vabs = float(np.nanmax(np.abs(data)))
    return TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)


def plot_one_turn_gaps(phi_bz, bz_line, gaps, selected_gap, ek_mev, output_path):
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    phi_deg = np.rad2deg(phi_bz)
    ax.plot(phi_deg, bz_line, lw=1.8, color="black", label="Bz on SEO")
    for gap in gaps:
        color = "#f2c94c" if gap is not selected_gap else "#eb5757"
        alpha = 0.28 if gap is not selected_gap else 0.42
        if gap["wraps"]:
            ax.axvspan(gap["start_deg"], phi_deg[-1], color=color, alpha=alpha)
            ax.axvspan(phi_deg[0], gap["end_deg"], color=color, alpha=alpha)
        else:
            ax.axvspan(gap["start_deg"], gap["end_deg"], color=color, alpha=alpha)
        ax.axvline(gap["center_deg"], color=color, lw=1.1, ls="--")
        ax.text(gap["center_deg"], 0.92, f"G{gap['id']}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)

    ax.axhline(GAP_THRESHOLD_T, color="#2f80ed", lw=0.9, ls=":")
    ax.axhline(-GAP_THRESHOLD_T, color="#2f80ed", lw=0.9, ls=":")
    ax.set_xlabel("phi in one-turn SEO Bz [deg]")
    ax.set_ylabel("Bz [T]")
    ax.set_title(f"{ek_mev:.0f} MeV OPERA 3D SEO Bz gap detection")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_full_ring(bmap, phi_full, r_row, points, selected_point, ek_mev, output_path):
    x_bg, y_bg, bz_bg = build_bz_background(bmap)
    norm = make_bz_norm(bz_bg)
    x_orbit = r_row * np.cos(phi_full)
    y_orbit = r_row * np.sin(phi_full)

    fig, ax = plt.subplots(figsize=(9.0, 8.0), constrained_layout=True)
    mesh = ax.pcolormesh(x_bg, y_bg, bz_bg, shading="auto", cmap="RdBu_r", norm=norm)
    ax.plot(x_orbit, y_orbit, color="white", lw=1.1, alpha=0.85, label=f"{ek_mev:.0f} MeV SEO")
    ax.scatter(
        [point["x_m"] for point in points],
        [point["y_m"] for point in points],
        c="black",
        s=20,
        alpha=0.88,
        label="candidate gap centers",
    )
    ax.scatter(
        [selected_point["x_m"]],
        [selected_point["y_m"]],
        c="#f2c94c",
        s=74,
        edgecolors="black",
        linewidths=0.8,
        label="selected point",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{ek_mev:.0f} MeV OPERA 3D gap candidates on mid-plane Bz")
    ax.text(
        0.015,
        0.015,
        f"selected: phi={selected_point['phi_deg']:.3f} deg, r={selected_point['r_m']:.8f} m\n"
        f"pr={selected_point['pr']:.8f}, Bz={selected_point['bz_T']:.3e} T",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 4},
    )
    ax.legend(loc="upper right")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Bz [T]")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_points_csv(points, output_path):
    header = "Ek_MeV,gap_id_in_template,repeat_id,phi_rad,phi_deg,r_m,pr,x_m,y_m,bz_T,gap_width_deg"
    rows = [
        [
            point["Ek_MeV"],
            point["gap_id_in_template"],
            point["repeat_id"],
            point["phi_rad"],
            point["phi_deg"],
            point["r_m"],
            point["pr"],
            point["x_m"],
            point["y_m"],
            point["bz_T"],
            point["gap_width_deg"],
        ]
        for point in points
    ]
    np.savetxt(output_path, np.asarray(rows, dtype=np.float64), delimiter=",", header=header, comments="", fmt="%.12e")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seo_path = os.path.join(BMAP_PATH, SEO_FOLDER)
    bmap = FFAG_Bfield_3D_opera(BMAP_PATH)
    seo = load_seo_inputs(seo_path)

    seo_idx, ek_selected = select_nearest_energy(seo["ek_axis"], TARGET_EK_MEV)
    bz_idx, bz_ek_selected = select_nearest_energy(seo["bz_ek_axis"], ek_selected)
    if abs(bz_ek_selected - ek_selected) > 1e-6:
        raise ValueError(f"No matching Bz row for Ek={ek_selected:.6f} MeV.")

    phi_full = seo["phi_full"]
    r_row = seo["seo_r"][seo_idx]
    pr_row = seo["seo_pr"][seo_idx]
    phi_two_cell = seo["phi_two_cell"]
    bz_line = seo["seo_bz_two_cell"][bz_idx]

    dphi_bz = float(np.median(np.diff(phi_two_cell)))
    if phi_two_cell[-1] > 2.0 * np.pi:
        one_turn_mask = phi_two_cell < 2.0 * np.pi
        phi_bz = phi_two_cell[one_turn_mask]
        bz_line = bz_line[one_turn_mask]
    else:
        phi_bz = phi_two_cell

    gaps = find_periodic_gaps(
        phi_bz,
        bz_line,
        threshold=GAP_THRESHOLD_T,
        min_gap_width=MIN_GAP_WIDTH_POINTS,
    )
    if len(gaps) == 0:
        raise RuntimeError(f"No gap found with threshold={GAP_THRESHOLD_T} T.")

    straight_gaps = [gap for gap in gaps if gap["width_deg"] >= MIN_STRAIGHT_GAP_WIDTH_DEG]
    if len(straight_gaps) == 0:
        straight_gaps = gaps

    selected_gap = straight_gaps[TARGET_GAP_INDEX % len(straight_gaps)]
    points = gap_points_from_full_ring(straight_gaps, phi_full, r_row, pr_row, ek_selected, bmap)
    selected_points = [point for point in points if point["gap_id_in_template"] == selected_gap["id"]]
    selected_point = selected_points[0] if selected_points else points[0]

    csv_path = os.path.join(OUTPUT_DIR, f"opera3d_{ek_selected:.0f}MeV_gap_points.csv")
    one_turn_png = os.path.join(OUTPUT_DIR, f"opera3d_{ek_selected:.0f}MeV_bz_one_turn_gaps.png")
    full_ring_png = os.path.join(OUTPUT_DIR, f"opera3d_{ek_selected:.0f}MeV_gap_points_on_Bz.png")

    save_points_csv(points, csv_path)
    plot_one_turn_gaps(phi_bz, bz_line, straight_gaps, selected_gap, ek_selected, one_turn_png)
    plot_full_ring(bmap, phi_full, r_row, points, selected_point, ek_selected, full_ring_png)

    print(f"Bmap path          : {os.path.abspath(BMAP_PATH)}")
    print(f"SEO path           : {os.path.abspath(seo_path)}")
    print(f"Selected Ek        : {ek_selected:.6f} MeV")
    print(f"Bz input span      : {phi_two_cell[0]:.6f} to {phi_two_cell[-1]:.6f} rad, dphi={dphi_bz:.6g} rad")
    print(f"Gap search span    : {phi_bz[0]:.6f} to {phi_bz[-1]:.6f} rad")
    print(f"Gap threshold      : |Bz| < {GAP_THRESHOLD_T:.6f} T")
    print(f"Low-field gaps     : {len(gaps)}")
    print(f"Straight candidates: {len(straight_gaps)} (width >= {MIN_STRAIGHT_GAP_WIDTH_DEG:.3f} deg)")
    for gap in straight_gaps:
        print(
            f"  G{gap['id']}: center={gap['center_deg']:.6f} deg, "
            f"width={gap['width_deg']:.6f} deg, npts={gap['npts']}"
        )
    print("Selected point:")
    print(
        f"  phi={selected_point['phi_rad']:.10f} rad ({selected_point['phi_deg']:.6f} deg), "
        f"r={selected_point['r_m']:.10f} m, pr={selected_point['pr']:.10f}, "
        f"Bz={selected_point['bz_T']:.6e} T"
    )
    print(f"Saved CSV          : {os.path.abspath(csv_path)}")
    print(f"Saved Bz plot      : {os.path.abspath(one_turn_png)}")
    print(f"Saved full plot    : {os.path.abspath(full_ring_png)}")


if __name__ == "__main__":
    main()

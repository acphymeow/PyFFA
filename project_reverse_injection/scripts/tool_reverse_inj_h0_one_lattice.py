"""Back-track one neutral H0 atom from the 300 MeV foil point.

H0 has zero electric charge, so the magnetic lattice does not deflect it.  The
trajectory is therefore propagated ballistically in Cartesian coordinates;
the analytical B map is sampled only for diagnostics and visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tool_reverse_inj_compare_k_analytical as base


BMAP_DIR = Path("./Bmaps/Bmap_FD16_k7p63_B1p65_R465_F0_s49p97")
OUTPUT_DIR = Path("./project_reverse_injection/h0_reverse_one_lattice_output")

TARGET_EK_MEV = 300.0
MAX_ORDER = 2
TIME_STEP_NS = 0.02
TRACK_TIME_NS = 120.0
APERTURE_R_MAX_M = 12.0
APERTURE_Z_ABS_M = 0.10
STRAIGHT_SECTION_OFFSET_CCW = 2
MIRROR_FINAL_PLOT_ABOUT_X0 = True
COMPACT_VIEW_MARGIN_M = 1.25

# CODATA-compatible rest energies.  Binding energy is subtracted for H0.
PROTON_REST_MEV = 938.27208816
ELECTRON_REST_MEV = 0.51099895069
HYDROGEN_BINDING_MEV = 13.598434599702e-6
H0_REST_MEV = PROTON_REST_MEV + ELECTRON_REST_MEV - HYDROGEN_BINDING_MEV
LIGHT_SPEED_M_S = 299_792_458.0


def relativistic_speed(ek_mev: float, rest_mev: float) -> tuple[float, float, float]:
    gamma = 1.0 + float(ek_mev) / float(rest_mev)
    beta = float(np.sqrt(1.0 - gamma**-2))
    return beta * LIGHT_SPEED_M_S, beta, gamma


def reverse_direction_cartesian(point: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return reversed unit vectors in local (r,z,phi) and Cartesian frames."""
    local_forward = np.array([np.tan(point["pr"]), np.tan(point["pz"]), 1.0], dtype=float)
    local_reverse = -local_forward / np.linalg.norm(local_forward)
    ur, uz, uphi = local_reverse
    phi = float(point["phi_rad"])
    cart_reverse = np.array(
        [ur * np.cos(phi) - uphi * np.sin(phi), ur * np.sin(phi) + uphi * np.cos(phi), uz],
        dtype=float,
    )
    return local_reverse, cart_reverse / np.linalg.norm(cart_reverse)


def propagate_h0(point: dict, bmap) -> dict:
    speed, beta, gamma = relativistic_speed(TARGET_EK_MEV, H0_REST_MEV)
    local_direction, cart_direction = reverse_direction_cartesian(point)
    dt_s = TIME_STEP_NS * 1.0e-9
    time_s = np.arange(0.0, TRACK_TIME_NS * 1.0e-9 + 0.5 * dt_s, dt_s)

    xyz0 = np.array([point["x_m"], point["y_m"], point["z_m"]], dtype=float)
    xyz = xyz0[None, :] + time_s[:, None] * speed * cart_direction[None, :]
    x, y, z = xyz.T
    r = np.hypot(x, y)
    phi = np.unwrap(np.arctan2(y, x))

    aperture_r_max = min(float(bmap.r_max), APERTURE_R_MAX_M)
    valid = (
        (r >= float(bmap.r_min))
        & (r <= aperture_r_max)
        & (np.abs(z) <= APERTURE_Z_ABS_M)
    )
    if not valid[0]:
        raise RuntimeError("The foil point is outside the analytical B-map aperture.")
    invalid_indices = np.flatnonzero(~valid)
    end = int(invalid_indices[0]) if invalid_indices.size else len(time_s)
    if end < 2:
        raise RuntimeError("H0 leaves the valid B-map range before one full time step.")

    time_s = time_s[:end]
    x, y, z, r, phi = x[:end], y[:end], z[:end], r[:end], phi[:end]
    bz, br, bphi, bx, by, b_abs = base.get_bfields_for_points(
        bmap, r, phi, z, MAX_ORDER
    )
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    s = np.r_[0.0, np.cumsum(ds)]
    return {
        "time_s": time_s,
        "s": s,
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "phi": phi,
        "bz": bz,
        "br": br,
        "bphi": bphi,
        "bx": bx,
        "by": by,
        "b_abs": b_abs,
        "speed_m_s": speed,
        "beta": beta,
        "gamma": gamma,
        "local_direction": local_direction,
        "cart_direction": cart_direction,
        "stop_reason": "left analytical B-map/aperture range" if invalid_indices.size else "maximum tracking time",
    }


def save_track_csv(traj: dict, output_path: Path) -> None:
    data = np.column_stack(
        (
            traj["time_s"],
            traj["s"],
            traj["x"],
            traj["y"],
            traj["z"],
            traj["r"],
            traj["phi"],
            np.rad2deg(np.mod(traj["phi"], 2.0 * np.pi)),
            traj["br"],
            traj["bphi"],
            traj["bz"],
            traj["bx"],
            traj["by"],
            traj["b_abs"],
        )
    )
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header=(
            "time_s,s_m,x_m,y_m,z_m,r_m,phi_rad,phi_mod_deg,"
            "Br_T,Bphi_T,Bz_T,Bx_T,By_T,Babs_T"
        ),
        comments="",
        fmt="%.12e",
    )


def plot_track_on_bz(
    bmap,
    foil: dict,
    orbit: dict,
    traj: dict,
    output_path: Path,
    compact_output_path: Path | None = None,
) -> None:
    x_bg, y_bg, bz_bg = base.build_bz_background(bmap, MAX_ORDER)
    vabs = float(np.nanmax(np.abs(bz_bg)))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    plot_x_sign = -1.0 if MIRROR_FINAL_PLOT_ABOUT_X0 else 1.0
    mirror_note = " (display mirrored about x=0)" if MIRROR_FINAL_PLOT_ABOUT_X0 else ""

    fig, ax = plt.subplots(figsize=(9.4, 8.2), constrained_layout=True)
    mesh = ax.pcolormesh(plot_x_sign * x_bg, y_bg, bz_bg, shading="auto", cmap="RdBu_r", norm=norm)
    ax.plot(plot_x_sign * orbit["x"], orbit["y"], color="white", lw=1.0, alpha=0.84, label="300 MeV SEO")
    ax.plot(plot_x_sign * traj["x"], traj["y"], color="#111111", lw=2.2, label="H0 reverse track (q = 0)")
    ax.scatter(
        [plot_x_sign * foil["x_m"]], [foil["y_m"]], c="#ffd43b", s=78,
        edgecolors="black", linewidths=0.8, zorder=5, label="foil point",
    )
    ax.scatter(
        [plot_x_sign * traj["x"][-1]], [traj["y"][-1]], c="#2d9cdb", s=48,
        marker="s", edgecolors="black", linewidths=0.6, zorder=5, label="valid end",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"{BMAP_DIR.name}\n300 MeV H0 reverse trajectory on analytical Bz "
        f"(max_order=2){mirror_note}"
    )
    ax.text(
        0.015,
        0.015,
        "H0 is neutral: ballistic straight line\n"
        f"physical foil: phi={foil['phi_deg']:.3f} deg, R={foil['r_m']:.6f} m\n"
        f"valid path={traj['s'][-1]:.3f} m, time={traj['time_s'][-1] * 1e9:.2f} ns\n"
        f"sampled max |Bz|={np.max(np.abs(traj['bz'])):.4f} T",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4},
    )
    ax.legend(loc="upper right")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Bz [T]")
    fig.savefig(output_path, dpi=210)
    plt.close(fig)

    if compact_output_path is not None:
        fig, ax = plt.subplots(figsize=(4.4, 3.4), constrained_layout=True)
        ax.pcolormesh(plot_x_sign * x_bg, y_bg, bz_bg, shading="auto", cmap="RdBu_r", norm=norm)
        x_track = plot_x_sign * traj["x"]
        y_track = traj["y"]
        ax.plot(x_track, y_track, color="#111111", lw=2.2)
        ax.scatter(
            [plot_x_sign * foil["x_m"]], [foil["y_m"]], c="#ffd43b", s=46,
            edgecolors="black", linewidths=0.7, zorder=5,
        )
        ax.scatter(
            [x_track[-1]], [y_track[-1]], c="#2d9cdb", s=32, marker="s",
            edgecolors="black", linewidths=0.55, zorder=5,
        )
        margin = COMPACT_VIEW_MARGIN_M
        ax.set_xlim(float(np.min(x_track) - margin), float(np.max(x_track) + margin))
        ax.set_ylim(float(np.min(y_track) - margin), float(np.max(y_track) + margin))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(False)
        fig.savefig(compact_output_path, dpi=190)
        plt.close(fig)


def plot_bz_along_s(traj: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.plot(traj["s"], traj["bz"], color="#202020", lw=1.7)
    ax.set_xlabel("s along H0 reverse trajectory [m]")
    ax.set_ylabel("sampled Bz [T]")
    ax.set_title("Magnetic field sampled along the neutral H0 path")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bmap = base.FFAG_Bfield_analytical(str(BMAP_DIR / "config_Bmap.json"), MAX_ORDER)
    seo = base.load_seo_inputs(str(BMAP_DIR / "resultsSEO"))
    base.TARGET_GAP_INDEX = STRAIGHT_SECTION_OFFSET_CCW
    foil_pack = base.select_foil_point(bmap, seo, MAX_ORDER)
    foil = foil_pack["selected_point"]
    traj = propagate_h0(foil, bmap)

    seo_idx = foil_pack["seo_idx"]
    phi_full = seo["phi_full"]
    r_full = seo["seo_r"][seo_idx]
    orbit = {
        "x": r_full * np.cos(phi_full),
        "y": r_full * np.sin(phi_full),
    }

    mirror_tag = "_mirror_x0" if MIRROR_FINAL_PLOT_ABOUT_X0 else ""
    stem = (
        f"{BMAP_DIR.name}_300MeV_H0_reverse_"
        f"ccw{STRAIGHT_SECTION_OFFSET_CCW}_straight_sections{mirror_tag}"
    )
    track_csv = OUTPUT_DIR / f"{stem}_track.csv"
    r_fi_bz_csv = OUTPUT_DIR / f"{stem}_r_fi_bz.csv"
    overlay_png = OUTPUT_DIR / f"{stem}_on_Bz.png"
    compact_png = OUTPUT_DIR / f"{stem}_on_Bz_compact.png"
    bz_s_png = OUTPUT_DIR / f"{stem}_Bz_along_s.png"
    summary_json = OUTPUT_DIR / f"{stem}_summary.json"

    save_track_csv(traj, track_csv)
    np.savetxt(
        r_fi_bz_csv,
        np.column_stack((traj["r"], traj["phi"], traj["bz"])),
        delimiter=",",
        header="r,fi,bz",
        comments="",
        fmt="%.12e",
    )
    plot_track_on_bz(bmap, foil, orbit, traj, overlay_png, compact_png)
    plot_bz_along_s(traj, bz_s_png)

    summary = {
        "particle": "H0",
        "charge_e": 0.0,
        "kinetic_energy_MeV": TARGET_EK_MEV,
        "rest_energy_MeV": H0_REST_MEV,
        "beta": traj["beta"],
        "gamma": traj["gamma"],
        "speed_m_s": traj["speed_m_s"],
        "time_step_ns": TIME_STEP_NS,
        "max_order": MAX_ORDER,
        "straight_section_offset_ccw": STRAIGHT_SECTION_OFFSET_CCW,
        "final_plot_mirrored_about_x0": MIRROR_FINAL_PLOT_ABOUT_X0,
        "lattice": BMAP_DIR.name,
        "foil_phi_deg": foil["phi_deg"],
        "foil_r_m": foil["r_m"],
        "foil_pr_rad": foil["pr"],
        "foil_bz_T": foil["bz_T"],
        "display_foil_x_m": float(-foil["x_m"] if MIRROR_FINAL_PLOT_ABOUT_X0 else foil["x_m"]),
        "display_foil_y_m": float(foil["y_m"]),
        "reverse_local_unit_r_z_phi": traj["local_direction"].tolist(),
        "reverse_cartesian_unit_x_y_z": traj["cart_direction"].tolist(),
        "valid_samples": int(traj["r"].size),
        "valid_time_ns": float(traj["time_s"][-1] * 1.0e9),
        "path_length_m": float(traj["s"][-1]),
        "r_min_m": float(np.min(traj["r"])),
        "r_max_m": float(np.max(traj["r"])),
        "end_r_m": float(traj["r"][-1]),
        "end_phi_deg": float(np.rad2deg(np.mod(traj["phi"][-1], 2.0 * np.pi))),
        "sampled_bz_min_T": float(np.min(traj["bz"])),
        "sampled_bz_max_T": float(np.max(traj["bz"])),
        "sampled_max_abs_bz_T": float(np.max(np.abs(traj["bz"]))),
        "stop_reason": traj["stop_reason"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"track_csv={track_csv.resolve()}")
    print(f"r_fi_bz_csv={r_fi_bz_csv.resolve()}")
    print(f"overlay_png={overlay_png.resolve()}")
    print(f"compact_png={compact_png.resolve()}")
    print(f"bz_s_png={bz_s_png.resolve()}")
    print(f"summary_json={summary_json.resolve()}")


if __name__ == "__main__":
    main()

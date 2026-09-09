import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import tool_reverse_inj_compare_k_analytical as base


TARGET_EK_MEV = 300.0
OUTPUT_DIR = "./analytical_reverse_injection_offset_scan_output"

FIELD_SPEC = {
    "key": "k7p63_B1p55_R365",
    "label": "Bmap_FD16_k7p63_B1p55_R365",
    "path": "./Bmaps/Bmap_FD16_k7p63_B1p55_R365",
    "color": "#111111",
    "max_order": 2,
}

RADIAL_OFFSETS_M = [0.0, -0.015, -0.030, -0.045, -0.060]
OFFSET_COLORS = ["#111111", "#2f80ed", "#27ae60", "#f2994a", "#eb5757", "#9b51e0", "#00a6a6"]


def offset_key(offset_m):
    offset_mm = int(round(offset_m * 1000.0))
    if offset_mm == 0:
        return "off0mm"
    prefix = "p" if offset_mm > 0 else "m"
    return f"off{prefix}{abs(offset_mm)}mm"


def offset_label(offset_m):
    return f"dR = {offset_m * 1000.0:+.0f} mm"


def make_offset_point(closed_point, bmap, max_order, offset_m):
    point = dict(closed_point)
    point["closed_r_m"] = closed_point["r_m"]
    point["closed_pr"] = closed_point["pr"]
    point["closed_bz_T"] = closed_point["bz_T"]
    point["radial_offset_m"] = float(offset_m)
    point["r_m"] = closed_point["r_m"] + float(offset_m)
    point["x_m"] = float(point["r_m"] * np.cos(point["phi_rad"]))
    point["y_m"] = float(point["r_m"] * np.sin(point["phi_rad"]))
    point["bz_T"] = float(base.get_bfields_for_points(
        bmap,
        np.array([point["r_m"]]),
        np.array([point["phi_rad"]]),
        np.array([0.0]),
        max_order,
    )[0][0])
    return point


def process_one_offset(base_spec, bmap, seo, closed_foil_pack, offset_m, color):
    closed_point = closed_foil_pack["selected_point"]
    point = make_offset_point(closed_point, bmap, base_spec["max_order"], offset_m)
    spec = dict(base_spec)
    spec["key"] = f"{base_spec['key']}_{offset_key(offset_m)}"
    spec["label"] = offset_label(offset_m)
    spec["color"] = color

    _, particle_v, particle_p_reversed = base.make_reverse_particle(point)
    dump_dir, particle_path = base.run_reverse_tracking(spec, bmap, seo, particle_v)
    traj = base.load_and_augment_trajectory(particle_path, particle_p_reversed, bmap, spec["max_order"])
    traj_valid = base.first_valid_segment(traj, bmap)

    seo_idx = closed_foil_pack["seo_idx"]
    phi_full = seo["phi_full"]
    r_row = seo["seo_r"][seo_idx]
    orbit = {
        "phi": phi_full,
        "r": r_row,
        "x": r_row * np.cos(phi_full),
        "y": r_row * np.sin(phi_full),
    }

    return {
        "key": spec["key"],
        "label": spec["label"],
        "field_label": base_spec["label"],
        "path": base_spec["path"],
        "color": color,
        "max_order": spec["max_order"],
        "bmap": bmap,
        "seo": seo,
        "dump_dir": dump_dir,
        "particle_path": particle_path,
        "foil": point,
        "closed_foil": closed_point,
        "selected_gap": closed_foil_pack["selected_gap"],
        "phi_bz": closed_foil_pack["phi_bz"],
        "bz_line": closed_foil_pack["bz_line"],
        "orbit": orbit,
        "traj": traj,
        "traj_valid": traj_valid,
    }


def save_offset_scan_summary_csv(results, output_path):
    rows = []
    for item in results:
        foil = item["foil"]
        closed = item["closed_foil"]
        traj = item["traj_valid"]
        rows.append([
            item["key"],
            item["field_label"],
            foil["Ek_MeV"],
            foil["phi_rad"],
            foil["phi_deg"],
            closed["r_m"],
            closed["pr"],
            closed["bz_T"],
            foil["radial_offset_m"],
            foil["r_m"],
            foil["pr"],
            foil["bz_T"],
            traj["time_s"][-1] * 1e9,
            traj["s"][-1],
            np.nanmin(traj["r"]),
            np.nanmax(traj["r"]),
            traj["r"][-1],
            np.rad2deg(np.mod(traj["phi"][-1], 2.0 * np.pi)),
            np.nanmin(traj["bz"]),
            np.nanmax(traj["bz"]),
        ])
    header = (
        "key,field_label,Ek_MeV,phi_rad,phi_deg,"
        "closed_R_m,closed_pr,closed_Bz_T,radial_offset_m,"
        "start_R_m,start_pr,start_Bz_T,"
        "track_time_ns,path_length_m,R_min_m,R_max_m,end_R_m,end_phi_deg,Bz_min_T,Bz_max_T"
    )
    np.savetxt(output_path, np.asarray(rows, dtype=object), delimiter=",", header=header, comments="", fmt="%s")


def plot_offsets_on_bz(results, output_path):
    first = results[0]
    x_bg, y_bg, bz_bg = base.build_bz_background(first["bmap"], first["max_order"])
    vabs = float(np.nanmax(np.abs(bz_bg)))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    orbit = first["orbit"]
    closed = first["closed_foil"]

    fig, ax = plt.subplots(figsize=(9.0, 8.0), constrained_layout=True)
    mesh = ax.pcolormesh(x_bg, y_bg, bz_bg, shading="auto", cmap="RdBu_r", norm=norm)
    ax.plot(orbit["x"], orbit["y"], color="white", lw=1.0, alpha=0.85, label="300 MeV SEO")
    ax.scatter([closed["x_m"]], [closed["y_m"]], c="white", s=64, edgecolors="black", linewidths=0.8, label="closed-orbit foil")

    for item in results:
        traj = item["traj_valid"]
        foil = item["foil"]
        ax.plot(traj["x"], traj["y"], color=item["color"], lw=1.8, label=item["label"])
        ax.scatter([foil["x_m"]], [foil["y_m"]], color=item["color"], s=38, edgecolors="black", linewidths=0.45)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{first['field_label']} reverse injection offset scan on Bz")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Bz [T]")
    base.save_and_maybe_open(fig, output_path, dpi=200, open_after_run=True)


def plot_offsets_r_and_bz_vs_s(results, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True, constrained_layout=True)
    for item in results:
        traj = item["traj_valid"]
        axes[0].plot(traj["s"], traj["r"], color=item["color"], lw=1.6, label=item["label"])
        axes[1].plot(traj["s"], traj["bz"], color=item["color"], lw=1.6, label=item["label"])

    axes[0].set_ylabel("R [m]")
    axes[0].set_title(f"{results[0]['field_label']} reverse trajectory offset scan")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("s along reverse trajectory [m]")
    axes[1].set_ylabel("Bz [T]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    base.save_and_maybe_open(fig, output_path, dpi=180, open_after_run=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base.OUTPUT_DIR = OUTPUT_DIR
    base.PLOTS_TO_OPEN.clear()

    print(f"Loading selected field: {FIELD_SPEC['label']} ...")
    bmap = base.FFAG_Bfield_analytical(os.path.join(FIELD_SPEC["path"], "config_Bmap.json"), FIELD_SPEC["max_order"])
    seo = base.load_seo_inputs(os.path.join(FIELD_SPEC["path"], "resultsSEO"))
    closed_foil_pack = base.select_foil_point(bmap, seo, FIELD_SPEC["max_order"])
    closed = closed_foil_pack["selected_point"]

    results = []
    for idx, offset_m in enumerate(RADIAL_OFFSETS_M):
        color = OFFSET_COLORS[idx % len(OFFSET_COLORS)]
        print(f"Processing offset {offset_m * 1000.0:+.0f} mm ...")
        item = process_one_offset(FIELD_SPEC, bmap, seo, closed_foil_pack, offset_m, color)
        results.append(item)

        key = offset_key(offset_m)
        traj_csv = os.path.join(OUTPUT_DIR, f"{FIELD_SPEC['key']}_{key}_300MeV_reverse_injection_Rle12_track.csv")
        r_fi_bz_csv = os.path.join(OUTPUT_DIR, f"{FIELD_SPEC['key']}_{key}_300MeV_reverse_injection_Rle12_r_fi_bz.csv")
        base.save_trajectory_csv(item["traj_valid"], traj_csv)
        base.save_r_fi_bz_csv(item["traj_valid"], r_fi_bz_csv)
        item["traj_csv"] = traj_csv
        item["r_fi_bz_csv"] = r_fi_bz_csv

    summary_csv = os.path.join(OUTPUT_DIR, f"{FIELD_SPEC['key']}_offset_scan_summary.csv")
    overlay_png = os.path.join(OUTPUT_DIR, f"{FIELD_SPEC['key']}_offset_scan_reverse_tracks_on_Bz.png")
    r_bz_s_png = os.path.join(OUTPUT_DIR, f"{FIELD_SPEC['key']}_offset_scan_R_and_Bz_vs_s.png")
    save_offset_scan_summary_csv(results, summary_csv)
    plot_offsets_on_bz(results, overlay_png)
    plot_offsets_r_and_bz_vs_s(results, r_bz_s_png)

    print(f"Output dir          : {os.path.abspath(OUTPUT_DIR)}")
    print(f"Selected field      : {FIELD_SPEC['label']}")
    print(f"Target Ek           : {TARGET_EK_MEV:.6f} MeV")
    print(f"Closed foil phi/R/pr: {closed['phi_deg']:.6f} deg, {closed['r_m']:.10f} m, {closed['pr']:.10e}")
    print(f"Offsets             : {', '.join(f'{x * 1000.0:+.0f} mm' for x in RADIAL_OFFSETS_M)}")
    for item in results:
        foil = item["foil"]
        traj = item["traj_valid"]
        print(f"{item['label']}:")
        print(f"  start R/pr/Bz     : {foil['r_m']:.10f} m, {foil['pr']:.10e}, {foil['bz_T']:.6e} T")
        print(f"  valid time/path   : {traj['time_s'][-1] * 1e9:.6f} ns, {traj['s'][-1]:.6f} m")
        print(f"  R range/end       : {np.nanmin(traj['r']):.6f} to {np.nanmax(traj['r']):.6f} m, end {traj['r'][-1]:.10f} m")
        print(f"  r-fi-bz CSV       : {os.path.abspath(item['r_fi_bz_csv'])}")
    print(f"Saved summary CSV   : {os.path.abspath(summary_csv)}")
    print(f"Saved overlay plot  : {os.path.abspath(overlay_png)}")
    print(f"Saved R/Bz vs s     : {os.path.abspath(r_bz_s_png)}")
    base.open_output_plots()


if __name__ == "__main__":
    main()

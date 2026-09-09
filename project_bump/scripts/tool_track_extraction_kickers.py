import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from FFAG_Bunch import FFAG_Bunch
from FFAG_Distribution import LoadSEOParams
from FFAG_Field import FFAG_BField_Error, FFAG_BumpKickerField, FFAG_EField_new
from FFAG_ParasAndConversion import FFAG_ConversionTools, FFAG_GlobalParameters
from FFAG_dump import StepDump
from FFAG_track import FFAG_RungeKutta, FunctionForAccelerationBunch_dt
from main_track import load_bmap_auto
from tool_plot_bmap_with_kickers import (
    add_kicker_regions_cartesian,
    evaluate_base_field,
    symmetric_norm,
)


SC_GRID_SIZE = (8, 16, 16, 10.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Track one proton for one pass from a closed orbit with the configured "
            "horizontal extraction kickers, and compare with a kicker-off reference."
        )
    )
    parser.add_argument("-j", "--config", default="config_track.json")
    parser.add_argument("--energy-mev", type=float, default=600.0)
    parser.add_argument(
        "--start-phi-deg", type=float, default=140.0,
        help="Initial global azimuth upstream of EK1 (default: 140 deg).",
    )
    parser.add_argument("--time-step-ns", type=float, default=0.02)
    parser.add_argument(
        "--turns", type=int, default=1,
        help="Number of turns to track; the continuous flat-top kickers act every turn.",
    )
    parser.add_argument(
        "-o", "--output-dir", default="extraction_single_particle_output",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def resolve_config_path(value, config_path):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def load_seo(seo_dir, energy_mev):
    r_data = np.loadtxt(seo_dir / "SEO_r.txt", skiprows=1)
    pr_data = np.loadtxt(seo_dir / "SEO_pr.txt", skiprows=1)
    phi_axis = np.asarray(r_data[0, 1:], dtype=np.float64)
    energy_axis = np.asarray(r_data[1:, 0], dtype=np.float64)
    idx = int(np.argmin(np.abs(energy_axis - energy_mev)))
    selected_energy = float(energy_axis[idx])
    if abs(selected_energy - energy_mev) > 1.0e-6:
        raise ValueError(
            f"SEO table has no {energy_mev:.6f} MeV row; nearest is {selected_energy:.6f} MeV."
        )
    if not np.allclose(pr_data[0, 1:], phi_axis):
        raise ValueError("SEO_r.txt and SEO_pr.txt use different azimuth axes.")
    if not np.allclose(pr_data[1:, 0], energy_axis):
        raise ValueError("SEO_r.txt and SEO_pr.txt use different energy axes.")

    seo_at_energy, _, seo_info = LoadSEOParams(str(seo_dir / "SEO_ini.txt"), selected_energy)
    return {
        "phi_axis": phi_axis,
        "energy_axis": energy_axis,
        "r_matrix": np.asarray(r_data[1:, 1:], dtype=np.float64),
        "pr_matrix": np.asarray(pr_data[1:, 1:], dtype=np.float64),
        "row_index": idx,
        "energy_mev": selected_energy,
        "r_row": np.asarray(r_data[idx + 1, 1:], dtype=np.float64),
        "pr_row": np.asarray(pr_data[idx + 1, 1:], dtype=np.float64),
        "seo_at_energy": seo_at_energy,
        "seo_info": seo_info,
        "revolution_period_s": float(seo_at_energy[3]),
    }


def make_closed_orbit_particle(seo, start_phi_deg):
    phi = math.radians(start_phi_deg)
    phi_mod = phi % (2.0 * math.pi)
    r = float(np.interp(phi_mod, seo["phi_axis"], seo["r_row"]))
    pr_normalized = float(np.interp(phi_mod, seo["phi_axis"], seo["pr_row"]))
    if abs(pr_normalized) >= 1.0:
        raise ValueError(f"Invalid normalized SEO radial momentum: {pr_normalized}.")
    # SEO uses pr = P_r/P, while ConvertPrzek2Vrzek_boris expects the
    # trajectory angle alpha with tan(alpha) = v_r/v_phi.  Thus
    # pr = sin(alpha), so alpha = asin(pr).
    pr_angle = math.asin(pr_normalized)

    particle_p = np.zeros((1, 16), dtype=np.float64)
    particle_p[0, 0] = r
    particle_p[0, 1] = pr_angle
    particle_p[0, 4] = phi
    particle_p[0, 5] = seo["energy_mev"]
    particle_p[0, 6] = 0.0
    particle_p[0, 13:16] = 0.0
    particle_v = FFAG_ConversionTools().ConvertPrzek2Vrzek_boris(particle_p)
    return particle_p, particle_v, pr_normalized


def make_global_parameters(bmap, config, seo, config_path, kicker_enabled):
    global_parameters = FFAG_GlobalParameters()
    global_parameters.AddBMap(bmap)

    emap_path = resolve_config_path(config["Emap"]["maps"], config_path)
    global_parameters.AddEMap(FFAG_EField_new(str(emap_path), False))
    global_parameters.AddSEOInfo(seo["seo_info"])
    global_parameters.AddBHarmonics(FFAG_BField_Error([], False))

    kicker_cfg = config["BumpKicker"]
    global_parameters.AddBumpKickers(
        FFAG_BumpKickerField(kicker_cfg["kickers"], kicker_enabled)
    )

    aperture = config["Aperture"]
    global_parameters.Aperture_m = np.array(
        [aperture["Rmin"], aperture["Rmax"], aperture["Zmin"], aperture["Zmax"]],
        dtype=np.float64,
    )
    global_parameters.Aperture_enable = bool(aperture["enable"])
    return global_parameters


def run_case(case_name, bmap, config, seo, config_path, particle_v, particle_p,
             run_dir, time_step_s, turns, kicker_enabled):
    dump_dir = run_dir / f"dump_{case_name}"
    revolution_period_s = seo["revolution_period_s"]
    n_steps = int(np.ceil(turns * revolution_period_s / time_step_s))
    tracked_time_s = n_steps * time_step_s

    step_dump = StepDump(
        start_time=-1.0e-15,
        end_time=tracked_time_s + 2.0 * time_step_s,
        time_interval=time_step_s,
        tracked_particle_id_global=[0],
        save_folder=str(dump_dir),
    )
    local_parameters = {
        "stop_condition": {
            "max_stepsN": n_steps,
            "max_turn": 10.0,
            "max_time": tracked_time_s + time_step_s,
        },
        "time_step": time_step_s,
        "step_dumps": step_dump,
        "enable_SC": False,
        "SC_type": 2,
        "SC_step": 20,
    }

    bunch = FFAG_Bunch(particle_v.copy(), marcosize=1.0, sc_grid_size=SC_GRID_SIZE)
    bunch.SEO_fi_axis = seo["phi_axis"]
    bunch.SEO_Ek_axis = seo["energy_axis"]
    bunch.SEO_r_matrix = seo["r_matrix"]
    bunch.SEO_pr_matrix = seo["pr_matrix"]
    bunch.SEO_perimeter = seo["seo_info"][:, 8]

    global_parameters = make_global_parameters(
        bmap, config, seo, config_path, kicker_enabled
    )
    FFAG_RungeKutta().rk4_solve_dt_bunch3_withSC_boris(
        FunctionForAccelerationBunch_dt,
        0.0,
        bunch,
        global_parameters,
        local_parameters,
    )
    if not step_dump.save_flag:
        step_dump.dump_all_particles()
        step_dump.save_flag = True

    particle_path = dump_dir / "particle_0_rank_0.npz"
    if not particle_path.exists():
        raise RuntimeError(f"StepDump did not produce {particle_path}.")
    dumped = np.load(particle_path)["particles"]
    initial_row = np.hstack((particle_p[0], [0.0]))
    rows = [initial_row, *dumped]

    final_p = FFAG_ConversionTools().ConvertVrzek2Przek_boris(bunch.LocalBunch.copy())[0]
    if np.all(np.isfinite(final_p[:6])):
        final_time = tracked_time_s
        if final_p[8] == 0.0 and dumped.shape[0] > 0:
            final_time = min(tracked_time_s, float(dumped[-1, -1]) + time_step_s)
        if not np.allclose(np.asarray(rows[-1])[:6], final_p[:6], rtol=0.0, atol=1.0e-13):
            rows.append(np.hstack((final_p, [final_time])))

    trajectory = np.asarray(rows, dtype=np.float64)
    trajectory = trajectory[np.all(np.isfinite(trajectory[:, [0, 2, 4, 5, -1]]), axis=1)]
    return {
        "name": case_name,
        "trajectory": trajectory,
        "bunch": bunch,
        "tracked_time_s": tracked_time_s,
        "n_steps": n_steps,
        "kicker_enabled": kicker_enabled,
    }


def augment_trajectory(case, bmap, kickers, seo):
    raw = case["trajectory"]
    r = raw[:, 0]
    phi = raw[:, 4]
    z = raw[:, 2]
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    flags = np.ones(r.size, dtype=np.bool_)

    bz_base = np.zeros_like(r)
    br_base = np.zeros_like(r)
    bfi_base = np.zeros_like(r)
    bmap.GetBfields(
        r, phi, z, flags, flags, bz_base, br_base, bfi_base, SearchMod=False
    )
    bz_kicker = np.zeros_like(r)
    br_kicker = np.zeros_like(r)
    bfi_kicker = np.zeros_like(r)
    if case["kicker_enabled"]:
        kickers.AddBumpKicker_to_Bmap_njit(
            r, phi, z, flags, flags, bz_kicker, br_kicker, bfi_kicker
        )

    ds = np.hypot(np.diff(x), np.diff(y))
    ds = np.sqrt(ds**2 + np.diff(z)**2)
    s = np.r_[0.0, np.cumsum(ds)]
    phi_mod = np.mod(phi, 2.0 * np.pi)
    r_seo = np.interp(phi_mod, seo["phi_axis"], seo["r_row"])

    case.update({
        "raw": raw,
        "time_s": raw[:, -1],
        "r": r,
        "pr": raw[:, 1],
        "z": z,
        "pz": raw[:, 3],
        "phi": phi,
        "phi_mod_deg": np.rad2deg(phi_mod),
        "energy_mev": raw[:, 5],
        "survive": raw[:, 8],
        "x": x,
        "y": y,
        "s": s,
        "r_seo": r_seo,
        "dr_mm": (r - r_seo) * 1.0e3,
        "bz_base": bz_base,
        "br_base": br_base,
        "bfi_base": bfi_base,
        "bz_kicker": bz_kicker,
        "br_kicker": br_kicker,
        "bfi_kicker": bfi_kicker,
        "bz_total": bz_base + bz_kicker,
        "br_total": br_base + br_kicker,
        "bfi_total": bfi_base + bfi_kicker,
    })
    return case


def save_case_csv(case, output_path):
    data = np.column_stack((
        case["time_s"], case["s"], case["r"], case["pr"],
        case["z"], case["pz"], case["phi"], case["phi_mod_deg"],
        case["energy_mev"], case["x"], case["y"], case["r_seo"],
        case["dr_mm"], case["br_base"], case["bfi_base"], case["bz_base"],
        case["br_kicker"], case["bfi_kicker"], case["bz_kicker"],
        case["br_total"], case["bfi_total"], case["bz_total"], case["survive"],
    ))
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header=(
            "time_s,s_m,r_m,pr,z_m,pz,phi_rad,phi_mod_deg,Ek_MeV,x_m,y_m,"
            "r_SEO_m,dr_from_SEO_mm,Br_base_T,Bphi_base_T,Bz_base_T,"
            "Br_kicker_T,Bphi_kicker_T,Bz_kicker_T,Br_total_T,Bphi_total_T,"
            "Bz_total_T,survive"
        ),
        comments="",
        fmt="%.12e",
    )


def case_summary(case, aperture_rmax, kicker_cfg, start_phi_deg, turns):
    outside = np.where(case["r"] > aperture_rmax)[0]
    extracted = bool(outside.size > 0 or case["bunch"].LostNum_Local > 0)
    summary = {
        "extracted_beyond_outer_aperture": extracted,
        "samples": int(case["r"].size),
        "tracked_time_ns": float(case["tracked_time_s"] * 1.0e9),
        "path_length_m": float(case["s"][-1]),
        "r_min_m": float(np.min(case["r"])),
        "r_max_m": float(np.max(case["r"])),
        "max_abs_dr_from_seo_mm": float(np.max(np.abs(case["dr_mm"]))),
        "final_phi_unwrapped_deg": float(np.rad2deg(case["phi"][-1])),
        "final_phi_mod_deg": float(case["phi_mod_deg"][-1]),
        "final_r_m": float(case["r"][-1]),
    }
    if outside.size:
        idx = int(outside[0])
        summary["outer_aperture_crossing"] = {
            "time_ns": float(case["time_s"][idx] * 1.0e9),
            "path_length_m": float(case["s"][idx]),
            "r_m": float(case["r"][idx]),
            "phi_unwrapped_deg": float(np.rad2deg(case["phi"][idx])),
            "phi_mod_deg": float(case["phi_mod_deg"][idx]),
            "x_m": float(case["x"][idx]),
            "y_m": float(case["y"][idx]),
        }

    if case["kicker_enabled"]:
        after_kickers = []
        for kicker in kicker_cfg:
            phi_end = math.radians(
                float(kicker["phi_start_deg"]) + float(kicker["phi_span_deg"])
            )
            eligible = np.where(case["phi"] >= phi_end)[0]
            if not eligible.size:
                break
            idx = int(eligible[0])
            after_kickers.append({
                "name": kicker["name"],
                "phi_deg": float(np.rad2deg(case["phi"][idx])),
                "r_m": float(case["r"][idx]),
                "dr_from_seo_mm": float(case["dr_mm"][idx]),
                "pr_rad": float(case["pr"][idx]),
            })
        summary["immediately_after_kickers"] = after_kickers

    advance_deg = np.rad2deg(case["phi"] - math.radians(start_phi_deg))
    per_turn = []
    for turn_idx in range(turns):
        low = 360.0 * turn_idx
        high = 360.0 * (turn_idx + 1)
        mask = (advance_deg >= low) & (advance_deg <= high)
        if not np.any(mask):
            break
        r_turn = case["r"][mask]
        dr_turn = case["dr_mm"][mask]
        r_max_turn = float(np.max(r_turn))
        per_turn.append({
            "turn": turn_idx + 1,
            "r_min_m": float(np.min(r_turn)),
            "r_max_m": r_max_turn,
            "minimum_outer_aperture_clearance_mm": float((aperture_rmax - r_max_turn) * 1.0e3),
            "max_abs_dr_from_seo_mm": float(np.max(np.abs(dr_turn))),
        })
    summary["per_turn"] = per_turn
    return summary


def plot_results(bmap, seo, kicker_cfg, on_case, off_case, start_phi_deg, turns, output_path):
    r_axis = np.linspace(bmap.r_min, bmap.r_max, 220)
    phi_axis = np.linspace(0.0, 2.0 * np.pi, 1441)
    r_grid, phi_grid = np.meshgrid(r_axis, phi_axis, indexing="ij")
    bz_grid, _, _ = evaluate_base_field(bmap, 0, r_grid, phi_grid, 0.0)
    x_grid = r_grid * np.cos(phi_grid)
    y_grid = r_grid * np.sin(phi_grid)

    fig, (ax_xy, ax_dr) = plt.subplots(
        1, 2, figsize=(15.2, 7.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.08, 0.92]},
    )
    levels = np.linspace(-np.max(np.abs(bz_grid)), np.max(np.abs(bz_grid)), 61)
    field_plot = ax_xy.contourf(
        x_grid, y_grid, bz_grid, levels=levels, cmap="RdBu_r",
        norm=symmetric_norm(bz_grid), extend="both",
    )
    add_kicker_regions_cartesian(ax_xy, kicker_cfg, labels=True)
    seo_x = seo["r_row"] * np.cos(seo["phi_axis"])
    seo_y = seo["r_row"] * np.sin(seo["phi_axis"])
    ax_xy.plot(seo_x, seo_y, color="white", lw=1.0, ls="--", alpha=0.9, label="600 MeV SEO")
    ax_xy.plot(off_case["x"], off_case["y"], color="#55d6e8", lw=1.35, label="kicker off")
    on_line, = ax_xy.plot(
        on_case["x"], on_case["y"], color="#ffd84d", lw=2.2, label="kicker on",
    )
    on_line.set_path_effects([
        path_effects.Stroke(linewidth=3.8, foreground="black"),
        path_effects.Normal(),
    ])
    ax_xy.scatter(
        [on_case["x"][0]], [on_case["y"][0]], s=42, c="lime",
        edgecolors="black", linewidths=0.7, zorder=8, label="start",
    )
    if np.any(on_case["r"] > bmap.r_max):
        idx = int(np.where(on_case["r"] > bmap.r_max)[0][0])
        ax_xy.scatter(
            [on_case["x"][idx]], [on_case["y"][idx]], marker="X", s=80,
            c="#ff4d4d", edgecolors="black", linewidths=0.8, zorder=9,
            label="R=12.2 m exit",
        )
    ax_xy.set_aspect("equal", adjustable="box")
    limit = bmap.r_max + 0.35
    ax_xy.set_xlim(-limit, limit)
    ax_xy.set_ylim(-limit, limit)
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title(f"{turns}-turn trajectory on OPERA mid-plane Bz")
    ax_xy.grid(True, alpha=0.18)
    ax_xy.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(field_plot, ax=ax_xy, pad=0.015)
    cbar.set_label("base Bz [T]")

    advance_on = np.rad2deg(on_case["phi"] - math.radians(start_phi_deg))
    advance_off = np.rad2deg(off_case["phi"] - math.radians(start_phi_deg))
    ax_dr.plot(advance_off, off_case["dr_mm"], color="#168aad", lw=1.4, label="kicker off")
    ax_dr.plot(advance_on, on_case["dr_mm"], color="#d62828", lw=1.8, label="kicker on")
    for turn_idx in range(turns):
        for idx, kicker in enumerate(kicker_cfg):
            start = (float(kicker["phi_start_deg"]) - start_phi_deg) % 360.0 + 360.0 * turn_idx
            end = start + float(kicker["phi_span_deg"])
            ax_dr.axvspan(start, end, color="#f4a261", alpha=0.28)
            if turn_idx == 0:
                label_y = 0.98 if idx % 2 == 0 else 0.90
                ax_dr.text(
                    0.5 * (start + end), label_y, str(kicker["name"]),
                    transform=ax_dr.get_xaxis_transform(), ha="center", va="top",
                    rotation=90, fontsize=7,
                )
    ax_dr.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax_dr.set_xlim(0.0, 360.0 * turns)
    ax_dr.set_xlabel(f"azimuth advance from {start_phi_deg:g} deg [deg]")
    ax_dr.set_ylabel("r - rSEO [mm]")
    ax_dr.set_title("Radial departure from the 600 MeV closed orbit")
    ax_dr.grid(True, alpha=0.28)
    ax_dr.legend(loc="best")

    status = "outer aperture crossed" if np.any(on_case["r"] > bmap.r_max) else "outer aperture not crossed"
    fig.suptitle(
        f"600 MeV proton, {turns}-turn continuous-kicker tracking ({status})",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_aperture_clearance(on_case, off_case, kicker_cfg, start_phi_deg,
                            aperture_rmax, turns, output_path):
    advance_on = np.rad2deg(on_case["phi"] - math.radians(start_phi_deg))
    advance_off = np.rad2deg(off_case["phi"] - math.radians(start_phi_deg))
    closest_idx = int(np.argmax(on_case["r"]))
    closest_advance = float(advance_on[closest_idx])
    closest_r = float(on_case["r"][closest_idx])
    clearance_m = aperture_rmax - closest_r

    fig, (ax_r, ax_xy) = plt.subplots(
        1, 2, figsize=(14.8, 6.3), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.18, 0.82]},
    )

    # Direct extraction criterion: the tracked radius must cross R_aperture.
    ax_r.axhspan(aperture_rmax, aperture_rmax + 0.05, color="#ef476f", alpha=0.10)
    ax_r.axhline(
        aperture_rmax, color="#c1121f", lw=2.0, ls="--",
        label=f"outer aperture R = {aperture_rmax:.3f} m",
    )
    ax_r.plot(advance_off, off_case["r"], color="#168aad", lw=1.35, label="kicker off")
    ax_r.plot(advance_on, on_case["r"], color="#d62828", lw=1.8, label="kicker on")
    ax_r.plot(
        advance_on, on_case["r_seo"], color="#555555", lw=1.0, ls=":",
        label="600 MeV SEO",
    )
    for turn_idx in range(turns):
        for idx, kicker in enumerate(kicker_cfg):
            start = (float(kicker["phi_start_deg"]) - start_phi_deg) % 360.0 + 360.0 * turn_idx
            end = start + float(kicker["phi_span_deg"])
            ax_r.axvspan(start, end, color="#f4a261", alpha=0.28)
            if turn_idx == 0:
                label_y = 0.97 if idx % 2 == 0 else 0.88
                ax_r.text(
                    0.5 * (start + end), label_y, str(kicker["name"]),
                    transform=ax_r.get_xaxis_transform(), ha="center", va="top",
                    rotation=90, fontsize=7,
                )
    ax_r.scatter(
        [closest_advance], [closest_r], marker="X", s=72, c="#d62828",
        edgecolors="black", linewidths=0.7, zorder=7,
    )
    ax_r.annotate(
        "",
        xy=(closest_advance, aperture_rmax),
        xytext=(closest_advance, closest_r),
        arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.2},
    )
    ax_r.text(
        closest_advance + 5.0,
        0.5 * (closest_r + aperture_rmax),
        f"clearance = {clearance_m * 1.0e3:.1f} mm",
        va="center",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    ymin = min(
        float(np.min(on_case["r"])), float(np.min(off_case["r"])),
        float(np.min(on_case["r_seo"])),
    ) - 0.025
    ax_r.set_xlim(0.0, 360.0 * turns)
    ax_r.set_ylim(ymin, aperture_rmax + 0.025)
    ax_r.set_xlabel(f"azimuth advance from {start_phi_deg:g} deg [deg]")
    ax_r.set_ylabel("actual radius R [m]")
    ax_r.set_title("Direct aperture-crossing test")
    ax_r.grid(True, alpha=0.28)
    ax_r.legend(loc="lower right", fontsize=8)

    # Cartesian zoom at the point where the kicked trajectory comes closest
    # to the outer aperture.
    center_phi = float(on_case["phi"][closest_idx])
    half_window = math.radians(12.0)
    on_mask = np.abs(on_case["phi"] - center_phi) <= half_window
    off_mask = np.abs(off_case["phi"] - center_phi) <= half_window
    phi_arc = np.linspace(center_phi - half_window, center_phi + half_window, 500)
    aperture_x = aperture_rmax * np.cos(phi_arc)
    aperture_y = aperture_rmax * np.sin(phi_arc)

    ax_xy.plot(aperture_x, aperture_y, color="#c1121f", lw=2.2, ls="--", label="R=12.2 m aperture")
    ax_xy.plot(
        on_case["r_seo"][on_mask] * np.cos(on_case["phi"][on_mask]),
        on_case["r_seo"][on_mask] * np.sin(on_case["phi"][on_mask]),
        color="#555555", lw=1.2, ls=":", label="600 MeV SEO",
    )
    ax_xy.plot(
        off_case["x"][off_mask], off_case["y"][off_mask],
        color="#168aad", lw=1.5, label="kicker off",
    )
    ax_xy.plot(
        on_case["x"][on_mask], on_case["y"][on_mask],
        color="#d62828", lw=2.0, label="kicker on",
    )
    closest_x = float(on_case["x"][closest_idx])
    closest_y = float(on_case["y"][closest_idx])
    aperture_point_x = aperture_rmax * math.cos(center_phi)
    aperture_point_y = aperture_rmax * math.sin(center_phi)
    ax_xy.plot(
        [closest_x, aperture_point_x], [closest_y, aperture_point_y],
        color="black", lw=1.2, marker="|",
    )
    ax_xy.scatter(
        [closest_x], [closest_y], marker="X", s=72, c="#d62828",
        edgecolors="black", linewidths=0.7, zorder=7,
    )
    ax_xy.annotate(
        f"minimum clearance\n{clearance_m * 1.0e3:.1f} mm",
        xy=(0.5 * (closest_x + aperture_point_x), 0.5 * (closest_y + aperture_point_y)),
        xytext=(12, 12), textcoords="offset points", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.88, "pad": 3},
    )
    all_x = np.r_[aperture_x, on_case["x"][on_mask], off_case["x"][off_mask]]
    all_y = np.r_[aperture_y, on_case["y"][on_mask], off_case["y"][off_mask]]
    margin = 0.08
    ax_xy.set_xlim(float(np.min(all_x)) - margin, float(np.max(all_x)) + margin)
    ax_xy.set_ylim(float(np.min(all_y)) - margin, float(np.max(all_y)) + margin)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title(
        f"Cartesian zoom near closest approach\n"
        f"phi = {np.rad2deg(center_phi) % 360.0:.2f} deg"
    )
    ax_xy.grid(True, alpha=0.28)
    ax_xy.legend(loc="best", fontsize=8)

    status = "crossed" if clearance_m < 0.0 else "not crossed"
    fig.suptitle(
        f"600 MeV, {turns}-turn continuous-kicker tracking: outer aperture {status}", fontsize=14,
    )
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_first_last_turn_comparison(on_case, off_case, kicker_cfg, start_phi_deg,
                                    aperture_rmax, turns, output_path):
    if turns < 2:
        return

    start_phi_rad = math.radians(start_phi_deg)
    advance_on = np.rad2deg(on_case["phi"] - start_phi_rad)
    advance_off = np.rad2deg(off_case["phi"] - start_phi_rad)

    def turn_slice(case, advance, turn_number):
        low = 360.0 * (turn_number - 1)
        high = 360.0 * turn_number
        mask = (advance >= low) & (advance <= high)
        return {
            "phase_deg": advance[mask] - low,
            "r": case["r"][mask],
            "r_seo": case["r_seo"][mask],
        }

    first_on = turn_slice(on_case, advance_on, 1)
    first_off = turn_slice(off_case, advance_off, 1)
    last_on = turn_slice(on_case, advance_on, turns)
    last_off = turn_slice(off_case, advance_off, turns)

    fig, (ax_first, ax_last, ax_peak) = plt.subplots(
        1, 3, figsize=(16.0, 5.5), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.82]},
    )

    all_r = np.r_[first_on["r"], first_off["r"], last_on["r"], last_off["r"]]
    ymin = float(np.min(all_r)) - 0.025
    ymax = aperture_rmax + 0.025

    for ax, on_turn, off_turn, turn_number in (
        (ax_first, first_on, first_off, 1),
        (ax_last, last_on, last_off, turns),
    ):
        ax.axhspan(aperture_rmax, ymax, color="#ef476f", alpha=0.10)
        ax.axhline(aperture_rmax, color="#c1121f", lw=2.0, ls="--", label="R=12.2 m aperture")
        ax.plot(off_turn["phase_deg"], off_turn["r"], color="#168aad", lw=1.25, label="kicker off")
        ax.plot(on_turn["phase_deg"], on_turn["r"], color="#d62828", lw=1.8, label="kicker on")
        ax.plot(on_turn["phase_deg"], on_turn["r_seo"], color="#555555", lw=1.0, ls=":", label="600 MeV SEO")
        for idx, kicker in enumerate(kicker_cfg):
            start = (float(kicker["phi_start_deg"]) - start_phi_deg) % 360.0
            end = start + float(kicker["phi_span_deg"])
            ax.axvspan(start, end, color="#f4a261", alpha=0.28)
            label_y = 0.97 if idx % 2 == 0 else 0.88
            ax.text(
                0.5 * (start + end), label_y, str(kicker["name"]),
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                rotation=90, fontsize=7,
            )
        if on_turn["r"].size:
            idx_max = int(np.argmax(on_turn["r"]))
            clearance_mm = (aperture_rmax - float(on_turn["r"][idx_max])) * 1.0e3
            ax.scatter(
                [on_turn["phase_deg"][idx_max]], [on_turn["r"][idx_max]],
                marker="X", s=58, c="#d62828", edgecolors="black", linewidths=0.6, zorder=7,
            )
            ax.text(
                0.98, 0.04, f"minimum clearance = {clearance_mm:.1f} mm",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
            )
        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("azimuth within turn [deg]")
        ax.set_title(f"Turn {turn_number}")
        ax.grid(True, alpha=0.28)

    ax_first.set_ylabel("actual radius R [m]")
    ax_last.legend(loc="upper right", fontsize=7)

    turn_numbers = np.arange(1, turns + 1)
    peak_on = []
    peak_off = []
    for turn_number in turn_numbers:
        on_turn = turn_slice(on_case, advance_on, int(turn_number))
        off_turn = turn_slice(off_case, advance_off, int(turn_number))
        peak_on.append(float(np.max(on_turn["r"])) if on_turn["r"].size else np.nan)
        peak_off.append(float(np.max(off_turn["r"])) if off_turn["r"].size else np.nan)
    peak_on = np.asarray(peak_on)
    peak_off = np.asarray(peak_off)

    ax_peak.axhspan(aperture_rmax, ymax, color="#ef476f", alpha=0.10)
    ax_peak.axhline(aperture_rmax, color="#c1121f", lw=2.0, ls="--", label="outer aperture")
    ax_peak.plot(turn_numbers, peak_off, marker="o", color="#168aad", lw=1.4, label="kicker off")
    ax_peak.plot(turn_numbers, peak_on, marker="o", color="#d62828", lw=1.8, label="kicker on")
    closest_turn_idx = int(np.nanargmax(peak_on))
    closest_clearance_mm = (aperture_rmax - peak_on[closest_turn_idx]) * 1.0e3
    ax_peak.scatter(
        [turn_numbers[closest_turn_idx]], [peak_on[closest_turn_idx]], marker="X", s=68,
        c="#d62828", edgecolors="black", linewidths=0.7, zorder=7,
    )
    ax_peak.annotate(
        f"closest: turn {closest_turn_idx + 1}\n{closest_clearance_mm:.1f} mm remaining",
        xy=(turn_numbers[closest_turn_idx], peak_on[closest_turn_idx]),
        xytext=(10, 18), textcoords="offset points", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.88, "pad": 3},
    )
    ax_peak.set_xticks(turn_numbers)
    ax_peak.set_xlim(0.7, turns + 0.3)
    ax_peak.set_ylim(ymin, ymax)
    ax_peak.set_xlabel("turn number")
    ax_peak.set_ylabel("maximum R in each turn [m]")
    ax_peak.set_title("Turn-by-turn closest approach")
    ax_peak.grid(True, alpha=0.28)
    ax_peak.legend(loc="lower right", fontsize=7)

    fig.suptitle(
        f"Continuous flat-top kicker: turn 1 versus turn {turns}", fontsize=14,
    )
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def main():
    args = parse_args()
    if args.time_step_ns <= 0.0:
        raise ValueError("--time-step-ns must be positive.")
    if args.turns < 1:
        raise ValueError("--turns must be >= 1.")

    config_path = Path(args.config).expanduser().resolve()
    config = load_json(config_path)
    bmap_path = resolve_config_path(config["BmapAndSEO"]["maps"], config_path)
    if bmap_path.is_file() and bmap_path.name == "config_Bmap.json":
        bmap_path = bmap_path.parent
    seo_dir = bmap_path / config["BmapAndSEO"]["SEO_FileName"]

    bmap = load_bmap_auto(str(bmap_path), int(config["BmapAndSEO"].get("max_order", 0)))
    seo = load_seo(seo_dir, args.energy_mev)
    particle_p, particle_v, start_pr_normalized = make_closed_orbit_particle(
        seo, args.start_phi_deg
    )

    output_root = Path(args.output_dir).expanduser().resolve()
    run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    time_step_s = args.time_step_ns * 1.0e-9

    on_case = run_case(
        "kicker_on", bmap, config, seo, config_path, particle_v, particle_p,
        run_dir, time_step_s, args.turns, True,
    )
    off_case = run_case(
        "kicker_off", bmap, config, seo, config_path, particle_v, particle_p,
        run_dir, time_step_s, args.turns, False,
    )
    kickers = FFAG_BumpKickerField(config["BumpKicker"]["kickers"], True)
    augment_trajectory(on_case, bmap, kickers, seo)
    augment_trajectory(off_case, bmap, kickers, seo)

    on_csv = run_dir / "single_particle_600MeV_kicker_on.csv"
    off_csv = run_dir / "single_particle_600MeV_kicker_off.csv"
    figure_path = run_dir / "single_particle_600MeV_extraction.png"
    clearance_figure_path = run_dir / "single_particle_600MeV_aperture_clearance.png"
    turn_comparison_path = run_dir / "single_particle_600MeV_turn1_vs_last.png"
    summary_path = run_dir / "summary.json"
    save_case_csv(on_case, on_csv)
    save_case_csv(off_case, off_csv)
    plot_results(
        bmap, seo, config["BumpKicker"]["kickers"], on_case, off_case,
        args.start_phi_deg, args.turns, figure_path,
    )
    plot_aperture_clearance(
        on_case, off_case, config["BumpKicker"]["kickers"], args.start_phi_deg,
        float(config["Aperture"]["Rmax"]), args.turns, clearance_figure_path,
    )
    plot_first_last_turn_comparison(
        on_case, off_case, config["BumpKicker"]["kickers"], args.start_phi_deg,
        float(config["Aperture"]["Rmax"]), args.turns, turn_comparison_path,
    )

    summary = {
        "config": str(config_path),
        "kicker_pulse_model": "continuous_flat_top",
        "kickers": config["BumpKicker"]["kickers"],
        "bmap": str(bmap_path),
        "seo_dir": str(seo_dir),
        "energy_mev": seo["energy_mev"],
        "start_phi_deg": args.start_phi_deg,
        "start_r_m": float(particle_p[0, 0]),
        "start_pr_normalized": start_pr_normalized,
        "start_trajectory_angle_rad": float(particle_p[0, 1]),
        "time_step_ns": args.time_step_ns,
        "turns_requested": args.turns,
        "revolution_period_ns": float(seo["revolution_period_s"] * 1.0e9),
        "aperture_rmax_m": float(config["Aperture"]["Rmax"]),
        "nominal_single_kicker_deflection_mrad": [
            0.299792458 * abs(float(kicker["BL_Tm"]))
            / (math.sqrt((seo["energy_mev"] + 938.2723) ** 2 - 938.2723 ** 2) / 1000.0)
            * 1000.0
            for kicker in config["BumpKicker"]["kickers"]
        ],
        "kicker_on": case_summary(
            on_case, float(config["Aperture"]["Rmax"]), config["BumpKicker"]["kickers"],
            args.start_phi_deg, args.turns,
        ),
        "kicker_off": case_summary(
            off_case, float(config["Aperture"]["Rmax"]), config["BumpKicker"]["kickers"],
            args.start_phi_deg, args.turns,
        ),
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print(f"Run directory       : {run_dir}")
    print(f"600 MeV start       : phi={args.start_phi_deg:.6f} deg, "
          f"r={particle_p[0, 0]:.9f} m, SEO pr={start_pr_normalized:+.9e}, "
          f"angle={particle_p[0, 1]:+.9e} rad")
    print(f"One-turn period     : {seo['revolution_period_s'] * 1.0e9:.6f} ns")
    print(f"Turns requested     : {args.turns}")
    for label, case in (("kicker on", on_case), ("kicker off", off_case)):
        result = summary[label.replace(" ", "_")]
        print(f"{label:<17}: extracted={result['extracted_beyond_outer_aperture']}, "
              f"Rmax={result['r_max_m']:.9f} m, "
              f"max|dR|={result['max_abs_dr_from_seo_mm']:.3f} mm")
        if "outer_aperture_crossing" in result:
            crossing = result["outer_aperture_crossing"]
            print(f"  aperture crossing: t={crossing['time_ns']:.6f} ns, "
                  f"phi={crossing['phi_mod_deg']:.6f} deg, r={crossing['r_m']:.9f} m")
    print(f"Saved summary       : {summary_path}")
    print(f"Saved kicker-on CSV : {on_csv}")
    print(f"Saved kicker-off CSV: {off_csv}")
    print(f"Saved figure        : {figure_path}")
    print(f"Saved clearance plot: {clearance_figure_path}")
    if args.turns > 1:
        print(f"Saved turn comparison: {turn_comparison_path}")


if __name__ == "__main__":
    main()

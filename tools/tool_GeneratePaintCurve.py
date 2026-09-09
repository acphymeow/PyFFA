import argparse
import os
import sys
from pathlib import Path

import matplotlib

if "--batch" in sys.argv or "--no-show-plots" in sys.argv or (os.name != "nt" and not os.environ.get("DISPLAY")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ========================= user config =========================
BMAP_PATH = Path("./Bmaps/OperaFieldMap")
SEO_DIR_NAME = "resultsSEO"
SEO_INI_PATH = None  # set a file path here to override BMAP_PATH/SEO_DIR_NAME

EK_INJ_MEV = 300.0
MAX_BUNCH_NUM = 1000
OUTPUT_DIR = Path("./PaintCurvesNew")

SAVE_PAINT_FILES = True
SAVE_PLOTS = False
SHOW_PLOTS = True

DELTA_PR_MAX = 0.0
DELTA_Z_MAX = 0.0

# Correlated scan: x/r and py/pz both grow from small to large.
# RMAX_MM_LIST = np.array([20.0])
# PZMAX_MRAD_LIST = np.array([2.0])
# R_PROFILE_TYPE = "up_power"
# PZ_PROFILE_TYPE = "up_power"
# ALPHA_LIST_R = np.array([0.25, 0.40, 0.60, 0.80])
# ALPHA_LIST_Z = np.array([0.25, 0.40, 0.60, 0.80])

# Anti-correlated scan: x/r grows from small to large, py/pz decreases from large to small.
RMAX_MM_LIST = np.array([20.0])
PZMAX_MRAD_LIST = np.array([2.0])
R_PROFILE_TYPE = "up_power"
PZ_PROFILE_TYPE = "down_power"
ALPHA_LIST_R = np.array([0.25, 0.40, 0.60, 0.80])
ALPHA_LIST_Z = np.array([1.30, 1.60, 2.00, 2.50])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PyFFA paint curve files from SEO frequency data.")
    parser.add_argument("--bmap-path", default=str(BMAP_PATH), help="Directory containing the SEO result directory.")
    parser.add_argument("--seo-dir-name", default=SEO_DIR_NAME, help="SEO result directory name under --bmap-path.")
    parser.add_argument("--seo-ini", default=SEO_INI_PATH, help="Explicit path to SEO_ini.txt.")
    parser.add_argument("--ek-inj-mev", type=float, default=EK_INJ_MEV, help="Injection kinetic energy in MeV.")
    parser.add_argument("--max-bunch-num", type=int, default=MAX_BUNCH_NUM, help="Maximum bunch number.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory for generated paint curves.")
    parser.add_argument("--batch", action="store_true", help="Batch mode: do not show plots.")
    parser.add_argument("--save-paint-files", dest="save_paint_files", action="store_true", default=None)
    parser.add_argument("--no-save-paint-files", dest="save_paint_files", action="store_false")
    parser.add_argument("--save-plots", dest="save_plots", action="store_true", default=None)
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    parser.add_argument("--show-plots", dest="show_plots", action="store_true", default=None)
    parser.add_argument("--no-show-plots", dest="show_plots", action="store_false")
    return parser.parse_args()


args = parse_args()
BMAP_PATH = Path(args.bmap_path)
SEO_DIR_NAME = args.seo_dir_name
SEO_INI_PATH = args.seo_ini
EK_INJ_MEV = args.ek_inj_mev
MAX_BUNCH_NUM = args.max_bunch_num
OUTPUT_DIR = Path(args.output_dir)
if args.save_paint_files is not None:
    SAVE_PAINT_FILES = args.save_paint_files
if args.save_plots is not None:
    SAVE_PLOTS = args.save_plots
if args.batch:
    SHOW_PLOTS = False
if args.show_plots is not None:
    SHOW_PLOTS = args.show_plots


def profile_curve(frac, profile):
    if profile.startswith("up_power_"):
        return frac ** float(profile.removeprefix("up_power_"))
    if profile.startswith("down_power_"):
        return 1.0 - frac ** float(profile.removeprefix("down_power_"))
    raise ValueError(f"Unsupported profile: {profile}")


def profile_dir(profile):
    return "up" if profile.startswith("up_") else "down"


def tag(value, digits):
    return f"{value:.{digits}f}".replace(".", "_")


# ========================= SEO frequency =========================
if MAX_BUNCH_NUM < 2:
    raise ValueError("MAX_BUNCH_NUM must be at least 2")
if R_PROFILE_TYPE not in {"up_power", "down_power"} or PZ_PROFILE_TYPE not in {"up_power", "down_power"}:
    raise ValueError("R_PROFILE_TYPE and PZ_PROFILE_TYPE must be 'up_power' or 'down_power'")
if np.any(ALPHA_LIST_R <= 0.0) or np.any(ALPHA_LIST_Z <= 0.0):
    raise ValueError("All alpha values must be positive")

seo_ini = Path(SEO_INI_PATH) if SEO_INI_PATH is not None else BMAP_PATH / SEO_DIR_NAME / "SEO_ini.txt"
if not seo_ini.is_file() and SEO_INI_PATH is None:
    found = sorted(BMAP_PATH.rglob("SEO_ini.txt")) if BMAP_PATH.exists() else []
    if len(found) == 1:
        seo_ini = found[0]
    elif len(found) > 1:
        raise FileNotFoundError("Multiple SEO_ini.txt files found. Set SEO_INI_PATH explicitly.")
if not seo_ini.is_file():
    raise FileNotFoundError(f"SEO_ini.txt not found: {seo_ini}")

seo_data = np.atleast_2d(np.loadtxt(seo_ini, skiprows=2))
if seo_data.shape[1] < 5:
    raise ValueError(f"SEO_ini.txt needs at least 5 columns: {seo_ini}")

seo_ek = seo_data[:, 1]
seo_freq = seo_data[:, 4]
order = np.argsort(seo_ek)
seo_ek, seo_freq = seo_ek[order], seo_freq[order]
if EK_INJ_MEV < seo_ek[0] or EK_INJ_MEV > seo_ek[-1]:
    raise ValueError(f"Ek_inj={EK_INJ_MEV:.3f} MeV outside SEO range {seo_ek[0]:.3f}-{seo_ek[-1]:.3f} MeV")

f_inj = float(np.interp(EK_INJ_MEV, seo_ek, seo_freq))
if not np.isfinite(f_inj) or f_inj <= 0.0:
    raise ValueError(f"Interpolated injection frequency is invalid: f_inj={f_inj}")
time_interval_ns = 1.0 / f_inj * 1e9

print(f"SEO_ini: {seo_ini}")
print(f"Interpolated f_inj = {f_inj:.6e} Hz")
print(f"Interpolated T_inj = {time_interval_ns:.6f} ns")


# ========================= generate curves =========================
OUTPUT_DIR.mkdir(exist_ok=True)
num_rows = MAX_BUNCH_NUM + 1  # endpoint keeps floor(max_time / interval) from losing one bunch
turn_axis = np.arange(num_rows)
frac_axis = turn_axis / MAX_BUNCH_NUM
time_axis_ns = turn_axis * time_interval_ns
header = "time_ns\tdelta_r(m)\tdelta_pr(rad)\tdelta_z(m)\tdelta_pz(rad)"

profile_sets = []
for alpha_r in ALPHA_LIST_R:
    for alpha_z in ALPHA_LIST_Z:
        r_profile = f"{R_PROFILE_TYPE}_{alpha_r:.2f}"
        pz_profile = f"{PZ_PROFILE_TYPE}_{alpha_z:.2f}"
        r_dir, pz_dir = profile_dir(r_profile), profile_dir(pz_profile)
        corr_tag = {("up", "up"): "cor_u", ("down", "down"): "cor_d", ("up", "down"): "anti_u", ("down", "up"): "anti_d"}[(r_dir, pz_dir)]
        profile_sets.append((float(alpha_r), float(alpha_z), r_profile, pz_profile, r_dir, pz_dir, corr_tag))

r_curves, pz_curves, generated = {}, {}, []
for rmax_mm in RMAX_MM_LIST:
    for alpha_r, _, r_profile, *_ in profile_sets:
        r_curves[(float(rmax_mm), alpha_r, r_profile)] = rmax_mm * 1e-3 * profile_curve(frac_axis, r_profile)
for pzmax_mrad in PZMAX_MRAD_LIST:
    for _, alpha_z, _, pz_profile, *_ in profile_sets:
        pz_curves[(float(pzmax_mrad), alpha_z, pz_profile)] = pzmax_mrad * 1e-3 * profile_curve(frac_axis, pz_profile)

for rmax_mm in RMAX_MM_LIST:
    for pzmax_mrad in PZMAX_MRAD_LIST:
        for alpha_r, alpha_z, r_profile, pz_profile, r_dir, pz_dir, corr_tag in profile_sets:
            dr = r_curves[(float(rmax_mm), alpha_r, r_profile)]
            dpz = pz_curves[(float(pzmax_mrad), alpha_z, pz_profile)]
            data = np.column_stack([time_axis_ns, dr, np.full(num_rows, DELTA_PR_MAX), np.full(num_rows, DELTA_Z_MAX), dpz])
            name = f"{corr_tag}_rmax_{tag(rmax_mm, 1)}mm_pzmax_{tag(pzmax_mrad, 1)}mrad_alphaR_{tag(alpha_r, 2)}_alphaZ_{tag(alpha_z, 2)}.paint"
            output_path = OUTPUT_DIR / name
            if SAVE_PAINT_FILES:
                np.savetxt(output_path, data, fmt="%.9e", delimiter="\t", header=header, comments="")
            generated.append((corr_tag, float(rmax_mm), float(pzmax_mrad), alpha_r, alpha_z, r_dir, pz_dir, dr, dpz, output_path))
            print(f"Saved: {output_path}")

print("=" * 72)
print(f"R ranges={len(RMAX_MM_LIST)}, pz ranges={len(PZMAX_MRAD_LIST)}, alpha sets={len(profile_sets)}")
print(f"Expected files={len(RMAX_MM_LIST) * len(PZMAX_MRAD_LIST) * len(profile_sets)}, generated files={len(generated)}, rows per file={num_rows}")
for alpha_r, alpha_z, _, _, r_dir, pz_dir, corr_tag in profile_sets:
    print(f"type={corr_tag}, R={r_dir}, pz={pz_dir}, alphaR={alpha_r:.2f}, alphaZ={alpha_z:.2f}")


# ========================= plots =========================
linestyles = ["-", "--", "-.", ":", (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
figs = []

fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
for i, ((rmax_mm, alpha_r, r_profile), dr) in enumerate(r_curves.items()):
    ax.plot(turn_axis, dr * 1e3, linestyle=linestyles[i % len(linestyles)], lw=1.8, label=rf"$R_{{max}}={rmax_mm:.0f}$ mm, $\alpha_R={alpha_r:.2f}$, {profile_dir(r_profile)}")
ax.set(xlabel="Turns", ylabel=r"$\Delta r$ [mm]")
ax.tick_params(labelsize=15)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, ncol=2, frameon=False)
if SAVE_PLOTS:
    fig.savefig(OUTPUT_DIR / "paint_delta_r.png", dpi=180)
figs.append(fig)

fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
for i, ((pzmax_mrad, alpha_z, pz_profile), dpz) in enumerate(pz_curves.items()):
    ax.plot(turn_axis, dpz * 1e3, linestyle=linestyles[i % len(linestyles)], lw=1.8, label=rf"$p_{{z,max}}={pzmax_mrad:.0f}$ mrad, $\alpha_Z={alpha_z:.2f}$, {profile_dir(pz_profile)}")
ax.set(xlabel="Turns", ylabel=r"$\Delta p_z$ [mrad]")
ax.tick_params(labelsize=15)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, ncol=2, frameon=False)
if SAVE_PLOTS:
    fig.savefig(OUTPUT_DIR / "paint_delta_pz.png", dpi=180)
figs.append(fig)

fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
for corr_tag, rmax_mm, pzmax_mrad, alpha_r, alpha_z, *_rest, dr, dpz, _ in generated:
    ax.plot(dr * 1e3, dpz * 1e3, lw=1.4, label=f"{corr_tag}: ({rmax_mm:.0f} mm, {pzmax_mrad:.0f} mrad), " rf"$\alpha_R={alpha_r:.2f}$, $\alpha_Z={alpha_z:.2f}$")
ax.set(xlabel=r"$\Delta r$ [mm]", ylabel=r"$\Delta p_z$ [mrad]")
ax.tick_params(labelsize=15)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2, frameon=False)
if SAVE_PLOTS:
    fig.savefig(OUTPUT_DIR / "paint_delta_r_delta_pz.png", dpi=180)
figs.append(fig)

if SHOW_PLOTS:
    plt.show()
else:
    for fig in figs:
        plt.close(fig)

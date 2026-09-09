"""Validation 4: bare nonlinear/chromatic optics missing from the fixed toy."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as REPO, CANONICAL_TUNE
sys.path.insert(0, str(REPO))
from FFAG_BetaFunc import LoadBMapAuto
from FFAG_ParasAndConversion import FFAG_ConversionTools, FFAG_GlobalParameters
from FFAG_track import FunctionForSEOUnified


SEO = REPO / "Bmap_FD16" / "resultsSEO_noError_X"
OUT = HERE / "results" / "validation_4_nonlinear_optics"
sys.path.insert(0, str(HERE))
import comparison_core as analysis  # noqa: E402
from toy_tracking import build_lattice, tune_fft  # noqa: E402

ENERGY_SCAN = np.array([296., 298., 300., 302., 304.])
JR_SCAN = np.array([0.5, 1., 2., 5., 10., 20., 40., 80.])
JZ_SCAN = np.array([0.5, 1., 2., 5., 7., 10., 20., 40., 80.])
REFERENCE = (300., 5., 7.)
N_TRACK, TUNE_START, STEP_RAD = 110, 10, 0.01


def table(name):
    return np.loadtxt(SEO / name, skiprows=1)


seo_ini = np.loadtxt(SEO / "SEO_ini.txt", skiprows=2)
beta_r, beta_z = table("BetaFuncR.txt"), table("BetaFuncZ.txt")
alpha_r, alpha_z = table("AlphaFuncR.txt"), table("AlphaFuncZ.txt")
lattice = build_lattice()


def at_energy(data, energy):
    return np.interp(energy, data[1:, 0], data[1:, 1])


def make_scan():
    points = [(energy, REFERENCE[1], REFERENCE[2], "energy") for energy in ENERGY_SCAN]
    points += [(REFERENCE[0], action, REFERENCE[2], "Jr") for action in JR_SCAN]
    points += [(REFERENCE[0], REFERENCE[1], action, "Jz") for action in JZ_SCAN]
    unique = []
    for point in points:
        if point[:3] not in [old[:3] for old in unique]:
            unique.append(point)
    return unique


scan = make_scan()
energy = np.array([point[0] for point in scan])
jr = np.array([point[1] for point in scan])*1e-6
jz = np.array([point[2] for point in scan])*1e-6
br, bz = at_energy(beta_r, energy), at_energy(beta_z, energy)
ar, az = at_energy(alpha_r, energy), at_energy(alpha_z, energy)
r0 = np.interp(energy, seo_ini[:, 1], seo_ini[:, 6])
pr0 = np.interp(energy, seo_ini[:, 1], seo_ini[:, 7])
dr, dz = np.sqrt(2*br*jr), np.sqrt(2*bz*jz)
state = np.column_stack((
    r0 + dr, pr0 - ar/br*dr, dz, -az/bz*dz,
    FFAG_ConversionTools().Ek2P(energy), np.arange(len(scan)),
))

global_parameters = FFAG_GlobalParameters()
global_parameters.AddBMap(LoadBMapAuto(str(REPO / "Bmap_FD16"), 2, flag3D=True))
n_step = round(2*np.pi/STEP_RAD)
h = 2*np.pi/n_step
history = []
phi = 0.0
for turn in range(N_TRACK):
    history.append(state.copy())
    for _ in range(n_step):
        k1 = h*FunctionForSEOUnified(phi, state, global_parameters)
        k2 = h*FunctionForSEOUnified(phi + h/2, state + k1/2, global_parameters)
        k3 = h*FunctionForSEOUnified(phi + h/2, state + k2/2, global_parameters)
        k4 = h*FunctionForSEOUnified(phi + h, state + k3, global_parameters)
        state += (k1 + 2*k2 + 2*k3 + k4)/6
        phi += h
history = np.asarray(history)[TUNE_START:]
qr = tune_fft(history[:, :, 0] - r0, history[:, :, 1] - pr0,
              br, ar, lattice["qr"])
qz = tune_fft(history[:, :, 2], history[:, :, 3], bz, az, lattice["qz"])

scan_array = np.array([(point[0], point[1], point[2], qr[i], qz[i])
                       for i, point in enumerate(scan)])
OUT.mkdir(parents=True, exist_ok=True)
np.savetxt(OUT / "bare_nonlinear_scan.csv", scan_array, delimiter=",",
           header="energy_mev,Jr_pi_mm_mrad,Jz_pi_mm_mrad,Qr,Qz", comments="")

# Energy-dependent Twiss-envelope change across the conservative 296--304 MeV span.
def profile_at_energy(data, value):
    return np.array([np.interp(value, data[1:, 0], data[1:, column])
                     for column in range(1, data.shape[1])])


br_reference = profile_at_energy(beta_r, 300.0)
bz_reference = profile_at_energy(beta_z, 300.0)
twiss_summary = []
for value in ENERGY_SCAN:
    ratio_r = np.sqrt(profile_at_energy(beta_r, value)/br_reference)
    ratio_z = np.sqrt(profile_at_energy(beta_z, value)/bz_reference)
    twiss_summary.append((value, ratio_r.min(), ratio_r.max(),
                          ratio_z.min(), ratio_z.max(),
                          at_energy(beta_r, value), at_energy(beta_z, value),
                          at_energy(alpha_r, value), at_energy(alpha_z, value)))
np.savetxt(OUT / "twiss_energy_summary.csv", twiss_summary, delimiter=",",
           header=("energy_mev,sqrt_beta_r_ratio_min,sqrt_beta_r_ratio_max,"
                   "sqrt_beta_z_ratio_min,sqrt_beta_z_ratio_max,beta_r0_m,"
                   "beta_z0_m,alpha_r0,alpha_z0"), comments="")

# Determine which actions and energies actually populate the lower 1% Qr edge.
tunes = np.load(CANONICAL_TUNE / "results" / "lsc_on" / "withlsc500_first100_3d_tunes.npz")
dump = REPO / "HPC_dir" / "data_coupling_withLsc500" / "FFT_Tune_Dump_0"
first = np.stack([np.load(dump / name)["particles"][0, [0, 1, 2, 3, 5]]
                  for name in tunes["filenames"]])[None, :, :]
x, xp, z, zp, ek = (value[0] for value in analysis.project_3d_coordinates(first))
br0, bz0 = lattice["beta_r"][0], lattice["beta_z"][0]
ar0, az0 = lattice["alpha_r"][0], lattice["alpha_z"][0]
jr_3d = 0.5*((1 + ar0*ar0)/br0*x*x + 2*ar0*x*xp + br0*xp*xp)*1e6
jz_3d = 0.5*((1 + az0*az0)/bz0*z*z + 2*az0*z*zp + bz0*zp*zp)*1e6
core = tunes["qr_fft"] <= np.percentile(tunes["qr_fft"], 1)
quantiles = np.array([.05, .50, .95])
core_summary = np.column_stack((quantiles,
                                np.quantile(ek[core], quantiles),
                                np.quantile(jr_3d[core], quantiles),
                                np.quantile(jz_3d[core], quantiles)))
np.savetxt(OUT / "endpoint_core_summary.csv", core_summary, delimiter=",",
           header="quantile,energy_mev,Jr_pi_mm_mrad,Jz_pi_mm_mrad", comments="")

reference_index = next(i for i, point in enumerate(scan) if point[:3] == REFERENCE)
qr_reference, qz_reference = qr[reference_index], qz[reference_index]
mask_e = (jr == REFERENCE[1]*1e-6) & (jz == REFERENCE[2]*1e-6)
mask_jr = (energy == REFERENCE[0]) & (jz == REFERENCE[2]*1e-6)
mask_jz = (energy == REFERENCE[0]) & (jr == REFERENCE[1]*1e-6)


def scan_interp(mask, coordinate, value, tune):
    order = np.argsort(coordinate[mask])
    return np.interp(value, coordinate[mask][order], tune[mask][order])


core_prediction = []
for quantile, value_e, value_jr, value_jz in core_summary:
    dqr = (scan_interp(mask_e, energy, value_e, qr) - qr_reference
           + scan_interp(mask_jr, jr*1e6, value_jr, qr) - qr_reference
           + scan_interp(mask_jz, jz*1e6, value_jz, qr) - qr_reference)
    dqz = (scan_interp(mask_e, energy, value_e, qz) - qz_reference
           + scan_interp(mask_jr, jr*1e6, value_jr, qz) - qz_reference
           + scan_interp(mask_jz, jz*1e6, value_jz, qz) - qz_reference)
    core_prediction.append((quantile, dqr, dqz))
np.savetxt(OUT / "endpoint_core_predicted_bare_correction.csv", core_prediction,
           delimiter=",", header="quantile,delta_Qr,delta_Qz", comments="")

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
for axis, mask, coordinate, xlabel in (
    (axes[0], mask_e, energy, "Kinetic energy [MeV]"),
    (axes[1], mask_jr, jr*1e6, r"$J_r$ [$\pi$ mm mrad]"),
    (axes[2], mask_jz, jz*1e6, r"$J_z$ [$\pi$ mm mrad]"),
):
    axis.plot(coordinate[mask], (qr[mask] - qr_reference)*1e3, "o-", label=r"$Q_r$")
    axis.plot(coordinate[mask], (qz[mask] - qz_reference)*1e3, "s-", label=r"$Q_z$")
    axis.axhline(0, color="black", ls="--", lw=1)
    axis.set(xlabel=xlabel, ylabel=r"$\Delta Q\times10^3$")
axes[0].legend()
fig.savefig(OUT / "validation_4_nonlinear_optics.png", dpi=250, bbox_inches="tight")
fig.savefig(OUT / "validation_4_nonlinear_optics.pdf", bbox_inches="tight")
plt.close(fig)

print(f"reference Qr,Qz = {qr_reference:.9f}, {qz_reference:.9f}")
print(f"energy effect on Qr over 296--304 MeV: {np.ptp(qr[mask_e]):.9f}")
print(f"Jr effect on Qr over 0.5--20: "
      f"{np.ptp(qr[mask_jr & (jr <= 20e-6)]):.9f}")
print(f"Jz effect on Qr over 0.5--30: "
      f"{np.ptp(qr[mask_jz & (jz <= 30e-6)]):.9f}")
twiss_summary = np.asarray(twiss_summary)
print("sqrt(beta_r(E)/beta_r(300)) range over 296--304 MeV: "
      f"{twiss_summary[:, 1:3].min():.6f}--{twiss_summary[:, 1:3].max():.6f}")
print("endpoint core [q05,q50,q95] rows:")
print(core_summary)
print("predicted additive bare corrections [quantile,dQr,dQz]:")
print(np.asarray(core_prediction))

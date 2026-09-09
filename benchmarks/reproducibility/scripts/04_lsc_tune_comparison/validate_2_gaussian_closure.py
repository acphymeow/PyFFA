"""Validation 2: test the fixed toy's transverse Gaussian closure."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT
DATA = REPO_ROOT / "HPC_dir" / "data_coupling_withLsc500"
SEO = REPO_ROOT / "Bmap_FD16" / "resultsSEO_noError_X" / "SEO_ini.txt"
OUT = HERE / "results" / "validation_2_gaussian_closure"
TURNS, N_SLICES, N_TOP, N_GRID, SIGMA_RANGE = (1, 51, 101), 128, 5, 128, 12.0


def load_turn(turn):
    blocks = [np.load(path)["particles"][:, [0, 2, 5, 7, 15, 16]]
              for path in sorted((DATA / f"TBT_Dump_{turn}").glob("crossing_angle_*_rank_*.npz"))]
    data = np.concatenate(blocks).astype(float)
    data = data[np.isfinite(data).all(axis=1) & (data[:, 3] == 1)]
    data = data[np.argsort(data[:, 5], kind="stable")]
    _, first = np.unique(data[:, 4].astype(np.int64), return_index=True)
    data = data[np.sort(first)]
    edges = np.linspace(data[:, 5].min(), data[:, 5].max(), N_SLICES + 1)
    bins = np.clip(np.searchsorted(edges, data[:, 5], side="right") - 1, 0, N_SLICES - 1)
    return data, bins, np.bincount(bins, minlength=N_SLICES)


def cic_density(x, z, x_grid, z_grid):
    dx, dz = x_grid[1] - x_grid[0], z_grid[1] - z_grid[0]
    density = np.zeros((N_GRID, N_GRID))
    rx, rz = (x - x_grid[0]) / dx, (z - z_grid[0]) / dz
    ix, iz = np.floor(rx).astype(int), np.floor(rz).astype(int)
    fx, fz = rx - ix, rz - iz
    ix, iz = np.clip(ix, 0, N_GRID - 2), np.clip(iz, 0, N_GRID - 2)
    for jj, ii, weight in (
        (iz, ix, (1 - fx)*(1 - fz)), (iz, ix + 1, fx*(1 - fz)),
        (iz + 1, ix, (1 - fx)*fz), (iz + 1, ix + 1, fx*fz),
    ):
        np.add.at(density, (jj, ii), weight)
    return density / (dx*dz)


def open_boundary_field(density, x_grid, z_grid):
    dx, dz = x_grid[1] - x_grid[0], z_grid[1] - z_grid[0]
    index = np.arange(2*N_GRID)
    offset = np.where(index < N_GRID, index, index - 2*N_GRID)
    xx, zz = np.meshgrid(offset*dx, offset*dz)
    self_radius = 0.5*np.hypot(dx, dz)
    kernel = -np.log(np.sqrt(np.maximum(xx*xx + zz*zz, self_radius**2))) / (2*np.pi)
    padded = np.zeros_like(kernel)
    padded[:N_GRID, :N_GRID] = density
    phi = np.fft.ifft2(np.fft.fft2(padded)*np.fft.fft2(kernel)).real
    phi = phi[:N_GRID, :N_GRID]*dx*dz
    ez, ex = np.gradient(-phi, dz, dx)
    return ex, ez


def core_slopes(ex, ez, x_grid, z_grid, sigma_x, sigma_z):
    interp_x = RegularGridInterpolator((z_grid, x_grid), ex)
    interp_z = RegularGridInterpolator((z_grid, x_grid), ez)
    x = np.linspace(0.15, 0.65, 11)*sigma_x
    z = np.linspace(0.15, 0.65, 11)*sigma_z
    field_x = (interp_x(np.c_[0*x, x]) - interp_x(np.c_[0*x, -x]))/2
    field_z = (interp_z(np.c_[z, 0*z]) - interp_z(np.c_[-z, 0*z]))/2
    return np.dot(x, field_x)/np.dot(x, x), np.dot(z, field_z)/np.dot(z, z)


def slice_metrics(block, seo):
    mean_energy = block[:, 2].mean()
    r0 = np.interp(mean_energy, seo[:, 1], seo[:, 11])
    vr0 = np.interp(mean_energy, seo[:, 1], seo[:, 12])
    pr0 = vr0/np.sqrt(1 - vr0*vr0)
    x = (block[:, 0] - r0)/np.sqrt(1 + pr0*pr0)
    z = block[:, 1].copy()
    x -= x.mean(); z -= z.mean()
    sigma_x, sigma_z = x.std(), z.std()
    x_grid = np.linspace(-SIGMA_RANGE*sigma_x, SIGMA_RANGE*sigma_x, N_GRID, endpoint=False)
    z_grid = np.linspace(-SIGMA_RANGE*sigma_z, SIGMA_RANGE*sigma_z, N_GRID, endpoint=False)
    actual = cic_density(x, z, x_grid, z_grid)
    xx, zz = np.meshgrid(x_grid, z_grid)
    gaussian = np.exp(-0.5*((xx/sigma_x)**2 + (zz/sigma_z)**2))/(2*np.pi*sigma_x*sigma_z)
    gaussian *= actual.sum()/gaussian.sum()
    actual_slope = core_slopes(*open_boundary_field(actual, x_grid, z_grid),
                               x_grid, z_grid, sigma_x, sigma_z)
    gaussian_slope = core_slopes(*open_boundary_field(gaussian, x_grid, z_grid),
                                 x_grid, z_grid, sigma_x, sigma_z)
    radius = np.sqrt((x/sigma_x)**2 + (z/sigma_z)**2)
    return sigma_x, sigma_z, np.mean(x*z)/(sigma_x*sigma_z), \
        np.mean(x**4)/sigma_x**4 - 3, np.mean(z**4)/sigma_z**4 - 3, \
        np.mean(radius < 1), np.mean(radius < 2), \
        actual_slope[0]/gaussian_slope[0], actual_slope[1]/gaussian_slope[1]


OUT.mkdir(parents=True, exist_ok=True)
seo = np.loadtxt(SEO, skiprows=2)
rows = []
for turn in TURNS:
    data, bins, counts = load_turn(turn)
    for slice_index in np.argsort(counts)[-N_TOP:][::-1]:
        metrics = slice_metrics(data[bins == slice_index], seo)
        rows.append((turn, slice_index, counts[slice_index], *metrics))

columns = ("turn,slice,count,sigma_x_m,sigma_z_m,corr_xz,kurtosis_x_excess,"
           "kurtosis_z_excess,p_r_lt_1,p_r_lt_2,gx_over_gaussian,gz_over_gaussian")
np.savetxt(OUT / "slice_metrics.csv", np.asarray(rows), delimiter=",", header=columns, comments="")

summary = []
for turn in TURNS:
    selected = [row for row in rows if row[0] == turn]
    weight = np.asarray([row[2] for row in selected])
    values = np.asarray([row[3:] for row in selected])
    summary.append((turn, weight.sum(), *np.average(values, axis=0, weights=weight)))
summary = np.asarray(summary)
np.savetxt(OUT / "turn_summary.csv", summary, delimiter=",",
           header="turn,top5_count," + columns.split("count,")[1], comments="")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
axes[0].plot(summary[:, 0], summary[:, -2], "o-", label=r"$G_x/G_{x,G}$")
axes[0].plot(summary[:, 0], summary[:, -1], "s-", label=r"$G_z/G_{z,G}$")
axes[0].axhline(1, color="black", ls="--", lw=1)
axes[0].set(xlabel="Turn", ylabel="core field-gradient ratio",
            title="Actual slice / matched Gaussian")
axes[0].legend()
axes[1].plot(summary[:, 0], summary[:, 5], "o-", label="x excess kurtosis")
axes[1].plot(summary[:, 0], summary[:, 6], "s-", label="z excess kurtosis")
axes[1].plot(summary[:, 0], summary[:, 4], "^-", label="corr(x,z)")
axes[1].axhline(0, color="black", ls="--", lw=1)
axes[1].set(xlabel="Turn", ylabel="shape metric", title="Departure from Gaussian closure")
axes[1].legend()
fig.savefig(OUT / "validation_2_gaussian_closure.png", dpi=250, bbox_inches="tight")
fig.savefig(OUT / "validation_2_gaussian_closure.pdf", bbox_inches="tight")
plt.close(fig)

for row in summary:
    print(f"turn {int(row[0]):3d}: sigma=({row[2]*1e3:.3f},{row[3]*1e3:.3f}) mm, "
          f"corr={row[4]:+.4f}, kurt=({row[5]:+.4f},{row[6]:+.4f}), "
          f"G/Gauss=({row[-2]:.4f},{row[-1]:.4f})")

"""Validation 5: compare stored 3-D SC fields with the Gaussian toy closure."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as ROOT, CANONICAL_TOY
FFT_DUMP = ROOT / "HPC_dir" / "data_coupling_withLsc500" / "FFT_Tune_Dump_0"
SEO_INI = ROOT / "Bmap_FD16" / "resultsSEO_noError_X" / "SEO_ini.txt"
LONG_CACHE = CANONICAL_TOY / "outputs" / "gauss_waterbag_longitudinal_pair.npz"
OUT = HERE / "results" / "validation_5_sc_kick_history"

N_RECORDS, N_SLICES, N_TOP = 100, 128, 5
SAMPLE_SCALE, Q_MACRO = 8.0, 1.0e8 * 1.602176634e-19
EPS0, MP_MEV = 8.8541878128e-12, 938.2720813
COLS = (0, 2, 5, 9, 10, 11, 12, 15, 16)
GL_X, GL_W = np.polynomial.legendre.leggauss(32)
GL_U, GL_W = 0.5*(GL_X + 1.0), 0.5*GL_W


def load_history():
    paths = sorted(FFT_DUMP.glob("particle_*.npz"))
    history = np.empty((len(paths), N_RECORDS, len(COLS)), dtype=np.float32)
    for i, path in enumerate(paths):
        block = np.load(path)["particles"]
        if len(block) < N_RECORDS:
            raise ValueError(f"{path.name} has only {len(block)} records")
        history[i] = block[:N_RECORDS, COLS]
        if (i + 1) % 5000 == 0:
            print(f"loaded {i + 1}/{len(paths)} files", flush=True)
    return history


def unwrap_to_short_arc(phase):
    ordered = np.sort(np.mod(phase, 2*np.pi))
    gaps = np.diff(np.r_[ordered, ordered[0] + 2*np.pi])
    start = ordered[(np.argmax(gaps) + 1) % len(ordered)]
    return np.mod(phase - start, 2*np.pi)


def gaussian_field(x, z, sx, sz, line_charge=1.0, gamma=1.0):
    ex, ez = np.zeros_like(x), np.zeros_like(z)
    t0 = max(sx*sx, sz*sz)
    for u, weight in zip(GL_U, GL_W):
        t = t0*u/(1-u)
        dt = t0/(1-u)**2
        ax, az = sx*sx + t, sz*sz + t
        common = weight*dt*np.exp(-0.5*(x*x/ax + z*z/az))/np.sqrt(ax*az)
        ex += common*x/ax
        ez += common*z/az
    factor = line_charge/(4*np.pi*EPS0*gamma*gamma)
    return factor*ex, factor*ez


def fit_slice(block, line_charge, r0, pr_raw):
    norm = 1.0 / np.sqrt(1.0 - pr_raw*pr_raw)
    pr_geom = pr_raw * norm
    x = (block[:, 0] - r0) / norm
    z = block[:, 1]
    ex = (block[:, 4] - pr_geom*block[:, 6]) / np.sqrt(1 + pr_geom*pr_geom)
    ez = block[:, 5]
    x -= x.mean(); z -= z.mean()
    sx, sz = x.std(), z.std()
    radius = np.sqrt((x/sx)**2 + (z/sz)**2)
    core = radius < 1.0
    design = np.c_[x[core], z[core], np.ones(core.sum())]
    gx = np.linalg.lstsq(design, ex[core], rcond=None)[0][0]
    gz = np.linalg.lstsq(design, ez[core], rcond=None)[0][1]
    gamma = 1.0 + block[:, 2].mean()/MP_MEV
    ex_gauss, ez_gauss = gaussian_field(x[core], z[core], sx, sz, line_charge, gamma)
    gx_gauss = np.linalg.lstsq(design, ex_gauss, rcond=None)[0][0]
    gz_gauss = np.linalg.lstsq(design, ez_gauss, rcond=None)[0][1]
    return sx, sz, gx, gz, gx_gauss, gz_gauss


def numerical_gaussian_ratio(sx, sz, grid_sx, grid_sz, n=128, scale=12.0,
                             kernel_aspect=1.0):
    x = np.linspace(-scale*grid_sx, scale*grid_sx, n, endpoint=False)
    z = np.linspace(-scale*grid_sz, scale*grid_sz, n, endpoint=False)
    dx, dz = x[1] - x[0], z[1] - z[0]
    xx, zz = np.meshgrid(x, z)
    rho = np.exp(-0.5*((xx/sx)**2 + (zz/sz)**2))/(2*np.pi*sx*sz)
    rho /= rho.sum()*dx*dz
    index = np.arange(2*n)
    offset = np.where(index < n, index, index - 2*n)
    kx, kz = np.meshgrid(offset*dx*np.sqrt(kernel_aspect),
                         offset*dz/np.sqrt(kernel_aspect))
    radius = np.hypot(kx, kz)
    radius[0, 0] = 0.5*np.hypot(dx, dz)
    kernel = -np.log(radius)/(2*np.pi)
    padded = np.zeros((2*n, 2*n))
    padded[:n, :n] = rho
    phi = np.fft.ifft2(np.fft.fft2(padded)*np.fft.fft2(kernel)).real[:n, :n]*dx*dz
    ez, ex = np.gradient(-phi, dz, dx)
    samples = np.linspace(0.15, 0.65, 11)
    ix = np.clip(np.floor((samples*sx - x[0])/dx).astype(int), 0, n - 2)
    iz = np.clip(np.floor((samples*sz - z[0])/dz).astype(int), 0, n - 2)

    def interp_axis(field, indices, coord, axis):
        if axis == 1:
            center = np.argmin(abs(z))
            frac = (coord - x[indices])/dx
            plus = (1-frac)*field[center, indices] + frac*field[center, indices + 1]
            mi = np.clip(np.floor((-coord - x[0])/dx).astype(int), 0, n - 2)
            mf = (-coord - x[mi])/dx
            minus = (1-mf)*field[center, mi] + mf*field[center, mi + 1]
        else:
            center = np.argmin(abs(x))
            frac = (coord - z[indices])/dz
            plus = (1-frac)*field[indices, center] + frac*field[indices + 1, center]
            mi = np.clip(np.floor((-coord - z[0])/dz).astype(int), 0, n - 2)
            mf = (-coord - z[mi])/dz
            minus = (1-mf)*field[mi, center] + mf*field[mi + 1, center]
        return 0.5*(plus - minus)

    xpos, zpos = samples*sx, samples*sz
    gx = np.dot(xpos, interp_axis(ex, ix, xpos, 1))/np.dot(xpos, xpos)
    gz = np.dot(zpos, interp_axis(ez, iz, zpos, 0))/np.dot(zpos, zpos)
    ex_gauss, _ = gaussian_field(xpos, np.zeros_like(xpos), sx, sz, 1.0, 1.0)
    _, ez_gauss = gaussian_field(np.zeros_like(zpos), zpos, sx, sz, 1.0, 1.0)
    gx_gauss = np.dot(xpos, ex_gauss)/np.dot(xpos, xpos)*EPS0
    gz_gauss = np.dot(zpos, ez_gauss)/np.dot(zpos, zpos)*EPS0
    return gx/gx_gauss, gz/gz_gauss


def toy_peak_density(turn):
    cache = np.load(LONG_CACHE)
    snap_turns = cache["snap_turns"].astype(int)
    s = cache["s_yes"][np.argmin(abs(snap_turns - turn))].astype(float)
    edges = np.linspace(s.min(), s.max(), N_SLICES + 1)
    counts = np.histogram(s, edges)[0]
    density = counts*(2.0e13*1.602176634e-19/len(s))/(edges[1] - edges[0])
    top = np.argsort(counts)[-N_TOP:]
    return np.average(density[top], weights=counts[top])


def analyze_one(block, turn, seo, sample_scale):
    phase = unwrap_to_short_arc(block[:, 3])
    edges = np.linspace(phase.min(), phase.max(), N_SLICES + 1)
    bins = np.clip(np.searchsorted(edges, phase, side="right") - 1, 0, N_SLICES - 1)
    counts = np.bincount(bins, minlength=N_SLICES)
    circumference = np.interp(block[:, 2].mean(), seo[:, 1], seo[:, 8])
    bunch_length = circumference*(phase.max() - phase.min())/(2*np.pi)
    x_all = np.empty(len(block))
    for slice_index in np.flatnonzero(counts):
        mask = bins == slice_index
        energy = block[mask, 2].mean()
        r0 = np.interp(energy, seo[:, 1], seo[:, 11])
        pr = np.interp(energy, seo[:, 1], seo[:, 12])
        x_all[mask] = (block[mask, 0] - r0)*np.sqrt(1.0 - pr*pr)
    global_sx, global_sz = x_all.std(), block[:, 1].std()
    selected = []
    for slice_index in np.argsort(counts)[-N_TOP:][::-1]:
        one = block[bins == slice_index]
        energy = one[:, 2].mean()
        r0 = np.interp(energy, seo[:, 1], seo[:, 11])
        pr = np.interp(energy, seo[:, 1], seo[:, 12])
        line_charge = counts[slice_index]*sample_scale*Q_MACRO*N_SLICES/bunch_length
        selected.append((counts[slice_index], line_charge, *fit_slice(one, line_charge, r0, pr)))
    selected = np.asarray(selected)
    avg = np.average(selected[:, 1:], axis=0, weights=selected[:, 0])
    sx, sz, gx, gz, gxg, gzg = avg[1:]
    ng_x, ng_z = numerical_gaussian_ratio(sx, sz, global_sx, global_sz)
    return (turn, bunch_length, *avg, gx/gxg, gz/gzg, ng_x, ng_z,
            global_sx, global_sz,
            toy_peak_density(turn) if turn % 10 == 0 else np.nan)


def analyze(history):
    seo = np.loadtxt(SEO_INI, skiprows=2)
    rows = []
    for record in range(N_RECORDS):
        turn = record + 10
        rows.append(analyze_one(history[:, record].astype(float), turn, seo, SAMPLE_SCALE))
        gx, gz, ng_x, ng_z = rows[-1][9:13]
        if turn % 10 == 0:
            print(f"turn {turn:3d}: G/Gauss=({gx:.4f},{gz:.4f}), "
                  f"grid/Gauss=({ng_x:.4f},{ng_z:.4f})", flush=True)
    return np.asarray(rows)


def analyze_full_checkpoints(turns=(1, 51, 101)):
    seo = np.loadtxt(SEO_INI, skiprows=2)
    rows = []
    for turn in turns:
        folder = FFT_DUMP.parent / f"TBT_Dump_{turn}"
        blocks = [np.load(path)["particles"][:, COLS]
                  for path in sorted(folder.glob("crossing_angle_*_rank_*.npz"))]
        block = np.concatenate(blocks).astype(float)
        order = np.argsort(block[:, 7], kind="stable")
        block = block[order]
        block = block[np.r_[True, np.diff(block[:, 7]) != 0]]
        row = analyze_one(block, turn, seo, 1.0)
        rows.append(row)
        print(f"full turn {turn}: N={len(block)}, G/Gauss=({row[9]:.4f},{row[10]:.4f})")
    return np.asarray(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = analyze(load_history())
    full_rows = analyze_full_checkpoints()
    header = ("turn,bunch_length_m,lambda_top5_C_per_m,sigma_x_m,sigma_z_m,"
              "gx_dump,gz_dump,gx_gaussian,gz_gaussian,gx_ratio,gz_ratio,"
              "gx_grid_ratio,gz_grid_ratio,global_sigma_x_m,global_sigma_z_m,"
              "lambda_toy_top5_C_per_m")
    np.savetxt(OUT / "turn_history.csv", rows, delimiter=",", header=header, comments="")
    np.savetxt(OUT / "full_checkpoint_history.csv", full_rows, delimiter=",",
               header=header, comments="")
    hann = np.hanning(len(rows))
    summary = {
        "plain_gx_ratio": np.mean(rows[:, 9]),
        "plain_gz_ratio": np.mean(rows[:, 10]),
        "hann_gx_ratio": np.average(rows[:, 9], weights=hann),
        "hann_gz_ratio": np.average(rows[:, 10], weights=hann),
        "grid_gx_ratio": np.mean(rows[:, 11]),
        "grid_gz_ratio": np.mean(rows[:, 12]),
    }
    sampled = np.isfinite(rows[:, 15])
    summary["lambda_3d_over_toy"] = np.mean(rows[sampled, 2]/rows[sampled, 15])
    reference = full_rows[1]
    factors = np.linspace(0.5, 2.0, 301)
    modeled = np.asarray([
        numerical_gaussian_ratio(reference[3], reference[4],
                                 reference[13], reference[14], kernel_aspect=f)
        for f in factors
    ])
    target = reference[9:11]
    best = np.argmin(np.sum((modeled - target)**2, axis=1))
    summary["best_cached_kernel_aspect_factor"] = factors[best]
    summary["best_cached_kernel_gx_ratio"] = modeled[best, 0]
    summary["best_cached_kernel_gz_ratio"] = modeled[best, 1]
    summary["turn1_global_sigma_x_over_z"] = full_rows[0, 13]/full_rows[0, 14]
    summary["turn51_global_sigma_x_over_z"] = full_rows[1, 13]/full_rows[1, 14]
    (OUT / "SUMMARY.txt").write_text("\n".join(f"{k}={v:.8f}" for k, v in summary.items()) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot(rows[:, 0], rows[:, 9], label=r"dump $G_x/G_{x,G}$")
    axes[0].plot(rows[:, 0], rows[:, 10], label=r"dump $G_z/G_{z,G}$")
    axes[0].plot(rows[:, 0], rows[:, 11], "--", label=r"128-grid $G_x/G_{x,G}$")
    axes[0].plot(rows[:, 0], rows[:, 12], "--", label=r"128-grid $G_z/G_{z,G}$")
    axes[0].axhline(1, color="black", lw=1)
    axes[0].set(xlabel="Turn", ylabel="core-gradient ratio", ylim=(0.55, 1.05))
    axes[0].legend(fontsize=8)
    axes[1].plot(rows[:, 0], rows[:, 2]*1e7, label="3-D dump reconstruction")
    axes[1].plot(rows[sampled, 0], rows[sampled, 15]*1e7, "o-", label="toy cache")
    axes[1].set(xlabel="Turn", ylabel=r"top-5 $\lambda$ [$10^{-7}$ C/m]")
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / "sc_kick_history.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUT / "sc_kick_history.pdf", bbox_inches="tight")
    plt.close(fig)
    print(summary)


if __name__ == "__main__":
    main()

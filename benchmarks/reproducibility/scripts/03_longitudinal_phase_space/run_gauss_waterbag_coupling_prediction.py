"""Tune-footprint prediction for HPC_dir/gauss_waterbag_coupling.

The configured bunch is transverse Gaussian / longitudinal waterbag:
300 MeV, 2.0e13 protons, 60 pi mm mrad RMS emittance, and a uniform
ellipse with full axes 72 ns x 8 MeV.  Identical longitudinal particles are
tracked with RF only and with RF + the 128-slice finite-disk LSC recipe.
Both transverse cases retain transverse space charge.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as REPO, SHARED_LONGITUDINAL
sys.path[:0] = [str(HERE), str(SHARED_LONGITUDINAL)]

from run_coupled_sc_toy import (
    OUT, SEO, N_CELLS, CELL_DEG, EK_MEV, MP_MEV, MP_KG, QE, EPS0,
    C_LIGHT, build_lattice, track, tune_fft, summarize,
)
from level5_bucket_sampled_tracking import load_seo_freq_table, rk4_step_numba, BETA_C
from level47_3d_recipe_in_1d import disk_sum_1d

N_PROTONS = 2.0e13
EMITTANCE_RMS = 60.0e-6
NS = 128
CACHE = OUT / "gauss_waterbag_longitudinal_pair.npz"


def _sigma_tag(value):
    return f"{float(value):g}".replace(".", "p")


def longitudinal_cache_path(density_smooth_sigma_bins=0.0):
    """Return a separate cache for each in-tracking density smoothing width."""
    sigma = float(density_smooth_sigma_bins)
    if sigma < 0.0:
        raise ValueError("density_smooth_sigma_bins must be non-negative")
    if sigma == 0.0:
        return CACHE
    return CACHE.with_name(
        f"{CACHE.stem}_density_smooth{_sigma_tag(sigma)}bin{CACHE.suffix}"
    )


def smooth_slice_counts(counts, sigma_bins=0.0):
    """Gaussian-smooth a uniform-grid line density while conserving charge."""
    sigma = float(sigma_bins)
    if sigma <= 0.0:
        return counts
    smoothed = gaussian_filter1d(
        np.asarray(counts, dtype=np.float64),
        sigma=sigma,
        mode="constant",
        cval=0.0,
        truncate=4.0,
    )
    raw_total = float(np.sum(counts))
    smooth_total = float(np.sum(smoothed))
    if smooth_total > 0.0:
        smoothed *= raw_total / smooth_total
    return smoothed


def initial_waterbag(n=20_000, seed=20260813):
    rng = np.random.default_rng(seed)
    radius = np.sqrt(rng.random(n))
    angle = 2*np.pi*rng.random(n)
    t_ns = 36.0*radius*np.cos(angle)
    de = 4.0*radius*np.sin(angle)
    return t_ns*1e-9, de


def longitudinal_pair(n_turns, substeps=64, density_smooth_sigma_bins=0.0):
    cache_path = longitudinal_cache_path(density_smooth_sigma_bins)
    OUT.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        d = np.load(cache_path)
        if int(d["snap_turns"][-1]) >= n_turns:
            return cache_path
    zd0, de0 = initial_waterbag()
    ek, freq = load_seo_freq_table(str(SEO / "SEO_ini.txt"))
    period = 1/freq
    tc = float(np.interp(EK_MEV, ek, period))
    etab = np.linspace(EK_MEV-30, EK_MEV+30, 20001)
    ttab = np.interp(etab, ek, period).astype(np.float64)
    emin, de_grid = etab[0], etab[1]-etab[0]
    dt = tc/substeps
    wanted = np.arange(0, n_turns+1, 10, dtype=int)
    if wanted[-1] != n_turns:
        wanted = np.r_[wanted, n_turns]
    wanted_set = set(map(int, wanted))
    result = {}
    for name, use_lsc in (("no", False), ("yes", True)):
        zd, de = zd0.copy(), de0.copy()
        ss, dd = [], []
        es = np.zeros_like(zd)
        qmacro = N_PROTONS*QE/len(zd)
        for turn in range(n_turns+1):
            if turn in wanted_set:
                ss.append((-BETA_C*zd).astype(np.float32))
                dd.append(de.astype(np.float32))
            if turn == n_turns:
                break
            for _ in range(substeps):
                if use_lsc:
                    s = -BETA_C*zd
                    lo, hi = float(s.min()), float(s.max())
                    sg = np.linspace(lo, hi, NS)
                    ds = sg[1]-sg[0]
                    ib = np.clip(((s-lo)/ds).astype(int), 0, NS-1)
                    counts = np.bincount(ib, minlength=NS)
                    counts_for_field = smooth_slice_counts(
                        counts, density_smooth_sigma_bins
                    )
                    es = disk_sum_1d(sg, counts_for_field*qmacro)[ib]
                    fsc = 1.0
                else:
                    es.fill(0.0)
                    fsc = 0.0
                zd, de = rk4_step_numba(
                    zd, de, dt, tc, 2*np.pi/tc, 0.0, 200e3,
                    emin, de_grid, ttab, fsc, BETA_C, es)
        result[f"s_{name}"] = np.stack(ss)
        result[f"de_{name}"] = np.stack(dd)
    np.savez_compressed(
        cache_path,
        snap_turns=wanted,
        substeps=substeps,
        density_smooth_sigma_bins=float(density_smooth_sigma_bins),
        **result,
    )
    return cache_path


def densities(cache_key, selected, n_turns, cache_path=CACHE):
    d = np.load(cache_path)
    st, snaps = d["snap_turns"], d[cache_key].astype(float)
    lam = np.empty((n_turns, len(selected)))
    sh = np.empty_like(lam)
    qtot = N_PROTONS*QE
    for turn in range(n_turns):
        k = np.clip(np.searchsorted(st, turn, side="right")-1, 0, len(st)-2)
        a = (turn-st[k])/(st[k+1]-st[k])
        s = (1-a)*snaps[k]+a*snaps[k+1]
        sh[turn] = s[selected]
        edges = np.linspace(s.min(), s.max(), NS+1)
        count, _ = np.histogram(s, bins=edges)
        rho = count*(qtot/len(s))/(edges[1]-edges[0])
        ib = np.clip(np.searchsorted(edges, sh[turn], side="right")-1, 0, NS-1)
        lam[turn] = rho[ib]
    return lam, sh


def smooth_grid(qr, qz, xr, yr):
    h, xe, ye = np.histogram2d(qr, qz, bins=90, range=(xr, yr))
    return gaussian_filter(h, 1.25).T, xe, ye


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=768)
    ap.add_argument("--turns", type=int, default=240)
    ap.add_argument("--steps-per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=22222)
    ap.add_argument(
        "--density-smooth-sigma-bins",
        type=float,
        default=0.0,
        help="Gaussian sigma for the 1-D line density in current slice widths",
    )
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tic = time.perf_counter()
    cache_path = longitudinal_pair(
        args.turns,
        density_smooth_sigma_bins=args.density_smooth_sigma_bins,
    )
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(20_000, args.particles, replace=False)
    lam_no, s_no = densities("s_no", selected, args.turns, cache_path)
    lam_yes, s_yes = densities("s_yes", selected, args.turns, cache_path)

    lat = build_lattice()
    X, P, Z, PZ = rng.normal(size=(4,args.particles))*np.sqrt(EMITTANCE_RMS)
    br0, bz0 = lat["beta_r"][0], lat["beta_z"][0]
    ar0, az0 = lat["alpha_r"][0], lat["alpha_z"][0]
    x0=np.sqrt(br0)*X; xp0=(P-ar0*X)/np.sqrt(br0)
    z0=np.sqrt(bz0)*Z; zp0=(PZ-az0*Z)/np.sqrt(bz0)
    sr=np.sqrt(lat["beta_r"]*EMITTANCE_RMS)
    sz=np.sqrt(lat["beta_z"]*EMITTANCE_RMS)
    gamma=1+EK_MEV/MP_MEV; beta=np.sqrt(1-gamma**-2)
    accel=QE/(gamma*MP_KG*beta**2*C_LIGHT**2)/(4*np.pi*EPS0*gamma**2)
    gx, gw=np.polynomial.legendre.leggauss(12); gs=.5*(gx+1); gw=.5*gw
    th=np.linspace(0,np.deg2rad(CELL_DEG),args.steps_per_cell+1)
    result={}
    for name, lam in (("no_longitudinal_sc",lam_no),("with_longitudinal_sc",lam_yes)):
        print("transverse tracking", name, flush=True)
        tr=track(x0,xp0,z0,zp0,lam,th,lat["theta"][0],
                 lat["theta"][1]-lat["theta"][0],lat["h"],lat["kr"],lat["kz"],
                 sr,sz,gs,gw,accel,N_CELLS)
        qr=tune_fft(tr[0],tr[1],br0,ar0,lat["qr"])
        qz=tune_fft(tr[2],tr[3],bz0,az0,lat["qz"])
        result[name]=(qr,qz)
        print(name, "Qr", summarize(qr,lat["qr"]), "Qz", summarize(qz,lat["qz"]))

    stem=OUT/"tune_2d_gauss_waterbag_coupling"
    np.savez_compressed(stem.with_suffix(".npz"),
        qr_no_lsc=result["no_longitudinal_sc"][0], qz_no_lsc=result["no_longitudinal_sc"][1],
        qr_lsc=result["with_longitudinal_sc"][0], qz_lsc=result["with_longitudinal_sc"][1],
        s_no_lsc=s_no.astype(np.float32), s_lsc=s_yes.astype(np.float32),
        qr_bare=lat["qr"], qz_bare=lat["qz"])
    allr=np.r_[result["no_longitudinal_sc"][0],result["with_longitudinal_sc"][0]]
    allz=np.r_[result["no_longitudinal_sc"][1],result["with_longitudinal_sc"][1]]
    xr=tuple(np.percentile(allr,[.3,99.7])+[-.004,.004])
    yr=tuple(np.percentile(allz,[.3,99.7])+[-.004,.004])
    fig, axes=plt.subplots(1,3,figsize=(15,4.8),sharex=True,sharey=True)
    colors={"no_longitudinal_sc":"tab:orange","with_longitudinal_sc":"tab:blue"}
    labels={"no_longitudinal_sc":"transverse SC + RF (LSC off)",
            "with_longitudinal_sc":"transverse SC + RF + LSC"}
    grids={n:smooth_grid(qr,qz,xr,yr) for n,(qr,qz) in result.items()}
    vmax=max(v[0].max() for v in grids.values())
    for ax,(name,(qr,qz)) in zip(axes[:2],result.items()):
        h,xe,ye=grids[name]
        ax.imshow(h,origin="lower",extent=[xe[0],xe[-1],ye[0],ye[-1]],
                  aspect="auto",cmap="viridis",vmin=0,vmax=vmax)
        ax.scatter(qr,qz,s=3,c="white",alpha=.18,linewidths=0)
        ax.set_title(labels[name])
    for name,(qr,qz) in result.items():
        h,xe,ye=grids[name]; vals=np.sort(h.ravel())[::-1]
        cum=np.cumsum(vals)/vals.sum()
        lev=sorted(vals[np.searchsorted(cum,f)] for f in (.95,.68))
        axes[2].contour((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2,h,
                        levels=lev,colors=colors[name],linewidths=(1.2,2.0))
        axes[2].plot([],[],color=colors[name],lw=2,label=labels[name])
    axes[2].axvline(lat["qr"],color=".4",ls="--",lw=1)
    axes[2].axhline(lat["qz"],color=".4",ls="--",lw=1)
    axes[2].set_title("68% and 95% density contours"); axes[2].legend(fontsize=8)
    for ax in axes: ax.set_xlabel(r"$Q_r$"); ax.grid(alpha=.15)
    axes[0].set_ylabel(r"$Q_z$")
    fig.suptitle(r"gauss_waterbag_coupling theory: 300 MeV, $N=2.0\times10^{13}$, "
                 r"$\epsilon_{r,z}=60\pi$ mm mrad")
    fig.tight_layout(); fig.savefig(stem.with_suffix(".png"),dpi=240); plt.close(fig)
    keys=list(summarize(result["no_longitudinal_sc"][0],lat["qr"]).keys())
    with open(stem.with_name(stem.name+"_summary.csv"),"w",encoding="utf-8") as f:
        f.write("case,plane,"+",".join(keys)+"\n")
        for name,(qr,qz) in result.items():
            for plane,q,qb in (("r",qr,lat["qr"]),("z",qz,lat["qz"])):
                d=summarize(q,qb); f.write(name+","+plane+","+",".join(str(d[k]) for k in keys)+"\n")
    print(f"done in {time.perf_counter()-tic:.1f}s; {stem}")


if __name__ == "__main__":
    main()

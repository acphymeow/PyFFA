"""Weakly self-consistent transverse + longitudinal space-charge toy model.

Longitudinal input
------------------
The script reads the validated level47 f_sc=0.7 cache.  That cache was
generated with RF motion plus the same 128-slice finite-disk longitudinal
field recipe used by the production 2.5-D solver.  Particle ordering is
preserved between snapshots, so s(t) and the local line density can be
reconstructed for each toy particle.

Transverse model
----------------
Matched 4-D Gaussian particles are tracked through Hill equations built from
the 300-MeV SEO/Twiss tables.  The nonlinear electric field of an elliptical
2-D Gaussian beam is evaluated along the real periodic lattice.  Its strength
is the local, turn-dependent line density lambda(s_i, turn).  The transverse
RMS envelope is frozen at the matched 60 pi mm mrad value; consequently this
is a weakly self-consistent model, not a replacement for the 6-D PIC solver.

Three cases are evaluated with identical transverse particles:
  bare     : no transverse SC
  frozen   : transverse SC with the turn-0 longitudinal density frozen
  dynamic  : transverse SC driven by RF + longitudinal-SC motion

Outputs are written beside this script under outputs/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.interpolate import CubicSpline


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as REPO
OUT = HERE / "outputs"

SEO = REPO / "Bmap_FD16" / "resultsSEO_noError_X"
LONG_CACHE = HERE / "outputs" / "level47_1d_cache_recipe3d_f0p7.npz"

N_CELLS = 16
CELL_DEG = 360.0 / N_CELLS
EK_MEV = 300.0
MP_MEV = 938.2720813
MP_KG = 1.67262192369e-27
QE = 1.602176634e-19
EPS0 = 8.8541878128e-12
C_LIGHT = 299792458.0
N_PROTONS_NOMINAL = 2.08e13
F_SC = 0.7
EMITTANCE_RMS = 60.0e-6       # geometric RMS emittance [m rad]
N_LONG_BINS = 128


def periodic_full(phi, y, xnew):
    phi = np.mod(np.asarray(phi, float), 2 * np.pi)
    order = np.argsort(phi)
    phi, y = phi[order], np.asarray(y, float)[order]
    phi, idx = np.unique(phi, return_index=True)
    y = y[idx]
    if abs(phi[0]) > 1e-13:
        phi, y = np.r_[0.0, phi], np.r_[y[-1], y]
    cs = CubicSpline(np.r_[phi, 2 * np.pi], np.r_[y, y[0]],
                     bc_type="periodic")
    return cs(np.mod(xnew, 2 * np.pi)), cs.derivative()(np.mod(xnew, 2*np.pi))


def periodic_cell_deg(x, y, xnew):
    x = np.mod(np.asarray(x, float), CELL_DEG)
    order = np.argsort(x)
    x, y = x[order], np.asarray(y, float)[order]
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    if abs(x[0]) > 1e-11:
        x, y = np.r_[0.0, x], np.r_[y[-1], y]
    cs = CubicSpline(np.r_[x, CELL_DEG], np.r_[y, y[0]],
                     bc_type="periodic")
    return cs(np.mod(xnew, CELL_DEG))


def periodic_resample(s, y, length, n):
    order = np.argsort(s)
    s, y = np.asarray(s)[order], np.asarray(y)[order]
    cs = CubicSpline(np.r_[s, length], np.r_[y, y[0]],
                     bc_type="periodic")
    su = np.linspace(0, length, n, endpoint=False)
    return su, cs(su)


def smooth_derivatives(y, length, cutoff=0.08, order=10):
    n = len(y)
    ds = length / n
    k = 2*np.pi*np.fft.fftfreq(n, d=ds)
    filt = np.exp(-(np.abs(k)/(cutoff*np.pi/ds))**order)
    Y = np.fft.fft(y)*filt
    return (np.fft.ifft(Y).real,
            np.fft.ifft(1j*k*Y).real,
            np.fft.ifft((1j*k)**2*Y).real)


def build_lattice(n_grid=300):
    sr = np.loadtxt(SEO / "SEO_r.txt", skiprows=1)
    fi, eks, rmat = sr[0, 1:], sr[1:, 0], sr[1:, 1:]
    br = np.loadtxt(SEO / "BetaFuncR.txt", skiprows=1)
    bz = np.loadtxt(SEO / "BetaFuncZ.txt", skiprows=1)
    beta_phi, beta_eks = br[0, 1:], br[1:, 0]
    theta_deg = np.linspace(0, CELL_DEG, n_grid, endpoint=False)
    theta = np.deg2rad(theta_deg)
    dtheta = theta[1]-theta[0]

    r_by_phi = np.array([np.interp(EK_MEV, eks, rmat[:, j])
                         for j in range(rmat.shape[1])])
    r, rp = periodic_full(fi, r_by_phi, theta)
    h = np.sqrt(r*r + rp*rp)
    ds_seg = 0.5*(h + np.roll(h, -1))*dtheta
    s = np.zeros(n_grid)
    s[1:] = np.cumsum(ds_seg[:-1])
    length = ds_seg.sum()

    beta_r_phi = np.array([np.interp(EK_MEV, beta_eks, br[1:, j+1])
                           for j in range(br.shape[1]-1)])
    beta_z_phi = np.array([np.interp(EK_MEV, beta_eks, bz[1:, j+1])
                           for j in range(bz.shape[1]-1)])
    beta_r_raw = periodic_cell_deg(beta_phi, beta_r_phi, theta_deg)
    beta_z_raw = periodic_cell_deg(beta_phi, beta_z_phi, theta_deg)
    su, beta_r_u = periodic_resample(s, beta_r_raw, length, n_grid)
    _, beta_z_u = periodic_resample(s, beta_z_raw, length, n_grid)

    beta_r_u, brp, brpp = smooth_derivatives(beta_r_u, length)
    beta_z_u, bzp, bzpp = smooth_derivatives(beta_z_u, length)
    alpha_r_u, alpha_z_u = -0.5*brp, -0.5*bzp
    gamma_r_u = (1 + alpha_r_u**2)/beta_r_u
    gamma_z_u = (1 + alpha_z_u**2)/beta_z_u
    kr_u = (-0.5*brpp + gamma_r_u)/beta_r_u
    kz_u = (-0.5*bzpp + gamma_z_u)/beta_z_u

    def back(y):
        cs = CubicSpline(np.r_[su, length], np.r_[y, y[0]],
                         bc_type="periodic")
        return cs(s)

    beta_r, beta_z = back(beta_r_u), back(beta_z_u)
    alpha_r, alpha_z = back(alpha_r_u), back(alpha_z_u)
    kr, kz = back(kr_u), back(kz_u)
    qr = N_CELLS*np.sum(1/beta_r_u)*(length/n_grid)/(2*np.pi)
    qz = N_CELLS*np.sum(1/beta_z_u)*(length/n_grid)/(2*np.pi)
    return dict(theta=theta, h=h, kr=kr, kz=kz,
                beta_r=beta_r, beta_z=beta_z,
                alpha_r=alpha_r, alpha_z=alpha_z,
                length_cell=length, qr=qr, qz=qz)


def longitudinal_line_density(n_turns, n_particles, seed):
    """Return local lambda [C/m], s histories and selected cache indices."""
    d = np.load(LONG_CACHE)
    snap_turns = d["snap_turns"].astype(int)
    s_snap = d["s_snap"].astype(float)
    if n_turns > snap_turns[-1] + 1:
        raise ValueError(f"cache only covers turn {snap_turns[-1]}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(s_snap.shape[1], n_particles, replace=False)
    turns = np.arange(n_turns)
    # Interpolate each longitudinal trajectory between the 10-turn snapshots.
    s_selected = np.empty((n_turns, n_particles))
    for j, ii in enumerate(idx):
        s_selected[:, j] = np.interp(turns, snap_turns, s_snap[:, ii])

    lam = np.empty_like(s_selected)
    total_charge = F_SC*N_PROTONS_NOMINAL*QE
    for it, turn in enumerate(turns):
        # Density uses all 20k longitudinal macro-particles, not only toy subset.
        k = np.searchsorted(snap_turns, turn, side="right") - 1
        k = np.clip(k, 0, len(snap_turns)-2)
        a = (turn-snap_turns[k])/(snap_turns[k+1]-snap_turns[k])
        sall = (1-a)*s_snap[k] + a*s_snap[k+1]
        lo, hi = float(sall.min()), float(sall.max())
        edges = np.linspace(lo, hi, N_LONG_BINS+1)
        counts, _ = np.histogram(sall, bins=edges)
        ds = edges[1]-edges[0]
        density = counts*(total_charge/len(sall))/ds
        ib = np.clip(np.searchsorted(edges, s_selected[it], side="right")-1,
                     0, N_LONG_BINS-1)
        lam[it] = density[ib]
    return lam, s_selected, idx


@njit(cache=True)
def interp_periodic(th, th0, dth, arr):
    n = len(arr)
    x = ((th-th0) % (n*dth))/dth
    i = int(np.floor(x)) % n
    a = x-np.floor(x)
    return (1-a)*arr[i] + a*arr[(i+1) % n]


@njit(cache=True)
def gaussian_field_unit(x, z, sx, sz, qs, qw):
    """Elliptical Gaussian field integral without lambda/(4 pi eps0 gamma^2)."""
    ex = 0.0
    ez = 0.0
    sx2, sz2 = sx*sx, sz*sz
    t0 = max(sx2, sz2)
    for iq in range(len(qs)):
        u = qs[iq]
        t = t0*u/(1-u)
        dt = t0/((1-u)*(1-u))
        ax, az = sx2+t, sz2+t
        root = np.sqrt(ax*az)
        e = np.exp(-0.5*(x*x/ax + z*z/az))
        ex += qw[iq]*dt*x/(ax*root)*e
        ez += qw[iq]*dt*z/(az*root)*e
    return ex, ez


@njit(cache=True)
def rhs(th, x, xp, z, zp, th0, dth, h, kr, kz, sr, sz,
        qs, qw, lambda_local, accel):
    hh = interp_periodic(th, th0, dth, h)
    kx = interp_periodic(th, th0, dth, kr)
    ky = interp_periodic(th, th0, dth, kz)
    sx = interp_periodic(th, th0, dth, sr)
    sy = interp_periodic(th, th0, dth, sz)
    ex, ez = gaussian_field_unit(x, z, sx, sy, qs, qw)
    f = accel*lambda_local
    return hh*xp, hh*(-kx*x + f*ex), hh*zp, hh*(-ky*z + f*ez)


@njit(cache=True)
def rhs_with_dispersion(th, xb, xbp, z, zp, th0, dth, h, kr, kz,
                        sr_beta, sz, dx, qs, qw, lambda_local, accel,
                        delta_ek_local, sigma_ek_slice):
    """Betatron equations with the physical x reconstructed from dispersion."""
    hh = interp_periodic(th, th0, dth, h)
    kx = interp_periodic(th, th0, dth, kr)
    ky = interp_periodic(th, th0, dth, kz)
    sx_beta = interp_periodic(th, th0, dth, sr_beta)
    sy = interp_periodic(th, th0, dth, sz)
    dx_here = interp_periodic(th, th0, dth, dx)
    x_physical = xb + dx_here * delta_ek_local
    sx = np.sqrt(sx_beta*sx_beta
                 + (dx_here*sigma_ek_slice)*(dx_here*sigma_ek_slice))
    ex, ez = gaussian_field_unit(x_physical, z, sx, sy, qs, qw)
    f = accel*lambda_local
    return hh*xbp, hh*(-kx*xb + f*ex), hh*zp, hh*(-ky*z + f*ez)


@njit(parallel=True, cache=True)
def track(x0, xp0, z0, zp0, lambda_turn, theta_steps, th0, dth_grid,
          h, kr, kz, sr, sz, qs, qw, accel, n_cells):
    nt, npart = lambda_turn.shape
    xo = np.empty((nt, npart)); xpo = np.empty_like(xo)
    zo = np.empty_like(xo); zpo = np.empty_like(xo)
    for ip in prange(npart):
        x, xp, z, zp = x0[ip], xp0[ip], z0[ip], zp0[ip]
        for it in range(nt):
            ll = lambda_turn[it, ip]
            for _ in range(n_cells):
                for j in range(len(theta_steps)-1):
                    th = theta_steps[j]
                    dt = theta_steps[j+1]-th
                    k1 = rhs(th,x,xp,z,zp,th0,dth_grid,h,kr,kz,sr,sz,qs,qw,ll,accel)
                    k2 = rhs(th+dt/2,x+dt*k1[0]/2,xp+dt*k1[1]/2,
                             z+dt*k1[2]/2,zp+dt*k1[3]/2,
                             th0,dth_grid,h,kr,kz,sr,sz,qs,qw,ll,accel)
                    k3 = rhs(th+dt/2,x+dt*k2[0]/2,xp+dt*k2[1]/2,
                             z+dt*k2[2]/2,zp+dt*k2[3]/2,
                             th0,dth_grid,h,kr,kz,sr,sz,qs,qw,ll,accel)
                    k4 = rhs(th+dt,x+dt*k3[0],xp+dt*k3[1],z+dt*k3[2],zp+dt*k3[3],
                             th0,dth_grid,h,kr,kz,sr,sz,qs,qw,ll,accel)
                    x += dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
                    xp += dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
                    z += dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
                    zp += dt*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6
            xo[it,ip], xpo[it,ip] = x, xp
            zo[it,ip], zpo[it,ip] = z, zp
    return xo, xpo, zo, zpo


@njit(parallel=True, cache=True)
def track_with_dispersion(
        xb0, xbp0, z0, zp0, lambda_turn, delta_ek_turn,
        sigma_ek_slice_turn, theta_steps, th0, dth_grid,
        h, kr, kz, sr_beta, sz, dx, qs, qw, accel, n_cells):
    """Track betatron coordinates while evaluating SC at x_beta+Dx*dEk."""
    nt, npart = lambda_turn.shape
    xo = np.empty((nt, npart)); xpo = np.empty_like(xo)
    zo = np.empty_like(xo); zpo = np.empty_like(xo)
    for ip in prange(npart):
        xb, xbp, z, zp = xb0[ip], xbp0[ip], z0[ip], zp0[ip]
        for it in range(nt):
            ll = lambda_turn[it, ip]
            dek = delta_ek_turn[it, ip]
            sek = sigma_ek_slice_turn[it, ip]
            for _ in range(n_cells):
                for j in range(len(theta_steps)-1):
                    th = theta_steps[j]
                    dt = theta_steps[j+1]-th
                    k1 = rhs_with_dispersion(
                        th, xb, xbp, z, zp, th0, dth_grid, h, kr, kz,
                        sr_beta, sz, dx, qs, qw, ll, accel, dek, sek)
                    k2 = rhs_with_dispersion(
                        th+dt/2, xb+dt*k1[0]/2, xbp+dt*k1[1]/2,
                        z+dt*k1[2]/2, zp+dt*k1[3]/2,
                        th0, dth_grid, h, kr, kz, sr_beta, sz, dx,
                        qs, qw, ll, accel, dek, sek)
                    k3 = rhs_with_dispersion(
                        th+dt/2, xb+dt*k2[0]/2, xbp+dt*k2[1]/2,
                        z+dt*k2[2]/2, zp+dt*k2[3]/2,
                        th0, dth_grid, h, kr, kz, sr_beta, sz, dx,
                        qs, qw, ll, accel, dek, sek)
                    k4 = rhs_with_dispersion(
                        th+dt, xb+dt*k3[0], xbp+dt*k3[1],
                        z+dt*k3[2], zp+dt*k3[3],
                        th0, dth_grid, h, kr, kz, sr_beta, sz, dx,
                        qs, qw, ll, accel, dek, sek)
                    xb += dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
                    xbp += dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
                    z += dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
                    zp += dt*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6
            xo[it, ip], xpo[it, ip] = xb, xbp
            zo[it, ip], zpo[it, ip] = z, zp
    return xo, xpo, zo, zpo


def tune_fft(x, p, beta, alpha, expected, pad=16, half_width=0.25):
    X = x/np.sqrt(beta)
    P = (alpha*x + beta*p)/np.sqrt(beta)
    u = (X-1j*P)
    u = (u-u.mean(axis=0))*np.hanning(len(u))[:,None]
    nfft = pad*len(u)
    spec = np.abs(np.fft.fft(u, n=nfft, axis=0))
    freq = np.fft.fftfreq(nfft)
    qfrac0 = expected % 1.0
    # The normalized complex coordinate rotates at a signed FFT frequency.
    # Represent fractional tunes above 0.5 by the equivalent negative value.
    q0 = qfrac0 if qfrac0 <= 0.5 else qfrac0-1.0
    # The full-intensity 80-pi case in test_0605_2 reaches approximately
    # Qr=3.077 and Qz=2.640, i.e. 0.16--0.18 below the bare tunes.  A 0.25
    # search half-width avoids artificial clipping at the former 0.13 edge.
    mask = np.abs(freq-q0) < half_width
    ids = np.flatnonzero(mask)
    out = np.empty(u.shape[1])
    for j in range(u.shape[1]):
        k = ids[np.argmax(spec[ids,j])]
        if 0 < k < nfft-1:
            y0,y1,y2 = spec[k-1,j],spec[k,j],spec[k+1,j]
            den = y0-2*y1+y2
            dk = 0.5*(y0-y2)/den if den != 0 else 0.0
            dk = min(0.5, max(-0.5, dk))
        else:
            dk = 0.0
        f_peak = freq[k] + dk/nfft
        out[j] = expected + (f_peak-q0)
    return out


def summarize(q, qbare):
    dq = q-qbare
    p = np.percentile(q, [1,5,16,50,84,95,99])
    return dict(mean=float(q.mean()), std=float(q.std()),
                p01=float(p[0]), p05=float(p[1]), p16=float(p[2]),
                median=float(p[3]), p84=float(p[4]), p95=float(p[5]),
                p99=float(p[6]), mean_shift=float(dq.mean()),
                spread_90=float(p[5]-p[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=384)
    ap.add_argument("--turns", type=int, default=220)
    ap.add_argument("--steps-per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    lat = build_lattice()
    lam, s_hist, long_idx = longitudinal_line_density(args.turns,
                                                       args.particles,
                                                       args.seed)
    rng = np.random.default_rng(args.seed+1)
    X, P, Z, PZ = rng.normal(size=(4,args.particles))*np.sqrt(EMITTANCE_RMS)
    b0r,b0z = lat["beta_r"][0],lat["beta_z"][0]
    a0r,a0z = lat["alpha_r"][0],lat["alpha_z"][0]
    x0=np.sqrt(b0r)*X; xp0=(P-a0r*X)/np.sqrt(b0r)
    z0=np.sqrt(b0z)*Z; zp0=(PZ-a0z*Z)/np.sqrt(b0z)
    sig_r=np.sqrt(lat["beta_r"]*EMITTANCE_RMS)
    sig_z=np.sqrt(lat["beta_z"]*EMITTANCE_RMS)
    gx=1+EK_MEV/MP_MEV
    br=np.sqrt(1-1/gx**2)
    # field = lambda/(4 pi eps0 gamma^2)*integral; transverse acceleration
    # factor is q/(gamma m beta^2 c^2).
    accel = QE/(gx*MP_KG*br**2*C_LIGHT**2)/(4*np.pi*EPS0*gx**2)
    qx,qw=np.polynomial.legendre.leggauss(10)
    qs=0.5*(qx+1); qw=0.5*qw
    th_steps=np.linspace(0,np.deg2rad(CELL_DEG),args.steps_per_cell+1)

    cases = {
        "bare": np.zeros_like(lam),
        "frozen": np.broadcast_to(lam[0], lam.shape).copy(),
        "dynamic": lam,
    }
    result={}
    for name, ll in cases.items():
        print(f"tracking {name}: {args.particles} particles x {args.turns} turns")
        tr=track(x0,xp0,z0,zp0,ll,th_steps,lat["theta"][0],
                 lat["theta"][1]-lat["theta"][0],lat["h"],lat["kr"],lat["kz"],
                 sig_r,sig_z,qs,qw,accel,N_CELLS)
        qr=tune_fft(tr[0],tr[1],b0r,a0r,lat["qr"])
        qz=tune_fft(tr[2],tr[3],b0z,a0z,lat["qz"])
        result[name]=(qr,qz)
        print(name,"Qr",summarize(qr,lat["qr"]),"Qz",summarize(qz,lat["qz"]))

    rows=[]
    for name,(qr,qz) in result.items():
        for plane,q,qb in [("r",qr,lat["qr"]),("z",qz,lat["qz"])]:
            d=summarize(q,qb)
            rows.append([name,plane]+[d[k] for k in d])
    keys=list(summarize(result["bare"][0],lat["qr"]).keys())
    with open(OUT/"tune_summary.csv","w",encoding="utf-8") as f:
        f.write("case,plane,"+",".join(keys)+"\n")
        for row in rows: f.write(",".join(map(str,row))+"\n")
    np.savez_compressed(OUT/"coupled_sc_toy_results.npz",
        qr_bare=result["bare"][0],qz_bare=result["bare"][1],
        qr_frozen=result["frozen"][0],qz_frozen=result["frozen"][1],
        qr_dynamic=result["dynamic"][0],qz_dynamic=result["dynamic"][1],
        s_history=s_hist.astype(np.float32),lambda_history=lam.astype(np.float32),
        long_indices=long_idx,qr_lattice=lat["qr"],qz_lattice=lat["qz"],
        emittance_rms=EMITTANCE_RMS,f_sc=F_SC)

    fig,ax=plt.subplots(2,2,figsize=(10,7.5))
    colors={"bare":"0.35","frozen":"tab:orange","dynamic":"tab:blue"}
    for j,(plane,qb) in enumerate([("r",lat["qr"]),("z",lat["qz"])]):
        ax[0,j].axvline(result["bare"][j].mean()-qb, color=colors["bare"],
                        lw=1.5, label="bare")
        pooled=[]
        for name,(qr,qz) in result.items():
            q=qr if plane=="r" else qz
            if name == "bare":
                continue
            ax[0,j].hist(q-qb,bins=45,histtype="step",density=True,
                         lw=1.5,color=colors[name],label=name)
            pooled.append(q-qb)
        pooled=np.concatenate(pooled)
        lo,hi=np.percentile(pooled,[0.5,99.5])
        ax[0,j].set_xlim(min(lo,-0.005),max(hi,0.005))
        ax[0,j].set_xlabel(rf"$\Delta Q_{plane}$")
        ax[0,j].set_ylabel("density")
        ax[0,j].legend(frameon=False)
        qdyn=result["dynamic"][j]
        ax[1,j].scatter(s_hist[0],qdyn-qb,s=5,alpha=.35,color="tab:blue")
        ax[1,j].set_xlabel(r"initial $s$ [m]")
        ax[1,j].set_ylabel(rf"dynamic $\Delta Q_{plane}$")
    fig.suptitle("Transverse tune footprint: RF + longitudinal SC driven toy (f=0.7)")
    fig.tight_layout()
    fig.savefig(OUT/"coupled_sc_tune_spread.png",dpi=220)
    plt.close(fig)

    fig,ax=plt.subplots(2,1,figsize=(8,6),sharex=True)
    ax[0].plot(np.std(s_hist,axis=1)); ax[0].set_ylabel(r"$\sigma_s$ [m]")
    ax[1].plot(np.mean(lam,axis=1),label="particle mean")
    ax[1].plot(np.max(lam,axis=1),label="sample max",alpha=.8)
    ax[1].set_ylabel(r"$\lambda$ [C/m]"); ax[1].set_xlabel("turn")
    ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT/"longitudinal_driver.png",dpi=220); plt.close(fig)
    print(f"elapsed {time.perf_counter()-t_start:.1f} s; outputs: {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct measurement of the RF voltage delivered by an Emap through the
PRODUCTION interpolation path (FFAG_EField_spiral.Interpolation2D_EMap).

Method: place test points on a dense azimuth grid at a given radius
(z=0), evaluate Ephi with v0_volt=V0, then integrate

    V_gap = \int E_phi(r, fi) * r * dfi

over each gap's hit region (HitGapIndex selects the gap).
This is exactly the energy gain per gap crossing at RF crest
(sin_RF = 1).  Sum over gaps = V_turn at crest.

The 3D runs (Longitudinal_0p0/0p5) nominally used 2 gaps x 100 kV,
but dump analysis gives f_s ~ 9.8 kHz <-> V_eff ~ 91-95 kV.

Run:  python tools/tool_check_emap_voltage.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from FFAG_Field import FFAG_EField_spiral, FFAG_EField_new

# ================================================================
# CONFIG — edit here
# ================================================================

FREQ_CURVE = ("HPC_dir/Longitudinal/Emaps/FrequencyCurve_LSC.txt")
V0_NOMINAL = 100e3          # [V] per gap, from acc_voltage_paint in header
R_TEST = [11.1867, 11.29, 11.4]   # [m] test radii (300 MeV closed orbit ~11.29)
N_FI = 20001                # azimuth samples around the ring
OUT_FIG = Path("outputs") / "tool_check_emap_voltage.png"

# ================================================================


def measure_voltage(efield, r_test, v0, n_fi=N_FI):
    fi = np.linspace(0.0, 2.0 * np.pi, n_fi, endpoint=False)
    dfi = fi[1] - fi[0]
    r = np.full(n_fi, r_test)
    z = np.zeros(n_fi)
    Er = np.zeros(n_fi)
    Ez = np.zeros(n_fi)
    Efi = np.zeros(n_fi)
    hit = np.full(n_fi, -1, dtype=np.int64)

    efield.Interpolation2D_EMap(r, fi, z, Er, Ez, Efi, hit, v0_volt=v0)

    gaps = {}
    for igap, az in enumerate(efield.gap_azimuths):
        m = hit == igap
        if np.any(m):
            v_gap = float(np.sum(Efi[m] * r[m]) * dfi)   # int E*dl
            gaps[igap] = (np.rad2deg(az), v_gap, int(np.sum(m)))
    return fi, Efi, hit, gaps


def main():
    # ---- analytical (fixed-gap) class: what main_track.py ACTUALLY uses ----
    print("=" * 60)
    print("FFAG_EField_new (analytical/fixed) — production class in main_track")
    print("=" * 60)
    efield_fix = FFAG_EField_new(FREQ_CURVE, True)
    for r_t in [11.29]:
        fi, Efi, hit, gaps = measure_voltage(efield_fix, r_t, V0_NOMINAL)
        v_tot = sum(g[1] for g in gaps.values())
        print(f"r = {r_t:.4f} m:")
        for igap, (az_deg, v_gap, npt) in gaps.items():
            print(f"  gap {igap} (azimuth {az_deg:6.1f} deg): "
                  f"V_gap = {v_gap/1e3:8.2f} kV  ({npt} samples)")
        print(f"  --> V_turn(crest) = {v_tot/1e3:8.2f} kV   "
              f"(nominal 2 x {V0_NOMINAL/1e3:.0f} = "
              f"{2*V0_NOMINAL/1e3:.0f} kV)")

    # ---- spiral class: intended class for this map ----
    print("\n" + "=" * 60)
    print("FFAG_EField_spiral — intended class for this map")
    print("=" * 60)
    efield = FFAG_EField_spiral(FREQ_CURVE, True)
    print("=== Emap header ===")
    print(f"gap_azimuths (deg) = {np.rad2deg(efield.gap_azimuths)}")
    print(f"rf_shift (deg)     = {np.rad2deg(efield.rf_shift)}")
    print(f"gap_width          = {efield.gap_width}")
    print(f"acc_Etot_uniform   = {efield.acc_Etot_uniform}")
    print(f"harmonic           = {efield.harmonic}")
    print(f"map r range        = [{efield.map_rmin:.4f}, "
          f"{efield.map_rmax:.4f}] m, nr={efield.nr}, dr={efield.dr:.5f}")
    print(f"map phi range      = [{efield.map_phimin:.6f}, "
          f"{efield.map_phimax:.6f}] rad "
          f"([{np.rad2deg(efield.map_phimin):.3f}, "
          f"{np.rad2deg(efield.map_phimax):.3f}] deg), "
          f"nphi={efield.nphi}, dphi={efield.dphi:.6e}")
    print(f"Ephi map peak      = {np.max(np.abs(efield.Ephi_map)):.6f}")
    print(f"V0 in freq curve   = {efield.freq_curve[0,2]:.3e} V "
          f"(first row)")

    print("\n=== delivered voltage (crest, sin_RF=1) ===")
    results = {}
    for r_t in R_TEST:
        fi, Efi, hit, gaps = measure_voltage(efield, r_t, V0_NOMINAL)
        v_tot = sum(g[1] for g in gaps.values())
        results[r_t] = (fi, Efi, hit, gaps, v_tot)
        print(f"\nr = {r_t:.4f} m:")
        for igap, (az_deg, v_gap, npt) in gaps.items():
            print(f"  gap {igap} (azimuth {az_deg:6.1f} deg): "
                  f"V_gap = {v_gap/1e3:8.2f} kV  ({npt} samples)")
        print(f"  --> V_turn(crest) = {v_tot/1e3:8.2f} kV   "
              f"(nominal 2 x {V0_NOMINAL/1e3:.0f} = "
              f"{2*V0_NOMINAL/1e3:.0f} kV)")

    # ---- figure: Ephi vs fi at the closed-orbit radius ----
    r_show = R_TEST[1] if len(R_TEST) > 1 else R_TEST[0]
    fi, Efi, hit, gaps, v_tot = results[r_show]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(np.rad2deg(fi), Efi / 1e3, lw=1.0)
    for igap, (az_deg, v_gap, npt) in gaps.items():
        ax.axvline(az_deg, color="r", ls="--", alpha=0.5,
                   label=f"gap {igap} center {az_deg:.0f} deg: "
                         f"{v_gap/1e3:.1f} kV")
    ax.set_xlabel(r"$\varphi$ [deg]")
    ax.set_ylabel(r"$E_\varphi$ [kV/m]")
    ax.set_title(f"Emap delivered field at r={r_show:.3f} m, "
                 f"V0={V0_NOMINAL/1e3:.0f} kV  ->  "
                 f"V_turn(crest) = {v_tot/1e3:.1f} kV "
                 f"(nominal {2*V0_NOMINAL/1e3:.0f} kV)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"\n[saved] {OUT_FIG}")
    plt.close(fig)


if __name__ == "__main__":
    main()

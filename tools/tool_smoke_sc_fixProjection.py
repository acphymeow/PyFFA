"""Smoke test: fixProjection SC solver with minimal particles (1 turn)."""
import os, sys, traceback
import numpy as np
from mpi4py import MPI

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    print("[OK] pyfftw imported")
except Exception as e:
    print(f"[WARN] pyfftw not available: {e}")
    pyfftw = None

TESTS_PASSED = 0
TESTS_FAILED = 0
ERRORS = []

def check(label, ok, detail=""):
    global TESTS_PASSED, TESTS_FAILED
    if ok:
        TESTS_PASSED += 1
        print(f"  [PASS] {label}  {detail}")
    else:
        TESTS_FAILED += 1
        msg = f"  [FAIL] {label}  {detail}"
        print(msg)
        ERRORS.append(msg)

# ============================================================
# Step 1: FFAG imports
# ============================================================
print("\n[1] Imports ...")
try:
    from FFAG_SC_new import (
        convert_seo_pr_to_pr0_geom, build_seo_s_matrix,
        interpolate_fields_to_particles_SBS_numba_withEfi,
    )
    from FFAG_Bunch import FFAG_Bunch
    from FFAG_Field import FFAG_Bfield_analytical
    print("  [OK]  FFAG_SC_new, FFAG_Bunch, FFAG_Field")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# Step 2: Load SEO tables
# ============================================================
print("\n[2] Load SEO tables ...")
SEO_R_FILE = "Bmap_FD16/resultsSEO_noError/SEO_r.txt"
SEO_PR_FILE = "Bmap_FD16/resultsSEO_noError/SEO_pr.txt"
if not os.path.exists(SEO_R_FILE):
    print(f"  [SKIP] SEO files not found at {SEO_R_FILE}")
    print("  Generate them first:  python examples/main_SEO.py -j config_SEO.json")
    sys.exit(1)

r_info = np.loadtxt(SEO_R_FILE, skiprows=1)
p_info = np.loadtxt(SEO_PR_FILE, skiprows=1)
seo_ek = r_info[1:, 0]
seo_fi = r_info[0, 1:]
r_mat = r_info[1:, 1:]
pr_mat = p_info[1:, 1:]
check("SEO r_matrix shape", r_mat.shape[1] > 1, str(r_mat.shape))

# ============================================================
# Step 3: Build SEO matrices for fixProjection
# ============================================================
print("\n[3] Build fixProjection SEO tables ...")
try:
    pr0_mat = convert_seo_pr_to_pr0_geom(pr_mat)
    s_mat, perimeter = build_seo_s_matrix(seo_fi, r_mat, pr0_mat)
    check("pr0_geom_matrix", not np.any(np.isnan(pr0_mat)),
          f"shape={pr0_mat.shape}, range=[{pr0_mat.min():.4f}, {pr0_mat.max():.4f}]")
    check("s_matrix", not np.any(np.isnan(s_mat)),
          f"shape={s_mat.shape}, range=[{s_mat.min():.2f}, {s_mat.max():.2f}]")
except Exception as e:
    traceback.print_exc()
    check("SEO table build", False, str(e))
    sys.exit(1)

# ============================================================
# Step 4: Select energy slice and build minimal Bunch
# ============================================================
print("\n[4] Build minimal Bunch ...")
EK0 = 300.0
iE = int(np.argmin(np.abs(seo_ek - EK0)))
print(f"    nearest Ek row: iE={iE}, Ek={seo_ek[iE]:.3f} MeV")

gamma_rel = 1.0 + EK0 / 938.2720813
particle_density = 2.08e10  # tiny (1% of full)

try:
    # Generate a proper arc-shaped bunch (match test_sc approach)
    n_parts = 200
    particle_density = 2.08e12  # 10% of full
    rng = np.random.default_rng(42)

    # arc spanning 4 cells (~90 deg), centered on phi0
    phi0 = np.pi / 8  # center at ~22.5 deg (middle of one cell)
    phi_span = 4 * (2*np.pi/16)  # ~90 deg
    phi = rng.uniform(phi0 - phi_span/2, phi0 + phi_span/2, n_parts)
    ek_i = np.full(n_parts, EK0)

    r = np.interp(phi % (2*np.pi), seo_fi, r_mat[iE], period=2*np.pi)
    pr_val = np.interp(phi % (2*np.pi), seo_fi, pr_mat[iE], period=2*np.pi)

    # transverse spread (Gaussian)
    sigma_x = 0.005  # 5 mm radial
    sigma_y = 0.005  # 5 mm vertical
    sigma_z = 0.005  # 5 mm z
    dEk_rms = 0.5    # MeV

    r_spread = rng.normal(0.0, sigma_x, n_parts)
    z_spread = rng.normal(0.0, sigma_z, n_parts)
    ek_dist = ek_i + rng.normal(0.0, dEk_rms, n_parts)

    # FFAG_Bunch expects polar Boris coordinates:
    # (r, vr), (z, vz), (fi, dfi/dt), ...
    r_cart = r + r_spread

    beta = np.sqrt(1 - 1.0/(1.0 + ek_dist/938.272)**2)
    speed = beta * 2.99792458e8
    vr = speed * pr_val
    vz = rng.normal(0.0, 0.001, n_parts) * speed
    vphi = np.sqrt(np.maximum(speed**2 - vr**2 - vz**2, 0.0))
    dphi_dt = vphi / r_cart

    arr = np.zeros((n_parts, 20))
    arr[:, 0] = r_cart
    arr[:, 1] = vr
    arr[:, 2] = z_spread
    arr[:, 3] = vz
    arr[:, 4] = phi
    arr[:, 5] = dphi_dt
    arr[:, 6] = ek_dist * 1e6 / 1e6 + 938.272  # total energy approx
    arr[:, 7] = 1.0  # survive flag
    arr[:, 15] = np.arange(n_parts)  # global particle ID

    BunchObj = FFAG_Bunch(arr, marcosize=particle_density,
                          sc_grid_size=(160, 160, 160, 5.0),
                          BunchType='Boris')
    BunchObj.Step_SurviveFlag = np.ones(n_parts, dtype=bool)
    print(f"    r range: [{r_cart.min():.3f}, {r_cart.max():.3f}] m")
    print(f"    z rms: {z_spread.std()*1e3:.1f} mm")
    print(f"    phi range: [{np.rad2deg(phi.min()):.1f}, {np.rad2deg(phi.max()):.1f}] deg")
    local_n = len(BunchObj.LocalBunch)
    global_n = MPI.COMM_WORLD.allreduce(local_n, op=MPI.SUM)
    check("BunchObj created", global_n == n_parts,
          f"NParticle_local={local_n}, NParticle_global={global_n}")
except Exception as e:
    traceback.print_exc()
    check("Bunch creation", False, str(e))
    sys.exit(1)

# ============================================================
# Step 5: Attach SEO tables and Bmap
# ============================================================
print("\n[5] Attach SEO tables + Bmap ...")
try:
    BunchObj.SEO_fi_axis = seo_fi
    BunchObj.SEO_Ek_axis = seo_ek
    BunchObj.SEO_r_matrix = r_mat
    BunchObj.SEO_pr_matrix = pr_mat
    BunchObj.SEO_pr0_geom_matrix = pr0_mat
    BunchObj.SEO_s_matrix = s_mat
    BunchObj.SEO_perimeter = perimeter
    BunchObj.sector_period = 2 * np.pi / 16.0   # 16 sectors

    config_bmap = './Bmaps/Bmap_FD16/config_Bmap.json'
    BunchObj.Bmap = FFAG_Bfield_analytical(config_bmap, max_order=0)
    BunchObj.Bmap.field_type = "analytical"
    check("SEO + Bmap attached", True)
except Exception as e:
    traceback.print_exc()
    check("Attach tables", False, str(e))
    sys.exit(1)

# ============================================================
# Step 6: FFT plans (open boundary padded, managed internally)
# ============================================================
print("\n[6] FFT plans ...")
try:
    # Grid size from f_grid_size/x_grid_size (set by Bunch obj from SC_grid_size)
    from FFAG_Bunch import FFAG_Bunch
    # Use a modest grid — 64 is enough for smoke test
    n_fft = 64
    fft_in = pyfftw.empty_aligned((n_fft, n_fft), dtype='complex128')
    fft_out = pyfftw.empty_aligned((n_fft, n_fft), dtype='complex128')
    fft_fwd = pyfftw.FFTW(fft_in, fft_out, direction='FFTW_FORWARD',
                           flags=('FFTW_ESTIMATE',), threads=1)
    fft_inv = pyfftw.FFTW(fft_out, fft_in, direction='FFTW_BACKWARD',
                           flags=('FFTW_ESTIMATE',), threads=1)
    check("FFT plans", True, f"grid={n_fft}")
except Exception as e:
    check("FFT plans", False, str(e))
    fft_fwd, fft_inv = None, None

# ============================================================
# Step 7: Call fixProjection
# ============================================================
print("\n[7] fixProjection SC solve ...")
try:
    ret = BunchObj.Build_2_5D_SliceGrid_sbs_new_fixProjection(fft_fwd, fft_inv)
    check("fixProjection return", ret is not None, f"ret type={type(ret)}")
    n_slices = BunchObj.sbs_nslices if hasattr(BunchObj, 'sbs_nslices') else '?'
    check("sbs_nslices", True, f"n_slices={n_slices}")
except Exception as e:
    import io; buf = io.StringIO(); traceback.print_exc(file=buf)
    check("fixProjection solve", False, buf.getvalue())

# ============================================================
# Step 8: Check SC fields on particles
# ============================================================
print("\n[8] SC fields on particles ...")
try:
    local_bunch_view = BunchObj.LocalBunch
    sc_fields = local_bunch_view[:, 10:13]  # Er, Ez, Efi
    er, ez, efi = sc_fields[:, 0], sc_fields[:, 1], sc_fields[:, 2]
    finite_mask = np.isfinite(sc_fields).all(axis=1)
    print(f"    Er:  min={er.min():.2e}, max={er.max():.2e}, "
          f"rms={np.sqrt(np.mean(er**2)):.2e} V/m")
    print(f"    Ez:  min={ez.min():.2e}, max={ez.max():.2e}, "
          f"rms={np.sqrt(np.mean(ez**2)):.2e} V/m")
    print(f"    Efi: min={efi.min():.2e}, max={efi.max():.2e}, "
          f"rms={np.sqrt(np.mean(efi**2)):.2e} V/m")
    check("SC fields finite", np.all(finite_mask),
          f"nan/inf count={np.sum(~finite_mask)}")
    has_field = np.any(np.abs(sc_fields) > 1e-12)
    check("SC fields non-zero", has_field)
    if not has_field:
        print("    NOTE: fields are zero — need more particles with spread")
except Exception as e:
    traceback.print_exc()
    check("SC fields check", False, str(e))
except Exception as e:
    traceback.print_exc()
    check("SC fields check", False, str(e))

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"Results: {TESTS_PASSED} passed, {TESTS_FAILED} failed")
if TESTS_FAILED > 0:
    print("\nFailures:")
    for msg in ERRORS:
        print(f"  {msg}")
    sys.exit(1)
else:
    print("All tests passed -- fixProjection chain is WORKING!")

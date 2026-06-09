import re
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from numba import njit, prange
from FFAG_MathTools import Bilinear_interp_2D_vect_uniform

# ============================================================
# 参数
# ============================================================
DATA_DIR = "../output/NewBenchmark/FFT_Tune_Gauss40_uniform72"
SEO_BASE = "./Bmap_FD16/resultsSEO_noError_X"
SEO_r_filepath, SEO_pr_filepath = f"{SEO_BASE}/SEO_r.txt", f"{SEO_BASE}/SEO_pr.txt"
betaR_filepath, betaZ_filepath = f"{SEO_BASE}/BetaFuncR.txt", f"{SEO_BASE}/BetaFuncZ.txt"

COL_R, COL_PR, COL_Z, COL_PZ, COL_FI, COL_EK = 0, 1, 2, 3, 4, 5
SUBTRACT_CLOSED_ORBIT = True

MAX_FILES = 20000
N_THEORY_PARTICLES = 2000
N_SKIP, MIN_TURNS, RNG_SEED = 10, 128, 12345

PAD_FACTOR = 8

QR_BARE_ABS, QZ_BARE_ABS = 3.230202618580, 2.813029944565
QR_CORE_ABS, QZ_CORE_ABS = 2.897641931285, 2.494797856827
# QR_CORE_ABS, QZ_CORE_ABS = 2.98508403, 2.57436813
# QR_CORE_ABS, QZ_CORE_ABS = 3.04655948, 2.65002719

QX_DISPLAY_SHIFT = 0.00
QR_BARE_PLOT = QR_BARE_ABS + QX_DISPLAY_SHIFT
QR_CORE_PLOT = QR_CORE_ABS + QX_DISPLAY_SHIFT

COMPLEX_SIGN_R, COMPLEX_SIGN_Z = -1.0, -1.0

N_CELLS = 16
CELL_START_DEG, CELL_END_DEG = 0.0, 360.0 / N_CELLS

N_THETA_CELL = 300
N_TURNS_THEORY = 128
N_STEPS_PER_CELL = 50
N_QUAD = 12

N_PROTONS, BUNCH_LENGTH_M, KINETIC_ENERGY_MEV = 0.0e13, 13.0, 300.0
EMITTANCE_IS_NORMALIZED = False
EMITTANCE_40PI_MM_MRAD = 40.0

EPS0, QE = 8.8541878128e-12, 1.602176634e-19
MP_MEV, MP_KG, C_LIGHT = 938.2720813, 1.67262192369e-27, 299792458.0

QR_XLIM, QZ_XLIM = (2.80, 3.30), (2.40, 2.90)

# ============================================================
# 读取 SEO / Twiss
# ============================================================
seo_r = np.loadtxt(SEO_r_filepath, skiprows=1)
seo_pr = np.loadtxt(SEO_pr_filepath, skiprows=1)

fi_axis, Ek_axis = seo_r[0, 1:], seo_r[1:, 0]
r_mat, pr_mat = seo_r[1:, 1:], seo_pr[1:, 1:]

betaR_data = np.loadtxt(betaR_filepath, skiprows=1)
betaZ_data = np.loadtxt(betaZ_filepath, skiprows=1)

theta_axis_deg = betaR_data[0, 1:]
twiss_Ek_axis = betaR_data[1:, 0]

betaR_mat = betaR_data[1:, 1:]
betaZ_mat = betaZ_data[1:, 1:]


def get_pid(path):
    m = re.search(r"particle_(\d+)_", path.name)
    return int(m.group(1)) if m else -1


def sub_co(r, pr, fi, Ek):
    fi_mod = np.mod(fi, 2*np.pi)

    r0 = np.zeros_like(r)
    pr0 = np.zeros_like(pr)
    f1 = np.zeros_like(r)
    f2 = np.zeros_like(pr)

    Bilinear_interp_2D_vect_uniform(
        Ek_axis[0], Ek_axis[1] - Ek_axis[0], len(Ek_axis),
        fi_axis[0], fi_axis[1] - fi_axis[0], len(fi_axis),
        r_mat, Ek, fi_mod, r0, f1,
    )

    Bilinear_interp_2D_vect_uniform(
        Ek_axis[0], Ek_axis[1] - Ek_axis[0], len(Ek_axis),
        fi_axis[0], fi_axis[1] - fi_axis[0], len(fi_axis),
        pr_mat, Ek, fi_mod, pr0, f2,
    )

    return r - r0, pr - pr0


def parabolic_interpolate(amp, k):
    y0, y1, y2 = amp[k-1], amp[k], amp[k+1]
    denom = y0 - 2.0*y1 + y2
    if denom == 0:
        return float(k)
    return k + 0.5*(y0-y2)/denom


def cfft_frac_interp(x, p, sign=-1.0, pad_factor=8):
    x = np.asarray(x, float)
    p = np.asarray(p, float)

    if len(x) < MIN_TURNS or np.std(x) == 0 or np.std(p) == 0:
        return np.nan

    x = x - np.mean(x)
    p = p - np.mean(p)

    scale = np.std(x) / np.std(p)
    u = x + 1j * sign * scale * p

    n = len(u)
    nfft = int(pad_factor * n)

    spec = np.abs(np.fft.fft(u, n=nfft))
    freq = np.fft.fftfreq(nfft, d=1.0)

    spec[0] = 0.0

    k = int(np.argmax(spec))
    k_use = min(max(k, 1), nfft - 2)

    k_peak = parabolic_interpolate(spec, k_use)
    df = freq[1] - freq[0]

    q_frac = np.mod(k_peak * df, 1.0)

    return q_frac


def abs_tune(qf, qb, qc):
    n0 = int(np.floor(qb))
    qmin = min(qb, qc) - 0.10
    qmax = max(qb, qc) + 0.10

    cand = np.array([
        n0 - 2 + qf,
        n0 - 1 + qf,
        n0     + qf,
        n0 + 1 + qf,
        n0 + 2 + qf,
    ])

    inside = cand[(cand >= qmin) & (cand <= qmax)]

    if len(inside) > 0:
        return inside[np.argmin(np.abs(inside - 0.5*(qb+qc)))]

    return cand[np.argmin(np.abs(cand-qb))]


def fft_range(x, xp, beta0, alpha0, q_low, q_high):
    X = x / np.sqrt(beta0)
    P = (alpha0*x + beta0*xp) / np.sqrt(beta0)
    u = X - 1j*P

    spec = np.abs(np.fft.fft((u - np.mean(u)) * np.hanning(len(u))))
    freq = np.fft.fftfreq(len(u))

    spec[np.argmin(np.abs(freq))] = 0.0

    best_q = np.nan
    best_a = -1.0

    for k, f in enumerate(freq):
        qf = np.mod(f, 1.0)

        for n in range(int(np.floor(q_low))-2, int(np.ceil(q_high))+3):
            q = n + qf

            if q_low <= q <= q_high and spec[k] > best_a:
                best_q = q
                best_a = spec[k]

    return best_q


# ============================================================
# Numba 理论 nonlinear tracking
# ============================================================
@njit(cache=True)
def interp_uniform(th, th0, dth, arr):
    n = arr.shape[0]
    u = (th - th0) / dth
    i = int(u)

    if i < 0:
        return arr[0]

    if i >= n - 1:
        return arr[n - 1]

    a = u - i

    return (1.0 - a) * arr[i] + a * arr[i + 1]


@njit(cache=True)
def gfield_scalar(x, z, sx, sz, quad_s, quad_w, field_coef):
    Ex = 0.0
    Ez = 0.0

    sx2 = sx * sx
    sz2 = sz * sz

    t0 = sx2 if sx2 > sz2 else sz2

    for iq in range(quad_s.shape[0]):
        s = quad_s[iq]
        w = quad_w[iq]

        t = t0 * s / (1.0 - s)
        dt = t0 / ((1.0 - s) * (1.0 - s))

        ax = sx2 + t
        az = sz2 + t

        root = np.sqrt(ax * az)
        expo = np.exp(-0.5 * (x*x/ax + z*z/az))

        Ex += w * dt * x / (ax * root) * expo
        Ez += w * dt * z / (az * root) * expo

    return field_coef * Ex, field_coef * Ez


@njit(cache=True)
def rhs_scalar(
    th, x, xp, z, zp,
    theta0, dtheta_grid,
    ds_dtheta, KR, KZ, sigmaR, sigmaZ,
    quad_s, quad_w, field_coef, force_factor,
):
    h = interp_uniform(th, theta0, dtheta_grid, ds_dtheta)
    kr = interp_uniform(th, theta0, dtheta_grid, KR)
    kz = interp_uniform(th, theta0, dtheta_grid, KZ)
    sx = interp_uniform(th, theta0, dtheta_grid, sigmaR)
    sz = interp_uniform(th, theta0, dtheta_grid, sigmaZ)

    Ex, Ez = gfield_scalar(x, z, sx, sz, quad_s, quad_w, field_coef)

    dx = h * xp
    dxp = h * (-kr * x + force_factor * Ex)

    dz = h * zp
    dzp = h * (-kz * z + force_factor * Ez)

    return dx, dxp, dz, dzp


@njit(parallel=True, cache=True)
def track_theory_numba(
    AR, AZ, theta_steps, theta0, dtheta_grid,
    ds_dtheta, KR, KZ, sigmaR, sigmaZ,
    quad_s, quad_w, field_coef, force_factor,
    n_cells, n_turns,
):
    n_part = AR.shape[0]
    n_steps = theta_steps.shape[0] - 1

    x_turn = np.zeros((n_turns, n_part))
    xp_turn = np.zeros((n_turns, n_part))
    z_turn = np.zeros((n_turns, n_part))
    zp_turn = np.zeros((n_turns, n_part))

    for ip in prange(n_part):
        x = AR[ip]
        xp = 0.0
        z = AZ[ip]
        zp = 0.0

        for it in range(n_turns):
            for ic in range(n_cells):
                for js in range(n_steps):
                    th = theta_steps[js]
                    dth = theta_steps[js + 1] - theta_steps[js]

                    k1x, k1xp, k1z, k1zp = rhs_scalar(
                        th, x, xp, z, zp,
                        theta0, dtheta_grid,
                        ds_dtheta, KR, KZ, sigmaR, sigmaZ,
                        quad_s, quad_w, field_coef, force_factor,
                    )

                    k2x, k2xp, k2z, k2zp = rhs_scalar(
                        th + 0.5*dth,
                        x + 0.5*dth*k1x,
                        xp + 0.5*dth*k1xp,
                        z + 0.5*dth*k1z,
                        zp + 0.5*dth*k1zp,
                        theta0, dtheta_grid,
                        ds_dtheta, KR, KZ, sigmaR, sigmaZ,
                        quad_s, quad_w, field_coef, force_factor,
                    )

                    k3x, k3xp, k3z, k3zp = rhs_scalar(
                        th + 0.5*dth,
                        x + 0.5*dth*k2x,
                        xp + 0.5*dth*k2xp,
                        z + 0.5*dth*k2z,
                        zp + 0.5*dth*k2zp,
                        theta0, dtheta_grid,
                        ds_dtheta, KR, KZ, sigmaR, sigmaZ,
                        quad_s, quad_w, field_coef, force_factor,
                    )

                    k4x, k4xp, k4z, k4zp = rhs_scalar(
                        th + dth,
                        x + dth*k3x,
                        xp + dth*k3xp,
                        z + dth*k3z,
                        zp + dth*k3zp,
                        theta0, dtheta_grid,
                        ds_dtheta, KR, KZ, sigmaR, sigmaZ,
                        quad_s, quad_w, field_coef, force_factor,
                    )

                    x += dth * (k1x + 2.0*k2x + 2.0*k3x + k4x) / 6.0
                    xp += dth * (k1xp + 2.0*k2xp + 2.0*k3xp + k4xp) / 6.0

                    z += dth * (k1z + 2.0*k2z + 2.0*k3z + k4z) / 6.0
                    zp += dth * (k1zp + 2.0*k2zp + 2.0*k3zp + k4zp) / 6.0

            x_turn[it, ip] = x
            xp_turn[it, ip] = xp
            z_turn[it, ip] = z
            zp_turn[it, ip] = zp

    return x_turn, xp_turn, z_turn, zp_turn


# ============================================================
# PyFFA tune 和振幅，读取 20000 个 tracking 粒子
# ============================================================
paths = sorted(Path(DATA_DIR).glob("*.npz"), key=get_pid)

if MAX_FILES is not None and len(paths) > MAX_FILES:
    rng = np.random.default_rng(RNG_SEED)
    paths = sorted(list(rng.choice(paths, MAX_FILES, replace=False)), key=get_pid)

pid = []
QR_py = []
QZ_py = []
AR = []
AZ = []

t0 = time.time()

for i, path in enumerate(paths):
    arr = np.load(path)["particles"]
    arr = arr[arr[:, -1].argsort()]
    arr = arr[N_SKIP:]

    r = arr[:, COL_R].astype(float)
    pr = arr[:, COL_PR].astype(float)
    z = arr[:, COL_Z].astype(float)
    pz = arr[:, COL_PZ].astype(float)
    fi = arr[:, COL_FI].astype(float)
    Ek = arr[:, COL_EK].astype(float)

    if SUBTRACT_CLOSED_ORBIT:
        x, px = sub_co(r, pr, fi, Ek)
    else:
        x, px = r, pr

    qRf = cfft_frac_interp(x, px, COMPLEX_SIGN_R, PAD_FACTOR)
    qZf = cfft_frac_interp(z, pz, COMPLEX_SIGN_Z, PAD_FACTOR)

    if not (np.isfinite(qRf) and np.isfinite(qZf)):
        continue

    pid.append(get_pid(path))

    QR_py.append(abs_tune(qRf, QR_BARE_ABS, QR_CORE_ABS))
    QZ_py.append(abs_tune(qZf, QZ_BARE_ABS, QZ_CORE_ABS))

    AR.append(0.5 * (x.max() - x.min()))
    AZ.append(0.5 * (z.max() - z.min()))

    if (i + 1) % 1000 == 0:
        print(f"Read PyFFA particle {i+1}/{len(paths)}")

pid = np.array(pid)
QR_py = np.array(QR_py)
QZ_py = np.array(QZ_py)
AR = np.array(AR)
AZ = np.array(AZ)

A_total_mm = np.sqrt((AR*1e3)**2 + (AZ*1e3)**2)
Np_py = len(pid)

print("\n" + "="*72)
print("PyFFA tune extracted from tracking")
print("="*72)
print(f"n_particles_pyffa = {Np_py}")
print(f"QR_py min/mean/max = {QR_py.min():.8f}, {QR_py.mean():.8f}, {QR_py.max():.8f}")
print(f"QZ_py min/mean/max = {QZ_py.min():.8f}, {QZ_py.mean():.8f}, {QZ_py.max():.8f}")
print(f"PyFFA data reading elapsed time = {time.time()-t0:.2f} s")


# ============================================================
# 从 PyFFA 20000 个点中抽 2000 个点用于理论积分
# ============================================================
rng = np.random.default_rng(RNG_SEED + 1)

n_th = min(N_THEORY_PARTICLES, Np_py)
idx_th = rng.choice(Np_py, size=n_th, replace=False)

pid_th_ref = pid[idx_th]
QR_py_ref = QR_py[idx_th]
QZ_py_ref = QZ_py[idx_th]

AR_th = AR[idx_th]
AZ_th = AZ[idx_th]
A_total_th_mm = A_total_mm[idx_th]

print("\n" + "="*72)
print("Theory particles selected from PyFFA sample")
print("="*72)
print(f"n_particles_theory = {n_th}")


# ============================================================
# cell 上 lattice / sigma
# ============================================================
theta_deg = np.linspace(CELL_START_DEG, CELL_END_DEG, N_THETA_CELL)
theta = np.deg2rad(theta_deg)

r_E = np.array([
    np.interp(KINETIC_ENERGY_MEV, Ek_axis, r_mat[:, j])
    for j in range(len(fi_axis))
])

r_cell = np.interp(
    np.mod(theta, 2*np.pi),
    np.r_[fi_axis, fi_axis[0]+2*np.pi],
    np.r_[r_E, r_E[0]],
)

dr_dtheta = np.gradient(r_cell, theta)
ds_dtheta = np.sqrt(r_cell**2 + dr_dtheta**2)

betaR_E = np.array([
    np.interp(theta_deg, theta_axis_deg, betaR_mat[i])
    for i in range(len(twiss_Ek_axis))
])

betaZ_E = np.array([
    np.interp(theta_deg, theta_axis_deg, betaZ_mat[i])
    for i in range(len(twiss_Ek_axis))
])

betaR = np.array([
    np.interp(KINETIC_ENERGY_MEV, twiss_Ek_axis, betaR_E[:, j])
    for j in range(N_THETA_CELL)
])

betaZ = np.array([
    np.interp(KINETIC_ENERGY_MEV, twiss_Ek_axis, betaZ_E[:, j])
    for j in range(N_THETA_CELL)
])

dbR_ds = np.gradient(betaR, theta) / ds_dtheta
dbZ_ds = np.gradient(betaZ, theta) / ds_dtheta

alphaR = -0.5 * dbR_ds
alphaZ = -0.5 * dbZ_ds

KR = (np.gradient(alphaR, theta) / ds_dtheta + (1.0 + alphaR**2) / betaR) / betaR
KZ = (np.gradient(alphaZ, theta) / ds_dtheta + (1.0 + alphaZ**2) / betaZ) / betaZ

betaR0 = betaR[0]
betaZ0 = betaZ[0]

alphaR0 = alphaR[0]
alphaZ0 = alphaZ[0]

gamma_rel = 1.0 + KINETIC_ENERGY_MEV / MP_MEV
beta_rel = np.sqrt(1.0 - 1.0/gamma_rel**2)

eps_rms = EMITTANCE_40PI_MM_MRAD * 1e-6

if EMITTANCE_IS_NORMALIZED:
    eps_rms /= beta_rel * gamma_rel

sigmaR = np.sqrt(betaR * eps_rms)
sigmaZ = np.sqrt(betaZ * eps_rms)

line_charge = N_PROTONS * QE / BUNCH_LENGTH_M
force_factor = QE / (gamma_rel * MP_KG * beta_rel**2 * C_LIGHT**2)
field_coef = line_charge / (4.0*np.pi*EPS0*gamma_rel**2)

quad_x, quad_w = np.polynomial.legendre.leggauss(N_QUAD)
quad_s = 0.5 * (quad_x + 1.0)
quad_w = 0.5 * quad_w

theta_steps = np.deg2rad(np.linspace(CELL_START_DEG, CELL_END_DEG, N_STEPS_PER_CELL + 1))

theta0 = theta[0]
dtheta_grid = theta[1] - theta[0]

print("\n" + "="*72)
print("Theory model parameters")
print("="*72)
print(f"line_charge = {line_charge:.8e} C/m")
print(f"beta_rel = {beta_rel:.8e}, gamma_rel = {gamma_rel:.8e}")
print(f"eps_rms = {eps_rms:.8e} m rad")
print(f"sigmaR mean = {sigmaR.mean()*1e3:.6f} mm")
print(f"sigmaZ mean = {sigmaZ.mean()*1e3:.6f} mm")
print(f"N_STEPS_PER_CELL = {N_STEPS_PER_CELL}")
print(f"N_TURNS_THEORY = {N_TURNS_THEORY}")
print(f"Np_theory = {n_th}")
print(f"QX_DISPLAY_SHIFT = {QX_DISPLAY_SHIFT:.6f}")


# ============================================================
# 作用量：J = JR + JZ，单位 mm mrad
# ============================================================
JR_py = AR**2 / (2.0 * betaR0)
JZ_py = AZ**2 / (2.0 * betaZ0)
Jtot_py = (JR_py + JZ_py) * 1e6

JR_th_ref = AR_th**2 / (2.0 * betaR0)
JZ_th_ref = AZ_th**2 / (2.0 * betaZ0)
Jtot_th_ref = (JR_th_ref + JZ_th_ref) * 1e6


# ============================================================
# Numba warm-up
# ============================================================
print("\nStart Numba warm-up...")
t0 = time.time()

_ = track_theory_numba(
    AR_th[:2].astype(np.float64).copy(),
    AZ_th[:2].astype(np.float64).copy(),
    theta_steps.astype(np.float64),
    theta0,
    dtheta_grid,
    ds_dtheta.astype(np.float64),
    KR.astype(np.float64),
    KZ.astype(np.float64),
    sigmaR.astype(np.float64),
    sigmaZ.astype(np.float64),
    quad_s.astype(np.float64),
    quad_w.astype(np.float64),
    field_coef,
    force_factor,
    N_CELLS,
    1,
)

print(f"Numba warm-up finished. Elapsed time = {time.time()-t0:.2f} s")


# ============================================================
# 正式理论积分，只跑 2000 个点
# ============================================================
print("\nStart full Numba theory tracking...")
t0 = time.time()

x_turn, xp_turn, z_turn, zp_turn = track_theory_numba(
    AR_th.astype(np.float64),
    AZ_th.astype(np.float64),
    theta_steps.astype(np.float64),
    theta0,
    dtheta_grid,
    ds_dtheta.astype(np.float64),
    KR.astype(np.float64),
    KZ.astype(np.float64),
    sigmaR.astype(np.float64),
    sigmaZ.astype(np.float64),
    quad_s.astype(np.float64),
    quad_w.astype(np.float64),
    field_coef,
    force_factor,
    N_CELLS,
    N_TURNS_THEORY,
)

print(f"Full Numba theory tracking finished. Elapsed time = {time.time()-t0:.2f} s")


# ============================================================
# 理论 tune 提取
# ============================================================
print("\nExtracting theory tunes...")
t0 = time.time()

QR_LOW = QR_CORE_ABS - 0.05
QR_HIGH = QR_BARE_ABS + 0.03

QZ_LOW = QZ_CORE_ABS - 0.05
QZ_HIGH = QZ_BARE_ABS + 0.03

QR_th_raw = np.array([
    fft_range(x_turn[:, i], xp_turn[:, i], betaR0, alphaR0, QR_LOW, QR_HIGH)
    for i in range(n_th)
])

QZ_th_raw = np.array([
    fft_range(z_turn[:, i], zp_turn[:, i], betaZ0, alphaZ0, QZ_LOW, QZ_HIGH)
    for i in range(n_th)
])

good_th = np.isfinite(QR_th_raw) & np.isfinite(QZ_th_raw)

pid_th_ref = pid_th_ref[good_th]

QR_py_ref = QR_py_ref[good_th]
QZ_py_ref = QZ_py_ref[good_th]

AR_th = AR_th[good_th]
AZ_th = AZ_th[good_th]

A_total_th_mm = A_total_th_mm[good_th]
Jtot_th_ref = Jtot_th_ref[good_th]

QR_th_raw = QR_th_raw[good_th]
QZ_th_raw = QZ_th_raw[good_th]

QR_th_plot = QR_th_raw + QX_DISPLAY_SHIFT
QZ_th_plot = QZ_th_raw

dQR_err_raw = QR_py_ref - QR_th_raw
dQZ_err_raw = QZ_py_ref - QZ_th_raw

dQR_err_plot = QR_py_ref - QR_th_plot
dQZ_err_plot = QZ_py_ref - QZ_th_plot

print(f"Theory tune extraction elapsed time = {time.time()-t0:.2f} s")

print("\n" + "="*72)
print("Theory tune from nonlinear Hill tracking")
print("="*72)
print(f"n_particles_theory_valid = {len(QR_th_raw)}")
print(f"QR_th_raw min/mean/max = {QR_th_raw.min():.8f}, {QR_th_raw.mean():.8f}, {QR_th_raw.max():.8f}")
print(f"QR_th_plot min/mean/max = {QR_th_plot.min():.8f}, {QR_th_plot.mean():.8f}, {QR_th_plot.max():.8f}")
print(f"QZ_th min/mean/max = {QZ_th_raw.min():.8f}, {QZ_th_raw.mean():.8f}, {QZ_th_raw.max():.8f}")

print("\n" + "="*72)
print("PyFFA selected particles - theory")
print("="*72)
print(f"dQR raw err min/mean/max = {dQR_err_raw.min():.8e}, {dQR_err_raw.mean():.8e}, {dQR_err_raw.max():.8e}")
print(f"dQR plot err min/mean/max = {dQR_err_plot.min():.8e}, {dQR_err_plot.mean():.8e}, {dQR_err_plot.max():.8e}")
print(f"dQZ err min/mean/max = {dQZ_err_raw.min():.8e}, {dQZ_err_raw.mean():.8e}, {dQZ_err_raw.max():.8e}")
print(f"QR RMSE raw  = {np.sqrt(np.mean(dQR_err_raw**2)):.8e}")
print(f"QR RMSE plot = {np.sqrt(np.mean(dQR_err_plot**2)):.8e}")
print(f"QZ RMSE      = {np.sqrt(np.mean(dQZ_err_raw**2)):.8e}")


# ============================================================
# 保存
# ============================================================
out_py = np.column_stack([
    pid,
    QR_py,
    QZ_py,
    AR*1e3,
    AZ*1e3,
    A_total_mm,
    Jtot_py,
])

np.savetxt(
    "pyffa_tracking_tune_20000_hist2d.csv",
    out_py,
    delimiter=",",
    header="pid,QR_py,QZ_py,AR_mm,AZ_mm,A_total_mm,J_total_mm_mrad",
    comments="",
)

out_th = np.column_stack([
    pid_th_ref,
    AR_th*1e3,
    AZ_th*1e3,
    A_total_th_mm,
    Jtot_th_ref,
    QR_py_ref,
    QZ_py_ref,
    QR_th_raw,
    QZ_th_raw,
    QR_th_plot,
    QZ_th_plot,
    QR_py_ref - QR_th_raw,
    QZ_py_ref - QZ_th_raw,
    QR_py_ref - QR_th_plot,
    QZ_py_ref - QZ_th_plot,
])

np.savetxt(
    "theory_tracking_tune_2000_scatter.csv",
    out_th,
    delimiter=",",
    header=(
        "pid,AR_mm,AZ_mm,A_total_mm,J_total_mm_mrad,"
        "QR_py_ref,QZ_py_ref,"
        "QR_th_raw,QZ_th_raw,"
        "QR_th_plot,QZ_th_plot,"
        "QR_py_minus_th_raw,QZ_py_minus_th_raw,"
        "QR_py_minus_th_plot,QZ_py_minus_th_plot"
    ),
    comments="",
)

print("\nSaved: pyffa_tracking_tune_20000_hist2d.csv")
print("Saved: theory_tracking_tune_2000_scatter.csv")


# ============================================================
# 图 1：PyFFA tracking，用 hist2d
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)

h = ax.hist2d(
    QR_py,
    QZ_py,
    bins=[256, 256],
    range=[QR_XLIM, QZ_XLIM],
    norm=LogNorm(),
    cmap="viridis",
)

ax.axvline(QR_BARE_PLOT, ls="--", lw=2.0, color="tab:blue")
ax.axhline(QZ_BARE_ABS, ls="--", lw=2.0, color="tab:blue")
ax.axvline(QR_CORE_PLOT, ls=":", lw=2.4, color="tab:blue")
ax.axhline(QZ_CORE_ABS, ls=":", lw=2.4, color="tab:blue")
# bare tune
ax.scatter(
    QR_BARE_PLOT,
    QZ_BARE_ABS,
    marker="o",
    s=160,
    facecolor="white",
    edgecolor="blue",
    linewidth=2.8,
    zorder=12,
    label="Bare tune"
)
# 最大空间电荷频移点
ax.scatter(
    QR_CORE_PLOT,
    QZ_CORE_ABS,
    marker="*",
    s=300,
    color="red",
    edgecolor="black",
    linewidth=2.4,
    zorder=13,
    label="Max SC tune shift"
)

ax.set_xlabel(r"$Q_x$", fontsize=24)
ax.set_ylabel(r"$Q_z$", fontsize=24)
# ax.set_title("Tune footprint from PyFFA tracking", fontsize=18)
ax.set_xlim(QR_XLIM)
ax.set_ylim(QZ_XLIM)
ax.tick_params(axis="both", labelsize=20)
# ax.grid(True, alpha=0.25)
ax.legend(fontsize=15, loc="upper left")

cb = plt.colorbar(h[3], ax=ax)
cb.set_label("Counts", fontsize=19)
cb.ax.tick_params(labelsize=12)

import os
os.makedirs("./prabfig", exist_ok=True)
plt.savefig("./prabfig/fig_tune_gauss_80_pic.png",dpi=300,bbox_inches="tight")


# ============================================================
# 图 2：Theory tracking，横向 tune 显示右移 0.03    label="Theoretic tune",
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)

sc = ax.scatter(
    QR_th_plot,
    QZ_th_plot,
    c=Jtot_th_ref,
    cmap="plasma",
    s=18,
    alpha=0.75,
    linewidths=0,
    rasterized=True,
)

ax.axvline(QR_BARE_PLOT, ls="--", lw=2.0, color="tab:blue")
ax.axhline(QZ_BARE_ABS, ls="--", lw=2.0, color="tab:blue")
ax.axvline(QR_CORE_PLOT, ls=":", lw=2.4, color="tab:blue")
ax.axhline(QZ_CORE_ABS, ls=":", lw=2.4, color="tab:blue")

# bare tune
ax.scatter(
    QR_BARE_PLOT,
    QZ_BARE_ABS,
    marker="o",
    s=160,
    facecolor="white",
    edgecolor="blue",
    linewidth=2.8,
    zorder=12,
    label="Bare tune"
)
# 最大空间电荷频移点
ax.scatter(
    QR_CORE_PLOT,
    QZ_CORE_ABS,
    marker="*",
    s=300,
    color="red",
    edgecolor="black",
    linewidth=2.4,
    zorder=13,
    label="Theoretic max \nSC tune shift")

ax.set_xlabel(r"$Q_x$", fontsize=24)
ax.set_ylabel(r"$Q_z$", fontsize=24)
# ax.set_title("Tune footprint from nonlinear theory tracking", fontsize=18)
ax.set_xlim(QR_XLIM)
ax.set_ylim(QZ_XLIM)
ax.tick_params(axis="both", labelsize=20)
# ax.grid(True, alpha=0.25)
ax.legend(fontsize=15, loc="upper left")

cb = plt.colorbar(sc, ax=ax)
cb.set_label(r"$J_x + J_z$ [$\pi$ mm mrad]", fontsize=19)
cb.ax.tick_params(labelsize=12)

# import os
os.makedirs("./prabfig", exist_ok=True)
plt.savefig("./prabfig/fig_tune_gauss_80_theory.png",dpi=300,bbox_inches="tight")

plt.show()
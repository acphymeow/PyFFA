import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from matplotlib.colors import LogNorm
from FFAG_MathTools import Bilinear_interp_2D_vect_uniform

# -------------------- 配置区 --------------------
# DATA_DIR0 = "../output/simulation_tunespread_10/FFT_Tune_Dump_0"
# DATA_DIR0 = "../output/benchmark_SC_tune_kvAna/kv30"
# DATA_DIR0 = "../output/simulation_resonance/NoError_SC/FFT_Tune_Dump_0"
# DATA_DIR0 = "../output/simulation_resonance/Error_SC/FFT_Tune_Dump_0"
# DATA_DIR0 = "../output/simulation_resonance/25GsB4_NonL/FFT_Tune_Dump_0"
# DATA_DIR0 = "../output/simulation_resonance/25GsB4_L/FFT_Tune_Dump_0"
# DATA_DIR0 = "./output/simulation6/FFT_Tune_Dump_0"
DATA_DIR0 = "../output/FD16_paint_1_1/FFT_Tune_Dump_0"
# DATA_DIR0 = "../output/NewBenchmark/FFT_Tune_Gauss80_uniform72"
# DATA_DIR0 = "../output/NewBenchmark/FFT_Tune_Gauss80_uniform72_noNonL"

COL_X, COL_XP = 0, 1
COL_Z, COL_ZP = 2, 3
COL_fi, COL_Ek = 4, 5

PAD_FACTOR = 8
USE_SAVGOL = False
SG_WINDOW, SG_ORDER = 18, 4

# -------------------- 工具函数 --------------------
def parabolic_interpolate(amp, k):
    y0, y1, y2 = amp[k-1], amp[k], amp[k+1]
    denom = (y0 - 2*y1 + y2)
    if denom == 0:
        return float(k)
    return k + 0.5*(y0 - y2) / denom


def tune_from_xpx(x, px, pad_factor=PAD_FACTOR, ignore_dc=True, px_scale=None):
    x = np.asarray(x)
    px = np.asarray(px)
    N = len(x)
    if N != len(px):
        raise ValueError("x and px must have same length")

    if px_scale is None:
        sx = x.std(ddof=0)
        sp = px.std(ddof=0)
        px_scale = sp / sx if sx > 0 and sp > 0 else 1.0

    z = x + 1j * (px / px_scale)
    z = z - z.mean()
    Nfft = int(pad_factor * N)
    Z = np.fft.fft(z, n=Nfft)
    freqs = np.fft.fftfreq(Nfft, d=1.0)
    mag = np.abs(Z)
    if ignore_dc:
        mag[0] = 0.0

    k = int(np.argmax(mag))
    k_use = min(max(k, 1), Nfft - 2)
    k_peak = parabolic_interpolate(mag, k_use)
    df = freqs[1] - freqs[0]
    tune = (k_peak % Nfft) * df
    return tune, 0, 0

# -------------------- 主处理 --------------------
pairs, pairs2 = [], []
# SEO_r_info = np.loadtxt(os.path.join('./Bmap/resultsSEO', "SEO_r.txt"), skiprows=1)
# SEO_pr_info = np.loadtxt(os.path.join('./Bmap/resultsSEO', "SEO_pr.txt"), skiprows=1)
# SEO_r_info = np.loadtxt(os.path.join('./Bmap_sp35/resultsSEO_noError', "SEO_r.txt"), skiprows=1)
# SEO_pr_info = np.loadtxt(os.path.join('./Bmap_sp35/resultsSEO_noError', "SEO_pr.txt"), skiprows=1)
# SEO_r_info = np.loadtxt(os.path.join('./Bmap_sp35/resultsSEO_5GsB4', "SEO_r.txt"), skiprows=1)
# SEO_pr_info = np.loadtxt(os.path.join('./Bmap_sp35/resultsSEO_5GsB4', "SEO_pr.txt"), skiprows=1)
SEO_r_info = np.loadtxt(os.path.join('./Bmap_FD16/resultsSEO_noError', "SEO_r.txt"), skiprows=1)
SEO_pr_info = np.loadtxt(os.path.join('./Bmap_FD16/resultsSEO_noError', "SEO_pr.txt"), skiprows=1)

SEO_fi_axis, SEO_Ek_axis, SEO_r_matrix, SEO_pr_matrix = (
    SEO_r_info[0, 1:], SEO_r_info[1:, 0], SEO_r_info[1:, 1:], SEO_pr_info[1:, 1:])

all_paths = sorted(Path(DATA_DIR0).glob("particle_*.npz"))

for path in all_paths:
    arr = np.load(path)['particles']
    arr = arr[arr[:, -1].argsort()]  # 按时间排序
    if len(arr) < SG_WINDOW:
        continue

    x, px, z, pz = arr[:, COL_X], arr[:, COL_XP], arr[:, COL_Z], arr[:, COL_ZP]
    fi, Ek_MeV = arr[:, COL_fi], arr[:, COL_Ek]
    coords_fi_mod = np.mod(fi, np.pi*2)

    Malloc_r_SEO = np.zeros_like(x)
    Malloc_pr_SEO = np.zeros_like(x)
    flag1 = np.zeros_like(x)
    flag2 = np.zeros_like(x)

    Bilinear_interp_2D_vect_uniform(SEO_Ek_axis[0], SEO_Ek_axis[1]-SEO_Ek_axis[0],
                                    len(SEO_Ek_axis),
                                    SEO_fi_axis[0], SEO_fi_axis[1]-SEO_fi_axis[0],
                                    len(SEO_fi_axis),
                                    SEO_r_matrix,
                                    Ek_MeV, coords_fi_mod,
                                    Malloc_r_SEO, flag1)

    Bilinear_interp_2D_vect_uniform(SEO_Ek_axis[0], SEO_Ek_axis[1]-SEO_Ek_axis[0],
                                    len(SEO_Ek_axis),
                                    SEO_fi_axis[0], SEO_fi_axis[1]-SEO_fi_axis[0],
                                    len(SEO_fi_axis),
                                    SEO_pr_matrix,
                                    Ek_MeV, coords_fi_mod,
                                    Malloc_pr_SEO, flag2)

    if USE_SAVGOL:
        win = SG_WINDOW if SG_WINDOW % 2 else SG_WINDOW + 1
        x = x - savgol_filter(x, win, SG_ORDER)
        px = px - savgol_filter(px, win, SG_ORDER)
        z = z - savgol_filter(z, win, SG_ORDER)
        pz = pz - savgol_filter(pz, win, SG_ORDER)

    x -= Malloc_r_SEO
    px -= Malloc_pr_SEO

    # x=x[:201]; px=px[:201]; z=z[:201]; pz=pz[:201]

    try:
        vx, _, _ = tune_from_xpx(x, px)
        vz, _, _ = tune_from_xpx(z, pz)
        pairs.append((vx, vz))
        pairs2.append((arr[-1, 0], arr[-1, 1], arr[-1, 2], arr[-1, 3]))
    except Exception as e:
        print("FFT tune计算失败:", path.name, e)

vx_all, vz_all = np.array(pairs).T if pairs else (np.array([]), np.array([]))
qqq = np.array(pairs2) if pairs2 else np.empty((0, 4))

# -------------------- 绘制 tune 散点 --------------------
vx_all_norm = 1 - vx_all
vz_all_norm = 1 - vz_all
# vx_all_norm = vx_all
# vz_all_norm = vz_all

# "../output/simulation_g1/FFT_Tune_Dump_0"
# flag0 = (vx_all_norm < 0.45) | (vz_all_norm > 0.25)
# "../output/simulation_g1_70/FFT_Tune_Dump_0"

flag0 = (vx_all_norm > 0.45)
vx_all_norm[flag0] -= 1

# flag0 = (vz_all_norm > 0.25)
# vz_all_norm[flag0] -= 1

# flag0 = (vx_all_norm > 0.3) * (vx_all_norm <0.7) * (vz_all_norm <0.3)
# flag1 = vx_all_norm > 0.8
#
# vx_all_norm[flag1] -= 1
# vz_all_norm[~flag0] -= 1

plt.figure(figsize=(6, 5))
if vx_all.size:
    # plt.scatter(vx_all_norm, vz_all_norm, s=3)
    plt.hist2d(vx_all_norm+3, vz_all_norm+2, bins=[256, 256],range=[[-0.7+3, 0.3+3], [0.9+1, 0.9+2]],norm=LogNorm())
    # plt.hist2d(vx_all_norm + 3+0.15, vz_all_norm + 2 - 0.1, bins=[256, 256], range=[[-0.6 + 3+0.15, 0.4 + 3+0.15], [-0.1 + 2 - 0.1, 0.9 + 2 - 0.1]],
    #            norm=LogNorm())
# print(f"delta vz: {2.17 - np.min(vz_all_norm + 2)}")
# print(f"delta vx: {2.67 - np.min(vx_all_norm + 2)}")
# plt.scatter(0.23+3+0.15, 0.83+2 -0.1, marker='.', s=200, c='red')
# plt.xlim([1.8, 2.8])
# plt.ylim([1.2, 2.2])
plt.xlabel("Qx", fontsize=22)
plt.ylabel("Qz", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
import os
os.makedirs("./prabfig", exist_ok=True)
plt.savefig(
        "./prabfig/fig_app_tuneshift.png",
        dpi=300,
        bbox_inches="tight"
    )
plt.show()

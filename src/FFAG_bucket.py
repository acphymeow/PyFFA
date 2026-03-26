import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from FFAG_MathTools import FFAG_interpolation


@njit(cache=True)
def _interp1d_vec(xg, yg, x):
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    xg0 = xg[0]; xgN = xg[-1]
    for i in range(n):
        xi = x[i]
        if xi <= xg0:
            out[i] = yg[0]
        elif xi >= xgN:
            out[i] = yg[-1]
        else:
            j = np.searchsorted(xg, xi) - 1
            x0 = xg[j]; x1 = xg[j+1]
            y0 = yg[j]; y1 = yg[j+1]
            out[i] = y0 + (y1 - y0) * (xi - x0) / (x1 - x0)
    return out

@njit(cache=True)
def DynamicFunc2_njit(t, X_in, E0, Tc, fi0, w0, U0, Ek_grid, T_grid):
    """
    X_in: (N,2) -> [:,0]=zd (s), [:,1]=dEk (MeV)
    返回 X_out: (N,2) -> zd_dot, dEk_dot (MeV/s)
    """
    zd = X_in[:, 0]
    dEk = X_in[:, 1]
    Ek = E0 + dEk
    T_Ek = _interp1d_vec(Ek_grid, T_grid, Ek)  # s

    X_out = np.empty_like(X_in)
    # zd_dot = ∂H/∂p
    X_out[:, 0] = -(Tc - T_Ek) / Tc
    # dEk_dot = -∂H/∂q
    X_out[:, 1] = (U0 * (np.sin(fi0 + w0 * zd) - np.sin(fi0)) / Tc) / 1e6
    return X_out

# =========================
# 简单参数容器
# =========================
class FFAG_VoidClass:
    def __init__(self):
        pass

class FFAG_bucket:
    def __init__(self, Bmapfoldname,
                 Ek0=300.0, fi0=30.0, U0=100e3, step_t=20e-9, step_N=15000):

        # ---------- 读取频率表并构造插值 ----------
        filename = os.path.join(Bmapfoldname, 'resultsSEO/SEO_ini.txt')
        data = np.loadtxt(filename, skiprows=2)
        Ek_SEO   = data[:, 1].astype(np.float64)
        Freq_SEO = data[:, 4].astype(np.float64)
        Period_SEO = 1.0 / Freq_SEO

        Func_EkFreq = FFAG_interpolation().linear_interpolation_vect(Ek_SEO, Freq_SEO)
        Func_EkPerd = FFAG_interpolation().linear_interpolation_vect(Ek_SEO, Period_SEO)

        order   = np.argsort(Ek_SEO)
        Ek_grid = np.ascontiguousarray(Ek_SEO[order], dtype=np.float64)
        T_grid  = np.ascontiguousarray(Period_SEO[order], dtype=np.float64)

        # ---------- 全局参数 G（VoidClass） ----------
        self.G = FFAG_VoidClass()
        self.G.Func_EkFreq = Func_EkFreq
        self.G.Func_EkPerd = Func_EkPerd
        self.G.Ek_grid = Ek_grid
        self.G.T_grid  = T_grid
        self.G.E0   = float(Ek0)
        self.G.fi0  = np.deg2rad(fi0)
        self.G.U0   = float(U0)
        self.G.step_t   = float(step_t)
        self.G.max_stepsN = int(step_N)
        # 派生量
        self.G.Tc = float(self.G.Func_EkPerd(np.array([self.G.E0], dtype=np.float64))[0])
        self.G.w0 = float(self.G.Func_EkFreq(np.array([self.G.E0], dtype=np.float64))[0] * 2*np.pi)

        # ---------- 生成初始点：分离器附近 + 中心线 ----------
        phi_s = float(self.G.fi0); w0 = float(self.G.w0)
        phi_u = np.pi - phi_s
        self.zd_u = (phi_u - phi_s) / w0
        M_sep = 20
        r_z_ns_sep = 5.0
        r_E_sep    = 0.5
        theta_sep  = np.linspace(0, 2*np.pi, M_sep, endpoint=False)
        zd_ring_sep  = self.zd_u + (r_z_ns_sep * 1e-9) * np.cos(theta_sep)
        dEk_ring_sep = 0.0 + (r_E_sep) * np.sin(theta_sep)
        self.X_sep   = np.column_stack((zd_ring_sep, dEk_ring_sep)).astype(np.float64)

        N_RINGS = 20
        ZLIM_NS = 0.8 * abs(self.zd_u) * 1e9
        z_line_ns = np.linspace(-ZLIM_NS, 0.0, N_RINGS, dtype=np.float64)
        self.X_center = np.column_stack((z_line_ns * 1e-9, np.zeros(N_RINGS, np.float64)))

        self.X_in = np.vstack([self.X_sep, self.X_center])

        # ---------- 先跑轨迹与稳定性 ----------
        self.tr_fwd = self.rk4_solve_dt(self.dynamic, 0.0, self.X_in)
        self.stable_fwd, _ = self.classify_stability(self.tr_fwd)
        self.tr_rev = self.rk4_solve_dt(self.dynamic_neg, 0.0, self.X_in)
        self.stable_rev, _ = self.classify_stability(self.tr_rev)

        # ---------- 坐标范围 ----------
        self.x_min_plot, self.x_max_plot, self.y_min_plot, self.y_max_plot = self._compute_axes_limits()
        self.x_min_sample, self.x_max_sample, self.y_min_sample, self.y_max_sample = self._compute_sample_limits()

        # 预留：H 相关属性
        self.x_ns = self.y_Ek = None
        self.q_grid = self.p_grid = None
        self.Hq_rel = self.Hp_rel = None
        self.H00 = self.H_map = None
        self.H_sep = None

        # 预留：中心线 H-L 曲线缓存
        self._H_arr_center = None
        self._L_mono_center = None

    # ======= 纵向动力学方程 =======
    def dynamic(self, t, X_in):
        G = self.G
        X_in = np.ascontiguousarray(X_in, dtype=np.float64)
        return DynamicFunc2_njit(float(t), X_in,
                                 float(G.E0), float(G.Tc), float(G.fi0),
                                 float(G.w0), float(G.U0),
                                 G.Ek_grid, G.T_grid)

    def dynamic_neg(self, t, X_in):
        return -self.dynamic(t, X_in)

    # ======= 四阶 RK 与求解器 =======
    def rk4_step(self, func, t, r, h):
        k1 = h * func(t, r)
        k2 = h * func(t + 0.5*h, r + 0.5*k1)
        k3 = h * func(t + 0.5*h, r + 0.5*k2)
        k4 = h * func(t + h,     r + k3)
        return (k1 + 2*k2 + 2*k3 + k4) / 6

    def rk4_solve_dt(self, func, t_start, r_start):
        r_points = r_start.copy()
        step_t = float(self.G.step_t)
        max_stepsN = int(self.G.max_stepsN)
        P = r_start.shape[0]
        tr_points = np.zeros((max_stepsN, P, 3), dtype=np.float64)
        t, i = float(t_start), 0
        while i < max_stepsN:
            # if i % 10000 == 0:
            #     print(f"current step id = {i}")
            r_pre = r_points.copy()
            tr_points[i, :, 0] = t
            tr_points[i, :, 1:] = r_pre
            dr = self.rk4_step(func, t, r_pre, step_t)
            r_points = r_pre + dr
            t += step_t
            i += 1
        return tr_points

    # ======= 稳定性判据 =======
    @staticmethod
    def _wrap_pi(x):
        return (x + np.pi) % (2*np.pi) - np.pi

    def classify_stability(self, tr_arr, growth_tol=0.2, frac_split=0.5):
        fi0 = float(self.G.fi0); w0 = float(self.G.w0)
        steps, P, _ = tr_arr.shape
        split = max(4, int(steps*frac_split))
        stable = np.ones(P, dtype=bool)
        reasons = [""]*P
        for pid in range(P):
            zd_hist = tr_arr[:, pid, 1]
            dEk_hist = tr_arr[:, pid, 2]
            phi_rel = self._wrap_pi(fi0 + w0*zd_hist - fi0)
            # if np.any(np.abs(phi_rel) >= np.pi):
            if np.any(np.abs(w0*zd_hist) >= np.pi):
                stable[pid] = False
                reasons[pid] = "crossed |phi-fi0|=pi"
                continue
            phi_amp_1 = np.max(np.abs(phi_rel[:split])); phi_amp_1 = max(phi_amp_1, 1e-12)
            phi_amp_2 = np.max(np.abs(phi_rel[split:])) if split < steps else phi_amp_1
            dE_amp_1  = np.max(np.abs(dEk_hist[:split])); dE_amp_1  = max(dE_amp_1, 1e-12)
            dE_amp_2  = np.max(np.abs(dEk_hist[split:])) if split < steps else dE_amp_1
            R_phi = phi_amp_2/phi_amp_1
            R_E   = dE_amp_2/dE_amp_1
            if (R_phi > 1.0 + growth_tol) or (R_E > 1.0 + growth_tol):
                stable[pid] = False
                reasons[pid] = f"amplitude growth (Rφ={R_phi:.2f}, R_E={R_E:.2f})"
            else:
                reasons[pid] = "bounded"
        return stable, reasons

    # ======= 抽样坐标范围 =======
    def _compute_sample_limits(self):
        _sample_region_Tmin = np.inf
        _sample_region_Tmax = -np.inf
        _sample_region_Emin = np.inf
        _sample_region_Emax = -np.inf

        for tr, mask in ((self.tr_fwd, self.stable_fwd), (self.tr_rev, self.stable_rev)):
            for pid in range(self.X_in.shape[0]):
                x_tr = tr[:, pid, 1] * 1e9
                y_tr = tr[:, pid, 2] + self.G.E0
                if mask[pid]:
                    _sample_region_Tmin = np.min((_sample_region_Tmin, np.min(x_tr)))
                    _sample_region_Tmax = np.max((_sample_region_Tmax, np.max(x_tr)))
                    _sample_region_Emin = np.min((_sample_region_Emin, np.min(y_tr)))
                    _sample_region_Emax = np.max((_sample_region_Emax, np.max(y_tr)))

        return _sample_region_Tmin, _sample_region_Tmax, _sample_region_Emin, _sample_region_Emax

    # ======= 画图坐标范围 =======
    def _compute_axes_limits(self):
        z_blocks, E_blocks = [], []
        for tr, mask in ((self.tr_fwd, self.stable_fwd), (self.tr_rev, self.stable_rev)):
            if mask.any():
                z_blocks.append((tr[:, mask, 1] * 1e9).ravel())
                E_blocks.append((tr[:, mask, 2] + self.G.E0).ravel())
        if z_blocks:
            z_all = np.concatenate(z_blocks)
            E_all = np.concatenate(E_blocks)
            x_min, x_max = z_all.min(), z_all.max()
            y_min, y_max = E_all.min(), E_all.max()
            dx = x_max - x_min; dy = y_max - y_min
            pad_x = 0.10 * dx if dx > 0 else 1.0
            pad_y = 0.10 * dy if dy > 0 else 0.1
            return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y
        else:
            return -20.0, 20.0, self.G.E0 - 2.0, self.G.E0 + 2.0

    # ======= 一维路径积分 =======
    @staticmethod
    def cumtrapz_from_grid_start(y, x):
        acc = np.zeros_like(x)
        for i in range(1, x.size):
            acc[i] = acc[i-1] + 0.5*(y[i-1]+y[i])*(x[i]-x[i-1])
        return acc

    # ======= A(p), B(q) =======
    def A_of_p(self, p_mev):
        Ek = self.G.E0 + p_mev
        T_Ek = np.interp(Ek, self.G.Ek_grid, self.G.T_grid,
                         left=self.G.T_grid[0], right=self.G.T_grid[-1])
        return -(self.G.Tc - T_Ek) / self.G.Tc

    def B_of_q(self, q_s):
        return (self.G.U0 * (np.sin(self.G.fi0 + self.G.w0 * q_s) - np.sin(self.G.fi0)) / self.G.Tc) / 1e6

    # ======= 构造 H(q,p) 并设置 H(0,0)=0 =======
    def build_H(self, Nq=241, Np=241):
        self.x_ns = np.linspace(self.x_min_plot, self.x_max_plot, Nq)
        self.y_Ek = np.linspace(self.y_min_plot, self.y_max_plot, Np)
        self.q_grid = self.x_ns * 1e-9
        self.p_grid = self.y_Ek - self.G.E0

        A_vals = self.A_of_p(self.p_grid)
        B_vals = self.B_of_q(self.q_grid)
        self.Hp_rel = self.cumtrapz_from_grid_start(A_vals, self.p_grid)   # ∫_{pmin}^p A dp
        self.Hq_rel = -self.cumtrapz_from_grid_start(B_vals, self.q_grid)  # -∫_{qmin}^q B dq

        H0_map = self.Hq_rel.reshape(-1,1) + self.Hp_rel.reshape(1,-1)
        Hq0 = np.interp(0.0, self.q_grid, self.Hq_rel, left=np.nan, right=np.nan)
        Hp0 = np.interp(0.0, self.p_grid, self.Hp_rel, left=np.nan, right=np.nan)
        if np.isnan(Hq0):
            Hq0 = self.Hq_rel[0] if abs(0.0 - self.q_grid[0]) < abs(0.0 - self.q_grid[-1]) else self.Hq_rel[-1]
        if np.isnan(Hp0):
            Hp0 = self.Hp_rel[0] if abs(0.0 - self.p_grid[0]) < abs(0.0 - self.p_grid[-1]) else self.Hp_rel[-1]
        self.H00 = Hq0 + Hp0
        self.H_map = H0_map - self.H00

        # 分离器能级（zd=zd_u, dEk=0）
        Hq_rel_s = np.interp(self.zd_u, self.q_grid, self.Hq_rel,
                             left=self.Hq_rel[0], right=self.Hq_rel[-1])
        Hp_rel_0 = np.interp(0.0, self.p_grid, self.Hp_rel,
                             left=self.Hp_rel[0], right=self.Hp_rel[-1])
        self.H_sep = (Hq_rel_s + Hp_rel_0) - self.H00
        pass

    # ======= 画bucket相图 =======
    def plot_phase(self, with_contours=True):
        fig, ax = plt.subplots(figsize=(8.6, 5.6))
        im = ax.imshow(self.H_map.T, origin='lower', aspect='auto',
                       extent=[self.x_min_plot, self.x_max_plot,
                               self.y_min_plot, self.y_max_plot], alpha=0.65)
        cbar = plt.colorbar(im, ax=ax, pad=0.01); cbar.set_label("H (arb.)")

        if with_contours:
            levels = np.linspace(np.nanmin(self.H_map), np.nanmax(self.H_map), 12)
            CS = ax.contour(self.x_ns, self.y_Ek, self.H_map.T, levels=levels, linewidths=0.9)
            ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")

        added_stable = False
        for tr, mask in ((self.tr_fwd, self.stable_fwd), (self.tr_rev, self.stable_rev)):
            for pid in range(self.X_in.shape[0]):
                x_tr = tr[:, pid, 1] * 1e9
                y_tr = tr[:, pid, 2] + self.G.E0
                if mask[pid]:
                    ax.plot(x_tr, y_tr, color='b', lw=0.9, alpha=1.0,
                            label='Stable' if not added_stable else None)
                    added_stable = True
        added_unstable = False
        for tr, mask in ((self.tr_fwd, self.stable_fwd), (self.tr_rev, self.stable_rev)):
            for pid in range(self.X_in.shape[0]):
                x_tr = tr[:, pid, 1] * 1e9
                y_tr = tr[:, pid, 2] + self.G.E0
                if not mask[pid]:
                    ax.plot(x_tr, y_tr, color='r', lw=0.9, alpha=0.35,
                            label='Unstable' if not added_unstable else None)
                    added_unstable = True

        ax.set_xlim(self.x_min_plot, self.x_max_plot)
        ax.set_ylim(self.y_min_plot, self.y_max_plot)
        ax.set_xlabel("zd (ns)"); ax.set_ylabel("Ek (MeV)")
        ax.set_title(f"H contours & heatmap  |  fi0={np.rad2deg(self.G.fi0):.1f}°, U0={self.G.U0/1e3:.0f} kV")
        ax.grid(True); ax.minorticks_on()
        if added_stable or added_unstable:
            ax.legend(loc='best')
        plt.tight_layout()
        return fig, ax

    # ======= 计算中心线的 H–投影束长=======
    def compute_center_H_vs_span(self):
        # 用 X_center 的稳定轨迹，得到 (H, L) 点
        n_sep = self.X_sep.shape[0]
        n_ctr = self.X_center.shape[0]
        center_pids = np.arange(n_sep, n_sep + n_ctr)

        q0s = self.X_center[:, 0]  # s
        p0s = self.X_center[:, 1]  # MeV
        H_center = (np.interp(q0s, self.q_grid, self.Hq_rel,
                              left=self.Hq_rel[0], right=self.Hq_rel[-1]) +
                    np.interp(p0s, self.p_grid, self.Hp_rel,
                              left=self.Hp_rel[0], right=self.Hp_rel[-1]) -
                    self.H00)

        H_list, L_list = [], []
        for i_local, pid in enumerate(center_pids):
            L_candidates = []
            if self.stable_fwd[pid]:
                z_hist = self.tr_fwd[:, pid, 1] * 1e9
                L_candidates.append(float(z_hist.max() - z_hist.min()))
            if self.stable_rev[pid]:
                z_hist = self.tr_rev[:, pid, 1] * 1e9
                L_candidates.append(float(z_hist.max() - z_hist.min()))
            if not L_candidates:
                continue
            L_use = min(L_candidates)  # 两个都稳定时取较小值
            H_list.append(float(H_center[i_local]))
            L_list.append(L_use)

        if not H_list:
            raise RuntimeError("[X_center] 没有可用的稳定轨迹用于束长。")

        H_arr = np.array(H_list); L_arr = np.array(L_list)
        order = np.argsort(H_arr)
        H_arr = H_arr[order]; L_arr = L_arr[order]

        # 缓存
        self._H_arr_center = H_arr
        self._L_mono_center = L_arr
        return H_arr, L_arr

    # ======= 画中心线 H–束长图=======
    def plot_center_H_vs_span(self):
        H_arr, L_mono = (self._H_arr_center, self._L_mono_center)
        if H_arr is None or L_mono is None:
            H_arr, L_mono = self.compute_center_H_vs_span()

        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        ax.plot(L_mono, H_arr, "-o", ms=3, lw=1.2, label="projection length (zd span)")
        ax.axhline(self.H_sep, ls="--", lw=1.2, label=r"$H_{\rm sep}$")
        ax.set_xlabel("bunch length(ns)")
        ax.set_ylabel("H (arb.)")
        ax.grid(True); ax.minorticks_on(); ax.legend(loc="best")
        plt.tight_layout()
        return fig, ax

    # ======= 散点筛选函数（输入 N_demo 与目标束长）=======
    def scatter_filter_by_span(self, N_demo, L_target_ns,  show=True):
        """
        根据中心线 H–束长曲线，将目标束长 L_target_ns 映射到阈值 H_thr，
        然后在当前相图范围内随机生成 N_demo 个粒子，筛选出 H <= H_thr 的粒子。

        Returns (若 return_arrays=True):
            (x_ns_sc, y_Ek_sc, H_pts, mask, H_thr)
        """

        # 确保 H 构建完、中心线曲线可用
        if self.H_map is None:
            self.build_H()
        H_arr, L_mono = (self._H_arr_center, self._L_mono_center)
        if H_arr is None or L_mono is None:
            H_arr, L_mono = self.compute_center_H_vs_span()

        # 目标束长->H 阈值
        L_min, L_max = float(L_mono.min()), float(L_mono.max())
        L_use = float(np.clip(L_target_ns, L_min, L_max))
        L_mono_idx = np.argsort(L_mono)
        L_mono_sorted = L_mono[L_mono_idx]
        H_arr_sorted = H_arr[L_mono_idx]
        temp=FFAG_interpolation().linear_interpolation(L_mono_sorted, H_arr_sorted)
        H_thr = temp(L_use)

        # 生成散点
        xy_sc_all = np.zeros((0,2))
        while np.size(xy_sc_all,0) < N_demo:
            rng = np.random.default_rng()
            x_ns_sc = rng.uniform(self.x_min_sample, self.x_max_sample, size=int(N_demo)*10)  # ns
            y_Ek_sc = rng.uniform(self.y_min_sample, self.y_max_sample, size=int(N_demo)*10)  # MeV
            # 计算散点 H
            q_pts = x_ns_sc * 1e-9
            p_pts = y_Ek_sc - self.G.E0
            Hq_rel_pts = np.interp(q_pts, self.q_grid, self.Hq_rel,
                                   left=self.Hq_rel[0], right=self.Hq_rel[-1])
            Hp_rel_pts = np.interp(p_pts, self.p_grid, self.Hp_rel,
                                   left=self.Hp_rel[0], right=self.Hp_rel[-1])
            H_pts = (Hq_rel_pts + Hp_rel_pts) - self.H00
            # 筛选规则：H <= H_thr
            mask = H_pts >= H_thr
            xy_sc_all=np.vstack((xy_sc_all, np.vstack((x_ns_sc[mask], y_Ek_sc[mask])).T))

        xy_sc_all =xy_sc_all[:N_demo]

        # 画图
        if show:
            fig, ax = plt.subplots(figsize=(8.0, 5.6))
            ax.scatter(x_ns_sc[~mask], y_Ek_sc[~mask], s=2, alpha=0.25, color='0.7', label='rejected')
            ax.scatter(xy_sc_all[:, 0], xy_sc_all[:, 1], s=3, alpha=0.85, color='g', label='accepted')
            ax.set_xlim(self.x_min_plot, self.x_max_plot)
            ax.set_ylim(self.y_min_plot, self.y_max_plot)
            ax.set_xlabel("zd (ns)"); ax.set_ylabel("Ek (MeV)")
            ax.set_title(f"Scatter filter by target bunch length L={L_target_ns:.3g} ns (H_thr={H_thr:.4g})")
            ax.grid(True); ax.minorticks_on(); ax.legend(loc='best')
            plt.tight_layout()

        return xy_sc_all[:, 0], xy_sc_all[:, 1], H_thr


if __name__ == "__main__":
    bucket = FFAG_bucket(Bmapfoldname="./Bmap", Ek0=300.0, fi0=3, U0=300e3, step_t=10e-9, step_N=30000)

    # 构建 H
    bucket.build_H(Nq=241, Np=241)

    # 相图
    bucket.plot_phase(with_contours=True)

    # 画中心线 H–束长
    bucket.plot_center_H_vs_span()

    # ===== 散点筛选：给定 N_demo 与 目标束长（单位 ns） =====
    x_ns_sc, y_Ek_sc, H_thr = bucket.scatter_filter_by_span(N_demo=30000, L_target_ns=80.0, show=True)

    plt.show()

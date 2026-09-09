# -*- coding: utf-8 -*-
"""
对数螺旋(2D 平面)：

    s(r) = s_ref + a * ln(r/r_ref)
    a = tan(alpha)

极坐标形式：
    r : 半径
    s : 极角

笛卡尔坐标：
    x = r cos(s)
    y = r sin(s)

目标：
    给定柱坐标网格点 (r,phi)，计算到螺旋线的最近距离
"""
import numpy as np
import matplotlib.pyplot as plt
import numba as nb


# ============================================================
# 核心函数：计算网格点到螺旋线的最近距离
# ============================================================
@nb.njit(parallel=True, fastmath=True)
def spiral_distance_direction_log(
    r_grid, phi_grid,
    r_ref, s_ref, alpha_rad,
    r_min, r_max,
    n_iter=6
):
    """
    返回：
      dist, ux, uy, flag
    flag:
      0 = 内部解（非端点）
      1 = 选择了 r_min 端点
      2 = 选择了 r_max 端点
    """
    a = np.tan(alpha_rad)
    a2 = a * a
    one_plus_a2 = 1.0 + a2  # ||c'(r)||^2 = 1 + a^2（对数螺旋）

    dist = np.empty(r_grid.shape, dtype=np.float64)
    ux   = np.empty(r_grid.shape, dtype=np.float64)
    uy   = np.empty(r_grid.shape, dtype=np.float64)
    flag = np.zeros(r_grid.shape, dtype=np.int32)

    n = r_grid.size
    rg = r_grid.ravel()
    pg = phi_grid.ravel()
    out_d  = dist.ravel()
    out_ux = ux.ravel()
    out_uy = uy.ravel()
    out_f  = flag.ravel()

    # 端点判断阈值：按尺度给一个相对量更稳
    eps = 1e-12 * (1.0 + r_max)

    for idx in nb.prange(n):
        r_p = rg[idx]
        phi_p = pg[idx]

        # 网格点 p 的笛卡尔
        x_p = r_p * np.cos(phi_p)
        y_p = r_p * np.sin(phi_p)

        # 初值并夹到定义域
        r = r_p
        if r < r_min:
            r = r_min
        if r > r_max:
            r = r_max
        if r <= 0.0:
            r = max(r_min, 1e-12)

        # 牛顿迭代（内部解）
        for _ in range(n_iter):
            s  = s_ref + a * np.log(r / r_ref)
            cs = np.cos(s)
            sn = np.sin(s)

            x = r * cs
            y = r * sn

            x1 = cs - a * sn
            y1 = sn + a * cs

            invr = 1.0 / r
            x2 = -(a * invr) * (sn + a * cs)
            y2 =  (a * invr) * (cs - a * sn)

            dx = x - x_p
            dy = y - y_p

            g  = dx * x1 + dy * y1
            gp = one_plus_a2 + dx * x2 + dy * y2

            if np.abs(gp) < 1e-14:
                break

            r_new = r - g / gp

            # 夹逼到定义域
            if r_new < r_min:
                r_new = r_min
            if r_new > r_max:
                r_new = r_max
            if r_new <= 0.0:
                r_new = max(r_min, 1e-12)

            if np.abs(r_new - r) <= 1e-12 * (1.0 + r):
                r = r_new
                break
            r = r_new

        # --------------------------------------------------
        # 若落在端点附近：计算两个端点，取更小的那个
        # --------------------------------------------------
        use_endpoint_check = (np.abs(r - r_min) <= eps) or (np.abs(r - r_max) <= eps)

        if use_endpoint_check:
            # 端点1：r_min
            s1 = s_ref + a * np.log(r_min / r_ref)
            x1 = r_min * np.cos(s1)
            y1 = r_min * np.sin(s1)
            d1x = x_p - x1
            d1y = y_p - y1
            d1  = np.sqrt(d1x*d1x + d1y*d1y)

            # 端点2：r_max
            s2 = s_ref + a * np.log(r_max / r_ref)
            x2p = r_max * np.cos(s2)
            y2p = r_max * np.sin(s2)
            d2x = x_p - x2p
            d2y = y_p - y2p
            d2  = np.sqrt(d2x*d2x + d2y*d2y)

            if d1 <= d2:
                # 选择 r_min 端点
                out_f[idx] = 1
                out_d[idx] = d1
                if d1 > 0.0:
                    out_ux[idx] = d1x / d1
                    out_uy[idx] = d1y / d1
                else:
                    out_ux[idx] = 0.0
                    out_uy[idx] = 0.0
            else:
                # 选择 r_max 端点
                out_f[idx] = 2
                out_d[idx] = d2
                if d2 > 0.0:
                    out_ux[idx] = d2x / d2
                    out_uy[idx] = d2y / d2
                else:
                    out_ux[idx] = 0.0
                    out_uy[idx] = 0.0

        else:
            # --------------------------------------------------
            # 正常内部解：用牛顿得到的 r 计算最近点
            # --------------------------------------------------
            out_f[idx] = 0

            s = s_ref + a * np.log(r / r_ref)
            x = r * np.cos(s)
            y = r * np.sin(s)

            vx = x_p - x
            vy = y_p - y
            d  = np.sqrt(vx*vx + vy*vy)

            out_d[idx] = d
            if d > 0.0:
                out_ux[idx] = vx / d
                out_uy[idx] = vy / d
            else:
                out_ux[idx] = 0.0
                out_uy[idx] = 0.0

    return dist, ux, uy, flag


# ============================================================
# 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 螺旋参数
    # --------------------------------------------------------

    alpha_deg = 25.0
    alpha_rad = np.deg2rad(alpha_deg)

    # 螺旋线参考点
    r_ref = 0.30
    s_ref = 0.30

    # 螺旋截断范围
    r_min = 0.20
    r_max = 1.20

    # --------------------------------------------------------
    # 极坐标网格
    # --------------------------------------------------------

    nr = 240
    nphi = 360

    r_axis = np.linspace(r_min,r_max,nr)
    phi_axis = np.linspace(-np.pi,np.pi,nphi)

    R,PHI = np.meshgrid(r_axis,phi_axis,indexing="ij")

    # --------------------------------------------------------
    # 要测试的点
    # --------------------------------------------------------

    XY_point = [[-0.9,0.3],[-0.8,0.2]]

    # --------------------------------------------------------
    # 计算最近距离
    # --------------------------------------------------------

    dist,ux,uy,_ = spiral_distance_direction_log(
        R,PHI,
        r_ref,s_ref,alpha_rad,
        r_min,r_max
    )

    # --------------------------------------------------------
    # 可视化
    # --------------------------------------------------------

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")

    # 画螺旋线
    a = np.tan(alpha_rad)

    r_curve = np.linspace(r_min,r_max,1000)

    s_curve = s_ref + a*np.log(r_curve/r_ref)

    x_curve = r_curve*np.cos(s_curve)
    y_curve = r_curve*np.sin(s_curve)

    ax.plot(x_curve,y_curve,lw=2)

    # 画参考圆
    th = np.linspace(-np.pi,np.pi,400)

    ax.plot(r_min*np.cos(th),r_min*np.sin(th),alpha=0.4)
    ax.plot(r_max*np.cos(th),r_max*np.sin(th),alpha=0.4)

    # --------------------------------------------------------
    # 绘制测试点和最近连线
    # --------------------------------------------------------

    for x,y in XY_point:

        r = np.sqrt(x*x+y*y)
        phi = np.arctan2(y,x)

        # 找到最近网格
        i = np.argmin(np.abs(r_axis-r))
        j = np.argmin(np.abs(phi_axis-phi))

        # 网格位置
        rp = R[i,j]
        phip = PHI[i,j]

        xp = rp*np.cos(phip)
        yp = rp*np.sin(phip)

        # 最近点
        d = dist[i,j]

        vx = ux[i,j]*d
        vy = uy[i,j]*d

        x_star = xp - vx
        y_star = yp - vy

        # 绘图
        ax.plot(x,y,"ro")
        ax.plot(x_star,y_star,"bx")
        ax.plot([x_star,x],[y_star,y],"k-")

    plt.show()


if __name__ == "__main__":
    main()
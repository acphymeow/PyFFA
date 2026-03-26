import numpy as np
from numba.typed import List
from numba import njit, prange
from FFAG_MathTools import fast_mod_njit
from mpi4py import MPI


@njit
def find_two_points_non_uniform_vect_njit(X, xi):
    right_point = np.searchsorted(X, xi, side='right')
    left_point = right_point - 1

    # Check if xi is out of the range of X
    idx_array_0, idx_array_1, flag = (
        left_point, right_point, np.zeros_like(xi))

    # if left_point < 0
    flag_left_point_less_0 = left_point < 0
    idx_array_0[flag_left_point_less_0] = 0
    idx_array_1[flag_left_point_less_0] = 1
    flag[flag_left_point_less_0] = 1

    # if right_right_point > len(X) - 1
    flag_right_point_larger_len = right_point > len(X) - 1
    idx_array_0[flag_right_point_larger_len] = len(X) - 2
    idx_array_1[flag_right_point_larger_len] = len(X) - 1
    flag[flag_right_point_larger_len] = 1

    idx_array = np.vstack((idx_array_0, idx_array_1))

    return idx_array, flag


@njit
def find_two_points_uniform_vectorized(grid, xi):
    """
    查找 xi 在均匀网格 grid 中对应的两个插值点下标。
    grid: 1D 均匀网格 (e.g., linspace)
    xi:   粒子坐标数组 (N,)

    返回：
        idx_array: (2, N)，每列为 (left, right)
        flag: (N,) 是否越界，0=正常，1=越界
    """
    x_min = grid[0]
    x_max = grid[-1]
    N = len(grid)
    dx = (x_max - x_min) / (N - 1)

    # 粗略估计插值点位置
    pos = (xi - x_min) / dx
    left = np.floor(pos).astype(np.int32)
    right = left + 1

    # 标记越界粒子
    flag = xi * 0.0

    mask_left = left < 0
    mask_right = right >= N

    left[mask_left] = 0
    right[mask_left] = 1
    flag[mask_left] = 1

    left[mask_right] = N - 2
    right[mask_right] = N - 1
    flag[mask_right] = 1

    # 拼成 (2, N) 索引数组
    idx_array = np.vstack((left, right))
    return idx_array, flag


@njit()
def GetLocalCoordinates_2p5(LocalBunch, Step_Survive,
                             malloc_r, malloc_fi, malloc_z,
                             malloc_gamma, malloc_mean_gamma, malloc_fiMod,Malloc_SC_surviveIDX,
                             speed_c):

    rows = LocalBunch.shape[0]
    j = 0
    sum_gamma = 0
    TWO_PI = 2 * np.pi
    for i in range(rows):
        if Step_Survive[i]:
            r = LocalBunch[i, 0]
            drdt = LocalBunch[i, 1]
            z = LocalBunch[i, 2]
            dzdt = LocalBunch[i, 3]
            fi = LocalBunch[i, 4]
            dphidt = LocalBunch[i, 5]
            r_dfidt = r * dphidt

            v2 = drdt ** 2 + dzdt ** 2 + r_dfidt ** 2
            gamma = 1.0 / np.sqrt(1.0 - v2 / speed_c ** 2)

            malloc_r[j] = r
            malloc_z[j] = z
            malloc_fi[j] = fi
            malloc_gamma[j] = gamma
            malloc_mean_gamma[j] = 1.0
            malloc_fiMod[j] = fi - TWO_PI * np.floor(fi / TWO_PI)
            Malloc_SC_surviveIDX[j]=i

            sum_gamma += gamma
            j += 1

    return (malloc_r[:j], malloc_fi[:j], malloc_z[:j],
            malloc_gamma[:j], malloc_mean_gamma[:j], malloc_fiMod[:j], Malloc_SC_surviveIDX[:j], j, sum_gamma)

@njit()
def GetLocalCoordinates_2p5_rk(LocalBunch, Step_Survive,
                             malloc_r, malloc_fi, malloc_z,
                             malloc_gamma, malloc_mean_gamma, malloc_fiMod,Malloc_SC_surviveIDX,
                             speed_c):
    E0_J = 938.2723e6 * 1.60217662e-19  # J
    rows = LocalBunch.shape[0]
    j = 0
    sum_gamma = 0
    TWO_PI = 2 * np.pi
    for i in range(rows):
        if Step_Survive[i]:
            r = LocalBunch[i, 0]
            drdt = LocalBunch[i, 1]
            z = LocalBunch[i, 2]
            dzdt = LocalBunch[i, 3]
            fi = LocalBunch[i, 4]
            # dphidt = LocalBunch[i, 5]
            # r_dfidt = r * dphidt
            Etot_J= LocalBunch[i, 5]

            # v2 = drdt ** 2 + dzdt ** 2 + r_dfidt ** 2
            # gamma = 1.0 / np.sqrt(1.0 - v2 / speed_c ** 2)
            gamma=Etot_J/E0_J

            malloc_r[j] = r
            malloc_z[j] = z
            malloc_fi[j] = fi
            malloc_gamma[j] = gamma
            malloc_mean_gamma[j] = 1.0
            malloc_fiMod[j] = fi - TWO_PI * np.floor(fi / TWO_PI)
            Malloc_SC_surviveIDX[j]=i

            sum_gamma += gamma
            j += 1

    return (malloc_r[:j], malloc_fi[:j], malloc_z[:j],
            malloc_gamma[:j], malloc_mean_gamma[:j], malloc_fiMod[:j], Malloc_SC_surviveIDX[:j], j, sum_gamma)

@njit()
def my_3dInterp_vect(x, y, z, values, xi, yi, zi):

    # 找到目标点的左右边界索引，并获取范围标志
    x_idx, x_flag = find_two_points_non_uniform_vect_njit(x, xi)
    y_idx, y_flag = find_two_points_non_uniform_vect_njit(y, yi)
    z_idx, z_flag = find_two_points_non_uniform_vect_njit(z, zi)

    interp_values = xi * 0.0

    # 合并范围标志，标记超出任意方向边界的点
    out_of_bounds_flag = x_flag + y_flag + z_flag

    for idxs in range(len(xi)):
        # 如果当前点超出网格范围，直接赋值为0
        if out_of_bounds_flag[idxs] > 0.01:
            interp_values[idxs] = 0.0
        else:
            xi_current, yi_current, zi_current = xi[idxs], yi[idxs], zi[idxs]

            x_left, x_right = x[x_idx[0, idxs]], x[x_idx[1, idxs]]
            y_left, y_right = y[y_idx[0, idxs]], y[y_idx[1, idxs]]
            z_left, z_right = z[z_idx[0, idxs]], z[z_idx[1, idxs]]

            x0_00 = values[x_idx[0, idxs], y_idx[0, idxs], z_idx[0, idxs]]
            x0_01 = values[x_idx[0, idxs], y_idx[0, idxs], z_idx[1, idxs]]
            x0_10 = values[x_idx[0, idxs], y_idx[1, idxs], z_idx[0, idxs]]
            x0_11 = values[x_idx[0, idxs], y_idx[1, idxs], z_idx[1, idxs]]

            x1_00 = values[x_idx[1, idxs], y_idx[0, idxs], z_idx[0, idxs]]
            x1_01 = values[x_idx[1, idxs], y_idx[0, idxs], z_idx[1, idxs]]
            x1_10 = values[x_idx[1, idxs], y_idx[1, idxs], z_idx[0, idxs]]
            x1_11 = values[x_idx[1, idxs], y_idx[1, idxs], z_idx[1, idxs]]

            x0_z0_interp = (yi_current - y_left) / (y_right - y_left) * x0_10 + (y_right - yi_current) / (
                        y_right - y_left) * x0_00
            x0_z1_interp = (yi_current - y_left) / (y_right - y_left) * x0_11 + (y_right - yi_current) / (
                        y_right - y_left) * x0_01
            x0_interp = (zi_current - z_left) / (z_right - z_left) * x0_z1_interp + (z_right - zi_current) / (
                        z_right - z_left) * x0_z0_interp

            x1_z0_interp = (yi_current - y_left) / (y_right - y_left) * x1_10 + (y_right - yi_current) / (
                        y_right - y_left) * x1_00
            x1_z1_interp = (yi_current - y_left) / (y_right - y_left) * x1_11 + (y_right - yi_current) / (
                        y_right - y_left) * x1_01
            x1_interp = (zi_current - z_left) / (z_right - z_left) * x1_z1_interp + (z_right - zi_current) / (
                        z_right - z_left) * x1_z0_interp

            interp_values[idxs] = (xi_current - x_left) / (x_right - x_left) * x1_interp + (x_right - xi_current) / (
                        x_right - x_left) * x0_interp

    return interp_values


@njit()
def DistributeCharge2D_Njit(x, z, q_macro, BunchLength, xmin, xmax, zmin, zmax, nx, nz):
    rho = np.zeros((nz,nx), np.float32) # 横向2维电荷密度
    dx = (xmax-xmin)/nx; dz = (zmax-zmin)/nz
    if dx<=0 or dz<=0 or BunchLength<=0:
        return rho
    if x.shape[0] == 0:
        return rho

    inv_area = 1.0/(dx*dz)
    # inv_area = 1.0
    q = q_macro / BunchLength * inv_area
    for k in range(x.shape[0]):
        rx=(x[k]-xmin)/dx; rz=(z[k]-zmin)/dz
        ix=int(np.floor(rx)); iz=int(np.floor(rz))
        # if ix<0 or ix>=nx-1 or iz<0 or iz>=nz-1: continue
        fx=rx-ix; fz=rz-iz
        # ---------- x 方向边缘裁剪 ----------
        if ix < 0:  # 左边界外
            ix = 0
            fx = 0.0  # 全部电荷给 ix
        elif ix >= nx - 1:  # 右边界外（含最右列中心线右侧）
            ix = nx - 2
            fx = 1.0  # 全部电荷给 ix+1

        # ---------- z 方向边缘裁剪 ----------
        if iz < 0:  # 下边界外
            iz = 0
            fz = 0.0
        elif iz >= nz - 1:  # 上边界外
            iz = nz - 2
            fz = 1.0

        rho[iz  ,ix  ] += (1-fx)*(1-fz)*q
        rho[iz  ,ix+1] +=  fx   *(1-fz)*q
        rho[iz+1,ix  ] += (1-fx)* fz   *q
        rho[iz+1,ix+1] +=  fx   * fz   *q

    return rho

@njit()
def DistributeChargeSBS_Njit(
    z_slices, x_slices, count_slices,
    q_macro, BunchLength,
    xmin, xmax, zmin, zmax,
    nx, nz, ns
):
    """
    Slice-By-Slice 2D 沉积：对每个 slice k，把 (x,z) 粒子沉积到 rho_3d[k, :, :]
    归一：q = q_macro / BunchLength / (dx*dz)
    """
    rho_3d = np.zeros((ns, nz, nx), np.float32)

    dx = (xmax - xmin) / nx
    dz = (zmax - zmin) / nz
    if dx <= 0.0 or dz <= 0.0 or BunchLength <= 0.0:
        return rho_3d

    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    # q = (q_macro / BunchLength) * (1.0 / (dx * dz))  # 面密度归一
    q = (q_macro / BunchLength * ns) * (1.0 / (dx * dz))  # 面密度归一

    for s in range(ns):
        cnt = count_slices[s]
        if cnt <= 0:
            continue

        # 该 slice 的有效 (x,z) 列表
        xs = x_slices[s, :cnt]
        zs = z_slices[s, :cnt]

        # 双线性沉积
        for i in range(cnt):
            rx = (xs[i] - xmin) * inv_dx
            rz = (zs[i] - zmin) * inv_dz

            ix = int(np.floor(rx))
            iz = int(np.floor(rz))

            fx = rx - ix
            fz = rz - iz

            # ---- 边界裁剪（把越界分量推回边缘单元）----
            if ix < 0:
                ix = 0
                fx = 0.0
            elif ix >= nx - 1:
                ix = nx - 2
                fx = 1.0

            if iz < 0:
                iz = 0
                fz = 0.0
            elif iz >= nz - 1:
                iz = nz - 2
                fz = 1.0

            w00 = (1.0 - fx) * (1.0 - fz) * q
            w10 = (       fx) * (1.0 - fz) * q
            w01 = (1.0 - fx) * (       fz) * q
            w11 = (       fx) * (       fz) * q

            rho_3d[s, iz    , ix    ] += w00
            rho_3d[s, iz    , ix + 1] += w10
            rho_3d[s, iz + 1, ix    ] += w01
            rho_3d[s, iz + 1, ix + 1] += w11

    return rho_3d


# calculate static electric field for all grid points
@njit()
def CalculateEFieldFromVoltage(x_grid, y_grid, z_grid, voltage_distribution):
    n, m, l = voltage_distribution.shape
    Ez_distribution = np.zeros((n, m, l))
    Ex_distribution = np.zeros((n, m, l))
    Ey_distribution = np.zeros((n, m, l))

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]

    for i in range(1, n - 1):
        for j in range(1, m - 1):
            for k in range(1, l - 1):
                # Calculate the partial derivatives using central difference
                Ex_distribution[i, j, k] = -(voltage_distribution[i + 1, j, k] - voltage_distribution[i - 1, j, k]) / (
                        2 * dx)
                Ey_distribution[i, j, k] = -(voltage_distribution[i, j + 1, k] - voltage_distribution[i, j - 1, k]) / (
                        2 * dy)
                Ez_distribution[i, j, k] = -(voltage_distribution[i, j, k + 1] - voltage_distribution[i, j, k - 1]) / (
                        2 * dz)

        # 处理边界点
        # x方向边界
        for j in range(m):
            for k in range(l):
                Ex_distribution[0, j, k] = -(voltage_distribution[1, j, k] - voltage_distribution[0, j, k]) / dx
                Ex_distribution[n - 1, j, k] = -(
                        voltage_distribution[n - 1, j, k] - voltage_distribution[n - 2, j, k]) / dx

        # y方向边界
        for i in range(n):
            for k in range(l):
                Ey_distribution[i, 0, k] = -(voltage_distribution[i, 1, k] - voltage_distribution[i, 0, k]) / dy
                Ey_distribution[i, m - 1, k] = -(
                        voltage_distribution[i, m - 1, k] - voltage_distribution[i, m - 2, k]) / dy

        # z方向边界
        for i in range(n):
            for j in range(m):
                Ez_distribution[i, j, 0] = -(voltage_distribution[i, j, 1] - voltage_distribution[i, j, 0]) / dz
                Ez_distribution[i, j, l - 1] = -(
                        voltage_distribution[i, j, l - 1] - voltage_distribution[i, j, l - 2]) / dz



    return Ex_distribution, Ey_distribution, Ez_distribution


@njit(parallel=False)
def ComputeSliceBoundsAndCoords_merged(coords_r, coords_z, coords_fi_mod, coords_r0,
                                       coords_x,
                                       slice_fi_step,
                                       fi_s_global,
                                       segments,
                                       r2d, z2d, slice_indices):
    """
    将粒子按 φ 均匀划分为多个 slice, 完成 r→x 坐标变换，
    并统计整个束团的 x/z 均值与标准差
    访问第 s 个 slice 的粒子坐标:
    r_slice_s = r_slices[s, :count_slices[s]], z_slice_s = z_slices[s, :count_slices[s]]

    Parameters
    ----------
    coords_r, coords_z : ndarray[float64, (N,)]
        粒子的 r, z 坐标
    coords_fi_mod : ndarray[float64, (N,)]
        粒子的 φ（弧度）∈ [0, 2π)
    coords_r0 : ndarray[float64, (N,)]
        每个粒子对应的参考闭轨半径 r₀
    coords_x : ndarray[float64, (N,)]
        输出：（预生成的）自然坐标 x = r - r₀
    slice_fi_step : float
        每个 slice φ 宽度
    fi_s_global : float
        φ 的全局起始角度
    segments : int
        slice 总数
    r2d, z2d : ndarray[float64, (segments, Nmax)]
        输出：各 slice 内粒子的 r/z 坐标列表
    slice_indices : ndarray[int32, (N,)]
        输出：每个粒子属于哪个 slice

    Returns
    -------
    r2d, z2d : 每个 slice 的粒子坐标
    counts   : 每个 slice 的粒子数
    slice_indices : 每粒子所属 slice 索引
    mean_x, sigma_x : 整个束团横向统计量
    mean_z, sigma_z : 整个束团纵向统计量
    """
    N = coords_r.shape[0]
    TWO_PI = 2.0 * np.pi

    counts   = np.zeros(segments, dtype=np.int32)
    sum_x    = 0.0
    sum_z    = 0.0
    sum_x2   = 0.0
    sum_z2   = 0.0

    for i in range(N):
        # x = r - r₀
        x = coords_r[i] - coords_r0[i]
        z = coords_z[i]
        coords_x[i] = x

        # 累加全局统计量
        sum_x  += x
        sum_z  += z
        sum_x2 += x * x
        sum_z2 += z * z

        # φ → slice 索引
        fi = coords_fi_mod[i]
        if fi < fi_s_global:
            fi += TWO_PI
        s = int((fi - fi_s_global) / slice_fi_step)
        if s < 0:
            s = 0
        elif s >= segments:
            s = segments - 1

        slice_indices[i] = s

        # 存入对应 slice 的坐标列表
        pos = counts[s]
        r2d[s, pos] = coords_r[i]
        z2d[s, pos] = z
        counts[s] += 1

    return coords_x[:N], r2d, z2d, counts, slice_indices, sum_x, sum_z, sum_x2, sum_z2


@njit(parallel=False)
def ComputeSliceBoundsAndCoordsSBS_merged(coords_r, coords_z, coords_fi_mod, coords_r0,
                                       coords_x,
                                       slice_fi_step,
                                       fi_s_global,
                                       segments,
                                       r2d, z2d, x2d, slice_indices):
    """
    将粒子按 φ 均匀划分为多个 slice, 完成 r→x 坐标变换，
    并统计整个束团的 x/z 均值与标准差
    访问第 s 个 slice 的粒子坐标:
    r_slice_s = r_slices[s, :count_slices[s]], z_slice_s = z_slices[s, :count_slices[s]]

    Parameters
    ----------
    coords_r, coords_z : ndarray[float64, (N,)]
        粒子的 r, z 坐标
    coords_fi_mod : ndarray[float64, (N,)]
        粒子的 φ（弧度）∈ [0, 2π)
    coords_r0 : ndarray[float64, (N,)]
        每个粒子对应的参考闭轨半径 r₀
    coords_x : ndarray[float64, (N,)]
        输出：（预生成的）自然坐标 x = r - r₀
    slice_fi_step : float
        每个 slice φ 宽度
    fi_s_global : float
        φ 的全局起始角度
    segments : int
        slice 总数
    r2d, z2d : ndarray[float64, (segments, Nmax)]
        输出：各 slice 内粒子的 r/z 坐标列表
    slice_indices : ndarray[int32, (N,)]
        输出：每个粒子属于哪个 slice

    Returns
    -------
    r2d, z2d : 每个 slice 的粒子坐标
    counts   : 每个 slice 的粒子数
    slice_indices : 每粒子所属 slice 索引
    mean_x, sigma_x : 整个束团横向统计量
    mean_z, sigma_z : 整个束团纵向统计量
    """
    N = coords_r.shape[0]
    TWO_PI = 2.0 * np.pi

    counts   = np.zeros(segments, dtype=np.int32)
    sum_x    = 0.0
    sum_z    = 0.0
    sum_x2   = 0.0
    sum_z2   = 0.0

    for i in range(N):
        # x = r - r₀
        x = coords_r[i] - coords_r0[i]
        z = coords_z[i]
        coords_x[i] = x

        # 累加全局统计量
        sum_x  += x
        sum_z  += z
        sum_x2 += x * x
        sum_z2 += z * z

        # φ → slice 索引
        fi = coords_fi_mod[i]
        if fi < fi_s_global:
            fi += TWO_PI
        s = int((fi - fi_s_global) / slice_fi_step) # 第几个slice
        if s < 0:
            s = 0
        elif s >= segments:
            s = segments - 1

        slice_indices[i] = s

        # 存入对应 slice 的坐标列表
        pos = counts[s]
        r2d[s, pos] = coords_r[i]
        z2d[s, pos] = z
        x2d[s, pos] = x
        counts[s] += 1

    return coords_x[:N], r2d, z2d, x2d, counts, slice_indices, sum_x, sum_z, sum_x2, sum_z2


@njit()
# @profile
def divide_arc2(points_fi):
    """返回 (arc_start, arc_end, arc_len)，角度均为弧度。"""
    shifts  = np.array([0.0,
                        np.pi,
                        np.pi * 0.5,
                        np.pi * 1.5], dtype=np.float64)

    min_v   = np.full(4,  1e30, dtype=np.float64)
    max_v   = np.full(4, -1e30, dtype=np.float64)

    # ---- 单次扫描 ----------------------------------
    for phi in points_fi:
        for k in range(4):
            p = fast_mod_njit(phi + shifts[k], 2*np.pi)
            if p < min_v[k]:
                min_v[k] = p
            if p > max_v[k]:
                max_v[k] = p
    # -----------------------------------------------

    spans = max_v - min_v
    best  = 0
    best_span = spans[0]
    for k in range(1, 4):
        if spans[k] < best_span:
            best_span = spans[k]
            best = k

    # 还原到原始坐标系
    arc_start = min_v[best] - shifts[best]
    arc_end   = max_v[best] - shifts[best]
    return arc_start, arc_end, best_span


def expand_arc(start, end):
    """将弧段展开成非负方向的线性区间（处理跨 0 情况）"""
    if end < start:
        end += 2 * np.pi
    return start, end


def merge_multiple_arcs(arcs):
    """
    合并多个弧段，返回一个最小连续弧段覆盖所有输入弧段。
    无效弧段（如 ±1e30）被忽略；如果全部无效则返回默认值。
    """
    expanded_arcs = []
    for s, e in arcs:
        s, e = expand_arc(s, e)
        # if (abs(s) > 1e10 or abs(e) > 1e10):
        #     continue
        expanded_arcs.append((s, e))

    if not expanded_arcs:
        return 0.0, np.pi*2

    base_center = (expanded_arcs[0][0] + expanded_arcs[0][1]) / 2
    adjusted_arcs = [expanded_arcs[0]]

    for s, e in expanded_arcs[1:]:
        candidates = [(s, e), (s + 2*np.pi, e + 2*np.pi), (s - 2*np.pi, e - 2*np.pi)]
        best_s, best_e = min(
            candidates,
            key=lambda arc: abs(((arc[0] + arc[1]) / 2) - base_center)
        )
        adjusted_arcs.append((best_s, best_e))

    starts, ends = np.array(adjusted_arcs)[:,0], np.array(adjusted_arcs)[:,1]
    merged_start = min(starts) % (2 * np.pi)
    merged_end = max(ends) % (2 * np.pi)
    return merged_start, merged_end

# def merge_multiple_arcs(arcs):
#     """
#     合并多个弧段，返回一个最小连续弧段覆盖所有输入弧段。
#     """
#     # Step 1: 展开所有弧段
#     expanded_arcs = [expand_arc(s, e) for s, e in arcs]
#
#     # Step 2: 以第一个为基准，将其余弧段平移 ±2π 后选择最接近的
#     base_center = (expanded_arcs[0][0] + expanded_arcs[0][1]) / 2
#     adjusted_arcs = [expanded_arcs[0]]
#
#     for s, e in expanded_arcs[1:]:
#         candidates = [(s, e), (s + 2*np.pi, e + 2*np.pi), (s - 2*np.pi, e - 2*np.pi)]
#         best_s, best_e = min(candidates, key=lambda arc: abs(((arc[0] + arc[1]) / 2) - base_center))
#         adjusted_arcs.append((best_s, best_e))
#
#     starts, ends = zip(*adjusted_arcs)
#     merged_start = min(starts) % (2 * np.pi)
#     merged_end = max(ends) % (2 * np.pi)
#     return merged_start, merged_end


@njit()
# @profile
def compute_field_from_potential_numba(phi, dr, dz, BunchLength, Ex, Ez, mean_gamma_global):
    nz, nx = phi.shape
    gamma = mean_gamma_global + 1.0  # mean_gamma_global 是 (γ-1)
    Gamma_2 = 1/(gamma * gamma)

    if dr==0 or dz==0 or BunchLength==0:
        return Ex, Ez

    inv2dr = 0.5 / dr
    inv2dz = 0.5 / dz

    # 1) 内部节点：既计算 Ex 又计算 Ez
    for j in range(1, nz-1):
        for i in range(1, nx-1):
            # ∂φ/∂r (中点差分)
            Ex[j, i] = -(phi[j, i+1] - phi[j, i-1]) * inv2dr * Gamma_2
            # ∂φ/∂z
            Ez[j, i] = -(phi[j+1, i] - phi[j-1, i]) * inv2dz * Gamma_2

    # 2) r 边界 i=0, i=nx-1
    for j in range(nz):
        # 左边界
        Ex[j, 0]     = -(phi[j, 1]   - phi[j, 0])     / dr  * Gamma_2
        Ez[j, 0]     = -(phi[min(j+1,nz-1), 0] - phi[max(j-1,0), 0]) / dz  * Gamma_2
        # 右边界
        Ex[j, nx-1]  = -(phi[j, nx-1] - phi[j, nx-2]) / dr  * Gamma_2
        Ez[j, nx-1]  = Ez[j, 0]  * Gamma_2  # 对于 z 方向边界，只要和 i=0 同理

    # 3) z 边界 j=0, j=nz-1，但排除角已在上面 i=0,nx-1 处理
    for i in range(1, nx-1):
        # 下边界
        Ex[0, i]     = -(phi[0, i+1]   - phi[0, i-1])   * inv2dr  * Gamma_2
        Ez[0, i]     = -(phi[1, i]     - phi[0, i])     / dz  * Gamma_2
        # 上边界
        Ex[nz-1, i]  = Ex[0, i]  * Gamma_2  # r 差分同 j=0
        Ez[nz-1, i]  = -(phi[nz-1, i] - phi[nz-2, i])   / dz  * Gamma_2

    return Ex, Ez


@njit(parallel=True)
def compute_field_from_potential_numba_SBS(
    phi,          # float32/float64, shape (nz, nx)
    k,                 # 处理的 slice 索引
    dr, dz,            # 网格步长
    BunchLength,
    Ex_cube, Ez_cube,
    mean_gamma_global  # = gamma-1
):
    """
    从 phi_cube[k] 计算 Ex_cube[k], Ez_cube[k]（双向差分 + 边界单边差分）。
    不返回新数组，直接就地写入 Ex_cube/Ez_cube。
    """

    # 取第 k 层 view（不会拷贝）
    Ex  = Ex_cube[k]
    Ez  = Ez_cube[k]

    nz, nx = phi.shape
    gamma = mean_gamma_global + 1.0
    inv_gamma2 = 1.0 / (gamma * gamma)
    # inv_gamma2 = 1.0

    if dr == 0.0 or dz == 0.0 or BunchLength==0:
        # 退化：清零该层
        for j in range(nz):
            for i in range(nx):
                Ex[j, i] = 0.0
                Ez[j, i] = 0.0
        return

    inv2dr = 0.5 / dr
    inv2dz = 0.5 / dz

    # 1) 内部节点（中心差分）
    for j in prange(1, nz-1):
        for i in range(1, nx-1):
            Ex[j, i] = -(phi[j, i+1] - phi[j, i-1]) * inv2dr * inv_gamma2
            Ez[j, i] = -(phi[j+1, i] - phi[j-1, i]) * inv2dz * inv_gamma2

    # 2) r 边界：i=0, i=nx-1（沿 r 单边差分；沿 z 用中心/夹取）
    for j in range(nz):
        # i=0
        Ex[j, 0]    = -(phi[j, 1]      - phi[j, 0])      / dr * inv_gamma2
        Ez[j, 0]    = -(phi[min(j+1,nz-1), 0] - phi[max(j-1,0), 0]) / dz * inv_gamma2
        # i=nx-1
        Ex[j, nx-1] = -(phi[j, nx-1]   - phi[j, nx-2])   / dr * inv_gamma2
        Ez[j, nx-1] = -(phi[min(j+1,nz-1), nx-1] - phi[max(j-1,0), nx-1]) / dz * inv_gamma2

    # 3) z 边界：j=0, j=nz-1（沿 z 单边差分；沿 r 用中心差分）
    for i in range(1, nx-1):
        # j=0
        Ex[0, i]     = -(phi[0, i+1]     - phi[0, i-1])   * inv2dr * inv_gamma2
        Ez[0, i]     = -(phi[1, i]       - phi[0, i])     / dz     * inv_gamma2
        # j=nz-1
        Ex[nz-1, i]  = -(phi[nz-1, i+1]  - phi[nz-1, i-1]) * inv2dr * inv_gamma2
        Ez[nz-1, i]  = -(phi[nz-1, i]    - phi[nz-2, i])   / dz     * inv_gamma2


@njit()
def make_fft_kernel_open_numba(nx: int, nz: int,
                             dx: float, dz: float) -> np.ndarray:
    """
    生成 (nz, nx) 周期边界 Green kernel: 1/(eps0*k^2)  (float32)。
    """
    eps0 = 8.854187817e-12

    dkx = np.float32(2.0 * np.pi / (nx * dx))
    dkz = np.float32(2.0 * np.pi / (nz * dz))

    kx = np.empty(nx, dtype=np.float32)
    for i in range(nx):
        idx = i if i <= nx // 2 else i - nx     # 频率 [-nx/2, nx/2)
        kx[i] = dkx * idx

    kz = np.empty(nz, dtype=np.float32)
    for j in range(nz):
        idz = j if j <= nz // 2 else j - nz
        kz[j] = dkz * idz

    kernel = np.empty((nz, nx), dtype=np.float32)

    for j in prange(nz):
        kz2 = kz[j] * kz[j]
        for i in range(nx):
            k2 = kx[i] * kx[i] + kz2
            kernel[j, i] = 0.0 if k2 == 0.0 else 1.0 / (eps0 * k2)

    return kernel


@njit()
def total_field_1d_numba(z: np.ndarray, q: np.ndarray) -> np.ndarray:
    EPS0 = 8.854_187_817e-12  # 真空介电常数 (F·m⁻¹)
    K = 1.0 / (4.0 * np.pi * EPS0)
    N  = z.size
    Ez = np.zeros(N)

    if z[1]-z[0]<=0.0:
        return K * Ez

    for i in range(N):                  # 并行外循环
        Ei = 0.0
        zi = z[i]
        for j in range(N):
            if i == j:
                continue
            r = zi - z[j]
            Ei += q[j] * np.sign(r) / (r*r)
        Ez[i] = Ei
    return K * Ez


@njit(parallel=True)
def interpolate_fields_to_particles_numba(Ex, Ez, Ef,
                                          coords_x, coords_z, slice_indices,
                                          coords_survive_idx,
                                          x0, z0,
                                          dx, dz, BunchLength,
                                          Ex_p, Ez_p, Ef_p, mat_in):
    """
    双线性插值：从网格 Ex, Ez 插值到粒子 (coords_r, coords_z) 上。
    返回 (Ex_p, Ez_p)，长度与 coords_r 相同。
    """
    nz, nx = Ex.shape
    N = coords_x.shape[0]
    # Ex_p = np.empty(N, dtype=np.float32)
    # Ez_p = np.empty(N, dtype=np.float32)
    # Ef_p = np.empty(N, dtype=np.float32)

    for n in prange(N):
        if dx <=0 or dz<=0 or BunchLength<=0:
            mat_in[coords_survive_idx[n], 10] = 0.0
            mat_in[coords_survive_idx[n], 11] = 0.0
            mat_in[coords_survive_idx[n], 12] = 0.0

        else:

            x = coords_x[n]
            z = coords_z[n]

            # 浮点索引
            t = (x - x0) / dx
            u = (z - z0) / dz

            # 基本索引
            # i0 = int(np.floor(t))
            # j0 = int(np.floor(u))

            i0 = int(t)
            j0 = int(u)
            if i0 < 0:
                i0 = 0
            elif i0 > nx-2:
                i0 = nx-2
            if j0 < 0:
                j0 = 0
            elif j0 > nz-2:
                j0 = nz-2

            alpha1 = t - i0
            beta1 = u - j0

            # 四角值
            Ex00 = Ex[j0    , i0    ]
            Ex10 = Ex[j0    , i0 + 1]
            Ex01 = Ex[j0 + 1, i0    ]
            Ex11 = Ex[j0 + 1, i0 + 1]

            Ez00 = Ez[j0    , i0    ]
            Ez10 = Ez[j0    , i0 + 1]
            Ez01 = Ez[j0 + 1, i0    ]
            Ez11 = Ez[j0 + 1, i0 + 1]

            # 双线性插值
            # Ex_p[n] = (1-alpha1)*(1-beta1)*Ex00 + alpha1*(1-beta1)*Ex10 + (1-alpha1)*beta1*Ex01 + alpha1*beta1*Ex11
            # Ez_p[n] = (1-alpha1)*(1-beta1)*Ez00 + alpha1*(1-beta1)*Ez10 + (1-alpha1)*beta1*Ez01 + alpha1*beta1*Ez11
            coef1, coef2, coef3, coef4 = (1-alpha1)*(1-beta1), alpha1*(1-beta1), (1-alpha1)*beta1, alpha1*beta1

            Ex_p[n] = coef1*Ex00 + coef2*Ex10 + coef3*Ex01 + coef4*Ex11
            Ez_p[n] = coef1*Ez00 + coef2*Ez10 + coef3*Ez01 + coef4*Ez11
            # Ef_p[n] = Ef[slice_indices[n]]

            mat_in[coords_survive_idx[n], 10] = coef1*Ex00 + coef2*Ex10 + coef3*Ex01 + coef4*Ex11
            mat_in[coords_survive_idx[n], 11] = coef1*Ez00 + coef2*Ez10 + coef3*Ez01 + coef4*Ez11
            # mat_in[coords_survive_idx[n], 12] = Ef[slice_indices[n]]

            # if abs(x)<0.01:
            #     mat_in[coords_survive_idx[n], 10] = 1e6*x
            # else:
            #     mat_in[coords_survive_idx[n], 10] = 1e2/x

            # mat_in[coords_survive_idx[n], 11] = 1e6*z
            # mat_in[coords_survive_idx[n], 12] = 0.0

    return Ex_p, Ez_p, Ef_p


@njit(parallel=True)
def interpolate_fields_to_particles_SBS_numba(
    Ex_cube, Ez_cube, Ef,
    coords_x, coords_z, coords_pr0, slice_indices,
    coords_survive_idx,
    x0, z0,
    dx, dz, BunchLength,
    Ex_p, Ez_p, Ef_p, mat_in
):
    """
    从分片的Ex_cube/Ez_cube (ns, nz, nx) 对粒子做双线性插值。
    结果写入 Ex_p/Ez_p/Ef_p 以及 mat_in[:, 10:13]
    - Ex_cube[k], Ez_cube[k] 为第 k 片的 (nz, nx) 场分量
    - Ef[k] 为第 k 片的一维纵向场
    """

    ns, nz, nx = Ex_cube.shape
    N = coords_x.shape[0]

    bad_grid = (dx <= 0.0) or (dz <= 0.0) or (BunchLength <= 0.0)

    for n in prange(N):
        idx_out = coords_survive_idx[n]

        if bad_grid:
            Ex_p[n] = 0.0
            Ez_p[n] = 0.0
            Ef_p[n] = 0.0
            # 就地写回
            mat_in[idx_out, 10] = 0.0
            mat_in[idx_out, 11] = 0.0
            mat_in[idx_out, 12] = 0.0
            continue

        # 该粒子所在 slice
        s = slice_indices[n]
        if s < 0:
            s = 0
        elif s >= ns:
            s = ns - 1

        Ex = Ex_cube[s]  # (nz, nx)
        Ez = Ez_cube[s]  # (nz, nx)

        # 连续网格坐标
        t = (coords_x[n] - x0) / dx  # x方向格点坐标
        u = (coords_z[n] - z0) / dz  # z方向格点坐标

        # 左下整数格点（floor）
        i0 = int(np.floor(t))
        j0 = int(np.floor(u))

        # 边界夹取，确保 i0∈[0, nx-2], j0∈[0, nz-2]
        if i0 < 0:
            i0 = 0
        elif i0 > nx - 2:
            i0 = nx - 2
        if j0 < 0:
            j0 = 0
        elif j0 > nz - 2:
            j0 = nz - 2

        # 小数部分
        alpha = t - i0  # ∈[0,1)（被夹取后）
        beta  = u - j0

        # 四角权重
        w00 = (1.0 - alpha) * (1.0 - beta)
        w10 = alpha         * (1.0 - beta)
        w01 = (1.0 - alpha) * beta
        w11 = alpha         * beta

        # 取四角值并插值
        # Ex
        e00 = Ex[j0    , i0    ]
        e10 = Ex[j0    , i0 + 1]
        e01 = Ex[j0 + 1, i0    ]
        e11 = Ex[j0 + 1, i0 + 1]
        er_val = w00*e00 + w10*e10 + w01*e01 + w11*e11

        # Ez
        z00 = Ez[j0    , i0    ]
        z10 = Ez[j0    , i0 + 1]
        z01 = Ez[j0 + 1, i0    ]
        z11 = Ez[j0 + 1, i0 + 1]
        ez_val = w00*z00 + w10*z10 + w01*z01 + w11*z11

        # 纵向场：若 Ef 为每 slice 的标量，一般直接索引；若是 1D 网格，请在外部先取值后传入
        ef_val = Ef[s]

        # 写回
        Ex_p[n] = er_val * (1/np.sqrt(1+coords_pr0[n]*coords_pr0[n]))
        Ez_p[n] = ez_val
        # Ef_p[n] = ef_val
        Ef_p[n] = er_val * (-coords_pr0[n]/np.sqrt(1+coords_pr0[n]*coords_pr0[n]))

        # 同步写回到粒子矩阵
        mat_in[idx_out, 10] = er_val * (1/np.sqrt(1+coords_pr0[n]*coords_pr0[n]))
        mat_in[idx_out, 11] = ez_val
        # mat_in[idx_out, 12] = ef_val
        mat_in[idx_out, 12] = er_val * (-coords_pr0[n]/np.sqrt(1+coords_pr0[n]*coords_pr0[n]))

    return Ex_p, Ez_p, Ef_p


@njit()
def coordinate_convert_SC(coords_r, mean_r_global, fi_s_global, fi_e_global, segments, slice_indices):
    """
    束流坐标变换：
    1. 将 r 坐标转为相对于中心线的 x 坐标。
    2. 计算 phi * R 累积积分作为纵向坐标。
    3. 给出总束长。

    参数：
        coords_r       : (N,) 粒子在全局坐标系下的径向位置
        mean_r_global  : (segments,) 每段的平均半径
        fi_s_global    : (segments,) 每段起始角
        fi_e_global    : (segments,) 每段终止角
        segments       : int，总段数
        slice_indices  : (N,) 每个粒子所在的段索引

    返回：
        coords_x       : (N,) 相对中心线的横向 x 坐标
        phi_times_R    : (segments,) φ * R 累积积分
        BunchLength    : float，总束长
    """
    N = coords_r.shape[0]
    coords_x = np.empty(N, dtype=coords_r.dtype)

    for i in range(N):
        coords_x[i] = coords_r[i] - mean_r_global[slice_indices[i]]

    sum_r = 0.0
    count = 0
    for i in range(segments):
        if mean_r_global[i] > 0.0:
            sum_r += mean_r_global[i]
            count += 1

    if count == 0:
        mean_r_avg = 0.0
    else:
        mean_r_avg = sum_r / count

    phi_times_R = np.zeros(segments, dtype=coords_r.dtype)
    dphi_R = (fi_e_global - fi_s_global) / segments
    # phi_times_R[0] = dphi_R[0] * mean_r_global[0]

    for i in range(0, segments):
        if mean_r_global[i] > 0.0:
            phi_times_R[i] = phi_times_R[i - 1] + dphi_R * mean_r_global[i]
        else:
            phi_times_R[i] = phi_times_R[i - 1] + dphi_R * mean_r_avg

    BunchLength = phi_times_R[-1]

    return coords_x, phi_times_R, BunchLength


def allgatherv_phi_3d(comm, phi_local, nz, nx):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 本地元素数
    send = np.ascontiguousarray(phi_local, dtype=np.float32)
    send_flat = send.ravel()
    local_cnt = np.array([send_flat.size], dtype=np.int64)

    # 收集所有元素数 -> counts
    counts = np.empty(size, dtype=np.int64)
    comm.Allgather(local_cnt, counts)

    # 位移（元素为单位）
    displs = np.zeros(size, dtype=np.int64)
    displs[1:] = np.cumsum(counts[:-1])
    total_elems = int(displs[-1] + counts[-1])

    # 接收缓冲
    recv_flat = np.empty(total_elems, dtype=np.float32)

    # Allgatherv（非 in-place 版）
    comm.Allgatherv([send_flat, MPI.FLOAT],
                    [recv_flat, (counts, displs), MPI.FLOAT])

    sum_S = total_elems // (nz * nx)  # 全局 slice 数
    phi_global = recv_flat.reshape(sum_S, nz, nx)
    return phi_global


def allgatherv_phi_3d_inplace(comm, phi_local, ns, nz, nx):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 本地元素个数（int64，单位是元素，不是字节）
    send = np.ascontiguousarray(phi_local, dtype=np.float32)
    send_flat = send.ravel()
    local_elems = np.array([send_flat.size], dtype=np.int64)

    counts = np.empty(size, dtype=np.int64)
    comm.Allgather(local_elems, counts)

    displs = np.zeros(size, dtype=np.int64)
    if size > 1:
        displs[1:] = np.cumsum(counts[:-1])

    total_elems = int(displs[-1] + counts[-1])
    assert total_elems == ns * nz * nx, "counts/displs 与 ns*nz*nx 不一致"

    recv_flat = np.empty(total_elems, dtype=np.float32)

    # 先把本地段放到全局缓冲的对应位置（counts 可能为 0，空切片也安全）
    beg = int(displs[rank]); end = beg + int(counts[rank])
    recv_flat[beg:end] = send_flat

    # 关键：in-place 调用，第一参数是 MPI.IN_PLACE（不是列表）
    comm.Allgatherv(MPI.IN_PLACE,
                    [recv_flat, (counts, displs), MPI.FLOAT])

    return recv_flat.reshape(ns, nz, nx)


@njit
def contiguous_block_for_rank(ns: int, size: int, rank: int):
    """
    把 [0, ns) 按尽量平均、连续块的方式分给 size 个进程。
    返回 (begin, end) 半开区间：该 rank 负责 k ∈ [begin, end)
    """
    base = ns // size
    rem  = ns %  size
    count = base + (1 if rank < rem else 0)
    begin = rank * base + min(rank, rem)
    end   = begin + count
    return begin, end, count




if __name__ == '__main__':
    pass

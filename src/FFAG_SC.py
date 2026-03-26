import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift

import time
from FFAG_Utils import FFAG_FlatBunch
# from numba import njit
from math import floor
import numba
from numba.typed import List
from numba import types
from numba import njit, int64, float64
from numba import njit, prange


# from FFAG_MathTools import my_3dInterp_vect


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

# @njit()
# def GetLocalCoordinates_2p5(LocalBunch):
#     """
#     返回本地 Bunch 中所有存活粒子的 x, y, z 笛卡尔坐标（n, 3 的 ndarray），
#     以及 LocalID 和 GlobalID（分别为 n, 的整数类型的 ndarray）。
#     """
#     # 获取属性的索引
#     r_idx = 0
#     fi_idx = 4
#     z_idx = 2
#     LID_idx, GID_idx = 13, 14
#     survive_flag_idx = 7
#
#     # 筛选出本地存活的粒子survive_flag == 1, 排除已损失或未注入的粒子
#     valid_particles_mask = LocalBunch[:, survive_flag_idx] == 1
#
#     # 提取有效粒子的 r, fi, z 坐标
#     r_polar = LocalBunch[valid_particles_mask, r_idx]
#     fi_polar = LocalBunch[valid_particles_mask, fi_idx]
#     z_polar = LocalBunch[valid_particles_mask, z_idx]
#
#     # 返回笛卡尔坐标和整数类型的 LocalID、GlobalID
#     LocalID, GlobalID = LocalBunch[valid_particles_mask, LID_idx], LocalBunch[valid_particles_mask, GID_idx]
#
#     return r_polar, fi_polar, z_polar, LocalID, GlobalID


@njit()
def GetLocalCoordinates_2p5(LocalBunch, Step_Survive,
                             malloc_r, malloc_fi, malloc_z,
                             malloc_lid, malloc_gid, malloc_fiMod):
    rows = LocalBunch.shape[0]
    j = 0
    TWO_PI = 2 * np.pi
    for i in range(rows):
        if Step_Survive[i]:
            malloc_r[j]   = LocalBunch[i, 0]
            malloc_z[j]   = LocalBunch[i, 2]
            malloc_fi[j]  = LocalBunch[i, 4]
            malloc_lid[j] = LocalBunch[i, 14]
            malloc_gid[j] = LocalBunch[i, 15]
            malloc_fiMod[j] = malloc_fi[j] - TWO_PI * np.floor(malloc_fi[j]/TWO_PI) # mod(fi, np.pi*2)
            # result[i] = a[i] - b * np.floor(a[i] / b)
            j += 1

    return (malloc_r[:j], malloc_fi[:j], malloc_z[:j],
            malloc_lid[:j], malloc_gid[:j], malloc_fiMod[:j], j)


# @njit()
# def GetLocalCoordinates_2p5(LocalBunch, Step_Survive, malloc_r, malloc_fi, malloc_z, malloc_lid, malloc_gid):
#     rows = LocalBunch.shape[0]
#
#     # 先统计存活粒子数
#     live_cnt = 0
#     for i in range(rows):
#         if Step_Survive[i]:   # survive_flag_idx
#             live_cnt += 1
#     # live_cnt  = int(np.sum(LocalBunch[:, 8]))
#     # live_cnt = int(np.sum(Step_Survive))
#
#     # r0     = np.empty(live_cnt, dtype=LocalBunch.dtype)
#     # fi0    = np.empty(live_cnt, dtype=LocalBunch.dtype)
#     # z0     = np.empty(live_cnt, dtype=LocalBunch.dtype)
#     # lid0   = np.empty(live_cnt, dtype=np.int64)   # LocalID
#     # gid0   = np.empty(live_cnt, dtype=np.int64)   # GlobalID
#
#     j = 0
#     for i in range(rows):
#         if Step_Survive[i]:
#             malloc_r[j]   = LocalBunch[i, 0]
#             malloc_z[j]   = LocalBunch[i, 2]
#             malloc_fi[j] = LocalBunch[i, 4]
#             malloc_lid[j] = LocalBunch[i, 14]
#             malloc_gid[j] = LocalBunch[i, 15]
#             j += 1
#
#     # r     = malloc_r[:live_cnt]
#     # fi    = malloc_fi[:live_cnt]
#     # z     = malloc_z[:live_cnt]
#     # lid   = malloc_lid[:live_cnt]
#     # gid   = malloc_gid[:live_cnt]
#
#     return malloc_r[:live_cnt], malloc_fi[:live_cnt], malloc_z[:live_cnt],malloc_lid[:live_cnt], malloc_gid[:live_cnt]
#

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


@njit
# @profile
def DistributeChargeNjit(idx_array_x, idx_array_y, idx_array_z, x_grid, y_grid, z_grid, charge_distribution,
                         gaussian_points, charge_scale):
    num_points = idx_array_x.shape[1]

    position_x = gaussian_points[:, 0]
    position_y = gaussian_points[:, 1]
    position_z = gaussian_points[:, 2]

    grid_spacing_x = x_grid[1] - x_grid[0]
    grid_spacing_y = y_grid[1] - y_grid[0]
    grid_spacing_z = z_grid[1] - z_grid[0]

    # 遍历每个散点，将每个散点的电荷量按距离加权分配到周围的8个顶点
    for index in range(num_points):
        x_index_0, x_index_1 = idx_array_x[0, index], idx_array_x[1, index]
        y_index_0, y_index_1 = idx_array_y[0, index], idx_array_y[1, index]
        z_index_0, z_index_1 = idx_array_z[0, index], idx_array_z[1, index]

        x0, x1 = x_grid[x_index_0], x_grid[x_index_1]
        y0, y1 = y_grid[y_index_0], y_grid[y_index_1]
        z0, z1 = z_grid[z_index_0], z_grid[z_index_1]

        wx0 = (x1 - position_x[index]) / grid_spacing_x
        wx1 = (position_x[index] - x0) / grid_spacing_x
        wy0 = (y1 - position_y[index]) / grid_spacing_y
        wy1 = (position_y[index] - y0) / grid_spacing_y
        wz0 = (z1 - position_z[index]) / grid_spacing_z
        wz1 = (position_z[index] - z0) / grid_spacing_z

        # 分配电荷到周围的8个顶点
        charge_distribution[x_index_0, y_index_0, z_index_0] += wx0 * wy0 * wz0 * charge_scale
        charge_distribution[x_index_1, y_index_0, z_index_0] += wx1 * wy0 * wz0 * charge_scale
        charge_distribution[x_index_0, y_index_1, z_index_0] += wx0 * wy1 * wz0 * charge_scale
        charge_distribution[x_index_1, y_index_1, z_index_0] += wx1 * wy1 * wz0 * charge_scale
        charge_distribution[x_index_0, y_index_0, z_index_1] += wx0 * wy0 * wz1 * charge_scale
        charge_distribution[x_index_1, y_index_0, z_index_1] += wx1 * wy0 * wz1 * charge_scale
        charge_distribution[x_index_0, y_index_1, z_index_1] += wx0 * wy1 * wz1 * charge_scale
        charge_distribution[x_index_1, y_index_1, z_index_1] += wx1 * wy1 * wz1 * charge_scale

    return charge_distribution


# @njit()
# # @profile
# def DistributeCharge2D_Njit(r, z, charge, rmin, rmax, zmin, zmax, nx, nz):
#     """
#     在 (r, z) 平面上将电荷分配到二维网格（双线性分配）。
#
#     参数：
#         r, z    : (N,) ndarray，粒子坐标（极坐标的 r 和 z）
#         charge  : float，宏粒子电荷量
#         rmin, rmax, zmin, zmax : 网格边界
#         nx, nz  : 网格划分数量（r 方向划分 nx，z 方向划分 nz）
#
#     返回：
#         rho[nz, nx] : 电荷密度矩阵（单位未归一化）
#     """
#     rho = np.zeros((nz, nx), dtype=np.float64)
#
#     if rmax == rmin or zmax == zmin:
#         return rho  # 空边界，直接返回
#
#     dr = (rmax - rmin) / nx
#     dz = (zmax - zmin) / nz
#
#     for i in range(r.shape[0]):
#         ri, zi, qi = r[i], z[i], charge
#
#         # 计算 r/z 在网格中的相对位置
#         rx = (ri - rmin) / dr
#         rz = (zi - zmin) / dz
#
#         ix = int(np.floor(rx))
#         iz = int(np.floor(rz))
#
#         # 边界检查
#         if ix < 0 or ix >= nx - 1:
#             continue
#         if iz < 0 or iz >= nz - 1:
#             continue
#
#         fx = rx - ix
#         fz = rz - iz
#
#         # 双线性权重分配到四个格点
#         rho[iz,   ix  ] += (1 - fx) * (1 - fz) * qi
#         rho[iz,   ix+1] += fx       * (1 - fz) * qi
#         rho[iz+1, ix  ] += (1 - fx) * fz       * qi
#         rho[iz+1, ix+1] += fx       * fz       * qi
#
#     return rho


@njit()
def DistributeCharge2D_Njit(x, z, q_macro, BunchLength, xmin, xmax, zmin, zmax, nx, nz):
    rho = np.zeros((nz,nx), np.float32) # 横向2维电荷密度
    dx = (xmax-xmin)/nx; dz = (zmax-zmin)/nz
    inv_area = 1.0/(dx*dz)
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


# @njit(parallel=True)
# def DistributeCharge2D_Njit(x, z, q_macro, BunchLength, xmin, xmax, zmin, zmax, nx, nz):
#     dx = (xmax - xmin) / nx
#     dz = (zmax - zmin) / nz
#     inv_area = 1.0 / (dx * dz)
#     q = q_macro / BunchLength * inv_area
#
#     nthreads = numba.get_num_threads()
#     rho_private = np.zeros((nthreads, nz, nx), dtype=np.float32)  # 每个线程一份私有副本
#
#     for k in prange(x.shape[0]):
#         rx = (x[k] - xmin) / dx
#         rz = (z[k] - zmin) / dz
#         ix = int(np.floor(rx))
#         iz = int(np.floor(rz))
#         fx = rx - ix
#         fz = rz - iz
#
#         # 边界处理
#         if ix < 0:
#             ix = 0
#             fx = 0.0
#         elif ix >= nx - 1:
#             ix = nx - 2
#             fx = 1.0
#         if iz < 0:
#             iz = 0
#             fz = 0.0
#         elif iz >= nz - 1:
#             iz = nz - 2
#             fz = 1.0
#
#         thread_id = numba.np.ufunc.parallel._get_thread_id()  # 当前线程号
#         rho_private[thread_id, iz  , ix  ] += (1 - fx) * (1 - fz) * q
#         rho_private[thread_id, iz  , ix+1] += fx * (1 - fz) * q
#         rho_private[thread_id, iz+1, ix  ] += (1 - fx) * fz * q
#         rho_private[thread_id, iz+1, ix+1] += fx * fz * q
#
#     # 归并所有线程的结果
#     rho = np.sum(rho_private, axis=0)
#     return rho


# @njit
# def DistributeCharge2D_Njit(r, z, charge, rmin, rmax, zmin, zmax, nx, nz, rho_pre_define):
#
#     rho = rho_pre_define * 0.0
#
#     if rmax == rmin or zmax == zmin:
#         return rho
#
#     dr = (rmax - rmin) / nx
#     dz = (zmax - zmin) / nz
#     idr = 1.0 / dr
#     idz = 1.0 / dz
#
#     n = r.shape[0]
#     for i in range(n):
#         ri = r[i]
#         zi = z[i]
#         qi = charge
#
#         # 提前过滤非法粒子
#         if ri < rmin or ri >= rmax:
#             continue
#         if zi < zmin or zi >= zmax:
#             continue
#
#         rx = (ri - rmin) * idr
#         rz = (zi - zmin) * idz
#
#         ix = floor(rx)
#         iz = floor(rz)
#
#         # 边界检查
#         if ix < 0 or ix >= nx - 1 or iz < 0 or iz >= nz - 1:
#             continue
#
#         fx = rx - ix
#         fz = rz - iz
#
#         # 电荷加权分配
#         rho[iz,   ix  ] += (1 - fx) * (1 - fz) * qi
#         rho[iz,   ix+1] += fx       * (1 - fz) * qi
#         rho[iz+1, ix  ] += (1 - fx) * fz       * qi
#         rho[iz+1, ix+1] += fx       * fz       * qi
#
#     return rho



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



def green_function(x, y, z):
    epsilon = 1e-8  # 避免除以零
    return 1.0 / (np.sqrt(x ** 2 + y ** 2 + z ** 2) + epsilon)


def ConvertToPolarElectricField(point_x, point_y, Ex_interp, Ey_interp, Ez_interp):
    """
    将插值后的电场分量从笛卡尔坐标 (Ex, Ey, Ez) 转换为极坐标系下的电场分量 (Er, Efi, Ez)。

    参数：
    - point_x, point_y: 粒子的位置坐标 (x, y)
    - Ex_interp, Ey_interp, Ez_interp: 插值得到的笛卡尔坐标电场分量 (Ex, Ey, Ez)

    返回：
    - Er_interp, Efi_interp, Ez_interp: 极坐标下的电场分量 (Er, Efi, Ez)
    """
    # 计算粒子的径向距离 r 和方位角 phi
    phi = np.arctan2(point_y, point_x)

    # 计算极坐标下的径向电场 Er 和方位角电场 Efi
    Er_interp = Ex_interp * np.cos(phi) + Ey_interp * np.sin(phi)
    Efi_interp = -Ex_interp * np.sin(phi) + Ey_interp * np.cos(phi)

    # Ez_interp 保持不变
    return Er_interp, Ez_interp, Efi_interp

@njit
def filter_by_slice_numba(r, z, slice_indices, target_slice):
    count = 0
    for i in range(len(slice_indices)):
        if slice_indices[i] == target_slice:
            count += 1

    r_out = np.empty(count, dtype=np.float64)
    z_out = np.empty(count, dtype=np.float64)

    j = 0
    for i in range(len(slice_indices)):
        if slice_indices[i] == target_slice:
            r_out[j] = r[i]
            z_out[j] = z[i]
            j += 1

    return r_out, z_out

@njit
# @profile
def filter_by_slice_numba_with_bounds(r, z, LID, slice_indices, target_slice):
    N = len(r)
    r_out = np.empty(N, dtype=np.float64)
    z_out = np.empty(N, dtype=np.float64)
    LID_out = np.empty(N, dtype=np.int64)

    j = 0
    rmin = np.inf
    rmax = -np.inf
    zmin = np.inf
    zmax = -np.inf

    for i in range(N):
        if slice_indices[i] == target_slice:
            ri = r[i]
            zi = z[i]
            LIDi = LID[i]

            r_out[j] = ri
            z_out[j] = zi
            LID_out[j] = LIDi

            if ri < rmin: rmin = ri
            if ri > rmax: rmax = ri
            if zi < zmin: zmin = zi
            if zi > zmax: zmax = zi
            j += 1

    return r_out[:j], z_out[:j], LID_out[:j], rmin, rmax, zmin, zmax


# @njit()
# @profile
def ComputeSliceBoundsAndCoords(coords_r, coords_z, coords_LID, slice_indices, segments):
    """
    计算每个 φ-slice 中的 r/z 边界（min/max），并返回每个 slice 的 (r,z) 粒子子集。

    参数：
        coords_r: (N,) ndarray，粒子的 r 坐标
        coords_z: (N,) ndarray，粒子的 z 坐标
        slice_indices: (N,) ndarray，每个粒子所属的 slice 索引（0 ~ segments-1）
        segments: 总划分数

    返回：
        local_rmin, local_rmax, local_zmin, local_zmax: 每段的边界值数组 (segments,)
        slice_r_list: list，每段内所有 r 坐标 (ndarray)
        slice_z_list: list，每段内所有 z 坐标 (ndarray)
    """
    local_rmin = np.zeros(segments)
    local_rmax = local_rmin * 1.0
    local_zmin = local_rmin * 1.0
    local_zmax = local_rmin * 1.0

    max_row_num = len(coords_r)

    slice_r_list = []
    slice_z_list = []
    slice_LID_list = []

    for i in range(segments):

        (r_i, z_i, LID_i,
         rmin_slice, rmax_slice,
         zmin_slice, zmax_slice) = (
            filter_by_slice_numba_with_bounds(coords_r, coords_z,coords_LID,slice_indices, i))

        slice_r_list.append(r_i)
        slice_z_list.append(z_i)
        slice_LID_list.append(LID_i)
        local_rmin[i] = rmin_slice
        local_rmax[i] = rmax_slice
        local_zmin[i] = zmin_slice
        local_zmax[i] = zmax_slice

    return local_rmin, local_rmax, local_zmin, local_zmax, slice_r_list, slice_z_list, slice_LID_list


# @njit
# def ComputeSliceBoundsAndCoords_merged(coords_r, coords_z, coords_LID, slice_indices, segments):
#     N = coords_r.shape[0]
#
#     # 1) 准备每个 slice 的边界数组
#     local_rmin = np.full(segments, np.inf, dtype=np.float64)
#     local_rmax = np.full(segments, -np.inf, dtype=np.float64)
#     local_zmin = np.full(segments, np.inf, dtype=np.float64)
#     local_zmax = np.full(segments, -np.inf, dtype=np.float64)
#
#     # 2) 用 typed.List 存放每个 slice 里的 coords
#     slice_r_list   = List()
#     slice_z_list   = List()
#     slice_LID_list = List()
#     for _ in range(segments):
#         # 占位，后面会 overwrite
#         slice_r_list.append(np.empty(0, dtype=coords_r.dtype))
#         slice_z_list.append(np.empty(0, dtype=coords_z.dtype))
#         slice_LID_list.append(np.empty(0, dtype=coords_LID.dtype))
#
#     # 3) 对每个 slice 做一次完整扫描，筛出属于它的粒子、更新边界
#     for s in range(segments):
#         # 暂存当前 slice 的所有粒子
#         tmp_r   = np.empty(N, dtype=np.float64)
#         tmp_z   = np.empty(N, dtype=np.float64)
#         tmp_LID = np.empty(N, dtype=np.int64)
#         j = 0
#         rmin, rmax = np.inf, -np.inf
#         zmin, zmax = np.inf, -np.inf
#
#         for i in range(N):
#             if slice_indices[i] == s:
#                 ri  = coords_r[i]
#                 zi  = coords_z[i]
#                 lid = coords_LID[i]
#
#                 tmp_r[j]   = ri
#                 tmp_z[j]   = zi
#                 tmp_LID[j] = lid
#
#                 # 更新边界
#                 if ri < rmin: rmin = ri
#                 if ri > rmax: rmax = ri
#                 if zi < zmin: zmin = zi
#                 if zi > zmax: zmax = zi
#
#                 j += 1
#
#         # 裁剪到真正长度并放入 typed.List
#         slice_r_list[s]   = tmp_r[:j]
#         slice_z_list[s]   = tmp_z[:j]
#         slice_LID_list[s] = tmp_LID[:j]
#
#         local_rmin[s] = rmin
#         local_rmax[s] = rmax
#         local_zmin[s] = zmin
#         local_zmax[s] = zmax
#
#     return (local_rmin, local_rmax,
#             local_zmin, local_zmax,
#             slice_r_list, slice_z_list, slice_LID_list)


@njit(parallel=True)
def ComputeSliceBoundsAndCoords_merged(coords_r, coords_z, coords_LID,
                                       coords_fi_mod, slice_fi_step,
                                       fi_s_global, fi_e_global,
                                       segments,
                                       r2d, z2d, LID2d, slice_indices, sum_r_array, sum_z_array, sum_r2_array, sum_z2_array):
    """
    将粒子按 phi slice 聚类，并为每 slice 计算 (r,z) 边界、坐标列表与计数。

    Parameters
    ----------
    coords_r, coords_z : 1D ndarray[float64]
        粒子的 r, z 坐标。
    coords_LID : 1D ndarray[int64]
        粒子的 LocalID。
    coords_fi_mod : 1D ndarray[float64]
        φ ∈ [0, 2π) 模后的角度。
    slice_fi_step : float
        每 slice 的 φ 宽度。
    fi_s_global : float
        全局 φ 的起始角度。
    fi_e_global : float
        全局 φ 的结束角度
    segments : int
        slice 总数。
    size_ratio : float
        用于边界放大的 σ 倍数。

    Returns
    -------
    local_rmin, local_rmax : 1D ndarray[float64]
    local_zmin, local_zmax : 1D ndarray[float64]
    r2d, z2d : 2D ndarray[float64]
    LID2d : 2D ndarray[int64]
    counts : 1D ndarray[int64]
    slice_indices
    """

    N = coords_r.shape[0]
    TWO_PI = 2 * np.pi

    counts = np.zeros(segments, dtype=np.int32)

    # 1. 逐个粒子划分到 slice 并写入数据行
    for i in range(N):
        fi = coords_fi_mod[i]
        if fi < fi_s_global:
            fi += TWO_PI
        s = int((fi - fi_s_global) / slice_fi_step)
        if s < 0:
            s = 0
        elif s >= segments:
            s = segments - 1

        slice_indices[i] = s

        pos = counts[s]
        r2d[s, pos] = coords_r[i]
        z2d[s, pos] = coords_z[i]
        LID2d[s, pos] = coords_LID[i]
        counts[s] += 1

    # 2. 并行计算每个 slice 的边界
    for s in prange(segments):
        if counts[s] == 0:
            continue

        sum_r = 0.0
        sum_z = 0.0
        sum_r2 = 0.0
        sum_z2 = 0.0
        for i in range(counts[s]):
            sum_r += r2d[s, i]
            sum_z += z2d[s, i]
            sum_r2 += r2d[s, i] * r2d[s, i]
            sum_z2 += z2d[s, i] * z2d[s, i]

        sum_r_array[s] = sum_r
        sum_z_array[s] = sum_z
        sum_r2_array[s] = sum_r2
        sum_z2_array[s] = sum_z2

    return (r2d, z2d, LID2d, counts, slice_indices, sum_r_array, sum_z_array, sum_r2_array, sum_z2_array)


@njit
def ComputeMeanAndSigmaFromReduced(sum_r_global, sum_r2_global,
                                   sum_z_global, sum_z2_global,
                                   count_global):
    segments = count_global.shape[0]
    mean_r = np.zeros(segments)
    sigma_r = np.zeros(segments)
    mean_z = np.zeros(segments)
    sigma_z = np.zeros(segments)

    for s in range(segments):
        count = count_global[s]
        if count > 0:
            mu_r = sum_r_global[s] / count
            mu_z = sum_z_global[s] / count
            var_r = sum_r2_global[s] / count - mu_r * mu_r
            var_z = sum_z2_global[s] / count - mu_z * mu_z

            mean_r[s] = mu_r
            mean_z[s] = mu_z
            sigma_r[s] = np.sqrt(max(var_r, 0.0))
            sigma_z[s] = np.sqrt(max(var_z, 0.0))
        else:
            mean_r[s] = 0.0
            mean_z[s] = 0.0
            sigma_r[s] = 0.0
            sigma_z[s] = 0.0

    return mean_r, sigma_r, mean_z, sigma_z


# @njit(parallel=True)
# # @profile
# def ComputeSliceBoundsAndCoords_merged(coords_r, coords_z, coords_LID,slice_indices, segments
#                                        ,size_ratio=1.0):
#     """
#         将粒子按 slice 聚类，返回每片边界与坐标列表。
#
#         Parameters
#         ----------
#         coords_r, coords_z : 1-D ndarray(float64)
#             粒子极坐标 (r, z)。
#         coords_LID        : 1-D ndarray(int64)
#             粒子 LocalID。
#         slice_indices     : 1-D ndarray(int64)
#             每个粒子所属 slice 编号 ∈ [0, segments-1]。
#         segments          : int
#             切片总数。
#         size_ratio        : float, optional (default=1.0)
#             边界放大倍数。
#
#         Returns
#         -------
#         (rmin, rmax, zmin, zmax,
#          r2d,  z2d,  LID2d,
#          counts)
#     """
#
#     N = coords_r.shape[0]
#
#     # 1) 初始化边界和计数
#     local_rmin = np.full(segments, np.inf,  dtype=np.float64)
#     local_rmax = np.full(segments, -np.inf, dtype=np.float64)
#     local_zmin = np.full(segments, np.inf,  dtype=np.float64)
#     local_zmax = np.full(segments, -np.inf, dtype=np.float64)
#
#     # 2) 分配二维输出：每行可容纳该 slice 的所有粒子
#     r2d   = np.empty((segments, N), dtype=np.float64)
#     z2d   = np.empty((segments, N), dtype=np.float64)
#     LID2d = np.empty((segments, N), dtype=np.int64)
#
#     # 行内写指针
#     counts = np.zeros(segments, dtype=np.int64)
#
#     # 3) 单次串行扫描：写入扁平数组并更新边界
#     for i in range(N):
#         s = slice_indices[i]
#         pos = counts[s]
#
#         ri = coords_r[i]
#         zi = coords_z[i]
#         lid = coords_LID[i]
#
#         # 写入
#         r2d[s, pos] = ri
#         z2d[s, pos] = zi
#         LID2d[s, pos] = lid
#         counts[s] += 1
#
#     scale = 3.0 * size_ratio  # μ ± scale·σ
#
#     # 4) for s in range(segments)改为prange
#     for s in range(segments):
#         if counts[s] == 0:  # —— 空 slice ——
#             local_rmin[s] = np.inf
#             local_rmax[s] = -np.inf
#             local_zmin[s] = np.inf
#             local_zmax[s] = -np.inf
#             continue
#
#         # 计算均值与方差
#         # --- 1) 求均值 ---
#         mu_r = 0.0
#         mu_z = 0.0
#         for i in range(counts[s]):
#             mu_r += r2d[s, i]
#             mu_z += z2d[s, i]
#         mu_r /= counts[s]
#         mu_z /= counts[s]
#
#         # --- 2) 求方差 ---
#         var_r = 0.0
#         var_z = 0.0
#         for i in range(counts[s]):
#             diff_r = r2d[s, i] - mu_r
#             diff_z = z2d[s, i] - mu_z
#             var_r += diff_r * diff_r
#             var_z += diff_z * diff_z
#         sigma_r = np.sqrt(var_r / counts[s])
#         sigma_z = np.sqrt(var_z / counts[s])
#
#         # --- 3) 设定边界 ---
#         local_rmin[s] = mu_r - scale * sigma_r
#         local_rmax[s] = mu_r + scale * sigma_r
#         local_zmin[s] = mu_z - scale * sigma_z
#         local_zmax[s] = mu_z + scale * sigma_z
#
#     return (local_rmin, local_rmax,
#             local_zmin, local_zmax,
#             r2d, z2d, LID2d,
#             counts)


@njit(parallel=True)
def compute_slice_charge_maps_prange(r_slices, z_slices, slice_counts,
                                      global_rmin, global_rmax, global_zmin, global_zmax,
                                      nx, nz):
    """
    对每个 φ-slice 并行生成电荷密度网格 (rho[nz, nx])，结果 shape = (segments, nz, nx)

    输入：
        r_slices, z_slices: shape=(segments, max_particle_per_slice)
        slice_counts: 每段实际粒子数
        global_rmin/rmax/zmin/zmax: 每段边界 (segments,)
        nx, nz: 网格数

    返回：
        charge_maps: shape=(segments, nz, nx)
    """
    segments = r_slices.shape[0]
    charge_maps = np.zeros((segments, nz, nx))

    for i in prange(segments):
        num = slice_counts[i]
        if num == 0:
            continue

        r_part = r_slices[i, :num]
        z_part = z_slices[i, :num]
        q_part = 1  # 电荷为 1，或替换为 q_slices[i, :num] 等

        # 创建空电荷网格作为 predefine 数组传入，避免重复分配
        charge_maps[i, :, :] = DistributeCharge2D_Njit(
            r_part, z_part, q_part,
            global_rmin[i], global_rmax[i],
            global_zmin[i], global_zmax[i],
            nx, nz
        )

    return charge_maps


def SC_calculator_fft_new(charge_distribution_out,
                          x_grid, y_grid, z_grid,
                          Xmatrix, Ymatrix, Zmatrix,
                          point_x, point_y, point_z):
    """
    使用FFT计算空间电荷场，输入为已知的3维电荷分布矩阵和3维网格矩阵。

    参数：
    - charge_distribution_out: numpy.ndarray
        包含网格点上电荷分布的3D数组，表示空间中的电荷密度分布。
    - x_grid, y_grid, z_grid: numpy.ndarray
        网格在x, y, z方向的坐标数组。
    - Xmatrix, Ymatrix, Zmatrix: numpy.ndarray
        通过x_grid, y_grid, z_grid生成的3D网格坐标矩阵，表示网格中的点的三维坐标。
    - point_x, point_y, point_z: numpy.ndarray
        粒子在x, y, z方向上的坐标数组，用于插值电场值到粒子位置。

    返回：
    - Ex_distribution, Ey_distribution, Ez_distribution: numpy.ndarray
        在x, y, z方向的电场分布，表示在网格上每个点的电场值。
    - Ex_interp, Ey_interp, Ez_interp: numpy.ndarray
        在粒子位置插值得到的x, y, z方向的电场值。
    - voltage_distribution_out: numpy.ndarray
        在网格上计算得到的电势分布。
    - charge_distribution_out: numpy.ndarray
        电荷分布矩阵，作为函数的输入返回。
    - x_grid, y_grid, z_grid: numpy.ndarray
        网格点的坐标。

    流程：
    1. 计算格林函数：使用空间坐标矩阵 (Xmatrix, Ymatrix, Zmatrix) 计算格林函数 G。
    2. 使用FFT计算电势分布：
       - 对电荷分布 `charge_distribution_out` 进行傅里叶变换得到频域电荷分布 `rho_fft`。
       - 通过格林函数的傅里叶变换 `G_fft` 和电荷分布的傅里叶变换计算电势的傅里叶表示。
       - 反傅里叶变换得到空间电势分布 `voltage_distribution_out`。
    3. 计算电场分布：
       - 通过电势分布计算网格上各点的电场 `Ex_distribution`, `Ey_distribution`, `Ez_distribution`。
    4. 插值电场到粒子位置：
       - 使用 `my_3dInterp_vect` 对网格电场进行插值，得到粒子位置的电场值 `Ex_interp`, `Ey_interp`, `Ez_interp`。
    5. 返回电场分布、电势分布及网格点信息。
    """

    epsilon_0 = 8.854187817e-12  # 真空介电常数

    # 计算格林函数
    G_function = green_function(Xmatrix-np.mean(x_grid), Ymatrix-np.mean(y_grid), Zmatrix-np.mean(z_grid))
    G_fft = fftn(G_function)

    # 计算电势分布
    rho_fft = fftn(charge_distribution_out)
    # 在频域中相乘
    voltage_distribution_fft = G_fft * rho_fft * 1 / (4 * np.pi * epsilon_0)
    # 结果中的频率或时间分量的顺序不同, 默认情况下是将零频率或零时间分量放在数组的起始位置。
    voltage_distribution_out = fftshift(np.real(ifftn(voltage_distribution_fft)))

    # 计算电场分布
    Ex_distribution, Ey_distribution, Ez_distribution = (
        CalculateEFieldFromVoltage(x_grid, y_grid, z_grid, voltage_distribution_out))

    # 插值电场到粒子位置
    Ex_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ex_distribution, point_x, point_y, point_z)
    Ey_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ey_distribution, point_x, point_y, point_z)
    Ez_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ez_distribution, point_x, point_y, point_z)

    # 将插值得到的电场分量从笛卡尔坐标 (Ex_interp, Ey_interp, Ez_interp) 转换为极坐标系 (Er, Efi, Ez)
    Er_interp, Ez_interp, Efi_interp = ConvertToPolarElectricField(point_x, point_y, Ex_interp, Ey_interp, Ez_interp)

    return (Ex_distribution, Ey_distribution, Ez_distribution, Er_interp, Ez_interp, Efi_interp,
            voltage_distribution_out, charge_distribution_out, x_grid, y_grid, z_grid)


def SC_calculator_fft_flat(charge_distribution_flat_expand,
                          r_grid_flat_expand, fr_grid_flat_expand, z_grid_flat_expand,
                          Rmatrix_flat_expand, FRmatrix_flat_expand, Zmatrix_flat_expand,
                          point_r_flat, point_fr_flat, point_z_flat):
    """
    使用FFT计算空间电荷场，输入为已知的3维电荷分布矩阵和3维网格矩阵。

    参数：
    - charge_distribution_out: numpy.ndarray
        包含网格点上电荷分布的3D数组，表示空间中的电荷密度分布。
    - x_grid, y_grid, z_grid: numpy.ndarray
        网格在x, y, z方向的坐标数组。
    - Xmatrix, Ymatrix, Zmatrix: numpy.ndarray
        通过x_grid, y_grid, z_grid生成的3D网格坐标矩阵，表示网格中的点的三维坐标。
    - point_x, point_y, point_z: numpy.ndarray
        粒子在x, y, z方向上的坐标数组，用于插值电场值到粒子位置。

    返回：
    - Ex_distribution, Ey_distribution, Ez_distribution: numpy.ndarray
        在x, y, z方向的电场分布，表示在网格上每个点的电场值。
    - Ex_interp, Ey_interp, Ez_interp: numpy.ndarray
        在粒子位置插值得到的x, y, z方向的电场值。
    - voltage_distribution_out: numpy.ndarray
        在网格上计算得到的电势分布。
    - charge_distribution_out: numpy.ndarray
        电荷分布矩阵，作为函数的输入返回。
    - x_grid, y_grid, z_grid: numpy.ndarray
        网格点的坐标。

    流程：
    1. 计算格林函数：使用空间坐标矩阵 (Xmatrix, Ymatrix, Zmatrix) 计算格林函数 G。
    2. 使用FFT计算电势分布：
       - 对电荷分布 `charge_distribution_out` 进行傅里叶变换得到频域电荷分布 `rho_fft`。
       - 通过格林函数的傅里叶变换 `G_fft` 和电荷分布的傅里叶变换计算电势的傅里叶表示。
       - 反傅里叶变换得到空间电势分布 `voltage_distribution_out`。
    3. 计算电场分布：
       - 通过电势分布计算网格上各点的电场 `Ex_distribution`, `Ey_distribution`, `Ez_distribution`。
    4. 插值电场到粒子位置：
       - 使用 `my_3dInterp_vect` 对网格电场进行插值，得到粒子位置的电场值 `Ex_interp`, `Ey_interp`, `Ez_interp`。
    5. 返回电场分布、电势分布及网格点信息。
    """

    epsilon_0 = 8.854187817e-12  # 真空介电常数


    # 计算格林函数
    G_function = green_function(Rmatrix_flat_expand-np.mean(r_grid_flat_expand),
                                FRmatrix_flat_expand-np.mean(fr_grid_flat_expand),
                                Zmatrix_flat_expand-np.mean(z_grid_flat_expand))
    G_fft = fftn(G_function)

    # 计算电势分布
    rho_fft = fftn(charge_distribution_flat_expand)
    # 在频域中相乘
    voltage_distribution_fft = G_fft * rho_fft * 1 / (4 * np.pi * epsilon_0)
    # 结果中的频率或时间分量的顺序不同, 默认情况下是将零频率或零时间分量放在数组的起始位置。
    voltage_distribution_out = fftshift(np.real(ifftn(voltage_distribution_fft)))

    # rho_fft = fftn(charge_distribution_out)
    # # 在频域中相乘
    # voltage_distribution_fft = G_fft * rho_fft * 1 / (4 * np.pi * epsilon_0)
    # # voltage_distribution_out = np.real(ifftn(voltage_distribution_fft))
    # # 结果中的频率或时间分量的顺序不同, 默认情况下是将零频率或零时间分量放在数组的起始位置。
    # voltage_distribution_out = np.real(fftshift(ifftn(voltage_distribution_fft)))

    # 计算电场分布
    Er_distribution, Ef_distribution, Ez_distribution = (
        CalculateEFieldFromVoltage(r_grid_flat_expand, fr_grid_flat_expand, z_grid_flat_expand, voltage_distribution_out))

    # 插值电场到粒子位置
    Er_interp = my_3dInterp_vect(r_grid_flat_expand, fr_grid_flat_expand, z_grid_flat_expand, Er_distribution, point_r_flat, point_fr_flat, point_z_flat)
    Ef_interp = my_3dInterp_vect(r_grid_flat_expand, fr_grid_flat_expand, z_grid_flat_expand, Ef_distribution, point_r_flat, point_fr_flat, point_z_flat)
    Ez_interp = my_3dInterp_vect(r_grid_flat_expand, fr_grid_flat_expand, z_grid_flat_expand, Ez_distribution, point_r_flat, point_fr_flat, point_z_flat)

    return (Er_distribution, Ef_distribution, Ez_distribution, Er_interp, Ef_interp, Ez_interp,
            voltage_distribution_out, charge_distribution_flat_expand,
            r_grid_flat_expand, fr_grid_flat_expand, z_grid_flat_expand)


def Bunch_SC_Calculator(Bunch_obj):
    """
    使用FFT方法计算Bunch_obj中所有粒子的自感应空间电荷电场。

    参数：
    - Bunch_obj: FFAG_Bunch 对象
        表示粒子束的对象，包含粒子的位置信息、电荷分布以及网格信息。

    流程：
    1. 检查网格信息是否存在。如果网格信息为空，则不进行计算。
    2. 提取存活粒子的坐标。
    3. 调用 SC_calculator_fft_new 函数计算电场分布，并将电场值插值到粒子位置。
    4. 更新 Bunch_obj 中的局部电场分量。

    返回：
    - Ex_interp, Ey_interp, Ez_interp: numpy.ndarray
        存活粒子位置的电场插值值。
    """

    # 判断 Bunch_obj 中的网格信息是否存在，如果为空则直接返回
    if Bunch_obj.xmax_Global is None or Bunch_obj.zmax_Global is None or Bunch_obj.Zgrid is None:
        print("Grid information is missing. Cannot calculate space charge electric field.")
        return None, None, None, None, None, None, None, None

    # 提取 FFAG_Bunch 中存活粒子的笛卡尔坐标
    local_points, Local_ID, Global_ID = Bunch_obj.GetLocalCoordinates()

    # 从 Bunch_obj 中提取电荷分布和网格信息
    charge_distribution_out = Bunch_obj.charge_distribution_global  # 全局电荷分布矩阵
    x_grid, y_grid, z_grid = Bunch_obj.Xgrid, Bunch_obj.Ygrid, Bunch_obj.Zgrid  # 网格坐标
    Xmatrix, Ymatrix, Zmatrix = Bunch_obj.Xmatrix, Bunch_obj.Ymatrix, Bunch_obj.Zmatrix  # 3D 网格矩阵

    # 提取存活粒子的 x, y, z 坐标
    point_x, point_y, point_z = local_points[:, 0], local_points[:, 1], local_points[:, 2]

    # 使用 SC_calculator_fft_new 计算空间电荷场并进行电场插值
    (Ex_distribution, Ey_distribution, Ez_distribution,
     Er_interp, Ez_interp, Efi_interp,
     voltage_distribution_out, charge_distribution_out, _, _, _) = (
        SC_calculator_fft_new(charge_distribution_out,
                              x_grid, y_grid, z_grid,
                              Xmatrix, Ymatrix, Zmatrix,
                              point_x, point_y, point_z)
    )

    # 将插值后的电场值存储到 Bunch_obj 中，更新 LocalBunch 的电场分量
    Bunch_obj.Update_SC_Efield_Local(Er_interp, Efi_interp, Ez_interp, Local_ID)

    return (Ex_distribution, Ey_distribution, Ez_distribution,
            Er_interp, Efi_interp, Ez_interp,
            voltage_distribution_out, charge_distribution_out)


def Bunch_SC_Calculator_FlatCoordinate(Bunch_obj):
    """
    使用FFT方法计算Bunch_obj中所有粒子的自感应空间电荷电场。

    参数：
    - Bunch_obj: FFAG_Bunch 对象
        表示粒子束的对象，包含粒子的位置信息、电荷分布以及网格信息。

    流程：
    1. 检查网格信息是否存在。如果网格信息为空，则不进行计算。
    2. 提取存活粒子的坐标。
    3. 调用 SC_calculator_fft_new 函数计算电场分布，并将电场值插值到粒子位置。
    4. 更新 Bunch_obj 中的局部电场分量。

    返回：
    - Ex_interp, Ey_interp, Ez_interp: numpy.ndarray
        存活粒子位置的电场插值值。
    """

    # 判断 Bunch_obj 中的网格信息是否存在，如果为空则直接返回
    if Bunch_obj.rmax_Global_flat is None or Bunch_obj.zmax_Global_flat is None or Bunch_obj.fmax_Global_flat is None:
        # print("No particle injected or survived. Grid information is missing.")
        return None, None, None, None, None, None, None, None, None, None, None

    # 提取 FFAG_Bunch 中存活粒子的笛卡尔坐标
    local_points_flat, Local_ID, Global_ID = Bunch_obj.GetLocalCoordinates_FlatCoordinate()

    # 从 Bunch_obj 中提取电荷分布和网格信息
    charge_distribution_out_flat = Bunch_obj.charge_distribution_global_flat  # 全局电荷分布矩阵
    r_grid_flat, f_grid_flat, z_grid_flat = Bunch_obj.Rgrid_flat, Bunch_obj.Fgrid_flat, Bunch_obj.Zgrid_flat  # 网格坐标
    Rmatrix_flat, Fmatrix_flat, Zmatrix_flat = Bunch_obj.Rmatrix_flat, Bunch_obj.Fmatrix_flat, Bunch_obj.Zmatrix_flat  # 3D 网格矩阵

    # 提取存活粒子的 x, y, z 坐标
    point_r_flat, point_f_flat, point_z_flat = local_points_flat[:, 0], local_points_flat[:, 1], local_points_flat[:, 2]
    point_f_flat = np.mod(point_f_flat, 2*np.pi)
    if Bunch_obj.fmin_Global_flat is not None and Bunch_obj.fmin_Global_flat < 0:
        point_f_flat[point_f_flat > np.pi] -= 2*np.pi

    # 扩展计算域,
    f_grid_flat_expand, r_grid_flat_expand, z_grid_flat_expand = (
        FFAG_FlatBunch().expand_axis(f_grid_flat, r_grid_flat, z_grid_flat))
    Rmatrix_flat_expand, Fmatrix_flat_expand, Zmatrix_flat_expand = (
        FFAG_FlatBunch().expand_matrix(f_grid_flat, Rmatrix_flat, Fmatrix_flat, Zmatrix_flat))
    charge_distribution_flat_expand, _ = (
        FFAG_FlatBunch().expand_data(f_grid_flat, charge_distribution_out_flat))

    # y方向量纲修正
    fr_grid_flat_expand = f_grid_flat_expand * Bunch_obj.rmean
    FRmatrix_flat_expand = Fmatrix_flat_expand * Bunch_obj.rmean
    point_fr_flat = point_f_flat * Bunch_obj.rmean

    # # 使用 SC_calculator_fft_new 计算空间电荷场并进行电场插值
    # (Er_distribution_flat, Ef_distribution_flat, Ez_distribution_flat,
    #  Er_interp, Ez_interp, Efi_interp,
    #  voltage_distribution_out, charge_distribution_out, _, _, _) = (
    #     SC_calculator_fft_new(charge_distribution_flat_expand,
    #                           r_grid_flat_expand, fr_grid_flat_expand, z_grid_flat_expand,
    #                           Rmatrix_flat_expand, FRmatrix_flat_expand, Zmatrix_flat_expand,
    #                           point_r_flat, point_fr_flat, point_z_flat)
    # )

    # 使用 SC_calculator_fft_flat 计算空间电荷场并进行电场插值
    (Er_distribution_flat, Ef_distribution_flat, Ez_distribution_flat,
     Er_interp, Efi_interp, Ez_interp,
     voltage_distribution_out, charge_distribution_out, _, _, _) = (
        SC_calculator_fft_flat(charge_distribution_flat_expand,
                              r_grid_flat_expand, fr_grid_flat_expand, z_grid_flat_expand,
                              Rmatrix_flat_expand, FRmatrix_flat_expand, Zmatrix_flat_expand,
                              point_r_flat, point_fr_flat, point_z_flat)
    )

    # 将插值后的电场值存储到 Bunch_obj 中，更新 LocalBunch 的电场分量
    Bunch_obj.Update_SC_Efield_Local(Er_interp, Efi_interp, Ez_interp, Local_ID)

    Bunch_obj.Er_distribution_flat_save = Er_distribution_flat
    Bunch_obj.Ef_distribution_flat_save = Ef_distribution_flat
    Bunch_obj.Ez_distribution_flat_save = Ez_distribution_flat
    Bunch_obj.voltage_distribution_flat_save = voltage_distribution_out

    Bunch_obj.Rmatrix_flat_save = Rmatrix_flat_expand
    Bunch_obj.Fmatrix_flat_save = Fmatrix_flat_expand
    Bunch_obj.Zmatrix_flat_save = Zmatrix_flat_expand
    Bunch_obj.charge_distribution_flat_save = charge_distribution_flat_expand

    return (Er_distribution_flat, Ef_distribution_flat, Ez_distribution_flat,
            Er_interp, Efi_interp, Ez_interp,
            voltage_distribution_out, charge_distribution_out,
            Rmatrix_flat_expand, Fmatrix_flat_expand, Zmatrix_flat_expand)



if __name__ == '__main__':
    pass
    # # Parameters for Gaussian distribution
    # num_points = 100000
    # mean = [0, 0, 0]  # mean at the origin
    # cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # diagonal covariance, points are not correlated
    # # # 设置随机数种子
    # # np.random.seed(42)
    # # Generating Gaussian distributed points
    # gaussian_points = np.random.multivariate_normal(mean, cov, num_points)
    # gaussian_points[:, 0] += 5
    #
    # # Using percentiles to remove outliers and determine the range
    # percentile_low = 0
    # percentile_high = 100
    #
    # # Defining the grid size
    # n, m, l = 40, 40, 40  # Number of divisions in x, y, and z directions
    # SC_calculator(gaussian_points, percentile_low, percentile_high, n, m, l)
    #
    # (Ex_distribution_direct, Ey_distribution_direct, Ez_distribution_direct,
    #  Ex_interp, Ey_interp, Ez_interp,
    #  voltage_distribution_direct, charge_distribution_direct, x_grid_direct, y_grid_direct, z_grid_direct) = (
    #     SC_calculator(gaussian_points, percentile_low, percentile_high, n, m, l))
    #
    # # Defining the grid size
    # SC_calculator_fft(gaussian_points, percentile_low, percentile_high, n, m, l)
    #
    # (Ex_distribution_fft, Ey_distribution_fft, Ez_distribution_fft,
    #  Ex_interp_fft, Ey_interp_fft, Ez_interp_fft,
    #  voltage_distribution_fft, charge_distribution_fft, x_grid_fft, y_grid_fft, z_grid_fft) = (
    #     SC_calculator_fft(gaussian_points, percentile_low, percentile_high, n, m, l))
    #
    #
    # xx_mesh_direct, yy_mesh_direct = np.meshgrid(x_grid_direct, y_grid_direct)
    # xx_mesh_fft, yy_mesh_fft = np.meshgrid(x_grid_fft, y_grid_fft)
    #
    # # visualization
    # # 选择中间层进行剖面图的绘制
    # xy_slice_direct = voltage_distribution_direct[:, :, l // 3 * 1]  # XY平面
    # xz_slice_direct = voltage_distribution_direct[:, :, l // 2 * 1]  # XZ平面
    # yz_slice_direct = voltage_distribution_direct[:, :, l // 3 * 2]  # YZ平面
    # xy_slice_fft = voltage_distribution_fft[:, :, l // 3 * 1]  # XY平面
    # xz_slice_fft = voltage_distribution_fft[:, :, l // 2 * 1]  # XZ平面
    # yz_slice_fft = voltage_distribution_fft[:, :, l // 3 * 2]  # YZ平面
    #
    # # Modifying the visualization to use the same color bar for all three plots for comparison
    #
    # # Find the global minimum and maximum voltage values for consistent color mapping
    # vmin = np.min(voltage_distribution_direct)
    # vmax = np.max(voltage_distribution_direct)
    # vmin_fft = np.min(voltage_distribution_fft)
    # vmax_fft = np.max(voltage_distribution_fft)
    #
    # fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    #
    # # 绘制_direct的结果
    # for i, slice_data in enumerate([xy_slice_direct, xz_slice_direct, yz_slice_direct]):
    #     im = axs[0, i].imshow(slice_data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    #     # axs[0, i].set_title(f'_direct Voltage Distribution at {["z/3", "z/2", "2z/3"][i]}')
    #     axs[0, i].set_xlabel('X axis')
    #     axs[0, i].set_ylabel('Y axis')
    #
    # # 绘制_fft的结果
    # for i, slice_data in enumerate([xy_slice_fft, xz_slice_fft, yz_slice_fft]):
    #     im = axs[1, i].imshow(slice_data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    #     # axs[1, i].set_title(f'_fft Voltage Distribution at {["z/3", "z/2", "2z/3"][i]}')
    #     axs[1, i].set_xlabel('X axis')
    #     axs[1, i].set_ylabel('Y axis')
    #
    # # 为所有图像创建一个统一的颜色条
    # cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.2, label='Voltage (V)')
    # plt.subplots_adjust(right=0.85)
    # # plt.tight_layout()
    #
    # # fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    # # # Plotting each slice with the same color scale
    # # cmap = 'viridis'  # Color map
    # # for i, slice_data in enumerate([xy_slice_direct, xz_slice_direct, yz_slice_direct]):
    # #     pos = axs[i].imshow(slice_data, cmap=cmap, origin='lower')
    # #     axs[i].set_title(f'Voltage Distribution at {["z/5", "z/2", "4z/5"][i]}')
    # #     axs[i].set_xlabel('X axis')
    # #     axs[i].set_ylabel('Y axis')
    # # # Create a single color bar for all plots
    # # fig.colorbar(pos, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label='Voltage (V)')
    #
    # # vx_slice_direct = voltage_distribution_direct[:, m // 2, l // 3 * 1]  # XY平面
    # # vx_slice_fft = voltage_distribution_fft[:, m // 2, l // 3 * 1]  # XY平面
    # # plt.figure()
    # # plt.plot(vx_slice_direct)
    # # plt.plot(vx_slice_fft)
    #
    # ra1 = np.arange(-5.0, -1.0, 0.01)
    # ra2 = np.arange(1.0, 5.0, 0.01)
    # epsilon_0 = 8.854187817e-12  # 真空介电常数
    # E_theory1 = (1 / (4 * np.pi * epsilon_0)) / (ra1) ** 2 * (ra1 / np.abs(ra1))
    # E_theory2 = (1 / (4 * np.pi * epsilon_0)) / (ra2) ** 2 * (ra2 / np.abs(ra2))
    #
    # ex_slice_direct = Ex_distribution_direct[:, m // 2, l // 2]  # XY平面
    # ex_slice_fft = Ex_distribution_fft[:, m // 2, l // 2]  # XY平面
    # plt.figure()
    # plt.plot(x_grid_direct, ex_slice_direct, label="PIC")
    # plt.plot(x_grid_direct, ex_slice_fft, linewidth=2.5, label="PIC-FFT")
    # # plt.plot(ra2, E_theory2, linestyle='dashed', color='red', linewidth=2.5, label="theoretical")
    # # plt.plot(ra1, E_theory1, linestyle='dashed', color='red', linewidth=2.5)
    # plt.xlabel('X(m)')
    # plt.ylabel('Ex(V/m)')
    # plt.legend(loc='best')
    # fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    # # 绘制_direct的结果
    # for i, slice_data in enumerate([xy_slice_direct, xz_slice_direct, yz_slice_direct]):
    #     axs[0, i].streamplot(xx_mesh_fft, yy_mesh_fft, Ey_distribution_fft[:, :, l // 2],
    #                          Ex_distribution_fft[:, :, l // 2], color='b', linewidth=1, density=2)
    #     # axs[0, i].set_xlabel('X axis')
    #     if i == 0:
    #         axs[0, i].set_ylabel('Y axis')
    #
    # # 绘制_fft的结果
    # for i, slice_data in enumerate([xy_slice_fft, xz_slice_fft, yz_slice_fft]):
    #     axs[1, i].streamplot(xx_mesh_direct, yy_mesh_direct, Ey_distribution_direct[:, :, l // 2],
    #                          Ex_distribution_direct[:, :, l // 2], color='b', linewidth=1, density=2)
    #     axs[1, i].set_xlabel('X axis')
    #     if i == 0:
    #         axs[1, i].set_ylabel('Y axis')
    #
    # plt.show()

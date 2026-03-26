import numpy as np
import re
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
import ast
from scipy.interpolate import interp1d
from FFAG_MathTools import (interpolate_spiral_emap_unique_gap_kernel, Bilinear_interp_2D_vect, Bmap_add_all_orders,
                            linear_interpolation_uniform_njit_two_vars)
from FFAG_ParasAndConversion import FFAG_ConversionTools, FFAG_GlobalParameters
from FFAG_Utils import FFAG_GeometryCalc, emap_kernel
import json
import numba as nb
from FFAG_MathTools import fast_mod_njit


class FFAG_Field:
    def __init__(self, filename):
        data = np.loadtxt(filename, skiprows=1)
        fi, r = data[0, 1:], data[1:, 0]
        self.r_axis, self.fi_axis, self.map_data = r, fi, data[1:, 1:]
        self.r_min, self.r_max, self.r_step = r[0], r[-1], r[1] - r[0]
        self.fi_min, self.fi_max, self.fi_step = fi[0], fi[-1], fi[1] - fi[0]
        self.r_size, self.fi_size = len(self.r_axis), len(self.fi_axis)


class FFAG_BField_new():

    def __init__(self, foldname, max_order, flag3D=True):
        self.max_order = max_order
        self.flag3D = flag3D

        # 初始化 Br, Bz, Bfi 的系数矩阵列表
        self.Br_coeff_matrices = []
        self.Bz_coeff_matrices = []
        self.Bfi_coeff_matrices = []

        # 检查 foldname 的类型并加载数据
        if isinstance(foldname, str):  # 如果 foldname 是路径字符串
            foldname = os.path.normpath(foldname)
            for order in range(max_order + 1):
                BrFileName = os.path.join(foldname, f"Br_order_{order}.npz")
                BzFileName = os.path.join(foldname, f"Bz_order_{order}.npz")
                BfiFileName = os.path.join(foldname, f"Bfi_order_{order}.npz")

                # 从 .npz 文件中加载各阶次系数矩阵
                self.Br_coeff_matrices.append(np.load(BrFileName)['Br_Taylor_coeff'])
                self.Bz_coeff_matrices.append(np.load(BzFileName)['Bz_Taylor_coeff'])
                self.Bfi_coeff_matrices.append(np.load(BfiFileName)['Bfi_Taylor_coeff'])

            # 设置 r 和 fi 的轴，从文件中读取数据
            self.BmapFileName = os.path.join(foldname, "Bmap.txt")
            data = np.loadtxt(self.BmapFileName, skiprows=1)
            fi, r = data[0, 1:], data[1:, 0]
            self.r_axis, self.fi_axis = r, fi
            self.map_data = self.Bz_coeff_matrices[0]

            # 读取 Bmap.txt 的表头以获取 Nsectors
            with open(self.BmapFileName, 'r') as file:
                match = re.search(r'Nsectors=(\d+)', file.readline().strip())
                self.Nsectors = int(match.group(1)) if match else None
        elif isinstance(foldname, np.ndarray):  # 如果 foldname 是 ndarray
            # 设置 r 和 fi 的轴，从数组中读取数据
            fi, r = foldname[0, 1:], foldname[1:, 0]
            self.r_axis, self.fi_axis = r, fi
            self.map_data = foldname[1:, 1:]
            self.Bz_coeff_matrices.append(self.map_data)
            self.Nsectors = None  # 如果是 ndarray 没有提供 Nsectors 信息
        else:
            raise TypeError("foldname 必须是路径字符串或 numpy.ndarray")

        # 通用属性计算
        self.r_min, self.r_max, self.r_step = r[0], r[-1], r[1] - r[0]
        self.fi_min, self.fi_max, self.fi_step = fi[0], fi[-1], fi[1] - fi[0]
        self.r_size, self.fi_size = len(self.r_axis), len(self.fi_axis)
        self.BMean = np.mean(self.map_data, axis=1)
        self.flutter = self.map_data / np.tile(self.BMean, (self.fi_size, 1)).T
        # self.f1 = FFAG_interpolation().linear_interpolation(self.r_axis, self.BMean)
        # 使用 interp1d 替换 FFAG_interpolation.linear_interpolation
        self.f1 = interp1d(self.r_axis, self.BMean, kind='linear', fill_value="extrapolate")
        self.rmin90 = self.r_min + (self.r_max - self.r_min) * 0.02
        Pmin90 = self.rmin90 * FFAG_GlobalParameters().q * self.f1(self.rmin90)
        self.rmax90 = self.r_max - (self.r_max - self.r_min) * 0.02
        Pmax90 = self.rmax90 * FFAG_GlobalParameters().q * self.f1(self.rmax90)
        self.Ekmin90 = FFAG_ConversionTools().P2Ek(Pmin90) / FFAG_GlobalParameters().q / 1e6
        self.Ekmax90 = FFAG_ConversionTools().P2Ek(Pmax90) / FFAG_GlobalParameters().q / 1e6

        # MPI 并行化打印信息
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            print('*' * 100)
            print(f'loading Bmap with order-{2*max_order + 1}- NonLinear terms ... ...')
            print('Bmap rmin = %.3fm, rmax = %.3fm' % (self.r_min, self.r_max))
            print(
                'Ek = %.3f MeV for R%.3fm orbit, Ek = %.3f MeV for R%.3fm orbit.' % (
                    self.Ekmin90, self.rmin90, self.Ekmax90, self.rmax90))
            print('*' * 100)

    def ExtendField(self, ext_Rsize_min, ext_Rsize_max, Binf):

        ext_min_arr = np.ones((ext_Rsize_min, self.fi_size)) * Binf * (-1)
        ext_max_arr = np.ones((ext_Rsize_max, self.fi_size)) * Binf

        r_min_axis_reverse = self.r_min - np.arange(1, ext_Rsize_min + 1, 1) * self.r_step
        r_max_axis = self.r_max + np.arange(1, ext_Rsize_max + 1, 1) * self.r_step
        r_min_axis = r_min_axis_reverse[::-1]

        r_axis_new = np.concatenate((r_min_axis, self.r_axis, r_max_axis))
        map_data_new = np.row_stack((ext_min_arr, self.map_data, ext_max_arr))
        data_without_fiAxis = np.column_stack((r_axis_new, map_data_new))

        data = np.row_stack((np.insert(self.fi_axis, 0, 0), data_without_fiAxis))

        fi, r = data[0, 1:], data[1:, 0]
        self.r_axis_ext, self.fi_axis_ext, self.map_data_ext = r, fi, data[1:, 1:]
        self.r_min_ext, self.r_max_ext, self.r_step_ext = r[0], r[-1], r[1] - r[0]
        self.fi_min_ext, self.fi_max_ext, self.fi_step_ext = fi[0], fi[-1], fi[1] - fi[0]
        self.r_size_ext, self.fi_size_ext = len(self.r_axis_ext), len(self.fi_axis_ext)

        self.BMean_ext = np.mean(self.map_data_ext, 1)
        # self.f1_ext = FFAG_interpolation().linear_interpolation(self.r_axis_ext, self.BMean_ext)
        # 替换 FFAG_interpolation.linear_interpolation
        self.f1_ext = interp1d(self.r_axis_ext, self.BMean_ext, kind='linear', fill_value="extrapolate")

    # @profile
    def Interpolation2DMap(self, r, fi, order, flag):
        """
        对某个阶次的矩阵进行插值
        flag: 0 -> Bz, 1 -> Br, 2 -> Bfi
        """
        if flag == 0:
            FieldMapTemp = self.Bz_coeff_matrices[order]
        elif flag == 1:
            FieldMapTemp = self.Bfi_coeff_matrices[order]
        elif flag == 2:
            FieldMapTemp = self.Br_coeff_matrices[order]
        elif flag == 3:
            FieldMapTemp = self.map_data_ext
        else:
            raise ValueError("flag 的值必须是 0（Bz）, 1（Br）, 2（Bfi）或 3（Bz_extend）")

        # 执行 2D 插值
        # Value, OutRangeFlag = Lagrange_interp_2D_vect(self.r_axis, self.fi_axis, FieldMapTemp, r, fi)
        Value, OutRangeFlag = Bilinear_interp_2D_vect(self.r_axis, self.fi_axis, FieldMapTemp, r, fi)

        return Value, OutRangeFlag

    # def Interpolation2DMapFast(self, r, fi, z, Bz0, Br, Bfi, order):
    #     """
    #     对某个阶次的矩阵进行插值
    #     flag: 0 -> Bz, 1 -> Br, 2 -> Bfi
    #     """
    #
    #     Bilinear_interp_2D_vect_uniform(self.r_min, self.r_step, self.r_size,
    #                                     self.fi_min, self.fi_step, self.fi_size,
    #                                     self.Bz_coeff_matrices[order], r, fi, z, Bz0, order)
    #
    #     Bilinear_interp_2D_vect_uniform(self.r_min, self.r_step, self.r_size,
    #                                     self.fi_min, self.fi_step, self.fi_size,
    #                                     self.Bfi_coeff_matrices[order], r, fi, z, Bfi, order)
    #
    #     Bilinear_interp_2D_vect_uniform(self.r_min, self.r_step, self.r_size,
    #                                     self.fi_min, self.fi_step, self.fi_size,
    #                                     self.Br_coeff_matrices[order], r, fi, z, Br, order)


class FFAG_EField_new:
    def __init__(self, freq_curve_path, EnableFlag):

        header, data = self.ParseEmapFile(freq_curve_path)
        self.GapType = "fixed"
        # self.acc_voltage = header["V0"]
        self.acc_phi0 = np.deg2rad(header["acc_phi_paint"])
        # print(type(header["gap_azimuth"]))
        # print(np.atleast_1d(header["gap_azimuth"]))
        self.gap_azimuths = np.deg2rad(np.atleast_1d(header["gap_azimuth"]))
        self.gap_width = header["gap_width"]
        self.rmin = header["E_rmin"]
        self.rmax = header["E_rmax"]
        self.freq_curve = data
        self.harmonic = header["harmonic"]
        self.EnableFlag = EnableFlag
        self.rf_shift = np.deg2rad(np.atleast_1d(header["rf_shift"]))

        self.acc_regions = self.compute_acc_region()  # 计算加速区域
        # # 包围盒
        # self.box_x_min = np.min(self.acc_region[:, 0])
        # self.box_x_max = np.max(self.acc_region[:, 0])
        # self.box_y_min = np.min(self.acc_region[:, 1])
        # self.box_y_max = np.max(self.acc_region[:, 1])
        # 分别为每个 gap 求包围盒
        self.box_bounds = []
        for region in self.acc_regions:
            x_min, x_max = np.min(region[:, 0]), np.max(region[:, 0])
            y_min, y_max = np.min(region[:, 1]), np.max(region[:, 1])
            self.box_bounds.append((x_min, x_max, y_min, y_max))

        self.acc_Etot_uniform = 1 / self.gap_width

        # self.freq_curve_interp = interp1d(self.freq_curve[:,0], self.freq_curve[:,1], kind='linear', fill_value="extrapolate")

        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"[Emap] load analytical Emap from {os.path.dirname(freq_curve_path)}")

    # def compute_acc_region(self):
    #     """ 计算加速区域的矩形边界 """
    #     theta = self.gap_azimuth  # 方位角
    #
    #     # 计算中心线起点和终点（极坐标 -> 直角坐标）
    #     center_start_x, center_start_y = self.rmin * np.cos(theta), self.rmin * np.sin(theta)
    #     center_end_x, center_end_y = self.rmax * np.cos(theta), self.rmax * np.sin(theta)
    #
    #     # 计算法线方向的单位向量（gap_azimuth 旋转 90°）
    #     nx = np.cos(theta + np.pi / 2)
    #     ny = np.sin(theta + np.pi / 2)
    #
    #     # 计算矩形四个顶点（沿法线方向平移 gap_width/2）
    #     left_start_x = center_start_x + (self.gap_width / 2) * nx
    #     left_start_y = center_start_y + (self.gap_width / 2) * ny
    #     left_end_x = center_end_x + (self.gap_width / 2) * nx
    #     left_end_y = center_end_y + (self.gap_width / 2) * ny
    #
    #     right_start_x = center_start_x - (self.gap_width / 2) * nx
    #     right_start_y = center_start_y - (self.gap_width / 2) * ny
    #     right_end_x = center_end_x - (self.gap_width / 2) * nx
    #     right_end_y = center_end_y - (self.gap_width / 2) * ny
    #
    #     # 返回矩形的四个角点
    #     return np.array([
    #         [left_start_x, left_start_y],
    #         [left_end_x, left_end_y],
    #         [right_end_x, right_end_y],
    #         [right_start_x, right_start_y]
    #     ])

    def compute_acc_region(self):
        """
        计算一个或多个加速间隙的矩形边界。
        返回：
            regions: list[np.ndarray], 每个元素是 shape (4, 2) 的矩形顶点坐标。
        """
        regions = []

        for theta in np.atleast_1d(self.gap_azimuths):
            # 中心线起止点
            center_start = np.array([self.rmin * np.cos(theta), self.rmin * np.sin(theta)])
            center_end = np.array([self.rmax * np.cos(theta), self.rmax * np.sin(theta)])

            # 法线方向（gap 方位 + 90°）
            n = np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])

            # 四个顶点（沿法线 ±gap_width/2 平移）
            left_start = center_start + 0.5 * self.gap_width * n
            left_end = center_end + 0.5 * self.gap_width * n
            right_end = center_end - 0.5 * self.gap_width * n
            right_start = center_start - 0.5 * self.gap_width * n

            region = np.stack([left_start, left_end, right_end, right_start])
            regions.append(region)

        return regions

    # @profile
    # def Interpolation2D_EMap(self, r, fi, Er, Ez, Efi):
    #     if self.EnableFlag:
    #         # x, y = r * np.cos(fi), r * np.sin(fi)
    #         # particle_xy = np.array([x, y]).T
    #         inside_flags = FFAG_GeometryCalc().point_in_convex_polygon_fast(r, fi, self.acc_region, self.box_x_min, self.box_x_max, self.box_y_min, self.box_y_max)
    #
    #         # 电场方向是垂直于gap方向
    #         theta_n = self.gap_azimuth + np.pi / 2
    #         delta_angle = theta_n - fi  # 对每个粒子计算相对角度
    #
    #         # 只对加速区内的粒子施加电场
    #         Er[inside_flags] = self.acc_Etot * np.cos(delta_angle[inside_flags])
    #         Efi[inside_flags] = self.acc_Etot * np.sin(delta_angle[inside_flags])
    #         Er[~inside_flags] = 0.0
    #         Efi[~inside_flags] = 0.0
    #
    #         return Er, Efi, Ez
    #     else:
    #         return np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)

    def Interpolation2D_EMap(self, r, fi, z, Er, Ez, Efi, Malloc_RF_shift, v0_volt=1.0):
        """
        支持多个加速间隙的电场叠加。
        Malloc_RF_shift[k]：记录第 k 个粒子所处 gap index；
        若粒子不在任何 gap 内，则为 -1。
        """
        if not self.EnableFlag:
            Er.fill(0.0)
            Efi.fill(0.0)
            Ez.fill(0.0)

        # 逐 gap 检查
        for gap_index, (region, theta, box) in enumerate(zip(self.acc_regions, self.gap_azimuths, self.box_bounds)):
            box_x_min, box_x_max, box_y_min, box_y_max = box
            emap_kernel(
                r, fi,
                self.acc_Etot_uniform * v0_volt,
                theta + np.pi / 2.0,  # 电场方向垂直于gap方向
                region[:, 0], region[:, 1],
                box_x_min, box_x_max, box_y_min, box_y_max,
                Er, Efi, Ez, Malloc_RF_shift,
                gap_index  # 当前 gap 的编号
            )


    def Interpolation1D_freq_curve(self, time):
        X0_time = self.freq_curve[0, 0]
        Y_freq, Z_volt = self.freq_curve[:, 1], self.freq_curve[:, 2]
        dX = self.freq_curve[1, 0] - self.freq_curve[0, 0]
        Y0_time, Z0_time, _ = linear_interpolation_uniform_njit_two_vars(X0_time, dX, Y_freq, Z_volt, time)
        return Y0_time, Z0_time

    @staticmethod
    def ParseEmapFile(file_path):
        # 初始化参数和行计数
        params = {}
        data_start_flag = False
        non_data_lines = 0

        # 读取文件，首先统计参数行的数量
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()

                # 检测数据部分的起始标志：固定表头
                if line.startswith("#Time (s)"):
                    non_data_lines += 1  # 记录表头行
                    break

                # 解析参数部分
                if "=" in line and not data_start_flag:
                    key, value = line.split("=", 1)
                    key, value = key.strip(), value.strip()

                    # 尝试将 value 转换成数值，如果失败则保留字符串
                    try:
                        if '.' in value or 'e' in value.lower():
                            params[key] = float(value)  # 处理浮点数
                        else:
                            params[key] = int(value)  # 处理整数
                    except ValueError:
                        # 列表/元组（如 [90, 180] 或 (90, 180)）
                        parsed = None
                        if (value.startswith('[') and value.endswith(']')) or (
                                value.startswith('(') and value.endswith(')')):
                            try:
                                obj = ast.literal_eval(value)
                                if isinstance(obj, (list, tuple)):
                                    # 统一转 float（保持简洁；需要 int 可自行改成 int）
                                    parsed = [float(x) for x in obj]
                            except Exception:
                                parsed = None

                        # 逗号分隔（如 "90, 180"）
                        if parsed is None and ',' in value:
                            parts = [p.strip() for p in value.split(',') if p.strip() != ""]
                            try:
                                parsed = [float(p) if ('.' in p or 'e' in p.lower()) else float(int(p)) for p in parts]
                            except Exception:
                                parsed = None

                        params[key] = parsed if parsed is not None else value  # 解析失败则保留原字符串

                # 统计非数据部分的行数
                non_data_lines += 1

        # 解析数据部分
        data = np.loadtxt(file_path, skiprows=non_data_lines)

        # for key, value in params.items():
        #     print(f"{key}: {value}")

        return params, data


class FFAG_EField_spiral:
    def __init__(self, freq_curve_path, EnableFlag):
        header, data = self.ParseEmapFile(freq_curve_path)
        self.GapType = "spiral"
        self.acc_phi0 = np.deg2rad(float(header["acc_phi_paint"]))
        self.gap_azimuths = np.deg2rad(np.atleast_1d(header["gap_azimuth"]).astype(np.float64))
        self.gap_width = float(header["gap_width"])
        self.freq_curve = data
        self.harmonic = int(header["harmonic"])
        self.EnableFlag = EnableFlag
        self.rf_shift = np.deg2rad(np.atleast_1d(header["rf_shift"]).astype(np.float64))

        emap_dir = os.path.dirname(os.path.abspath(freq_curve_path))
        er_file = header.get("Er_map_file", "Er_coef.txt")
        ez_file = header.get("Ez_map_file", "Ez_coef.txt")
        ephi_file = header.get("Ephi_map_file", "Ephi_coef.txt")

        self.Er_map, self.r_axis, self.phi_axis = self.LoadSpiralMap(os.path.join(emap_dir, er_file))
        self.Ezcoef_map, _, _ = self.LoadSpiralMap(os.path.join(emap_dir, ez_file))
        self.Ephi_map, _, _ = self.LoadSpiralMap(os.path.join(emap_dir, ephi_file))

        self.r0 = self.r_axis[0]
        self.dr = self.r_axis[1] - self.r_axis[0]
        self.nr = len(self.r_axis)

        self.phi0 = self.phi_axis[0]
        self.dphi = self.phi_axis[1] - self.phi_axis[0]
        self.nphi = len(self.phi_axis)

        self.map_rmin = self.r_axis[0]
        self.map_rmax = self.r_axis[-1]
        self.map_phimin = self.phi_axis[0]
        self.map_phimax = self.phi_axis[-1]

        self.acc_Etot_uniform = 1.0 / self.gap_width

        # --------------------------------------------------
        # 读取 spiral gap 交点信息，并构造按能量的一维插值器
        # --------------------------------------------------
        self.Ek_gap = np.atleast_1d(header["Ek_gap"]).astype(np.float64)
        self.r_gap = np.atleast_1d(header["r_gap"]).astype(np.float64)
        self.phi_gap_deg = np.atleast_1d(header["phi_gap_deg"]).astype(np.float64)
        self.phi_gap_unwrapped_deg = np.atleast_1d(header["phi_gap_unwrapped_deg"]).astype(np.float64)

        # 转成弧度
        self.phi_gap = np.deg2rad(self.phi_gap_deg)
        self.phi_gap_unwrapped = np.deg2rad(self.phi_gap_unwrapped_deg)

        # 若表头写入顺序异常，这里统一按能量升序排序
        sort_idx = np.argsort(self.Ek_gap)
        self.Ek_gap = self.Ek_gap[sort_idx]
        self.r_gap = self.r_gap[sort_idx]
        self.phi_gap = self.phi_gap[sort_idx]
        self.phi_gap_unwrapped = self.phi_gap_unwrapped[sort_idx]

        # 简单线性插值器
        def _make_linear_interp(x_nodes, y_nodes):
            x_nodes = np.asarray(x_nodes, dtype=np.float64)
            y_nodes = np.asarray(y_nodes, dtype=np.float64)

            def interp_func(x):
                x = np.asarray(x, dtype=np.float64)
                return np.interp(x, x_nodes, y_nodes)

            return interp_func

        self.r_gap_interp = _make_linear_interp(self.Ek_gap, self.r_gap)
        self.phi_gap_interp_unwrapped = _make_linear_interp(self.Ek_gap, self.phi_gap_unwrapped)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"[Emap] load spiral Emap from {emap_dir}")
            # print(f"[Emap] gap azimuths (deg) = {np.rad2deg(self.gap_azimuths)}")
            # print(f"[Emap] rf_shift (deg)     = {np.rad2deg(self.rf_shift)}")
            # print(
            #     f"[Emap] source phi range (deg) = "
            #     f"[{np.rad2deg(self.map_phimin):.6f}, {np.rad2deg(self.map_phimax):.6f}]"
            # )

    def LoadSpiralMap(self, filepath):
        raw = np.loadtxt(filepath)
        phi_axis = raw[0, 1:].astype(np.float64)
        r_axis = raw[1:, 0].astype(np.float64)
        data = raw[1:, 1:].astype(np.float64)
        return data, r_axis, phi_axis

    def Interpolation2D_EMap(self, r, fi, z, Er, Ez, Efi, HitGapIndex, v0_volt=1.0):
        """
        纯空间版本：
        1. 不考虑 RF 时间相位
        2. 每个点最多命中一个 gap
        3. HitGapIndex 记录命中的 gap 编号：
           -1 表示未命中
           0,1,2,... 表示命中的 gap 序号
        4. Ez 输出为真实 Ez，而不是 Ez/z
        """
        if not self.EnableFlag:
            Er.fill(0.0)
            Efi.fill(0.0)
            Ez.fill(0.0)
            HitGapIndex.fill(-1)
            return

        npt = len(r)

        tmp_phi = np.empty(npt, dtype=np.float64)
        tmp_flag = np.ones(npt, dtype=np.int32)

        rf_amp = v0_volt * self.acc_Etot_uniform

        interpolate_spiral_emap_unique_gap_kernel(
            r, fi, z,
            self.gap_azimuths,
            rf_amp,
            self.r0, self.dr, self.nr,
            self.phi0, self.dphi, self.nphi,
            self.Er_map, self.Ezcoef_map, self.Ephi_map,
            Er, Ez, Efi,
            HitGapIndex,
            tmp_phi, tmp_flag
        )

    def Interpolation1D_freq_curve(self, time):
        X0_time = self.freq_curve[0, 0]
        dX = self.freq_curve[1, 0] - self.freq_curve[0, 0]
        Y_freq = self.freq_curve[:, 1]
        Z_volt = self.freq_curve[:, 2]
        Y0_time, Z0_time, _ = linear_interpolation_uniform_njit_two_vars(
            X0_time, dX, Y_freq, Z_volt, time
        )
        return Y0_time, Z0_time

    @staticmethod
    def ParseEmapFile(file_path):
        params = {}
        non_data_lines = 0

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()

                if line.startswith("#Time (s)"):
                    non_data_lines += 1
                    break

                if "=" in line:
                    key, value = line.split("=", 1)
                    key, value = key.strip(), value.strip()

                    try:
                        if "." in value or "e" in value.lower():
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except ValueError:
                        parsed = None

                        if (value.startswith("[") and value.endswith("]")) or (
                                value.startswith("(") and value.endswith(")")):
                            obj = ast.literal_eval(value)
                            if isinstance(obj, (list, tuple)):
                                parsed = [float(x) for x in obj]

                        if parsed is None and "," in value:
                            parts = [p.strip() for p in value.split(",") if p.strip() != ""]
                            parsed = [float(p) for p in parts]

                        params[key] = parsed if parsed is not None else value

                non_data_lines += 1

        data = np.loadtxt(file_path, skiprows=non_data_lines)
        return params, data


# class FFAG_Bfield_analytical:
#     def __init__(self, config_Bmap_path, max_order, flag3D=True, AddNumricalMap=True):
#         # 读取配置文件
#         with open(config_Bmap_path, 'r') as f:
#             config_data = json.load(f)
#
#         Bmap = config_data['Bmap']
#         rmin_max_step_m = Bmap["rmin_max_step_m"]
#         SpiralAngle_deg = Bmap["SpiralAngle_deg"]
#         self.Nsectors = Bmap["NSector"]
#         rmin, rmax, rstep = rmin_max_step_m[0], rmin_max_step_m[1], rmin_max_step_m[2]
#         self.Bmean = Bmap["Bmean"]
#         self.r_axis_Bmean = Bmap["r_axis"]
#         config_data_config = config_data['config']
#         folder = config_data_config['folder']
#         expand_order = config_data_config['expand_order']
#
#         self.convex_params = self.BmapParams_to_ConvexParams(Bmap)
#         self.convex_params = np.array(self.convex_params)
#         self.max_order=expand_order
#         self.save_folder = folder
#         self.config_path = config_Bmap_path
#
#         theta_step_rad = Bmap["theta_step_rad"]
#         n_fi_steps = int((np.pi * 2 / self.Nsectors - 0.0) / theta_step_rad) + 1  # 确保包括fi_max
#         fi_axis = np.linspace(0, np.pi * 2 / self.Nsectors, n_fi_steps)
#
#         # 计算r轴步数
#         n_r_steps = int((rmax - rmin) / rstep) + 1  # 确保包括rmax
#         r_axis = np.linspace(rmin, rmax, n_r_steps)
#
#         # 拟合多项式
#         s_values_rad = self.calculate_s(r_axis, 0.0, SpiralAngle_deg)
#         s_rad_polynomial = polyfit_manual(r_axis, s_values_rad, 6)
#         s_rad_polynomial_prime = polyder_manual(s_rad_polynomial)
#
#         B0, k_value = Bmap['coefficients_tupel']
#         if self.Bmean is None:
#             self.f1 = None
#         else:
#             self.f1 = interp1d(self.r_axis_Bmean, self.Bmean, kind='linear', fill_value="extrapolate")
#
#         self.r_axis = r_axis
#         self.fi_axis = fi_axis
#         self.B0 = B0
#         self.k_value = k_value
#         self.s_rad_polynomial = s_rad_polynomial
#         self.s_rad_polynomial_prime = s_rad_polynomial_prime
#         self.r_min = rmin
#         self.r_max = rmax
#         self.r_step = rstep
#         self.flag3D = flag3D
#         self.Binf = 6.0
#         self.BmapFoldName = os.path.dirname(config_Bmap_path)
#
#         self.fi_min, self.fi_max, self.fi_step = self.fi_axis[0], self.fi_axis[-1], self.fi_axis[1] - self.fi_axis[0]
#         self.r_size, self.fi_size = len(self.r_axis), len(self.fi_axis)
#
#         self.max_order = max_order
#         self.flag3D = flag3D
#
#         if AddNumricalMap:
#             # 初始化 Br, Bz, Bfi 的系数矩阵列表
#             self.Br_coeff_matrices = []
#             self.Bz_coeff_matrices = []
#             self.Bfi_coeff_matrices = []
#
#             # 检查 foldname 的类型并加载数据
#             foldname = os.path.normpath(self.BmapFoldName)
#             for order in range(0, max_order + 1):
#                 BrFileName = os.path.join(foldname, f"Br_order_{order}.npz")
#                 BzFileName = os.path.join(foldname, f"Bz_order_{order}.npz")
#                 BfiFileName = os.path.join(foldname, f"Bfi_order_{order}.npz")
#
#                 # 从 .npz 文件中加载各阶次系数矩阵
#                 self.Br_coeff_matrices.append(np.load(BrFileName)['Br_Taylor_coeff'])
#                 self.Bz_coeff_matrices.append(np.load(BzFileName)['Bz_Taylor_coeff'])
#                 self.Bfi_coeff_matrices.append(np.load(BfiFileName)['Bfi_Taylor_coeff'])
#
#         if MPI.COMM_WORLD.Get_rank() == 0:
#             print(f"loading analytical Bmap from {self.config_path},\nloading numerical Bmap with order-{2*self.max_order+1}- nonlinear terms from {self.BmapFoldName}")

class FFAG_Bfield_analytical:
    def __init__(self, config_Bmap_path, max_order, flag3D=True, AddNumricalMap=True):

        with open(config_Bmap_path, "r") as f:
            config_data = json.load(f)

        Bmap = config_data["Bmap"]
        rmin_max_step_m = Bmap["rmin_max_step_m"]
        SpiralAngle_deg = Bmap["SpiralAngle_deg"]
        self.Nsectors = Bmap["NSector"]

        rmin, rmax, rstep = rmin_max_step_m
        self.Bmean = Bmap["Bmean"]
        self.r_axis_Bmean = Bmap["r_axis"]

        cfg = config_data["config"]
        folder = cfg["folder"]

        # ---------------------------------------------------- 基本几何
        theta_step = Bmap["theta_step_rad"]
        n_fi_steps = int((2 * np.pi / self.Nsectors) / theta_step) + 1
        fi_axis = np.linspace(0, 2 * np.pi / self.Nsectors, n_fi_steps)
        n_r_steps = int((rmax - rmin) / rstep) + 1
        r_axis = np.linspace(rmin, rmax, n_r_steps)

        # ---------------------------------------------------- 多项式拟合 / f1
        s_val = self.calculate_s(r_axis, 0.0, SpiralAngle_deg)
        self.s_rad_polynomial = polyfit_manual(r_axis, s_val, 6)
        self.s_rad_polynomial_prime = polyder_manual(self.s_rad_polynomial)
        B0, k_value = Bmap['coefficients_tupel']
        self.B0 = B0
        self.k_value = k_value

        if self.Bmean is None:
            self.f1 = None
        else:
            self.f1 = interp1d(self.r_axis_Bmean, self.Bmean,
                               kind="linear", fill_value="extrapolate")

        # ---------------------------------------------------- 保存网格
        self.r_axis, self.fi_axis = r_axis, fi_axis
        self.r_min, self.r_max, self.r_step = rmin, rmax, rstep
        self.fi_min, self.fi_max = fi_axis[0], fi_axis[-1]
        self.fi_step = fi_axis[1] - fi_axis[0]
        self.r_size, self.fi_size = len(r_axis), len(fi_axis)

        self.convex_params = self.BmapParams_to_ConvexParams(Bmap)
        self.convex_params = np.array(self.convex_params)
        self.max_order = max_order
        self.flag3D = flag3D
        self.save_folder = folder
        self.config_path = config_Bmap_path
        self.BmapFoldName = os.path.dirname(config_Bmap_path)
        self.Binf = 6.0

        # ---------------------------------------------------- 读取各阶矩阵并 stack
        if AddNumricalMap:
            Br_list, Bz_list, Bfi_list = [], [], []
            fold = os.path.normpath(self.BmapFoldName)
            for n in range(0, max_order + 1):
                Br_file = os.path.join(fold, f"Br_order_{n}.npz")
                Bz_file = os.path.join(fold, f"Bz_order_{n}.npz")
                Bfi_file = os.path.join(fold, f"Bfi_order_{n}.npz")

                Br_list.append(np.ascontiguousarray(np.load(Br_file)["Br_Taylor_coeff"]))
                Bz_list.append(np.ascontiguousarray(np.load(Bz_file)["Bz_Taylor_coeff"]))
                Bfi_list.append(np.ascontiguousarray(np.load(Bfi_file)["Bfi_Taylor_coeff"]))

            # stack  (max_order+1, NX, NY) 并保持 C 连续
            self.Br_stack = np.ascontiguousarray(np.stack(Br_list, axis=0))
            self.Bz_stack = np.ascontiguousarray(np.stack(Bz_list, axis=0))
            self.Bfi_stack = np.ascontiguousarray(np.stack(Bfi_list, axis=0))

        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"[Bmap] load analytical map  : {config_Bmap_path}")
            print(f"[Bmap] load numerical map : {self.BmapFoldName}, "
                  f"orders 1 to order {2 * max_order + 1}")

    def AddHigherOrderBmaps(self,
                               r, fi, z,
                               Bz0, Br, Bfi, max_order):
        """
        一次性把 order=1…self.max_order 全部累加到 (Bz0, Br, Bfi)
        0 阶和线性项仍由 GetBfields() 按理论公式计算。
        """

        Bmap_add_all_orders(
            self.r_min, self.r_step, self.r_size,
            self.fi_min, self.fi_step, self.fi_size,
            self.Bz_stack,
            self.Bfi_stack,
            self.Br_stack,
            r, fi, z,
            max_order,
            Bz0, Bfi, Br)


    def BmapParams_to_ConvexParams(self, Bmap):
        intervals = Bmap['interval_1']
        polarities = Bmap['positive_or_negative_1']
        fringe_widths = Bmap['fringe_width_1']
        num_sectors = Bmap['NSector']

        convex_params = []
        total_angle = 360.0 / num_sectors

        # 计算每个扇区的角度范围
        start_angle = 0
        for interval, polarity, fringe_width in zip(intervals, polarities, fringe_widths):
            if polarity == 0:
                # 无磁场，跳过
                start_angle += interval * total_angle
                continue

            # 计算该扇区的起止角度
            end_angle = start_angle + interval * total_angle

            # 判断凸/凹形及强度
            if polarity > 0:
                shape = (abs(polarity), 1.0)  # 凸
            else:
                shape = (abs(polarity), -1,0)  # 凹

            # 例如上升/下降沿宽度是10%角宽
            edge_width = (end_angle - start_angle) * fringe_width

            # 将当前扇区的参数添加到 convex_params, 单位转换为rad
            convex_params.append((np.deg2rad(start_angle), np.deg2rad(end_angle), np.deg2rad(edge_width), np.float64(shape[0]), np.float64(shape[1])))

            # 更新下一个扇区的起始角度
            start_angle = end_angle

        return convex_params


    def calculate_s(self, r_axis, s0, alpha_deg):
        """给定半径范围r0-rmax, 步长rstep, r0处的起始方位角s0(rad), 螺旋角alpha_deg"""

        # # 生成 r 的数组
        # r0, rmax, rstep = r_axis[0], r_axis[-1], r_axis[1]-r_axis[0]

        # 初始化 s 的数组
        s_values = np.zeros_like(r_axis)
        s_values[0] = s0

        # 角度转为弧度
        alpha = np.deg2rad(alpha_deg)
        tan_alpha = np.tan(alpha)

        # 遍历 r_axis 计算 s(r) 单位rad
        for i in range(1, len(r_axis)):
            r = r_axis[i]
            dr = r - r_axis[i - 1]
            ds = tan_alpha / r * dr
            s_values[i] = s_values[i - 1] + ds

        return s_values

    # @profile
    def GetBz(self, r, fi, SearchMod = False):
        # Bz = B0(r) * F[θ-s(r)]

        spiral_azimuth = polyval_manual(self.s_rad_polynomial, r)

        B0rAmplitude = GetB0R_nth_derivative(self.B0, self.k_value, 0, r)

        # fi_shift = (fi - spiral_azimuth)%(2 * np.pi / self.Nsectors)
        # fi_shift = np.mod(fi - spiral_azimuth, 2 * np.pi/self.Nsectors)
        fi_shift = fast_mod_njit(fi - spiral_azimuth, 2 * np.pi / self.Nsectors)

        B0F, _ = GetB0F_nth_derivative(fi_shift, self.convex_params, 0)
        Bz = B0rAmplitude * B0F

        if SearchMod:
            mask_low = r < self.r_min
            mask_high = r > self.r_max
            # 区间外直接赋值常量
            Bz[mask_low] = -self.Binf
            Bz[mask_high] = self.Binf

        return Bz

    # @profile
    def GetBfi(self, r, fi, SearchMod = False):
        # Bfi = (1/r) * dBz/dfi * z
        # Bz = B0(r) * F[θ-s(r)]
        # dBz / dfi = B0(r) * F'[θ-s(r)]
        # dBz / dr = B0'(r) * F[θ-s(r)] - B0(r) * F'[θ-s(r)] * s'(r)

        spiral_azimuth = polyval_manual(self.s_rad_polynomial, r)
        # spiral_azimuth_prime = polyval_manual(self.s_rad_polynomial_prime, r)

        B0rAmplitude = GetB0R_nth_derivative(self.B0, self.k_value, 0, r)
        # B0rAmplitude_prime = GetB0R_nth_derivative(self.B0, self.k_value, 1, r)

        # fi_shift = np.mod(fi - spiral_azimuth, 2 * np.pi / self.Nsectors)
        fi_shift = fast_mod_njit(fi - spiral_azimuth, 2 * np.pi / self.Nsectors)
        B0F, B0F_prime = GetB0F_nth_derivative(fi_shift, self.convex_params, 1)
        Bfi = B0rAmplitude * B0F_prime / r  # F'[θ-s(r)] ---> B0F_prime

        if SearchMod:
            mask_low = r < self.r_min
            mask_high = r > self.r_max
            # 区间外直接赋值常量
            Bfi[mask_low] = 0.0
            Bfi[mask_high] = 0.0

        return Bfi

    def GetBr(self, r, fi, SearchMod = False):
        # Br = dBz/dr * z
        # Bz = B0(r) * F[θ-s(r)]
        # dBz / dfi = B0(r) * F'[θ-s(r)]
        # dBz / dr = B0'(r) * F[θ-s(r)] - B0(r) * F'[θ-s(r)] * s'(r)

        spiral_azimuth = polyval_manual(self.s_rad_polynomial, r)
        spiral_azimuth_prime = polyval_manual(self.s_rad_polynomial_prime, r)

        B0rAmplitude = GetB0R_nth_derivative(self.B0, self.k_value, 0, r)
        B0rAmplitude_prime = GetB0R_nth_derivative(self.B0, self.k_value, 1, r)

        # fi_shift = np.mod(fi - spiral_azimuth, 2 * np.pi / self.Nsectors)
        fi_shift = fast_mod_njit(fi - spiral_azimuth, 2 * np.pi / self.Nsectors)
        B0F, B0F_prime = GetB0F_nth_derivative(fi_shift, self.convex_params, 1)

        Br = B0rAmplitude_prime *  B0F - B0rAmplitude * B0F_prime * spiral_azimuth_prime

        if SearchMod:
            mask_low = r < self.r_min
            mask_high = r > self.r_max
            # 区间外直接赋值常量
            Br[mask_low] = 0.0
            Br[mask_high] = 0.0

        return Br

    # @profile
    def GetBfields(self, r, fi, z, Inj_flag, Survive_flag, Bz, Br, Bfi, SearchMod=False):
        """
        一次性计算并返回 Bz, Br, Bfi 分量
        参数：
            r (array): 半径数组
            fi (array): 方位角数组
        返回：
            tuple: (Bz, Br, Bfi)
        """
        GetBfields_Njit(r, fi, z, Inj_flag, Survive_flag,
                        Bz, Br, Bfi,
                        self.s_rad_polynomial, self.s_rad_polynomial_prime,
                        self.B0, self.k_value, self.Nsectors, self.convex_params)

        # # 计算公共变量
        # spiral_azimuth = polyval_manual(self.s_rad_polynomial, r)
        # spiral_azimuth_prime = polyval_manual(self.s_rad_polynomial_prime, r)
        # B0rAmplitude = GetB0R_nth_derivative(self.B0, self.k_value, 0, r)
        # B0rAmplitude_prime = GetB0R_nth_derivative(self.B0, self.k_value, 1, r)
        # # fi_shift = np.mod(fi - spiral_azimuth, 2 * np.pi / self.Nsectors)
        # fi_shift = fast_mod_njit(fi - spiral_azimuth, 2 * np.pi / self.Nsectors)
        # # 计算 B0F 和 B0F_prime
        # B0F, B0F_prime = GetB0F_nth_derivative(fi_shift, self.convex_params, 1)
        #
        # # 计算 Bz, Br, Bfi
        # Bz = B0rAmplitude * B0F
        # Bfi = B0rAmplitude * B0F_prime / r
        # Br = B0rAmplitude_prime * B0F - B0rAmplitude * B0F_prime * spiral_azimuth_prime

        # return Bz, Br, Bfi


    def UpdataConfig(self, ):
        """
        生成中平面 Bz, Br, Bfi (0阶)，保存为 .npz，并绘图
        config_Bmap.json
        """

        # 读取配置文件

        os.makedirs(self.save_folder, exist_ok=True)

        r_mesh, fi_mesh = np.meshgrid(self.r_axis, self.fi_axis, indexing='ij')
        shape = r_mesh.shape

        Bz_mat = np.zeros(shape)
        Bmean = np.zeros_like(self.r_axis)

        for i, r in enumerate(self.r_axis):
            fi_array = self.fi_axis
            r_array = np.full_like(fi_array, r)

            Bz_mat[i, :] = self.GetBz(r_array, fi_array)
            Bmean[i] = np.mean(Bz_mat[i, :])

        self.Bmean = Bmean

        #更新json文件中的Bmean,保存到save_folder
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)

        config_data['Bmap']['Bmean'] = self.Bmean.tolist()  # 注意转为可序列化 list
        config_data['Bmap']['r_axis'] = self.r_axis.tolist()  # 注意转为可序列化 list

        # 保存新的 JSON 到 save_folder/config_Bmap.json
        config_json_name = 'config_Bmap.json'
        # with open(os.path.join(self.save_folder, self.config_path), 'w') as f:
        #     json.dump(config_data, f, indent=4)
        with open(os.path.join(self.save_folder, config_json_name), 'w') as f:
            json.dump(config_data, f, indent=4)

        print(f"[Bmap] Bmean 已保存到: {os.path.join(self.save_folder, config_json_name)}")



class FFAG_BField_Error:
    """
    柱坐标磁场谐波 Bz(r,phi):

        Bz(r, phi) = sum_k amp_k * cos(m_k * phi + phase_k)

    每个谐波只在自己的 [rmin_k, rmax_k] 半径范围内有效。

    初始化参数 harmonics_cfg 是一个列表, 每个元素为一个 dict:
        {
            "m": 1,
            "amp_Gs": 10.0,
            "phase_deg": 30.0,
            "rmin": 10.2,
            "rmax": 10.4,
        }
    """

    def __init__(self, harmonics_cfg, EnableFlag):
        # 保存原始配置
        self.harmonics_cfg = harmonics_cfg
        self.enable_flag = EnableFlag

        # 预先转成几条 1D 数组, 方便 Python 版和 Numba 版共用
        Nharm = len(harmonics_cfg)
        self.Nharm = Nharm

        self.m_arr = np.empty(Nharm, dtype=np.int64)
        self.amp_arr = np.empty(Nharm, dtype=np.float64)
        self.phase_arr = np.empty(Nharm, dtype=np.float64)
        self.rmin_arr = np.empty(Nharm, dtype=np.float64)
        self.rmax_arr = np.empty(Nharm, dtype=np.float64)

        for idx, h in enumerate(harmonics_cfg):
            m = int(h["m"])
            amp_Gs = float(h["amp_Gs"])
            amp_T = amp_Gs * 1e-4            # Gs -> Tesla
            phase_deg = float(h["phase_deg"])
            phase_rad = np.deg2rad(phase_deg)
            rmin = float(h["rmin"])
            rmax = float(h["rmax"])

            self.m_arr[idx] = m
            self.amp_arr[idx] = amp_T
            self.phase_arr[idx] = phase_rad
            self.rmin_arr[idx] = rmin
            self.rmax_arr[idx] = rmax

    # --------------------------------------------------------
    # Python 版 Bz: 支持标量/数组 + 广播, 逻辑简单清晰
    # --------------------------------------------------------
    def GetBzHarmonics(self, r, phi):
        """
        计算 Bz(r,phi).

        r, phi 可以是标量或 numpy 数组 (任意形状)，会自动广播。
        角度 phi 用弧度制.
        """
        r_arr = np.asarray(r, dtype=float)
        phi_arr = np.asarray(phi, dtype=float)

        B_harmonic_map = np.zeros_like(r_arr, dtype=float)

        for k in range(self.Nharm):
            m = self.m_arr[k]
            amp_T = self.amp_arr[k]
            phase_rad = self.phase_arr[k]
            rmin = self.rmin_arr[k]
            rmax = self.rmax_arr[k]

            inside = (r_arr >= rmin) & (r_arr <= rmax)
            if np.any(inside):
                B_harmonic_map[inside] += amp_T * np.cos(m * phi_arr[inside] + phase_rad)

        return B_harmonic_map

    # --------------------------------------------------------
    # Numba 版: 一堆粒子 (1D 数组), 预分配输出
    # --------------------------------------------------------
    def AddBHarmonics_to_Bmap_njit(self, r_arr, phi_arr, z_arr,
                                   Bz_out, Br_out, Bfi_out):
        """
        一次性叠加磁场谐波到 Bz / Br / Bfi

        r_arr, phi_arr, z_arr : 1D 粒子数组
        Bz_out, Br_out, Bfi_out : 外部预分配，直接累加
        """
        if not self.enable_flag:
            return

        bharmonics_kernel_1d(
            r_arr, phi_arr, z_arr,
            self.m_arr, self.amp_arr, self.phase_arr,
            self.rmin_arr, self.rmax_arr,
            Bz_out, Br_out, Bfi_out
        )


class FFAG_Quadrupole():

    def __init__(self, config_path):
        """
        4极铁加在直线节上, 沿着理想粒子速度方向放置, 对闭轨无影响。将r, fi, z转换为x, y, z,
        Bx=Gy,By=Gx,Bz=0

        从投影方向看，边界为矩形。根据闭轨信息，计算每个4极铁的矩形顶点信息，作为类属性。

        Args:
            config_path:

        """
        pass


    def AddQuadrupole(self,):
        # 根据闭轨信息，和config文件，计算每个4极铁的几何边界(四顶点)
        pass


    def GetBField(self, r, fi, z, Inj_flag, Survive_flag):
        # 根据粒子坐标，判断是否在4极铁几何范围内，计算其坐标对应的自然坐标磁场分量，然后转换为柱坐标或笛卡尔坐标分量
        Br = 0
        Bfi = 0
        Bz = 0
        return Br, Bfi, Bz

# @profile
@nb.njit()
def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

# @nb.njit(parallel=True)
# def sigmoid(x, x0, k):
#     y_malloc = x * 0.0
#     for i in nb.prange(x.size):
#         y_malloc[i] = 1.0 / (1.0 + np.exp(-k * (x[i] - x0)))
#     return y_malloc

# @nb.njit(parallel=True)
# def sigmoid(x, x0, k, y_malloc):
#     for i in nb.prange(x.size):
#         y_malloc[i] = 1.0 / (1.0 + np.exp(-k * (x[i] - x0)))
#     return y_malloc


@nb.njit
def sigmoid_nth_derivative(x0, k, order, x_values):
    """
    Calculate the higher order derivative polynomial coefficients for the new sigmoid.

    Parameters:
    x0 (float): The center parameter of the sigmoid function.
    k (float): The steepness parameter of the sigmoid function.
    order (int): The order of the derivative to calculate.
    x_values (array): x_axis values.

    Returns:
    list, array: Polynomial coefficients after the specified derivative and the evaluated polynomial derivative.
    """
    if order == 0:
        return None, sigmoid(x_values, x0, k)

    sig_x = sigmoid(x_values, x0, k)

    coeffs = np.array([-k, k, 0], dtype=np.float64)  # Coefficients for the first derivative polynomial (σ_prime(x) = kσ(x)(1-σ(x)))
    current_coeffs = coeffs

    for i in range(order - 1):  # Since we start with the first derivative
        # Calculate the first derivative of the current polynomial
        derivative_coeff = polyder_manual(current_coeffs)
        # Multiply the derivative by the first derivative polynomial using convolution
        current_coeffs = np.convolve(derivative_coeff, coeffs)
        current_coeffs = current_coeffs.astype(np.float64)

    # Compute the polynomial-based nth order derivative of the sigmoid function
    poly_derivative_y = polyval_manual(current_coeffs, sig_x)

    return current_coeffs, poly_derivative_y

@nb.njit
# @profile
def GetB0R_nth_derivative(B0, k_value, n, x_vals):
    """
    计算幂函数或幂级数函数的 n 阶导数。

    Parameters:
        B0 (float or list/array):
            如果是单个系数 (float)，表示幂函数的系数 a。
        k_value (float):
            - 如果 a_or_coefficients 是数值，则表示幂函数的幂次 k。
        n (int): 要求的导数阶数。
        x_vals (float or array): 自变量 x 的值，可以是单值或数组。

    Returns:
        ndarray: n 阶导数在各 x 值处的结果, B0 * x**k_value。
    """
    if n == 0:
        coefficient = B0
        exponent = k_value
    else:
        k_vals = np.arange(k_value, k_value - n, -1)
        k_vals = np.append(k_vals, 1)
        coefficient = B0 * np.prod(k_vals)
        exponent = k_value - n
    return coefficient * np.power(x_vals, exponent)
    # k_vals = np.arange(k_value, k_value - n, -1)
    # k_vals = np.append(k_vals, 1)
    # coefficient = B0 * np.prod(k_vals)
    # exponent = k_value - n
    # return coefficient * x_vals ** exponent


# @profile
# @nb.njit
def GetB0F_nth_derivative(x, convex_params, n):
    # y_total = np.zeros_like(x)
    y_total = x*0.0
    # 当 n=0 时，返回原函数的值两次
    if n == 0:
        for params in convex_params:
            x0, x1, W0, y01, y02 = params
            y_total += shape_function(x, x0, x1, W0, y01, y02)
        return y_total, y_total  # 返回原函数和原函数

    # 计算高阶导数的情况
    y_total_derivatives = np.zeros_like(x)
    for params in convex_params:
        x0, x1, W0, y01, y02 = params
        y_total += shape_function(x, x0, x1, W0, y01, y02)
        y_total_derivatives += shape_function_nth_derivative(x, x0, x1, W0, n, y01, y02)

    return y_total, y_total_derivatives


# 定义凸字形或凹字形函数

# @profile
@nb.njit()
def shape_function(x, x0, x1, W0, y01, y02):

    y_convex = y02 * np.abs(y01)
    y_flat = 0

    k = 16 / W0  # 控制陡峭程度
    # y1_malloc = x * 0.0
    # y2_malloc = x * 0.0

    y1 = sigmoid(x, x0, k)
    y2 = sigmoid(x, x1, -k)
    return y_flat + (y_convex - y_flat) * y1 * y2

# @nb.njit(fastmath=True)
# def shape_function(x, x0, x1, W0, y01, y02):
#     y_convex = y02 * np.abs(y01)
#     k = 16.0 / W0
#
#     z1 = -k * (x - x0)
#     z2 = k * (x - x1)
#
#     s1 = 1.0 / (1.0 + np.exp(z1))
#     s2 = 1.0 / (1.0 + np.exp(z2))
#
#     return y_convex * s1 * s2



@nb.njit
def polyder_manual(coeffs):
    """
    Manually calculate the derivative of a polynomial given its coefficients.

    Parameters:
    coeffs (array): Coefficients of the polynomial, ordered from highest degree to lowest.

    Returns:
    array: Coefficients of the derived polynomial.
    """
    n = len(coeffs) - 1  # Highest power of the polynomial
    derived_coeffs = np.empty(n, dtype=coeffs.dtype)

    for i in range(n):
        derived_coeffs[i] = coeffs[i] * (n - i)

    return derived_coeffs


@nb.njit
def polyval_manual(coeffs, x):
    """
    Manually evaluate a polynomial at a given value x.

    Parameters:
    coeffs (array): Coefficients of the polynomial, ordered from highest degree to lowest.
    x (float or array): The value(s) at which to evaluate the polynomial.

    Returns:
    float or array: Evaluated polynomial at x.
    """
    # result = np.zeros_like(x)
    result = x * 0.0
    power = len(coeffs) - 1

    for coeff in coeffs:
        result = result * x + coeff
        power -= 1

    return result


def polyfit_manual(x, fx, order):
    """
    使用 numpy.polyfit 拟合数据，返回从高次到低次排列的多项式系数。

    Parameters:
    x (array): 自变量数据。
    fx (array): 因变量数据。
    order (int): 拟合多项式的阶数。

    Returns:
    array: 高次到低次排列的多项式系数。
    """
    coeffs = np.polyfit(x, fx, order)  # 默认就是高次到低次
    return coeffs

# @profile
# @nb.njit
def shape_function_nth_derivative(x, x0, x1, W0, n, y01, y02):
    """
    生成凸字形或凹字形函数的高阶导数。

    参数：
    x (array): 输入值的数组。
    x0 (float): 函数形状变化的起始点。
    x1 (float): 函数形状变化的终止点。
    W0 (float): 控制函数形状变化区域的宽度。
    n (int): 导数的阶数。
    y01 (tuple): 三元组，表示函数在不同区域的值和形状类型。默认值是 (0, 0, 1)。
        - y01[0], y01[1]: 函数在变化区域两端的值, 不区分顺序。
        - y01[2]: 控制函数形状类型的方向，正值表示凸字形，负值表示凹字形。

    返回值：
    array: 函数在输入值 x 处的输出值的高阶导数。
    """
    direction = y02
    y_convex = direction * np.abs(y01)
    y_flat = 0

    k = 16 / W0  # 控制陡峭程度

    result = np.zeros_like(x)  # 初始化 result 为与 x 形状相同的数组
    for i in range(0, n+1):
        binomial_coeff = binomial_Manually(n, i)  # 计算二项式系数
        _, deriv_y1 = sigmoid_nth_derivative(x0, k, i, x)
        _, deriv_y2 = sigmoid_nth_derivative(x1, -k, n - i, x)
        result += binomial_coeff * deriv_y1 * deriv_y2
    return (y_convex - y_flat) * result

# @nb.njit
def factorial_Manually(n):
    """
    Manually calculate the factorial of n.
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# @nb.njit
def binomial_Manually(n, k):
    """
    Calculate the binomial coefficient "n choose k".

    Parameters:
    n (int): Total number of items.
    k (int): Number of chosen items.

    Returns:
    int: Binomial coefficient.
    """
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    return factorial_Manually(n) // (factorial_Manually(k) * factorial_Manually(n - k))


@nb.njit(parallel=True)
# @profile
def GetBfields_Njit(r, fi, z, Inj_flag, Survive_flag,
                    Bz, Br, Bfi,
                    s_rad_poly, s_rad_poly_prime,
                    B0, k_value,
                    Nsectors,
                    convex_params_2d):
    """
    r, fi               : (N,)  ndarray
    s_rad_poly[*]       : 1-D  ndarray  (最高次 最低次)
    s_rad_poly_prime[*] : 1-D  ndarray  (P' 系数，同上)
    convex_params_2d    : (M,5) ndarray [[x0,x1,W0,y01,y02], ...]
    返回:
        Bz, Br, Bfi ── 每个都是 (N,) ndarray
    """

    N      = r.size
    # Bz     = np.zeros_like(r)
    # Br     = np.zeros_like(r)
    # Bfi    = np.zeros_like(r)

    sector_period = 2.0 * np.pi / Nsectors
    M             = convex_params_2d.shape[0]

    for i in nb.prange(N):                       # 并行遍历每个粒子
        # 判断是否已注入 + 存活
        if Inj_flag[i] != 1 or Survive_flag[i] != 1:
            continue  # 跳过

        ri  = r[i]
        fii = fi[i]
        zi = z[i]

        # ---------- (1) P(ri) 及其导数 ----------
        val  = 0.0
        for a in s_rad_poly:
            val = val * ri + a
        spiral_azimuth = val

        val  = 0.0
        for a in s_rad_poly_prime:
            val = val * ri + a
        spiral_azimuth_prime = val

        # ---------- (2) B0 · r^k 及其一阶导数 ----------
        if k_value == 0.0:
            B0r     = B0
            B0r_p   = 0.0
        else:
            B0r     = B0 * ri ** k_value
            B0r_p   = B0 * k_value * ri ** (k_value - 1)

        # ---------- (3) 周期性平移 ----------
        fi_shift = fii - spiral_azimuth
        fi_shift -= sector_period * np.floor(fi_shift / sector_period)

        # ---------- (4) 计算 B0F 与 B0F' ----------
        B0F       = 0.0
        B0F_prime = 0.0

        for j in range(M):
            x0  = convex_params_2d[j, 0]
            x1  = convex_params_2d[j, 1]
            W0  = convex_params_2d[j, 2]
            y01 = convex_params_2d[j, 3]
            y02 = convex_params_2d[j, 4]

            y_convex = y02 * np.abs(y01)       # 这里 y_flat = 0

            k_sig = 16.0 / W0                  # 斜率

            # --- 两个 sigmoids ---
            s1 = 1.0 / (1.0 + np.exp(-k_sig * (fi_shift - x0)))
            s2 = 1.0 / (1.0 + np.exp( k_sig * (fi_shift - x1)))  # 注意 -k
            # s1 = 0.0
            # s2 = 0.0

            # --- 原函数值 ---
            y_local = y_convex * s1 * s2
            B0F += y_local

            # --- 一阶导数:  y' = y_convex·(s1' s2 + s1 s2') ---
            s1_prime =  k_sig * s1 * (1.0 - s1)
            s2_prime = -k_sig * s2 * (1.0 - s2)
            y_local_p = y_convex * (s1_prime * s2 + s1 * s2_prime)
            B0F_prime += y_local_p

        # ---------- (5) 三个分量 ----------
        Bz[i]  = B0r * B0F
        Bfi[i] = B0r * B0F_prime / ri * zi
        Br[i]  = (B0r_p * B0F - B0r * B0F_prime * spiral_azimuth_prime) * zi

    return Bz, Br, Bfi

# @nb.njit(parallel=True, fastmath=True)
# # @profile
# def GetBfields_Njit(r, fi, z, Inj_flag, Survive_flag,
#                     Bz, Br, Bfi,
#                     s_rad_poly, s_rad_poly_prime,
#                     B0, k_value,
#                     Nsectors,
#                     convex_params_2d):
#     """
#     r, fi               : (N,)  ndarray
#     s_rad_poly[*]       : 1-D  ndarray  (最高次→最低次)
#     s_rad_poly_prime[*] : 1-D  ndarray  (P' 系数，同上)
#     convex_params_2d    : (M,5) ndarray [[x0,x1,W0,y01,y02], ...]
#     返回:
#         Bz, Br, Bfi ── 每个都是 (N,) ndarray
#     """
#
#     N = r.size
#     sector_period = 2.0 * np.pi / Nsectors
#     M = convex_params_2d.shape[0]
#
#     for i in nb.prange(N):  # 并行遍历每个粒子
#         # 判断是否已注入 + 存活
#         if Inj_flag[i] != 1 or Survive_flag[i] != 1:
#             continue
#
#         ri  = r[i]
#         fii = fi[i]
#         zi  = z[i]
#
#         # ---------- (1) P(ri) 及其导数 ----------
#         val = 0.0
#         for a in s_rad_poly:
#             val = val * ri + a
#         spiral_azimuth = val
#
#         val = 0.0
#         for a in s_rad_poly_prime:
#             val = val * ri + a
#         spiral_azimuth_prime = val
#
#         # ---------- (2) B0 · r^k 及其一阶导数 ----------
#         if k_value == 0.0:
#             B0r   = B0
#             B0r_p = 0.0
#         else:
#             B0r   = B0 * ri ** k_value
#             B0r_p = (k_value / ri) * B0r   # 复用 B0r，避免再次幂运算
#
#         # ---------- (3) 周期性平移 ----------
#         fi_shift = fii - spiral_azimuth
#         fi_shift -= sector_period * np.floor(fi_shift / sector_period)
#
#         # ---------- (4) 计算 B0F 与 B0F'（稳定 sigmoid 写法） ----------
#         B0F       = 0.0
#         B0F_prime = 0.0
#
#         for j in range(M):
#             x0  = convex_params_2d[j, 0]
#             x1  = convex_params_2d[j, 1]
#             W0  = convex_params_2d[j, 2]
#             y01 = convex_params_2d[j, 3]
#             y02 = convex_params_2d[j, 4]
#
#             y_convex = y02 * np.abs(y01)  # y_flat = 0
#             k_sig    = 16.0 / W0
#
#             # # --- 两个稳定 sigmoid：s1 = σ(k*(fi-x0))，s2 = σ(-k*(fi-x1))
#             # # 采用早饱和阈值 L=12，减少不必要的 exp 调用
#             # L = 12.0
#             #
#             # d0 = k_sig * (fi_shift - x0)
#             # if d0 >= L:
#             #     s1 = 1.0
#             # elif d0 <= -L:
#             #     s1 = 0.0
#             # elif d0 >= 0.0:
#             #     e0 = np.exp(-d0)
#             #     s1 = 1.0 / (1.0 + e0)
#             # else:
#             #     e0 = np.exp(d0)
#             #     s1 = e0 / (1.0 + e0)
#             #
#             # # 注意原式：s2 = 1/(1+exp(+k*(fi-x1))) = σ(-d1)
#             # d1m = -k_sig * (fi_shift - x1)  # 等价于 -(k*(fi-x1))
#             # if d1m >= L:
#             #     s2 = 1.0
#             # elif d1m <= -L:
#             #     s2 = 0.0
#             # elif d1m >= 0.0:
#             #     e1 = np.exp(-d1m)
#             #     s2 = 1.0 / (1.0 + e1)
#             # else:
#             #     e1 = np.exp(d1m)
#             #     s2 = e1 / (1.0 + e1)
#             s1=0;s2=0
#             # --- 原函数值 ---
#             y_local = y_convex * s1 * s2
#             B0F += y_local
#
#             # --- 一阶导数： s' = k*s*(1-s)；注意 s2' 的负号
#             s1_prime =  k_sig * s1 * (1.0 - s1)
#             s2_prime = -k_sig * s2 * (1.0 - s2)
#             y_local_p = y_convex * (s1_prime * s2 + s1 * s2_prime)
#             B0F_prime += y_local_p
#
#         # ---------- (5) 三个分量 ----------
#         Bz[i]  = B0r * B0F
#         Bfi[i] = B0r * B0F_prime / ri * zi
#         Br[i]  = (B0r_p * B0F - B0r * B0F_prime * spiral_azimuth_prime) * zi
#
#     return Bz, Br, Bfi

@nb.njit()
def bharmonics_kernel_1d(r_arr, phi_arr, z_arr,
                          m_arr, amp_arr, phase_arr,
                          rmin_arr, rmax_arr,
                          Bz_out, Br_out, Bfi_out):
    """
    同时计算：
      Bz_h  =  A cos(m phi + phase)
      Br_h  =  z * dB/dr        ≈ 0  (当前模型)
      Bfi_h = (z/r) * dB/dphi  = -(z/r) * m A sin(...)
    """
    N = r_arr.shape[0]
    Nh = m_arr.shape[0]

    for i in range(N):
        r = r_arr[i]
        phi = phi_arr[i]
        z = z_arr[i]

        if r <= 0.0:
            continue

        Bz = 0.0
        Br = 0.0
        Bfi = 0.0

        for k in range(Nh):
            if rmin_arr[k] <= r <= rmax_arr[k]:
                m = m_arr[k]
                A = amp_arr[k]
                ph = phase_arr[k]

                c = np.cos(m * phi + ph)
                # s = np.sin(m * phi + ph)

                # Bz (0 阶)
                Bz += A * c

                # Br (一阶) —— 当前 r 无关模型 → 0
                # Br += z * dB/dr  = 0

                # Bfi (一阶，关键)
                # Bfi += -(z / r) * m * A * s

        Bz_out[i] += Bz
        Br_out[i] += Br
        Bfi_out[i] += Bfi


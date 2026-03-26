import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from FFAG_MathTools import FFAG_interpolation, Lagrange_interp_2D_vect
from FFAG_ParasAndConversion import FFAG_GlobalParameters, FFAG_ConversionTools
from FFAG_track import FFAG_SearchSEO
from FFAG_Field import FFAG_BField_new
from concurrent.futures import ProcessPoolExecutor
from numpy.polynomial import Polynomial
from FFAG_HighOrder import calculate_s, Generate_F_givenR, Generate_Bmap_Bz, BmapParams_to_ConvexParams
# from test_convex_para import BmapParams_to_ConvexParams
from scipy.interpolate import interp1d


def polyfit_shifted(x, y, x0, degree):
    """
    y = a0 + a1*(x-x0)^1 + a2*(x-x0)^2+a3*(x-x0)^3 + ...
    """
    x_shift = x0
    x_shifted = x - x_shift

    # 拟合多项式
    coefficients = np.polyfit(x_shifted, y, degree)
    inverse_powers = np.array([1 / (x_shifted[-1] ** i) for i in range(degree, -1, -1)])

    return coefficients, x_shift, inverse_powers


def polyval_shifted(coefficients, x_shift, x_input):
    """
    y = a0 + a1*(x-x0)^1 + a2*(x-x0)^2+a3*(x-x0)^3 + ...
    """
    x_input_shifted = x_input - x_shift
    y_output = np.polyval(coefficients, x_input_shifted)

    return y_output


# 生成磁场数据
def GenerateF(config_data, SpiralAngle=0.0, HomotopyCoef=0.0, PlotFlag=False):
    # 提取Bmap数据
    # HomotopyCoef = 0 ---> intervals = original intervals
    # HomotopyCoef = 1 ---> intervals = flat intervals:(1, 1, 1, 1, 1)

    Bmap = config_data['Bmap']
    intervals = Bmap['interval']
    values = Bmap['positive_or_negative']
    n = Bmap['NSector']
    Period_width = 360.0 / n
    cumulative_widths = np.cumsum(intervals) * Period_width

    FlatVals = np.array([1, 1, 1, 1, 1])
    RealVals = np.array(values)
    HomoVals = (FlatVals - RealVals) * HomotopyCoef + RealVals

    # Generate fi values
    fi_start, fi_end, fi_step = 0.0, Period_width, 0.1
    fi_values = np.arange(fi_start, fi_end, fi_step)

    # 初始化 B_values 数组
    B_values = np.zeros_like(fi_values)

    # Generate B values based on fi values and coefficients
    for j, dfi in enumerate(fi_values):
        mod_dfi = dfi % Period_width
        for i, (width, value) in enumerate(zip(intervals, HomoVals)):
            if mod_dfi < cumulative_widths[i]:
                if value != 0:
                    B_values[j] = value
                else:
                    B_values[j] = value
                break

    # Plot the generated waveform
    if PlotFlag:
        plt.figure(figsize=(12, 6))
        plt.plot(fi_values, B_values, label='Magnetic Field')
        plt.xlabel('Fi (degrees)')
        plt.ylabel('Magnetic Field')
        plt.title('Generated Magnetic Field vs Fi')
        plt.legend()
        plt.grid(True)

    return fi_values, B_values, n


def GetRawISO(f0, GlobalParas, rmin=None, rmax=None):
    E0 = GlobalParas.E0
    c = GlobalParas.c
    q = GlobalParas.q
    rinf = c / (2 * np.pi * f0)
    B0 = 2 * np.pi * f0 * E0 / q / c ** 2
    if rmin is None:
        r0 = 0.0
        r99 = rinf * 0.99
    else:
        r0 = rmin
        r99 = rmax

    r_axis_analytical = np.linspace(r0, r99, 1000)

    Biso_analytical = B0 / np.sqrt(1 - (r_axis_analytical / rinf) ** 2)
    P_analytical = q * Biso_analytical * r_axis_analytical
    Ek_analytical_J = FFAG_ConversionTools().P2Ek(P_analytical)
    Ek_analytical_MeV = Ek_analytical_J / q / 1e6

    return r_axis_analytical, Biso_analytical, Ek_analytical_MeV, rinf


class FFAG_BField_new_Iso():
    def __init__(self, data, flag3D=True):

        # data = np.loadtxt(BmapFileName, skiprows=1)
        self.data_dBdr = None
        self.data_dBdf = None
        self.Cr = None
        self.Cf = None
        self.Cz = None
        self.Cz2 = None

        fi, r = data[0, 1:], data[1:, 0]
        self.r_axis, self.fi_axis, self.map_data = r, fi, data[1:, 1:]
        self.r_min, self.r_max, self.r_step = r[0], r[-1], r[1] - r[0]
        self.fi_min, self.fi_max, self.fi_step = fi[0], fi[-1], fi[1] - fi[0]
        self.r_size, self.fi_size = len(self.r_axis), len(self.fi_axis)

        self.Nsectors = None

        self.BMean = np.mean(self.map_data, 1)
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
        self.r_axis_ext, self.fi_axis_ext, self.map_data_ext = None, None, None
        self.r_min_ext, self.r_max_ext, self.r_step_ext = None, None, None
        self.fi_min_ext, self.fi_max_ext, self.fi_step_ext = None, None, None
        self.r_size_ext, self.fi_size_ext = None, None
        self.BMean_ext, self.f1_ext = None, None

    def ExtendField(self, ext_Rsize_min, ext_Rsize_max, Binf):
        # ext_Rsize_min, ext_Rsize_max = 100, 100

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
        pass

    def Interpolation2DMapExtForSEO(self, r, fi):
        FieldMapTemp = self.map_data_ext

        Value, OutRangeFlag = Lagrange_interp_2D_vect(
            self.r_axis_ext, self.fi_axis_ext, FieldMapTemp, r, fi)
        return Value, OutRangeFlag

    def Interpolation2DMap(self, r, fi, flag):
        if flag == 0:
            FieldMapTemp = self.map_data
        elif flag == 1:
            FieldMapTemp = self.data_dBdf  # dBdfi
        elif flag == 2:
            FieldMapTemp = self.data_dBdr  # dBdr
        elif flag == 3:
            FieldMapTemp = self.Cr  # dB2dfi2
        elif flag == 4:
            FieldMapTemp = self.Cf  # dB2dr2
        elif flag == 5:
            FieldMapTemp = self.Cz  # dB2dr2
        else:
            FieldMapTemp = self.Cz2  # dSumdr

        Value, OutRangeFlag = Lagrange_interp_2D_vect(self.r_axis, self.fi_axis, FieldMapTemp, r, fi)

        return Value, OutRangeFlag


def GetOrbitFreqOld(coefficients, other_paras):
    # 计算回旋频率
    BmapInfo = other_paras["BmapInfo"]
    EkRange = other_paras["EkRange"]
    fi_values = other_paras["fi_axis"]
    r_axis_analytical = other_paras["r_axis"]
    objective_freq = other_paras["objective_freq_MHz"]
    s_values_FitCoef = other_paras["s_values"]
    ConvexParams = other_paras["ConvexParams"]

    BzR_coef_k = (coefficients, r_axis_analytical[0])
    SaveData, Fmean, _ = Generate_Bmap_Bz(r_axis_analytical, fi_values, ConvexParams, s_values_FitCoef, BzR_coef_k)

    BMapData = FFAG_BField_new(SaveData, 0, flag3D=False)
    BMapData.Nsectors = BmapInfo["NSector"]
    BMapData.ExtendField(100, 100, 8)

    # generate global parameters
    GlobalParas = FFAG_GlobalParameters()
    GlobalParas.AddBMap(BMapData)

    results = []
    SEO_obj = FFAG_SearchSEO(GlobalParas)
    for Ek_value in EkRange:
        r0, pr0, r0_analytical = SEO_obj.SearchSEOUsingInitialEkVect(Ek_value, verbose=True)
        t_points, r_points = SEO_obj.TrackAParticle(r0, pr0, Ek_value, np.pi * 2 / BMapData.Nsectors, 0.0010)
        r_of_fi, pr_of_fi = r_points[:, 0], r_points[:, 1]

        orbitalfreq,_ = SEO_obj.GetFreq(t_points, r_of_fi, pr_of_fi, Ek_value)
        results.append(orbitalfreq)
    print("orbital freq=", results)
    error_value = np.max(np.abs(np.array(results) - objective_freq * 1.0e6))
    return error_value


#
#
# def nelder_mead(obj_func, initial_simplex, ObjFuncPara, max_iter=150, alpha=1.0, gamma=2.0, rho=0.5,
#                 eps=50):
#     n = initial_simplex.shape[1]  # dimension
#     func_values = np.zeros(n + 1)  # target function value
#
#     # Calculate initial function values
#     for i in range(n + 1):
#         func_values[i] = obj_func(initial_simplex[i], ObjFuncPara)
#
#     # Write output info
#     with open("nelder_mead.txt", 'w') as fid:
#         fid.write("begins\n")
#
#     for j in range(max_iter):
#         indices = np.argsort(func_values)  # sort
#         best, second_best, worst = indices[0], indices[1], indices[-1]
#
#         # Print the best and worst function values for debugging purposes
#         print("Iteration:", j, "Best value:", func_values[best], "Worst value:", func_values[worst])
#         print("best vertex = ", initial_simplex[best])
#
#         with open("nelder_mead.txt", 'a') as fid:
#             fid.write(f"Iteration: {j}, Best value: {func_values[best]}, Worst value: {func_values[worst]}\n")
#             # fid.write(f"Best vertex: {initial_simplex[best]}\n")
#             fid.write("Best vertex: " + " ".join(f"{coord:.10f}" for coord in initial_simplex[best]) + "\n")
#
#         if func_values[best] < eps:
#             break
#
#         best_x, worst_x = initial_simplex[best], initial_simplex[worst]
#         center = np.mean(initial_simplex[indices[:-1]], axis=0)
#
#         # Reflect
#         reflect_x = center + alpha * (center - worst_x)
#         reflect_FuncValue = obj_func(reflect_x, ObjFuncPara)
#
#         if reflect_FuncValue < func_values[second_best] and reflect_FuncValue >= func_values[best]:
#             initial_simplex[worst] = reflect_x
#             func_values[worst] = reflect_FuncValue
#         elif reflect_FuncValue < func_values[best]:
#             # Expand
#             expand_x = center + gamma * (reflect_x - center)
#             expand_FuncValue = obj_func(expand_x, ObjFuncPara)
#             if expand_FuncValue < func_values[best]:
#                 initial_simplex[worst] = expand_x
#                 func_values[worst] = expand_FuncValue
#             else:
#                 initial_simplex[worst] = reflect_x
#                 func_values[worst] = reflect_FuncValue
#         else:
#             # Contract
#             contract_x = center + rho * (worst_x - center)
#             contract_FuncValue = obj_func(contract_x, ObjFuncPara)
#             if contract_FuncValue < func_values[worst]:
#                 initial_simplex[worst] = contract_x
#                 func_values[worst] = contract_FuncValue
#             else:
#                 # Shrink
#                 for i in indices[1:]:
#                     initial_simplex[i] = best_x + 0.5 * (initial_simplex[i] - best_x)
#                     func_values[i] = obj_func(initial_simplex[i], ObjFuncPara)
#
#         # Check for convergence
#
#         # if np.max(np.abs(func_values - np.mean(func_values))) < eps:
#         #     break
#         # if np.max(np.abs(func_values - np.mean(func_values))) < eps:
#         #     break
#
#     index_min = np.argmin(func_values)
#
#     return initial_simplex[index_min]


def search_iso(global_paras, Ek):
    """
    全局定义的函数，用于在不同能量下计算回旋频率。
    """
    ISO_obj = FFAG_SearchSEO(global_paras)
    return ISO_obj.task(Ek)


def search_iso_global(args):
    """
    全局函数，用于调用 search_iso，避免 lambda 引发的序列化问题。
    """
    global_paras, Ek = args
    return search_iso(global_paras, Ek)


def GetOrbitFreq(coefficients, other_paras, executor):
    # 计算回旋频率
    BmapInfo = other_paras["BmapInfo"]
    EkRange = other_paras["EkRange"]
    fi_values = other_paras["fi_axis"]
    r_axis_analytical = other_paras["r_axis"]
    objective_freq = other_paras["objective_freq_MHz"]
    s_values_FitCoef = other_paras["s_values"]
    ConvexParams = other_paras["ConvexParams"]

    BzR_coef_k = (coefficients, r_axis_analytical[0])
    SaveData, Fmean, _ = Generate_Bmap_Bz(r_axis_analytical, fi_values, ConvexParams, s_values_FitCoef, BzR_coef_k)

    BMapData = FFAG_BField_new(SaveData, 0, flag3D=False)
    BMapData.Nsectors = BmapInfo["NSector"]
    BMapData.ExtendField(100, 100, 6)

    # generate global parameters
    GlobalParas = FFAG_GlobalParameters()
    GlobalParas.AddBMap(BMapData)

    # 使用全局的 search_iso_global 函数进行并行计算
    results = list(executor.map(search_iso_global, [(GlobalParas, Ek) for Ek in EkRange]))

    orbital_freq = np.array(results)
    print("orbital freq=", orbital_freq)
    error_value = np.max(np.abs(orbital_freq - objective_freq * 1.0e6))
    return error_value


def nelder_mead(obj_func, initial_simplex, ObjFuncPara, max_iter=150, alpha=1.0, gamma=4.0, rho=0.5, eps=50):
    # 创建进程池
    with ProcessPoolExecutor() as executor:
        # 包装目标函数，使其接受进程池
        def wrapped_obj_func(coefficients, paras):
            return obj_func(coefficients, paras, executor)

        n = initial_simplex.shape[1]  # dimension
        func_values = np.zeros(n + 1)  # target function value

        # Calculate initial function values
        for i in range(n + 1):
            func_values[i] = wrapped_obj_func(initial_simplex[i], ObjFuncPara)

        # Write output info
        with open("nelder_mead.txt", 'w') as fid:
            fid.write("begins\n")

        for j in range(max_iter):
            indices = np.argsort(func_values)  # sort
            best, second_best, worst = indices[0], indices[1], indices[-1]

            print(f"Iteration: {j}, Best value: {func_values[best]}, Worst value: {func_values[worst]}")
            with open("nelder_mead.txt", 'a') as fid:
                fid.write(f"Iteration: {j}, Best value: {func_values[best]}, Worst value: {func_values[worst]}\n")
                fid.write("Best vertex: " + " ".join(f"{coord:.10f}" for coord in initial_simplex[best]) + "\n")

            if func_values[best] < eps:
                break

            best_x, worst_x = initial_simplex[best], initial_simplex[worst]
            center = np.mean(initial_simplex[indices[:-1]], axis=0)

            # Reflect
            reflect_x = center + alpha * (center - worst_x)
            reflect_FuncValue = wrapped_obj_func(reflect_x, ObjFuncPara)

            if reflect_FuncValue < func_values[second_best] and reflect_FuncValue >= func_values[best]:
                initial_simplex[worst] = reflect_x
                func_values[worst] = reflect_FuncValue
            elif reflect_FuncValue < func_values[best]:
                # Expand
                expand_x = center + gamma * (reflect_x - center)
                expand_FuncValue = wrapped_obj_func(expand_x, ObjFuncPara)
                if expand_FuncValue < func_values[best]:
                    initial_simplex[worst] = expand_x
                    func_values[worst] = expand_FuncValue
                else:
                    initial_simplex[worst] = reflect_x
                    func_values[worst] = reflect_FuncValue
            else:
                # Contract
                contract_x = center + rho * (worst_x - center)
                contract_FuncValue = wrapped_obj_func(contract_x, ObjFuncPara)
                if contract_FuncValue < func_values[worst]:
                    initial_simplex[worst] = contract_x
                    func_values[worst] = contract_FuncValue
                else:
                    # Shrink
                    for i in indices[1:]:
                        initial_simplex[i] = best_x + 0.5 * (initial_simplex[i] - best_x)
                        func_values[i] = wrapped_obj_func(initial_simplex[i], ObjFuncPara)

        index_min = np.argmin(func_values)

        return initial_simplex[index_min]


def generate_initial_simplex(coefficients, inverse_powers):
    n = np.size(coefficients, 0)
    initial_simplex = np.zeros((n + 1, n))

    initial_simplex[0, :] = coefficients

    for i in range(n):
        updated_coefficients = coefficients.copy()
        randnum = np.random.uniform(-1, 1)
        if i < len(inverse_powers):
            updated_coefficients[i] += inverse_powers[i] * 0.030 * randnum
        initial_simplex[i + 1, :] = updated_coefficients

    return initial_simplex


def Generate_Iso_maps(r_axis, fi_axis_rad, convex_paras, s_values_FitCoef_rad, Bz_FitCoef):
    """
    根据给定的半径和角度轴生成 B_z 分量的 Flutter 矩阵。
    这个矩阵使用凸形或凹形函数的参数以及拟合的 Bz 中心值来计算。

    参数:
    r_axis: 半径轴上的点数组
    fi_axis_rad: 角度轴上的点数组（以弧度为单位）
    convex_paras: 凸形/凹形函数的参数，用于生成 Flutter 函数
    s_values_FitCoef_rad: s(r) 的拟合多项式系数，用于调整角度函数
    Bz_FitCoef: Bz 中心值的拟合多项式系数，用于在半径方向上的插值

    返回值:
    Flutter_matrix: Bz 分量的 Flutter 矩阵，大小为 (len(r_axis), len(fi_axis_rad))
    每个元素表示在相应半径和角度下的 Bz 值，通过 Flutter 函数和中心磁场值的乘积计算得出。
    Flutter_Fmean: 半径方向上调变场的平均值
    """
    # 初始化 Flutter 矩阵，维度为 (半径数量, 角度数量)
    Flutter_matrix = np.zeros((len(r_axis), len(fi_axis_rad)), dtype=np.float64)
    Flutter_Fmean = np.zeros(len(r_axis), dtype=np.float64)

    # 遍历每个半径值，计算对应的 Flutter 矩阵
    for r_idx, r_vals in enumerate(r_axis):
        # 生成与给定 r 相关的 Flutter 函数
        F_givenR, _ = Generate_F_givenR(fi_axis_rad, convex_paras, r_vals, s_values_FitCoef_rad, order=1)
        # Flutter 的平均磁场
        Flutter_Fmean[r_idx] = np.mean(F_givenR)
        # 根据半径值计算 Bz 的中心磁场
        Bcentral_givenR = polyval_shifted(Bz_FitCoef, np.min(r_axis), r_vals) / Flutter_Fmean[r_idx]
        # 将 Flutter 函数与中心磁场值相乘，得到 Bz 的 Flutter 矩阵
        Flutter_matrix[r_idx, :] = F_givenR * Bcentral_givenR

    # 保存数据，将半径轴添加为第一列，角度轴添加为第一行
    # 首先，将 r 轴和 Flutter 矩阵整合为一个矩阵
    SaveData = np.hstack((np.reshape(r_axis, (-1, 1)), Flutter_matrix))

    # 然后，将角度轴添加为第一行
    SaveData = np.vstack((np.hstack((np.zeros((1, 1)), np.reshape(fi_axis_rad, (1, -1)))), SaveData))

    return SaveData, Flutter_Fmean


def CheckISOBmapConfig(config_data):
    # 从配置文件中获取能量范围
    energy_inj = config_data['machine']['energy_inj']
    energy_ext = config_data['machine']['energy_ext']

    # 读取 Bmap 信息
    BmapInfo = config_data['Bmap']
    BzInfo = config_data['BzInfo']
    NSector = BmapInfo['NSector']
    theta_step_rad = BmapInfo['theta_step_rad']
    OrbitalFreq_MHz = BmapInfo['orbital_freq_MHz']
    k_value = BmapInfo['k_value']
    B0_T = BmapInfo['B0_T']
    SpiralAngle_deg = BmapInfo['SpiralAngle_deg']
    rmin_max_step_m = BmapInfo['rmin_max_step_m']
    BmapType = BmapInfo['Type']
    ConvexParams = BmapParams_to_ConvexParams(BmapInfo)

    # 设置 fi 和 r 轴
    fi_axis = np.linspace(0, np.pi * 2 / NSector, int(round(np.pi * 2 / NSector / theta_step_rad)) + 1)
    r_min, r_max, r_step = rmin_max_step_m[0], rmin_max_step_m[1], rmin_max_step_m[2]
    r_axis = np.linspace(r_min, r_max, int(round((r_max - r_min) / r_step)) + 1)

    # 计算螺旋线方位角s(r)并进行多项式拟合
    s_values = calculate_s(r_axis, 0.0, SpiralAngle_deg)
    s_values_FitCoef = Polynomial.fit(r_axis, s_values, 6)

    # 初始化全局参数
    GlobalParas = FFAG_GlobalParameters()

    # 注意coefficients必须为中心线磁场的展开系数，输入的是平均磁场，需要转换
    r_axis_analytical, Biso_analytical, Ek_analytical_MeV, rinf = GetRawISO(
        OrbitalFreq_MHz * 1e6, GlobalParas, rmin=r_min, rmax=r_max)
    degree = 6  # 多项式的阶数
    coefficients, _, inverse_powers = polyfit_shifted(
        r_axis_analytical, Biso_analytical, np.min(r_axis_analytical), degree)

    if BmapType == 'ISOCHRONOUS':
        print("Bmap type = ", BmapType)
        coefficients_tupel = (tuple(coefficients), r_min)
    elif BmapType == 'SCALE':
        print("Bmap type = ", BmapType)
        coefficients_tupel = (k_value, B0_T)
    else:
        raise ValueError("Invalid BmapType: expected 'ISOCHRONOUS' or 'SCALE'")

    # 生成等效 B-map 数据
    SaveData, Fmean, Bcentral = Generate_Bmap_Bz(r_axis, fi_axis, ConvexParams, s_values_FitCoef, coefficients_tupel)
    BMapData = FFAG_BField_new(SaveData, 1, flag3D=False)

    # 更新coefficients为Bcentral
    coefficients_Bcentral, _, inverse_powers = polyfit_shifted(
        r_axis, Bcentral, np.min(r_axis), degree)
    if BmapType == 'ISOCHRONOUS':
        coefficients_tupel = (tuple(coefficients_Bcentral), r_min)
        BzInfo['RawIso'] = tuple(coefficients_Bcentral)

    # 检查 energy_inj, energy_ext 是否在 BMapData 的 Ekmin90 和 Ekmax90 范围内
    rmin90, rmax90, Ekmin90, Ekmax90 = BMapData.rmin90, BMapData.rmax90, BMapData.Ekmin90, BMapData.Ekmax90
    print(f"检查能量范围: \n"
          f"注入能量 (energy_inj): {energy_inj} MeV\n"
          f"提取能量 (energy_ext): {energy_ext} MeV\n"
          f"BMapData 的能量范围: [{Ekmin90}, {Ekmax90}] MeV\n")

    # 判断能量范围是否满足条件
    if energy_inj < Ekmin90 or energy_ext > Ekmax90:
        print("Error: 能量范围不在 BMapData 的有效范围内。程序终止。")
        raise ValueError("注入或提取能量不在 BMapData 的 Ekmin90 和 Ekmax90 范围内。")
    else:
        print("能量范围在 BMapData 的有效范围内。")

    return coefficients_tupel


def CheckISOBmapConfigNew(config_data):
    # 从配置文件中获取能量范围
    energy_inj = config_data['machine']['energy_inj']
    energy_ext = config_data['machine']['energy_ext']
    EkRange = np.linspace(energy_inj, energy_ext, num=7, endpoint=True)  # 能量范围

    # 读取 Bmap 信息
    BmapInfo = config_data['Bmap']
    BzInfo = config_data['BzInfo']
    NSector = BmapInfo['NSector']
    theta_step_rad = BmapInfo['theta_step_rad']
    OrbitalFreq_MHz = BmapInfo['orbital_freq_MHz']
    k_value = BmapInfo['k_value']
    B0_T = BmapInfo['B0_T']
    SpiralAngle_deg = BmapInfo['SpiralAngle_deg']
    rmin_max_step_m = BmapInfo['rmin_max_step_m']
    BmapType = BmapInfo['Type']
    ConvexParams = BmapParams_to_ConvexParams(BmapInfo)

    # 设置 fi 和 r 轴
    fi_axis = np.linspace(0, np.pi * 2 / NSector, int(round(np.pi * 2 / NSector / theta_step_rad)) + 1)
    r_min, r_max, r_step = rmin_max_step_m
    r_axis = np.linspace(r_min, r_max, int(round((r_max - r_min) / r_step)) + 1)

    # 计算螺旋线方位角 s(r) 并进行多项式拟合
    s_values = calculate_s(r_axis, 0.0, SpiralAngle_deg)
    s_values_FitCoef = Polynomial.fit(r_axis, s_values, 6)

    other_paras = {
        "BmapInfo": BmapInfo,
        "EkRange": EkRange,
        "fi_axis": fi_axis,
        "r_axis": r_axis,
        "objective_freq_MHz": OrbitalFreq_MHz,
        "s_values": s_values_FitCoef,
        "ConvexParams": ConvexParams
    }

    GlobalParas = FFAG_GlobalParameters()

    # 处理不同的 Bmap 类型
    if BmapType == 'ISOCHRONOUS':
        print("Bmap type = ISOCHRONOUS")
        # 获取原始等时性磁场的展开系数
        r_axis_analytical, Biso_analytical, Ek_analytical_MeV, rinf = GetRawISO(
            OrbitalFreq_MHz * 1e6, GlobalParas, rmin=r_min, rmax=r_max)
        degree = 6  # 多项式阶数
        coefficients, _, _ = polyfit_shifted(r_axis_analytical, Biso_analytical, np.min(r_axis_analytical), degree)
        coefficients_tupel = (tuple(coefficients), r_min)

        IsoError = GetOrbitFreqOld(coefficients, other_paras)
        BzInfo['IsoCoef'] = tuple(coefficients)
        BzInfo['IsoError'] = IsoError

    elif BmapType == 'SCALE':
        print("Bmap type = SCALE")
        # 等比磁场使用用户直接输入的 k 值和 B0
        coefficients_tupel = (B0_T, k_value)

    else:
        raise ValueError("Invalid BmapType: expected 'ISOCHRONOUS', or 'SCALE'")

    # 生成等效 B-map 数据
    SaveData, Fmean, Bcentral = Generate_Bmap_Bz(r_axis, fi_axis, ConvexParams, s_values_FitCoef, coefficients_tupel)
    # plt.figure()
    # plt.plot(r_axis, Bcentral)
    # plt.show()
    BMapData = FFAG_BField_new(SaveData, 1, flag3D=False)

    # 检查 energy_inj, energy_ext 是否在 BMapData 的 Ekmin90 和 Ekmax90 范围内
    rmin90, rmax90, Ekmin90, Ekmax90 = BMapData.rmin90, BMapData.rmax90, BMapData.Ekmin90, BMapData.Ekmax90
    print(f"检查能量范围: \n"
          f"注入能量 (energy_inj): {energy_inj} MeV\n"
          f"引出能量 (energy_ext): {energy_ext} MeV\n"
          f"BMapData 的能量范围: [{Ekmin90}, {Ekmax90}] MeV")

    if energy_inj < Ekmin90 or energy_ext > Ekmax90:
        print("Error: 能量范围不在 BMapData 的有效范围内。程序终止。")
        raise ValueError(f"注入或引出能量不在 BMapData 的 [{Ekmin90} 和 {Ekmax90} 范围内, 请调整磁场参数。")
    else:
        print("能量范围在 BMapData 的有效范围内。")

    return coefficients_tupel


if __name__ == '__main__':
    # 获取命令行参数
    if len(sys.argv) != 4:
        print("Usage: python script.py <config_file_path> <max_iter>")
        sys.exit(1)

    file_path = sys.argv[1]  # 配置文件路径
    expand_order = int(sys.argv[2]) # 展开阶数
    max_iter = int(sys.argv[3])  # 最大迭代次数

    # 读取JSON文件
    with open(file_path, 'r') as f:
        config_data = json.load(f)

    energy_inj = config_data['machine']['energy_inj']
    energy_ext = config_data['machine']['energy_ext']
    EkRange = np.linspace(energy_inj, energy_ext, num=4, endpoint=True)

    BmapInfo = config_data['Bmap']
    NSector = BmapInfo['NSector']
    theta_step_rad = BmapInfo['theta_step_rad']
    OrbitalFreq_MHz = BmapInfo['orbital_freq_MHz']
    k_value = BmapInfo['k_value']
    SpiralAngle_deg = BmapInfo['SpiralAngle_deg']
    rmin_max_step_m = BmapInfo['rmin_max_step_m']
    ConvexParams = BmapParams_to_ConvexParams(BmapInfo)

    fi_axis = np.linspace(0, np.pi * 2 / NSector, int(round(np.pi * 2 / NSector / theta_step_rad)) + 1)
    r_min, r_max, r_step = rmin_max_step_m[0], rmin_max_step_m[1], rmin_max_step_m[2]
    r_axis = np.linspace(r_min, r_max, int(round((r_max - r_min) / r_step)) + 1)

    s_values = calculate_s(r_axis, 0.0, SpiralAngle_deg)
    s_values_FitCoef = Polynomial.fit(r_axis, s_values, 6)

    other_paras = {"BmapInfo": BmapInfo,
                   "EkRange": EkRange,
                   "fi_axis": fi_axis,
                   "r_axis": r_axis,
                   "objective_freq_MHz": OrbitalFreq_MHz,
                   "s_values": s_values_FitCoef,
                   "ConvexParams": ConvexParams}

    GlobalParas = FFAG_GlobalParameters()

    # 生成初始simplex
    coefficients_input = np.array(config_data['BzInfo']['IsoCoef'])
    # degree = len(coefficients_input) - 1  # 多项式的阶数

    Biso_reconstruct = polyval_shifted(coefficients_input, np.min(r_axis), r_axis)
    coefficients_output, _, inverse_powers = polyfit_shifted(r_axis, Biso_reconstruct, np.min(r_axis), expand_order)
    initial_simplex = generate_initial_simplex(coefficients_output, inverse_powers)

    # 优化simplex
    best_coef = nelder_mead(GetOrbitFreq, initial_simplex, other_paras, max_iter=max_iter)

    IsoError0 = GetOrbitFreqOld(best_coef, other_paras)

    # 更新字典
    BzInfo = config_data['BzInfo']
    BzInfo['IsoCoef'] = list(best_coef)
    BzInfo['IsoError'] = IsoError0
    save_file_name = f"configII_ISO_optimized_{int(IsoError0)}.json"
    with open(save_file_name, 'w') as f_out:
        json.dump(config_data, f_out, indent=2, sort_keys=True)

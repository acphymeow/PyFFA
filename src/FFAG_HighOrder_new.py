import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from numpy.polynomial import Polynomial
from collections import defaultdict
import numba as nb
from math import factorial
from mpi4py import MPI
from concurrent.futures import ProcessPoolExecutor

def calculate_first_derivative(term):
    """
    Calculate the first derivative of a given term.

    Args:
        term (np.ndarray): An array where the first element is the order of G's derivative (n),
                           the second element is the coefficient (k),
                           and the remaining elements are the exponents for s and its derivatives (m_list).

    Returns:
        list: A list of np.ndarrays representing the first derivative terms.
    """
    n, k = term[0], term[1]
    m_list = term[2:]
    derivative_terms = []

    # Contribution from the derivative of G^(n) with chain rule factor
    new_m_list = np.copy(m_list)
    if len(new_m_list) > 1:
        new_m_list[1] += 1  # Increase the exponent of s'
    else:
        new_m_list = np.append(new_m_list, 1)  # Add the exponent for s'
    derivative_terms.append(np.concatenate(([n + 1, -k], new_m_list)))  # Add the term for G^(n+1) with the chain rule factor

    # Contribution from the derivative of each s^(i)
    for i in range(len(m_list)):
        if m_list[i] > 0:
            new_m_list = np.copy(m_list)
            new_m_list[i] -= 1  # Decrease the exponent of s^(i)
            if i + 1 < len(new_m_list):
                new_m_list[i + 1] += 1  # Increase the exponent of s^(i+1)
            else:
                new_m_list = np.append(new_m_list, 1)  # Add the exponent for s^(i+1)
            derivative_terms.append(np.concatenate(([n, k * m_list[i]], new_m_list)))  # Add the term for the derivative of s^(i)

    return derivative_terms

def merge_terms(terms):
    """
    Merge terms with the same order of G's derivative and the same list of exponents.

    Args:
        terms (list): A list of np.ndarrays representing terms.

    Returns:
        list: A merged list of np.ndarrays representing terms.
    """
    merged = defaultdict(lambda: 0.0)
    for term in terms:
        n, k = term[0], term[1]
        m_list = tuple(term[2:])  # Convert m_list to a tuple to use as a key
        key = (n, m_list)  # Use (n, m_list) as the key
        merged[key] += k  # Sum the coefficients for the same key

    # Convert the merged dictionary back to a list of np.ndarrays and remove zero coefficients
    result = []
    for (n, m_list), k in merged.items():
        if k != 0:
            result.append(np.concatenate(([n, k], np.array(m_list))))
    return result


def calculate_higher_order_derivative(term, order):
    """
    Calculate the higher order derivative of a given term.

    Args:
        term (np.ndarray): An array containing the initial order of G's derivative (n),
                           the coefficient (k), and the exponents for s and its derivatives (m_list).
        order (int): The order of the derivative to calculate.

    Returns:
        list: A list of np.ndarrays representing the higher order derivative terms.
    """
    current_terms = [term]
    for _ in range(order):
        next_terms = []
        for term in current_terms:
            next_terms.extend(calculate_first_derivative(term))  # Calculate the first derivative of each term
        current_terms = merge_terms(next_terms)  # Merge terms with the same (n, m_list)
    return current_terms


def dGn_dnR_Vals(order, fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals, order_plus=0):
    """
    # 输入参数：凹凸函数G的相关参数 convex_paras, fi_axis(rad)
    # 输入参数：螺旋角函数s的相关参数 s的拟合结果s_values_FitCoef(rad), r_axis
    # 根据current_terms重新构建高阶导数,其结构[n,k,[m0,m1,m2,...]]
    # G[f-s(r)]的高阶导数: G(n)*k*[s^m0 * s(1)^m1 * s(2)^m2 * ...]
    # 而s的n阶导数s(n)=fit_polynomial_derivative(s_values_FitCoef, order)
    # G的高阶导数G(n)=Generate_F(fi_deg, convex_params, fi_shift_deg, order)[1]
    # 返回给定r_vals上的n阶偏导Patial_G_patial_R值：G[f-s(r)] at given r
    """
    initial_term = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    current_terms = calculate_higher_order_derivative(initial_term, order)

    sr_rad = s_values_FitCoef_rad(r_vals)
    SumTerm = np.zeros_like(fi_axis_rad)

    for term in current_terms:
        n, k, m_list = np.int64(term[0]), term[1], term[2:]
        fi_shift_rad = sr_rad

        # 计算 G 的 n+order_plus 阶导数
        G_derivative_values = Generate_F(fi_axis_rad, convex_paras, fi_shift_rad, n+order_plus)[1]
        # [1] 获取 G 的 n 阶导数, n+order_plus=0时可以返回原函数

        # 计算 s 及其各阶导数
        Patial_S_patial_R_Terms = 1.0  # [s^m0 * s(1)^m1 * s(2)^m2 * ...]这一项
        for order, exponent in enumerate(m_list):
            # order是需要遍历的求导的次数,从0开始递增. exponent是导数项的幂次.
            dsr_n_at_givenR = fit_polynomial_derivative(s_values_FitCoef_rad, order)(r_vals)
            Patial_S_patial_R_Terms *= dsr_n_at_givenR ** exponent
            # print(f"Order: {order}, dsr_n: {dsr_n_at_givenR}")

        SumTerm += G_derivative_values * k * Patial_S_patial_R_Terms

    return SumTerm

# 定义 sigmoid 函数
@nb.njit
def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

# @nb.njit
# def sigmoid_nth_derivative(x0, k, order, x_values):
#     """
#     Calculate the higher order derivative polynomial coefficients for the new sigmoid.
#
#     Parameters:
#     x0 (float): The center parameter of the sigmoid function.
#     k (float): The steepness parameter of the sigmoid function.
#     order (int): The order of the derivative to calculate.
#     x_values (array): x_axis values.
#
#     Returns:
#     list, array: Polynomial coefficients after the specified derivative and the evaluated polynomial derivative.
#     """
#     if order == 0:
#         return None, sigmoid(x_values, x0, k)
#
#     sig_x = sigmoid(x_values, x0, k)
#
#     coeffs = np.array([0, k, -k], dtype=np.float64)  # Coefficients for the first derivative polynomial (σ_prime(x) = kσ(x)(1-σ(x)))
#     current_coeffs = coeffs
#
#     for i in range(order - 1):  # Since we start with the first derivative
#         # Calculate the first derivative of the current polynomial
#         derivative_coeff = np.polyder(current_coeffs[::-1])[::-1]
#         # Multiply the derivative by the first derivative polynomial using convolution
#         current_coeffs = np.convolve(derivative_coeff, coeffs)
#         current_coeffs = current_coeffs.astype(np.float64)
#
#     # Compute the polynomial-based nth order derivative of the sigmoid function
#     poly_derivative_y = np.polyval(current_coeffs[::-1], sig_x)
#
#     return current_coeffs, poly_derivative_y

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
    result = np.zeros_like(x)
    power = len(coeffs) - 1

    for coeff in coeffs:
        result = result * x + coeff
        power -= 1

    return result

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

# 定义凸字形或凹字形函数
def shape_function(x, x0, x1, W0, y01=(1, 1)):
    direction = y01[1]
    y_convex = direction * np.abs(y01[0])
    y_flat = 0
    # y_flat, y_convex = (np.min(y01[:-1]), np.max(y01[:-1])) if direction > 0 else (np.max(y01[:-1]), np.min(y01[:-1]))

    k = 16 / W0  # 控制陡峭程度
    y1 = sigmoid(x, x0, k)
    y2 = sigmoid(x, x1, -k)
    return y_flat + (y_convex - y_flat) * y1 * y2


# def shape_function_nth_derivative(x, x0, x1, W0, n, y01=(0, 0, 1)):
#     """
#     生成凸字形或凹字形函数的高阶导数。
#
#     参数：
#     x (array): 输入值的数组。
#     x0 (float): 函数形状变化的起始点。
#     x1 (float): 函数形状变化的终止点。
#     W0 (float): 控制函数形状变化区域的宽度。
#     n (int): 导数的阶数。
#     y01 (tuple): 三元组，表示函数在不同区域的值和形状类型。默认值是 (0, 0, 1)。
#         - y01[0], y01[1]: 函数在变化区域两端的值, 不区分顺序。
#         - y01[2]: 控制函数形状类型的方向，正值表示凸字形，负值表示凹字形。
#
#     返回值：
#     array: 函数在输入值 x 处的输出值的高阶导数。
#     """
#     direction = y01[2]
#     y_flat, y_convex = (np.min(y01[:-1]), np.max(y01[:-1])) if direction > 0 else (np.max(y01[:-1]), np.min(y01[:-1]))
#
#     k = 16 / W0  # 控制陡峭程度
#
#     result = 0
#     for i in range(0, n+1):
#         binomial_coeff = binom(n, i)  # 计算二项式系数
#         _, deriv_y1 = sigmoid_nth_derivative(x0, k, i, x)
#         _, deriv_y2 = sigmoid_nth_derivative(x1, -k, n - i, x)
#         result += binomial_coeff * deriv_y1 * deriv_y2
#     return (y_convex - y_flat) * result


@nb.njit
def factorial_Manually(n):
    """
    Manually calculate the factorial of n.
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@nb.njit
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

@nb.njit
def shape_function_nth_derivative(x, x0, x1, W0, n, y01=(1, 1)):
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
    direction = y01[1]
    y_convex = direction * np.abs(y01[0])
    y_flat = 0

    k = 16 / W0  # 控制陡峭程度

    result = np.zeros_like(x)  # 初始化 result 为与 x 形状相同的数组
    for i in range(0, n+1):
        binomial_coeff = binomial_Manually(n, i)  # 计算二项式系数
        _, deriv_y1 = sigmoid_nth_derivative(x0, k, i, x)
        _, deriv_y2 = sigmoid_nth_derivative(x1, -k, n - i, x)
        result += binomial_coeff * deriv_y1 * deriv_y2
    return (y_convex - y_flat) * result


# 生成多个凸字形和凹字形函数
def generate_shapes(x, convex_params):
    y_total = np.zeros_like(x)
    for params in convex_params:
        x0, x1, W0, y01 = params
        y_total += shape_function(x, x0, x1, W0, y01)
    return y_total


# 生成多个凸字形和凹字形函数的高阶导数
# def generate_shapes_with_derivatives(x, convex_params, n):
#     y_total = np.zeros_like(x)
#     y_total_derivatives = np.zeros_like(x)
#
#     for params in convex_params:
#         x0, x1, W0, y01 = params
#         y_total += shape_function(x, x0, x1, W0, y01)
#         y_total_derivatives += shape_function_nth_derivative(x, x0, x1, W0, n, y01)
#
#     return y_total, y_total_derivatives

def generate_shapes_with_derivatives(x, convex_params, n):
    y_total = np.zeros_like(x)

    # 当 n=0 时，返回原函数的值两次
    if n == 0:
        for params in convex_params:
            x0, x1, W0, y01 = params
            y_total += shape_function(x, x0, x1, W0, y01)
        return y_total, y_total  # 返回原函数和原函数

    # 计算高阶导数的情况
    y_total_derivatives = np.zeros_like(x)
    for params in convex_params:
        x0, x1, W0, y01 = params
        y_total += shape_function(x, x0, x1, W0, y01)
        y_total_derivatives += shape_function_nth_derivative(x, x0, x1, W0, n, y01)

    return y_total, y_total_derivatives


# 平移函数并保持周期性
def shift_function(x, y, delta_x):
    step_size = x[1] - x[0]
    shift_steps = int(delta_x / step_size)
    y_shifted = np.roll(y, shift_steps)
    return y_shifted


def calculate_s(r_axis, s0, alpha_deg):
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


def fit_polynomial(r_axis, s_values, order):
    """将 s(r) 拟合成多项式, s(r)求高阶导"""
    # 拟合多项式
    s_values_FitPoly = Polynomial.fit(r_axis, s_values, order)
    return s_values_FitPoly


def fit_polynomial_derivative(s_values_FitCoef, order):
    """计算s(r)多项式的高阶导数"""
    # 计算高阶导数
    s_dp_FitPoly = s_values_FitCoef.deriv(m=order)
    return s_dp_FitPoly


# def recursive_nth_derivative(a, k, n, x_vals):
#     """递推计算n阶幂函数的导数(包括非整数次幂, scale Bz(r)求导)"""
#     # a:系数   k:k或幂次   n:求导的阶数   x_vals:x坐标
#     k_vals = np.arange(k, k-n, -1)
#     k_vals = np.append(k_vals, 1)
#     coefficient = a * np.prod(k_vals)
#     exponent = k - n
#     return coefficient * x_vals**exponent


def recursive_nth_derivative(a_or_coefficients, k_or_x0, n, x_vals):
    """
    计算幂函数或幂级数函数的 n 阶导数。

    Parameters:
        a_or_coefficients (float or list/array):
            - 如果是单个系数 (float)，表示幂函数的系数 a。
            - 如果是系数列表或数组 (list or array)，表示幂级数各项的系数 [a0, a1, a2, ...]。
        k_or_x0 (float):
            - 如果 a_or_coefficients 是数值，则表示幂函数的幂次 k。
            - 如果 a_or_coefficients 是列表或数组，则表示幂级数的展开点 x0。
        n (int): 要求的导数阶数。
        x_vals (float or array): 自变量 x 的值，可以是单值或数组。

    Returns:
        ndarray: n 阶导数在各 x 值处的结果。
    """
    # 判断 a_or_coefficients 是数值还是数组
    if isinstance(a_or_coefficients, (int, float)):  # 处理幂函数的情况
        a = a_or_coefficients
        k = k_or_x0
        k_vals = np.arange(k, k - n, -1)
        k_vals = np.append(k_vals, 1)
        coefficient = a * np.prod(k_vals)
        exponent = k - n
        return coefficient * x_vals ** exponent

    else:  # 处理幂级数的情况
        coefficients = a_or_coefficients[::-1]
        x0 = k_or_x0
        derivative = np.zeros_like(x_vals)

        for i, a_i in enumerate(coefficients):
            if i >= n:  # 只有 i >= n 时，n 阶导数非零
                coeff_derivative = a_i * factorial(i) / factorial(i - n)
                term = coeff_derivative * (x_vals - x0) ** (i - n)
                derivative += term

        return derivative


def Generate_F_givenR(fi_rad, convex_params, rvals, s_rad_polynomial, order=1):
    fi_shift_rad = s_rad_polynomial(rvals)

    # 生成初始区间上的凹凸函数及其高阶导数
    Bz_initial, Bz_initial_derivative = generate_shapes_with_derivatives(fi_rad, convex_params, order)

    Bz_shifted = shift_function(fi_rad, Bz_initial, fi_shift_rad)
    Bz_shifted_derivative = shift_function(fi_rad, Bz_initial_derivative, fi_shift_rad)

    return Bz_shifted, Bz_shifted_derivative


def Generate_F(fi_rad, convex_params, fi_shift_rad, order=1):
    # 生成初始区间上的凹凸函数及其高阶导数
    Bz_initial, Bz_initial_derivative = generate_shapes_with_derivatives(fi_rad, convex_params, order)

    Bz_shifted = shift_function(fi_rad, Bz_initial, fi_shift_rad)
    Bz_shifted_derivative = shift_function(fi_rad, Bz_initial_derivative, fi_shift_rad)

    return Bz_shifted, Bz_shifted_derivative


def Generate_ScaleR(r_min, r_max, r_step, Bz_max, r_Bzmax, k, order=0):
    # 生成 r 的数组
    r_axis = np.arange(r_min, r_max + r_step, r_step, dtype=np.float64)
    # 常数
    a = Bz_max / r_Bzmax**k
    # 求导系数
    k_vals = np.arange(k, k-order, -1)
    coefficient = a * np.prod(k_vals)
    # 剩余指数
    exponent = k - order
    return coefficient * r_axis**exponent


def merge_similar_terms(matrix):
    """
    合并矩阵中前3列相同的行，将相同项的系数a相加。
    """
    term_dict = {}

    for row in matrix:
        key = (row[0], row[1], row[2])  # 前三列作为键
        if key in term_dict:
            term_dict[key] += row[3]
        else:
            term_dict[key] = row[3]

    # 将合并后的项转换回矩阵
    merged_matrix = []
    for key, value in term_dict.items():
        merged_matrix.append([key[0], key[1], key[2], value])

    return np.array(merged_matrix)


def apply_r_derivative(matrix):
    """
    对给定的n*4矩阵应用d/dr操作，返回新矩阵。
    每一行表示一个表达式的系数：m, n, k, a
    """
    new_matrix = []

    for row in matrix:  # 遍历每一行
        m, n, k, a = row
        m, n, k, a = int(m), int(n), int(k), int(a)  # 确保为整数

        # 计算乘以d/dr后的两行
        new_row1 = [m + 1, n, k, a]
        new_row2 = [m, n, k + 1, -a * k]

        # 添加第一项
        new_matrix.append(new_row1)
        # 添加第二项，只有当k不为0时
        if k != 0:
            new_matrix.append(new_row2)

    return np.array(new_matrix)  # 不需要转置


def apply_second_r_derivative(matrix):
    """
    对给定的n*4矩阵应用d^2/dr^2操作，返回新矩阵。
    每一行表示一个表达式的系数：m, n, k, a
    """
    # 首先应用一次d/dr
    first_derivative_matrix = apply_r_derivative(matrix)

    # 再应用一次d/dr
    second_derivative_matrix = apply_r_derivative(first_derivative_matrix)

    return second_derivative_matrix


def apply_theta_derivative(matrix):
    """
    对给定的n*4矩阵应用d/dtheta操作，返回新矩阵。
    每一行表示一个表达式的系数：m, n, k, a
    """
    new_matrix = []

    for row in matrix:  # 遍历每一行
        m, n, k, a = row
        m, n, k, a = int(m), int(n), int(k), int(a)  # 确保为整数

        # 计算乘以d/dtheta后的新行
        new_row = [m, n + 1, k, a]

        new_matrix.append(new_row)

    return np.array(new_matrix)  # 不需要转置


def apply_second_theta_derivative(matrix):
    """
    对给定的n*4矩阵应用d^2/dtheta^2操作，返回新矩阵。
    每一行表示一个表达式的系数：m, n, k, a
    """
    # 首先应用一次d/dtheta
    first_derivative_matrix = apply_theta_derivative(matrix)

    # 再应用一次d/dtheta
    second_derivative_matrix = apply_theta_derivative(first_derivative_matrix)

    return second_derivative_matrix


def apply_r_negative_power(matrix, power):
    """
    对给定的n*4矩阵乘以r^(-power)，返回新矩阵。
    每一行表示一个表达式的系数：m, n, k, a
    """
    new_matrix = []

    for row in matrix:  # 遍历每一行
        m, n, k, a = row
        m, n, k, a = int(m), int(n), int(k), int(a)  # 确保为整数

        # 计算乘以r^(-power)后的新行
        new_row = [m, n, k + power, a]

        new_matrix.append(new_row)

    return np.array(new_matrix)  # 不需要转置


def apply_laplacian_combined(matrix):
    """
    对给定的n*4矩阵应用极坐标系下的拉普拉斯算子，使用组合递推关系，返回新矩阵。
    每一行表示一个表达式的系数：m, n, k, a
    """
    # 计算d^2/dr^2
    second_r_derivative_matrix = apply_second_r_derivative(matrix)

    # 计算(1/r) * d/dr
    r_derivative_matrix = apply_r_derivative(matrix)
    r_derivative_matrix = apply_r_negative_power(r_derivative_matrix, 1)

    # 计算(1/r^2) * d^2/dtheta^2
    second_theta_derivative_matrix = apply_second_theta_derivative(matrix)
    second_theta_derivative_matrix = apply_r_negative_power(second_theta_derivative_matrix, 2)

    # 合并所有结果
    combined_matrix = np.vstack((second_r_derivative_matrix, r_derivative_matrix, second_theta_derivative_matrix))

    return combined_matrix


def nth_derivative_B_r(n, a, k, fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals):

    # 对于一个函数B(r, theta) = A(r)*G(theta-s(r)), 需要求d^nB/dr^n 和 d^nB/dtheta^n
    # 其中d^nG/dr^n = dGn_dnR_Vals(n, fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals)
    # 除n外其他参数都是已经给定的参量

    # d^nA/dr^n = recursive_nth_derivative(a, k, n, r_vals)
    # 除n外其他参数都是已经给定的参量

    # A = recursive_nth_derivative(a, k, 0, r_vals)
    # 除n外其他参数都是已经给定的参量

    # d^nG/dtheta^n = Generate_F_givenR(fi_rad, convex_params, rvals, s_rad_polynomial, order=n)
    # 除n外其他参数都是已经给定的参量

    # 求(d^(n)B)/(dr^n) 和 求(d^(m)B)/(dtheta^n)

    # 处理 n=0 的情况，直接返回函数本身的值
    if n == 0:
        G_val, _ = Generate_F_givenR(fi_axis_rad, convex_paras, r_vals, s_values_FitCoef_rad, order=0)
        A_val = recursive_nth_derivative(a, k, 0, r_vals)
        return A_val * G_val

    result = 0
    for i in range(0, n+1):
        binomial_coeff = binom(n, i)
        deriv_y1 = dGn_dnR_Vals(n, fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals)
        deriv_y2 = recursive_nth_derivative(a, k, n-i, r_vals)
        result += binomial_coeff * deriv_y1 * deriv_y2

    return result


def nth_derivative_B_theta(n, a, k, fi_axis_rad, convex_params, s_rad_polynomial, r_vals):
    # 对于一个函数B(r, theta) = A(r)*G(theta-s(r)), 需要求d^nB/dr^n 和 d^nB/dtheta^n
    # 其中d^nG/dr^n = dGn_dnR_Vals(n, fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals)
    # 除n外其他参数都是已经给定的参量

    # d^nA/dr^n = recursive_nth_derivative(a, k, n, r_vals)
    # 除n外其他参数都是已经给定的参量

    # A = recursive_nth_derivative(a, k, 0, r_vals)
    # 除n外其他参数都是已经给定的参量

    # d^nG/dtheta^n = Generate_F_givenR(fi_rad, convex_params, rvals, s_rad_polynomial, order=n)
    # 除n外其他参数都是已经给定的参量

    # 求(d^(n)B)/(dr^n) 和 求(d^(m)B)/(dtheta^n)

    # 处理 n=0 的情况，直接返回函数本身的值
    if n == 0:
        G_val = Generate_F_givenR(fi_axis_rad, convex_params, r_vals, s_rad_polynomial, order=0)[0]
        A_val = recursive_nth_derivative(a, k, 0, r_vals)
        return A_val * G_val

    A_r_vals = recursive_nth_derivative(a, k, 0, r_vals)
    _, dnG_dThetan = Generate_F_givenR(fi_axis_rad, convex_params, r_vals, s_rad_polynomial, order=n)
    result = A_r_vals * dnG_dThetan

    return result


def nth_derivative_B_r_theta(n, m, a, k, fi_axis_rad, convex_params, s_rad_polynomial, r_vals):
    """
        计算 B(r, theta) = A(r)*G(theta-s(r)) 的混合高阶导数 d^(n+m)B/(dr^n * dθ^m)。

        参数:
        - n: r 的偏导阶数
        - m: θ 的偏导阶数
        - a, k: 与 A(r) 相关的参数, a可以是数或数组,数代表等比FFAG情形，数组代表等时FFAG情形
        - a, k: 等比FFAG时,a为常系数,k为k值。等时FFAG时,a为展开系数,k为x0
        - fi_axis_rad: 角度坐标数组（弧度制）
        - convex_params: 凸/凹形参数
        - s_rad_polynomial: s(r) 的多项式系数
        - r_vals: r 的具体值

        返回值:
        - 混合导数的结果 d^(n+m)B/(dr^n * dθ^m)
        """

    result = 0
    for i in range(0, n+1):
        binomial_coeff = binom(n, i)
        deriv_y1 = dGn_dnR_Vals(n, fi_axis_rad, convex_params, s_rad_polynomial, r_vals, order_plus=m)
        deriv_y2 = recursive_nth_derivative(a, k, n-i, r_vals)
        result += binomial_coeff * deriv_y1 * deriv_y2

    return result


def Polar_Laplace_Vals(merge_Laplace_matrix, r_vals, fi_axis_rad, convex_paras, s_values_FitCoef_rad, BzR_coef_k):
    # 求拉普拉斯作用于B一共w次,或者再增加一次d/dr(或d/dtheta),取决于输入的matrix
    # 在等时情形时, BzR_coef_k[0]是B0(r)=a0+a1*(r-r0)+a2*(r-r0)^2+...关于(r-r0)的展开系数(a0,a1,a2,...), BzR_coef_k[1]是r0
    # 在等比情形时, BzR_coef_k[0]是B0(r)=a*r^k中的常系数a, BzR_coef_k[1]是k值

    result = np.zeros_like(fi_axis_rad)
    for term in merge_Laplace_matrix:
        n, m, k, a0 = term[0], term[1], term[2], term[3]
        # n:对r求偏导的阶数, m:对theta求偏导的阶数, k:r的-k次幂, a:其它系数
        # dBdR_n = nth_derivative_B_r(n, a, k, fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals)
        # dBdF_n = nth_derivative_B_theta(m, a, k, fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals)
        dB_dRn_dFm = nth_derivative_B_r_theta(n, m, BzR_coef_k[0], BzR_coef_k[1], fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals)
        # dB_coef_n = a / r_vals**k
        dB_coef_n = a0
        # result += dBdR_n * dBdF_n * dB_coef_n
        result += dB_dRn_dFm * dB_coef_n

        # plt.figure()
        # plt.plot(dB_dRn_dFm)
        # plt.show()

    return result


def Generate_Bmap_Bz(r_axis, fi_axis_rad, convex_paras, s_values_FitCoef_rad, BzR_coef_k):
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

    a_or_coefficients = BzR_coef_k[0]
    k_or_x0 = BzR_coef_k[1]

    # 初始化 Flutter 矩阵，维度为 (半径数量, 角度数量)
    Flutter_matrix = np.zeros((len(r_axis), len(fi_axis_rad)), dtype=np.float64)
    Flutter_Fmean = np.ones(len(r_axis), dtype=np.float64)
    Bcentral_analytical = np.ones(len(r_axis), dtype=np.float64)

    # 遍历每个半径值，计算对应的 Flutter 矩阵
    for r_idx, r_vals in enumerate(r_axis):
        # 生成与给定 r 相关的 Flutter 函数
        F_givenR = dGn_dnR_Vals(0, fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals, order_plus=0)

        if isinstance(a_or_coefficients, (int, float)):  # 处理幂函数(等比)的情况
            # 给出的是中心线磁场，无需转换
            pass
        else:  # 处理幂级数(等时)的情况
            # 给出的是平均场，需要根据Flutter的平均磁场转换为中心线磁场
            Flutter_Fmean[r_idx] = np.mean(F_givenR)

        # 根据半径值计算 Bz 的中心磁场
        Bcentral_givenR = recursive_nth_derivative(a_or_coefficients, k_or_x0, 0, r_vals) / Flutter_Fmean[r_idx]

        # 将 Flutter 函数与中心磁场值相乘，得到 Bz 的 Flutter 矩阵
        Flutter_matrix[r_idx, :] = F_givenR * Bcentral_givenR

        #保存中心线磁场
        Bcentral_analytical[r_idx] = Bcentral_givenR

    # 保存数据，将半径轴添加为第一列，角度轴添加为第一行
    # 首先，将 r 轴和 Flutter 矩阵整合为一个矩阵
    SaveData = np.hstack((np.reshape(r_axis, (-1, 1)), Flutter_matrix))

    # 然后，将角度轴添加为第一行
    SaveData = np.vstack((np.hstack((np.zeros((1, 1)), np.reshape(fi_axis_rad, (1, -1)))), SaveData))

    return SaveData, Flutter_Fmean, Bcentral_analytical


def Generate_Bmaps(MaxOrder, r_axis, fi_axis_rad, convex_paras, s_values_FitCoef_rad, BzR_coef_k):
    """
    为每个磁场分量 B_r, B_z, B_phi 生成从 0 到 MaxOrder 阶的泰勒展开系数矩阵。

    参数:
    MaxOrder: 泰勒展开的最高阶数
    r_axis: 半径轴上的点
    fi_axis_rad: 角度轴上的点 (以弧度为单位)
    convex_paras: 凸字形或凹字形函数参数
    s_values_FitCoef_rad: 拟合的 s(r) 的多项式系数
    BzR_coef_k: 径向场相关参数,是一个元组,包含2个元素。第一个元素是数时，对应等比情形。第一个元素是数组时，对应等时情形。

    返回值:
    Br_Taylor_matrices, Bz_Taylor_matrices, Bfi_Taylor_matrices:
    每阶 Br, Bz, Bfi 的泰勒展开系数矩阵列表，长度为 MaxOrder + 1
    """

    # 初始化 Br, Bz, Bfi 的泰勒展开系数矩阵列表，每一阶都有3个矩阵
    Br_Taylor_matrices = []
    Bz_Taylor_matrices = []
    Bfi_Taylor_matrices = []

    # 初始矩阵表示原始函数形式
    InitialMatrix = np.array([
        [0, 0, 0, 1],  # 表示原函数
    ])

    # 复制初始矩阵，作为拉普拉斯展开起点
    laplacian_matrix = np.copy(InitialMatrix)

    # 开始泰勒展开，从 0 阶到 MaxOrder
    for expansion_order in range(0, MaxOrder + 1):
        # 打印当前的阶次进度
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            print(f"正在计算第 {expansion_order} 阶展开系数...", flush=True)

        # 计算导数和拉普拉斯项
        laplacian_matrix_OneMoreDr = apply_r_derivative(laplacian_matrix)
        laplacian_matrix_OneMoreDf = apply_theta_derivative(laplacian_matrix)
        const_coef_2np1 = (-1) ** expansion_order / factorial(2 * expansion_order + 1)  # 常数项
        const_coef_2n = (-1) ** expansion_order / factorial(2 * expansion_order)  # 常数项

        # 合并相同项
        laplacian_matrix = merge_similar_terms(laplacian_matrix)
        laplacian_matrix_OneMoreDr = merge_similar_terms(laplacian_matrix_OneMoreDr)
        laplacian_matrix_OneMoreDf = merge_similar_terms(laplacian_matrix_OneMoreDf)

        # 初始化本阶的 Br, Bz, Bfi 矩阵
        Br_Taylor_matrix = np.zeros((len(r_axis), len(fi_axis_rad)), dtype=np.float64)
        Bz_Taylor_matrix = np.zeros((len(r_axis), len(fi_axis_rad)), dtype=np.float64)
        Bfi_Taylor_matrix = np.zeros((len(r_axis), len(fi_axis_rad)), dtype=np.float64)

        # 遍历 r 轴，填充每个 r 值对应的 fi 上的泰勒系数
        for r_idx, r_vals in enumerate(r_axis):
            # 生成与给定 r 相关的 Flutter 函数
            F_givenR = dGn_dnR_Vals(0, fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals, order_plus=0)
            MeanF = np.mean(F_givenR)
            # 填充每个分量的泰勒系数矩阵
            Bz_Taylor_coef = const_coef_2n * Polar_Laplace_Vals(laplacian_matrix, r_vals, fi_axis_rad, convex_paras,
                                                s_values_FitCoef_rad, BzR_coef_k) / MeanF
            Br_Taylor_coef = const_coef_2np1 * Polar_Laplace_Vals(laplacian_matrix_OneMoreDr, r_vals, fi_axis_rad, convex_paras,
                                                s_values_FitCoef_rad, BzR_coef_k) / MeanF
            Bfi_Taylor_coef = const_coef_2np1 / r_vals * Polar_Laplace_Vals(laplacian_matrix_OneMoreDf, r_vals, fi_axis_rad, convex_paras,
                                                 s_values_FitCoef_rad, BzR_coef_k) / MeanF

            # 将每个 r_vals 的结果填入对应矩阵
            Br_Taylor_matrix[r_idx, :] = Br_Taylor_coef
            Bz_Taylor_matrix[r_idx, :] = Bz_Taylor_coef
            Bfi_Taylor_matrix[r_idx, :] = Bfi_Taylor_coef

        # 保存当前阶次的泰勒系数矩阵
        Br_Taylor_matrices.append(Br_Taylor_matrix)
        Bz_Taylor_matrices.append(Bz_Taylor_matrix)
        Bfi_Taylor_matrices.append(Bfi_Taylor_matrix)

        # 更新拉普拉斯矩阵进行下一阶展开
        laplacian_matrix = apply_laplacian_combined(laplacian_matrix)

    # 返回所有阶次的泰勒系数矩阵
    return Br_Taylor_matrices, Bz_Taylor_matrices, Bfi_Taylor_matrices


def compute_taylor_coefficients_group(r_group, fi_axis_rad, laplacian_matrix, laplacian_matrix_OneMoreDr,
                                      laplacian_matrix_OneMoreDf, convex_paras, s_values_FitCoef_rad,
                                      BzR_coef_k, const_coef_2np1, const_coef_2n):
    """对一组 r 值计算泰勒展开系数矩阵"""
    Bz_group = []
    Br_group = []
    Bfi_group = []

    for r_vals in r_group:
        Bz_Taylor_coef = const_coef_2n * Polar_Laplace_Vals(
            laplacian_matrix, r_vals, fi_axis_rad, convex_paras, s_values_FitCoef_rad, BzR_coef_k)
        Br_Taylor_coef = const_coef_2np1 * Polar_Laplace_Vals(
            laplacian_matrix_OneMoreDr, r_vals, fi_axis_rad, convex_paras, s_values_FitCoef_rad, BzR_coef_k)
        Bfi_Taylor_coef = const_coef_2np1 / r_vals * Polar_Laplace_Vals(
            laplacian_matrix_OneMoreDf, r_vals, fi_axis_rad, convex_paras, s_values_FitCoef_rad, BzR_coef_k)

        Bz_group.append(Bz_Taylor_coef)
        Br_group.append(Br_Taylor_coef)
        Bfi_group.append(Bfi_Taylor_coef)

    return np.array(Bz_group), np.array(Br_group), np.array(Bfi_group)


def Generate_Bmaps_MultiProcess(MaxOrder, r_axis, fi_axis_rad, convex_paras, s_values_FitCoef_rad, BzR_coef_k):
    """
    为每个磁场分量 B_r, B_z, B_phi 生成从 0 到 MaxOrder 阶的泰勒展开系数矩阵（并行版本）。
    """
    Br_Taylor_matrices = []
    Bz_Taylor_matrices = []
    Bfi_Taylor_matrices = []

    InitialMatrix = np.array([[0, 0, 0, 1]])  # 表示原函数
    laplacian_matrix = np.copy(InitialMatrix)

    # 创建一次进程池
    with ProcessPoolExecutor(max_workers=8) as executor:
        for expansion_order in range(0, MaxOrder + 1):
            print(f"正在计算第 {2*expansion_order}, {2*expansion_order+1}阶展开系数...")

            # 计算导数和常数系数
            laplacian_matrix_OneMoreDr = apply_r_derivative(laplacian_matrix)
            laplacian_matrix_OneMoreDf = apply_theta_derivative(laplacian_matrix)
            const_coef_2np1 = (-1) ** expansion_order / factorial(2 * expansion_order + 1)
            const_coef_2n = (-1) ** expansion_order / factorial(2 * expansion_order)

            laplacian_matrix = merge_similar_terms(laplacian_matrix)
            laplacian_matrix_OneMoreDr = merge_similar_terms(laplacian_matrix_OneMoreDr)
            laplacian_matrix_OneMoreDf = merge_similar_terms(laplacian_matrix_OneMoreDf)

            # 初始化每阶的矩阵
            Br_Taylor_matrix = np.zeros((len(r_axis), len(fi_axis_rad)), dtype=np.float64)
            Bz_Taylor_matrix = np.zeros((len(r_axis), len(fi_axis_rad)), dtype=np.float64)
            Bfi_Taylor_matrix = np.zeros((len(r_axis), len(fi_axis_rad)), dtype=np.float64)

            # 分组任务：将 r_axis 分为若干组，每组包含多个 r 值
            group_size = 10
            r_axis_groups = [r_axis[i:i + group_size] for i in range(0, len(r_axis), group_size)]

            # 提交任务到进程池
            futures = [
                executor.submit(
                    compute_taylor_coefficients_group, r_group, fi_axis_rad, laplacian_matrix, laplacian_matrix_OneMoreDr,
                    laplacian_matrix_OneMoreDf, convex_paras, s_values_FitCoef_rad, BzR_coef_k,
                    const_coef_2np1, const_coef_2n
                )
                for r_group in r_axis_groups
            ]

            # 获取任务结果
            for i, future in enumerate(futures):
                Bz_group, Br_group, Bfi_group = future.result()
                start_idx = i * group_size
                end_idx = start_idx + Bz_group.shape[0]
                Bz_Taylor_matrix[start_idx:end_idx, :] = Bz_group
                Br_Taylor_matrix[start_idx:end_idx, :] = Br_group
                Bfi_Taylor_matrix[start_idx:end_idx, :] = Bfi_group

            # 保存本阶的结果

            Br_Taylor_matrices.append(Br_Taylor_matrix)
            Bz_Taylor_matrices.append(Bz_Taylor_matrix)
            Bfi_Taylor_matrices.append(Bfi_Taylor_matrix)

            laplacian_matrix = apply_laplacian_combined(laplacian_matrix)

    return Br_Taylor_matrices, Bz_Taylor_matrices, Bfi_Taylor_matrices


def BmapParams_to_ConvexParams(Bmap):
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
            shape = (abs(polarity), 1)  # 凸
        else:
            shape = (abs(polarity), -1)  # 凹

        # 假设上升/下降沿宽度是10%角宽
        edge_width = (end_angle - start_angle) * fringe_width

        # 将当前扇区的参数添加到 convex_params, 单位转换为rad
        convex_params.append((np.deg2rad(start_angle), np.deg2rad(end_angle), np.deg2rad(edge_width), shape))

        # 更新下一个扇区的起始角度
        start_angle = end_angle

    return convex_params

def nth_derivate_wrap(Bmap_data, faxis, raxis, laplace_coefs):

    # Bmap_laplace_coefs = Bmap_data
    Nr, Nf = Bmap_data.shape
    r_matrix = np.tile(raxis.reshape(-1, 1), (1, Nf))
    results = np.zeros_like(Bmap_data)
    for laplace_coef in laplace_coefs:
        nth, mth, rpower, const_coef = laplace_coef[0], laplace_coef[1], laplace_coef[2], laplace_coef[3]
        results += const_coef / r_matrix ** rpower * nth_derivate_drdf(Bmap_data, faxis, mth, raxis, nth)

    return results


def nth_derivate_laplace(Bmap_data, faxis, raxis):
    Nr, Nf = Bmap_data.shape
    dBdr = nth_derivate_dr(Bmap_data, raxis, 1)
    dBdr2 = nth_derivate_dr(Bmap_data, raxis, 2)
    dBdf2 = nth_derivate_df(Bmap_data, faxis, 2)
    r_matrix = np.tile(raxis.reshape(-1,1), (1, Nf))
    rpower_matrix = r_matrix ** 2

    Laplace_Bmap = dBdr2 + 1/r_matrix*dBdr + 1/rpower_matrix*dBdf2

    return Laplace_Bmap

def nth_derivate_drdf(Bmap_data, faxis, max_order_f, raxis, max_order_r):
    dBdx = nth_derivate_dr(Bmap_data, raxis, max_order_r)
    dBdx = nth_derivate_df(dBdx, faxis, max_order_f)
    return dBdx


def nth_derivate_df(Bmap_data, faxis, max_order):
    """
    计算二维磁场矩阵在 (r, φ) 方向上的高阶导数，使用 np.gradient 方法，并考虑 φ 方向的周期性。

    参数：
    - Bmap_data: 2D 磁场矩阵，形状 (Nr, Nf)
    - faxis: 1D 数组，φ 方向的网格点
    - raxis: 1D 数组，r 方向的网格点
    - max_order: 最高阶数

    返回：
    - derivatives: B对fi的n阶偏导
    """
    Nr, Nf = Bmap_data.shape

    # 在 φ 方向左右各复制一份，形成扩展矩阵
    Bmap_extended = np.hstack([Bmap_data[:,:-1], Bmap_data, Bmap_data[:,1:]])
    faxis_left = faxis[:-1]-(faxis[-1]-faxis[0])  # 到-1止是为了避免faxis[-1]和faxis[0]重复
    faxis_right = faxis[1:]+(faxis[-1]-faxis[0])  # 从1开始是为了避免faxis[-1]和faxis[0]重复
    faxis_extended = np.hstack([faxis_left, faxis, faxis_right])

    # 计算 φ 方向导数
    dBdf = Bmap_extended

    for order in range(0, max_order + 1):
        if order > 0:
            dBdf = np.gradient(dBdf, faxis_extended, axis=1, edge_order=2)

    derivatives = dBdf[:, Nf-1:Nf+Nf-1]  # 取回中间部分

    return derivatives


def nth_derivate_dr(Bmap_data, raxis, max_order):
    """
    计算二维磁场矩阵在 (r, φ) 方向上的高阶导数，使用 np.gradient 方法，并考虑 φ 方向的周期性。

    参数：
    - Bmap_data: 2D 磁场矩阵，形状 (Nr, Nf)
    - faxis: 1D 数组，φ 方向的网格点
    - raxis: 1D 数组，r 方向的网格点
    - max_order: 最高阶数

    返回：
    - derivatives: B对r的m阶偏导
    """
    Nr, Nf = Bmap_data.shape

    # 在 φ 方向左右各复制一份，形成扩展矩阵
    Bmap_extended = np.hstack([Bmap_data[:, :-1], Bmap_data, Bmap_data[:, 1:]])

    # 计算 φ 方向导数
    dBdf = Bmap_extended

    for order in range(0, max_order + 1):
        if order > 0:
            dBdf = np.gradient(dBdf, raxis, axis=0, edge_order=2)

    derivatives = dBdf[:, Nf - 1:Nf + Nf - 1]  # 取回中间部分

    return derivatives


def Generate_Bmaps_new(MaxOrder, r_axis, fi_axis_rad, Bmap_data):
    """
    为每个磁场分量 B_r, B_z, B_phi 生成从 0 到 MaxOrder 阶的泰勒展开系数矩阵。

    参数:
    MaxOrder: 泰勒展开的最高阶数
    r_axis: 半径轴上的点
    fi_axis_rad: 角度轴上的点 (以弧度为单位)
    convex_paras: 凸字形或凹字形函数参数
    s_values_FitCoef_rad: 拟合的 s(r) 的多项式系数
    BzR_coef_k: 径向场相关参数,是一个元组,包含2个元素。第一个元素是数时，对应等比情形。第一个元素是数组时，对应等时情形。

    返回值:
    Br_Taylor_matrices, Bz_Taylor_matrices, Bfi_Taylor_matrices:
    每阶 Br, Bz, Bfi 的泰勒展开系数矩阵列表，长度为 MaxOrder + 1
    """

    # 初始化 Br, Bz, Bfi 的泰勒展开系数矩阵列表，每一阶都有3个矩阵
    Br_Taylor_matrices = []
    Bz_Taylor_matrices = []
    Bfi_Taylor_matrices = []

    # 初始矩阵表示原始函数形式
    InitialMatrix = np.array([
        [0, 0, 0, 1],  # 表示原函数
    ])

    # 复制初始矩阵，作为拉普拉斯展开起点
    laplacian_matrix = np.copy(InitialMatrix)
    Nr, Nf = Bmap_data.shape
    r_matrix = np.tile(r_axis.reshape(-1, 1), (1, Nf))


    # 开始泰勒展开，从 0 阶到 MaxOrder
    for expansion_order in range(0, MaxOrder + 1):
        # 打印当前的阶次进度
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            print(f"正在计算第 {expansion_order} 阶展开系数...", flush=True)

        # 计算导数和拉普拉斯项
        laplacian_matrix_OneMoreDr = apply_r_derivative(laplacian_matrix)
        laplacian_matrix_OneMoreDf = apply_theta_derivative(laplacian_matrix)
        const_coef_2np1 = (-1) ** expansion_order / factorial(2 * expansion_order + 1)  # 常数项
        const_coef_2n = (-1) ** expansion_order / factorial(2 * expansion_order)  # 常数项

        # 合并相同项
        laplacian_matrix = merge_similar_terms(laplacian_matrix)
        laplacian_matrix_OneMoreDr = merge_similar_terms(laplacian_matrix_OneMoreDr)
        laplacian_matrix_OneMoreDf = merge_similar_terms(laplacian_matrix_OneMoreDf)

        Bz_Taylor_matrix = nth_derivate_wrap(Bmap_data, fi_axis_rad, r_axis, laplacian_matrix)*const_coef_2n
        Br_Taylor_matrix = nth_derivate_wrap(Bmap_data, fi_axis_rad, r_axis, laplacian_matrix_OneMoreDr)*const_coef_2np1
        Bfi_Taylor_matrix = nth_derivate_wrap(Bmap_data, fi_axis_rad, r_axis, laplacian_matrix_OneMoreDf)*const_coef_2np1/r_matrix

        # 保存当前阶次的泰勒系数矩阵
        Br_Taylor_matrices.append(Br_Taylor_matrix)
        Bz_Taylor_matrices.append(Bz_Taylor_matrix)
        Bfi_Taylor_matrices.append(Bfi_Taylor_matrix)

        # 更新拉普拉斯矩阵进行下一阶展开
        laplacian_matrix = apply_laplacian_combined(laplacian_matrix)

    # 返回所有阶次的泰勒系数矩阵
    return Br_Taylor_matrices, Bz_Taylor_matrices, Bfi_Taylor_matrices

if __name__ == '__main__':
    # 定义多个凹凸形函数参数 (x0, x1, W0, y01) - 左侧中心点,右侧中心点,上升下降沿宽度,高度参数
    # convex_params = [
    #     (np.deg2rad(10), np.deg2rad(25), np.deg2rad(6), (0, 1, 1)),
    #     (np.deg2rad(35), np.deg2rad(55), np.deg2rad(6), (0, -1, -1)),
    #     (np.deg2rad(65), np.deg2rad(80), np.deg2rad(6), (0, 1, 1))
    # ]

    # 对于一个函数B(r, theta) = A(r)*G(theta-s(r)), 需要求d^nB/dr^n 和 d^nB/dtheta^n
    # 其中d^nG/dr^n = dGn_dnR_Vals(n, fi_axis_rad, convex_paras, s_values_FitCoef_rad, r_vals)
    # d^nA/dr^n = recursive_nth_derivative(a, k, n, x_vals)
    # d^nG/dtheta^n = Generate_F(fi_rad, convex_params, fi_shift_rad, order=1)
    # 求(d^(n)B)/(dr^n) 和 求(d^(m)B)/(dtheta^n)

    convex_params = [
        (np.deg2rad(20), np.deg2rad(70), np.deg2rad(15), (1, 1)),
    ]

    # 平移fi_shift_deg度并保持周期性
    fi_shift_deg = 9.0  # 需要平移的角度
    r_min, r_max, r_step, s0 = 4.800, 6.800, 0.001, 0.00
    alpha_deg = 45.0  # spiral angle in degree
    order = 8  # 多项式阶数
    fi_min_deg, fi_max_deg, fi_step_deg = 0.0, 90.0, 0.01
    s0_deg = 0.0

    # 生成fi = np.arange(r0, rmax + rstep, rstep)
    fi_axis_deg = np.arange(fi_min_deg, fi_max_deg + fi_step_deg, fi_step_deg)
    r_axis_m = np.arange(r_min, r_max + r_step, r_step)
    fi_axis_rad = np.deg2rad(fi_axis_deg)
    # 计算 s(r)
    s_values_rad = calculate_s(r_axis_m, np.deg2rad(s0_deg), alpha_deg)
    # 拟合多项式
    s_rad_polynomial = fit_polynomial(r_axis_m, s_values_rad, order)
    print("Fitted polynomial coefficients:", s_rad_polynomial.convert().coef)
    # 获取r处的值: polynomial(r)

    n_fi, n_r = len(fi_axis_deg), len(r_axis_m)

    Bz_map = np.zeros((n_r, n_fi), dtype=np.float64)
    dBzdf_map = np.zeros((n_r, n_fi), dtype=np.float64)
    dBzdr_map = np.zeros((n_r, n_fi), dtype=np.float64)

    for i in range(n_r):
        current_r = r_axis_m[i]
        fi_shift_rad = s_rad_polynomial(current_r)
        Bz_at_given_r, dB_at_given_r = Generate_F(fi_axis_rad, convex_params, fi_shift_rad, order=1)
        Bz_map[i, :] = Bz_at_given_r

    # ff, rr = np.meshgrid(fi_axis_rad, r_axis_m)
    # plot_xx, plot_yy = rr*np.cos(ff), rr*np.sin(ff)
    # fig = plt.figure(6)
    # ax = fig.add_subplot(111, projection='3d')
    # # 绘制3D曲面
    # ax.plot_surface(plot_xx, plot_yy, Bz_map, cmap='plasma', rcount=120, ccount=120)
    # plt.show()

    initial_term = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    order = 12  # Compute the 4th derivative
    current_terms = calculate_higher_order_derivative(initial_term, order)
    SumTerm = dGn_dnR_Vals(order, fi_axis_rad, convex_params, s_rad_polynomial, r_axis_m[10])

    fi_shift_rad = s_rad_polynomial(r_axis_m[10])
    _, dB_at_given_r = Generate_F(fi_axis_rad, convex_params, fi_shift_rad, order=order)

    plt.figure(6)
    plt.plot(np.rad2deg(fi_axis_rad), SumTerm)
    plt.figure(7)
    plt.plot(np.rad2deg(fi_axis_rad), dB_at_given_r, linewidth=2, color='b')
    plt.show()
    pass

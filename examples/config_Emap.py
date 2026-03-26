# -*- coding: utf-8 -*-
import json
import time
import os


config_data = {}
###############################################################
# ------------------- General parameters ---------------------
###############################################################
config = dict()
config['date'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# 是否执行各个预处理模块
config['build_frequency_curve'] = True
config['build_spiral_map'] = True

# 是否绘图
config['plot_frequency_curve'] = True
config['plot_spiral_map'] = True

# 频率曲线输出文件
config['filename'] = './Emaps_test/FrequencyCurve_Lpaint_1.txt'

###############################################################
# ------------------- Machine parameters ---------------------
###############################################################
machine = dict()
machine['energy_inj'] = 300   # MeV
machine['energy_ext'] = 600   # MeV

###############################################################
# ------------------- Emap configuration -------------------- #
###############################################################
Emap = dict()

Emap['Bmap_fold'] = "./Bmap_FD16"       # 磁场文件目录
# Emap['Bmap_fold'] = "/hpcfs/APG/zhoukai/ffa/Bmap"

# ============================================================
# 同步相位程序
# ============================================================
# Emap['acc_phase_start'] = 0.0    # 涂抹相位（度）
# Emap['acc_phase_end'] = 30.0     # 加速相位（度）
# Emap['phase_ramp_start'] = 1000  # 平滑变化开始圈
# Emap['phase_ramp_end'] = 2000    # 平滑变化结束圈

Emap['acc_phase_start'] = 30.0      # 涂抹相位（度）
Emap['acc_phase_end'] = 30.0       # 加速相位（度）
Emap['phase_ramp_start'] = 1000       # 同步相位线性变化开始圈编号
Emap['phase_ramp_end'] = 2000         # 同步相位线性变化结束圈编号

# ============================================================
# 电压程序
# ============================================================
# Emap['acc_voltage_start'] = 100e3    # 涂抹电压
# Emap['acc_voltage_end'] = 300e3      # 加速电压
# Emap['voltage_ramp_start'] = 1000    # 前1000圈固定为 acc_voltage_start
# Emap['voltage_ramp_end'] = 1100      # 1000~1100圈线性过渡到 acc_voltage_end

Emap['acc_voltage_start'] = 100e3      # 涂抹电压
Emap['acc_voltage_end'] = 100e3        # 加速电压
Emap['voltage_ramp_start'] = 1000      # 加速电压线性变化开始圈编号
Emap['voltage_ramp_end'] = 1100        # 加速电压线性变化结束圈编号

# 相位说明：
#   涂抹阶段 (0 ~ phase_ramp_start)
#        加速相位固定为 acc_phase_start
#   过渡阶段 (phase_ramp_start ~ phase_ramp_end)
#        相位线性变化，从 acc_phase_start → acc_phase_end
#   加速阶段 (phase_ramp_end 以后)
#        加速相位固定为 acc_phase_end
#
# 电压说明：
#   涂抹阶段 (0 ~ voltage_ramp_start)
#        加速电压固定为 acc_voltage_start
#   过渡阶段 (voltage_ramp_start ~ voltage_ramp_end)
#        电压线性变化，从 acc_voltage_start → acc_voltage_end
#   加速阶段 (voltage_ramp_end 以后)
#        加速电压固定为 acc_voltage_end
#
# 注：上述 ramp 参数在配置文件中按“圈编号”给出；
#     程序内部若每圈存在多个 gap，会自动乘以 Ngap，
#     转换为对应的 gap-event 编号。

# ============================================================
# 多加速间隙设置
# ============================================================
Emap['harmonic'] = 1                  # RF 谐波数 h
Emap['Ngap'] = 2                      # 加速间隙个数
# ============================================================
# gap 模型选择
# ============================================================
# 'fixed'  : 旧代码模型，gap 方位固定为 gap_azimuth
# 'spiral' : gap 位置由 SEO 闭轨与基准 spiral 线交点确定，
#            其余 gap 按 Ngap 在整圈上等角间隔旋转复制得到
Emap['gap_model'] = 'spiral'

# ============================================================
# fixed gap 参数
# ============================================================
Emap['gap_azimuth'] = [90, ]       # fixed 模型下各 gap 的固定方位角（度）
Emap['gap_width'] = 1.0                 # 加速间隙宽度 [m]
Emap['rmin'] = 10.0                     # fixed 模型下电场有效最小半径 [m]
Emap['rmax'] = 12.5                     # fixed 模型下电场有效最大半径 [m]

# ============================================================
# spiral gap 参数（频率曲线 + spiral map 共用）
# ============================================================
# spiral 模型：
#     r(phi) = r_ref * exp((phi - s_ref)/tan(alpha))
#
# 说明：
#   spiral_alpha_deg : spiral 角 alpha（度）
#   spiral_r_ref     : 参考半径 r_ref（m）
#   spiral_s_ref_deg : 参考角 s_ref（度）
#   spiral_Nphi      : 在频率曲线求交时使用的 phi 扫描点数
Emap['spiral_alpha_deg'] = 50.00
Emap['spiral_r_ref'] = 8.5
Emap['spiral_s_ref_deg'] = 20.0
Emap['spiral_Nphi'] = 4000

# 高斯型 gap 场参数
Emap['spiral_U'] = 1.0                  # 归一化幅值
Emap['spiral_FWHM'] = 0.60              # 沿法向高斯分布的 FWHM [m]
# spiral map 计算范围 / 保存范围
# 说明：
#   spiral_rmin_calc    : 生成 spiral map 时实际计算半径下限
#   spiral_rmax_calc    : 生成 spiral map 时实际计算半径上限
#   spiral_rmin_save    : 最终保存到 Er/Ez/Ephi 表格中的半径下限
#   spiral_rmax_save    : 最终保存到 Er/Ez/Ephi 表格中的半径上限
Emap['spiral_rmin_calc'] = 7.5
Emap['spiral_rmax_calc'] = 14.0
Emap['spiral_rmin_save'] = 8.5
Emap['spiral_rmax_save'] = 13.0

# 最终保存到 Er/Ez/Ephi 表格中的网格参数
Emap['spiral_nr'] = 1001                # r 方向网格点数
Emap['spiral_nphi_full'] = 1441         # 用于估算角步长的整圈分辨率
Emap['spiral_pad_deg'] = 5.0            # 在 spiral 角窗口两端额外扩展的角度（度）
Emap['spiral_min_nphi'] = 64            # 最小角向网格点数

# 数值参数
Emap['spiral_newton_iter'] = 10         # 最近点搜索的 Newton 迭代次数
Emap['spiral_normal_side'] = 1          # 法向方向：+1 左法向，-1 右法向

# ============================================================
# spiral map 生成参数
# ============================================================
# 输出目录与文件名
Emap['emap_save_dir'] = './Emaps_test'
Emap['Er_map_file'] = 'Er_coef.txt'
Emap['Ez_map_file'] = 'Ez_coef.txt'
Emap['Ephi_map_file'] = 'Ephi_coef.txt'

# 多加速间隙相位说明：
# 1. 当前程序采用“公共同步相位程序 + 各 gap 自动相位偏移”的形式：
#        φ_eff(i, k) = φ_sync(k) + Δφ_i
#    其中：
#        φ_sync(k) : 第 k 个 gap-event 的公共同步相位程序
#        Δφ_i      : 第 i 个 gap 的固定附加相位偏移, 消除gap位置的影响，使得粒子穿越gaps时，在同一相位附近加速
#
# 2. Δφ_i 不再由用户手动输入，而是由程序根据谐波数 h 和 gap 个数 Ngap 自动生成：
#        Δφ_i = 2π * h * i / Ngap
#    若使用角度制，则为：
#        Δφ_i(deg) = 360 * h * i / Ngap
#    其中 i = 0, 1, ..., Ngap-1
#
# 3. 例如：
#    - Ngap = 2, h = 1  -> [0°, 180°]
#    - Ngap = 3, h = 1  -> [0°, 120°, 240°]
#    - Ngap = 4, h = 2  -> [0°, 180°, 360°, 540°]，按 360° 周期等效
#
# 4. 对应的第 i 个 gap 电场相位可写为：
#        E_i(t) = E0 * sin(2π * f_rf * t + φ_sync + Δφ_i)
#
# 5. 注意：
#    当前代码中的 Δφ_i 是每个 gap 的固定常数偏移，
#    不是每个 gap 各自独立、且随圈数变化的相位程序。

# ============================================================
# 曲线格式参数
# ============================================================
Emap['t_start'] = -200e-9               # 起始时间 [s]；请确保小于等于 0，避免时间序列混乱
Emap['num_turns'] = 9000                # 最大圈数，覆盖实际模拟圈数

config_data['config'] = config
config_data['machine'] = machine
config_data['Emap'] = Emap

###############################################################
# ------------------- Write JSON file -------------------------
###############################################################
if __name__ == "__main__":
    # 输出文件名与脚本同名
    current_file_name = os.path.splitext(os.path.basename(__file__))[0] + ".json"

    with open(current_file_name, 'w', encoding='utf-8') as f_out:
        json.dump(config_data, f_out, indent=2, sort_keys=True)
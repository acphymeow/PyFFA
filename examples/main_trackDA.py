import argparse
import os
import json
import numpy as np
from numba import set_num_threads
# set_num_threads(4)

# 注意：这里多加了 GetInitRAndFiForInjection, LoadSEOBetaFunc 的 import
from FFAG_Distribution import (
    GenerateBunches,
    LoadSEOParams,
    GetInitRAndFiForInjection,
    LoadSEOBetaFunc,
)
from FFAG_track import FFAG_RungeKutta, FunctionForAccelerationBunch_dt
from FFAG_ParasAndConversion import FFAG_GlobalParameters, FFAG_ConversionTools
from FFAG_Field import FFAG_EField_new, FFAG_Bfield_analytical, sigmoid_nth_derivative, sigmoid
from FFAG_Bunch import FFAG_Bunch, FFAG_ManageBunchAttribute
from FFAG_dump import StepDump, StepDumpBunch, PositionDump, Dumps, PositionDumpBunch
import matplotlib.pyplot as plt
import pyfftw


# =========================
#   生成 DA 扫描束团的函数
# =========================
def GenerateDAStudyBunch(BunchPara, DAScanPara):
    """
    生成用于动态孔径(DA)研究的粒子束。
    这个束团由多个单粒子组成，每个粒子代表一个特定的初始发射度。
    *** 水平 / 垂直发射度相同；支持非线性间隔，使大发射度更密集 ***

    参数:
    - BunchPara: dict
        包含通用束团参数的字典 (与 GenerateBunches 中使用的类似)。
    - DAScanPara: dict
        例如:
        {
            'emittance_start': 10,     # [pi mm mrad]
            'emittance_stop': 1000,    # [pi mm mrad]
            'num_particles': 20,
            'spacing_power': 0.5       # <1: 大发射度更密集
        }

    返回:
    - particles: numpy.ndarray
        生成的粒子束，每行一个粒子，格式为(r, pr, z, pz, ...)。
    """
    # 1. 从 BunchPara 提取通用参数
    SEOFileName       = BunchPara["SEO"]
    InjectionPosition = BunchPara["InjPosition"]
    Ek_c              = BunchPara["InjectEk"]
    InjTimeNanoSec    = BunchPara["InjTimeNanoSec"]

    # 2. 加载 SEO 数据和 Twiss 参数
    SEOiniFileName = os.path.join(SEOFileName, "SEO_ini.txt")
    SEO_R_FileName = os.path.join(SEOFileName, "SEO_r.txt")
    SEO_Pr_FileName = os.path.join(SEOFileName, "SEO_pr.txt")

    SEO_ini, _, _ = LoadSEOParams(SEOiniFileName, Ek_c)
    SEO_R  = np.loadtxt(SEO_R_FileName)
    SEO_Pr = np.loadtxt(SEO_Pr_FileName)

    if isinstance(InjectionPosition, str):
        InjectionCL = np.loadtxt(InjectionPosition, skiprows=1)
        fi_c, r_c, pr_c = GetInitRAndFiForInjection(SEO_R, SEO_Pr, InjectionCL, Ek_c)
    else:  # float: 直接用角度
        fi_c, r_c, pr_c = GetInitRAndFiForInjection(SEO_R, SEO_Pr, InjectionPosition, Ek_c)

    BetaZ_inj, BetaR_inj, AlphaZ_inj, AlphaR_inj = LoadSEOBetaFunc(SEOFileName, Ek_c, fi_c)

    # 3. 创建发射度扫描范围 (非线性)
    start_emit    = DAScanPara['emittance_start']   # [pi mm mrad]
    stop_emit     = DAScanPara['emittance_stop']    # [pi mm mrad]
    num_particles = DAScanPara['num_particles']
    power         = DAScanPara.get('spacing_power', 1.0)

    # 步骤 a: 生成一个从 0 到 1 的归一化线性空间
    normalized_space = np.linspace(0.0, 1.0, num_particles)

    # 步骤 b: 应用幂函数使其非线性化
    # power < 1: s^power > s (0<s<1)，点往右挤，大发射度更密集
    powered_space = normalized_space ** power

    # 步骤 c: 将非线性化的 [0,1] 空间映射到实际发射度范围
    emit_range = stop_emit - start_emit
    emit_values_mm_mrad = start_emit + powered_space * emit_range  # [pi mm mrad]

    # 单位转换: pi*mm*mrad -> pi*m*rad
    emit_values_m_rad = emit_values_mm_mrad / 1e6

    # 4. 为每个发射度生成一个粒子（水平 / 垂直发射度相同）
    particle_list   = []
    attr_mgr        = FFAG_ManageBunchAttribute()
    num_attributes  = attr_mgr.get_num_attributes()
    attr            = attr_mgr.Attribute

    for i, emit in enumerate(emit_values_m_rad):
        # 根据 Twiss 参数和当前发射度，计算水平和垂直方向的最大位移
        r_offset = np.sqrt(BetaR_inj * emit)
        z_offset = np.sqrt(BetaZ_inj * emit)

        # 根据 Twiss 参数计算匹配的动量偏移
        pr_offset = - (AlphaR_inj / BetaR_inj) * r_offset
        pz_offset = - (AlphaZ_inj / BetaZ_inj) * z_offset

        # 创建一个符合程序格式的粒子状态向量
        particle = np.zeros(num_attributes)

        # 水平方向 (r, pr)
        particle[attr['r']]  = r_c + r_offset
        particle[attr['vr']] = pr_c + pr_offset

        # 垂直方向 (z, pz)
        particle[attr['z']]  = z_offset
        particle[attr['vz']] = pz_offset

        # 其它自由度：相位 / 能量 / 注入时间
        particle[attr['fi']]   = fi_c
        particle[attr['Ek']]   = Ek_c
        particle[attr['inj_t']] = InjTimeNanoSec

        # ID
        particle[attr['Bunch_ID']]  = i
        particle[attr['Local_ID']]  = i
        particle[attr['Global_ID']] = i

        particle_list.append(particle)

    bunch_p_coords = np.array(particle_list)
    return bunch_p_coords


# 从 JSON 文件读取配置
def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config


# =========================
#           主函数
# =========================
def main(config_file, restart_bunch=None):
    # 读取 JSON 配置
    config = load_config_from_json(config_file)

    # 从配置中获取参数
    BmapPATHName = config['BmapAndSEO']['maps']
    max_order    = config['BmapAndSEO']['max_order']
    SEOPATHName  = os.path.join(BmapPATHName, 'resultsSEO')
    EmapPATHName = config['Emap']['maps']
    Ek_MeV       = config['track']['start_EkMeV']
    Azimuth_start = config['track']['start_azimuth']
    check        = config['check']

    # 粒子束参数
    PaintPara       = config['Paint']
    PaintEnable     = PaintPara['enable']
    PaintMaxNum     = PaintPara['MaxBunchNum']
    PaintTimeInterval = PaintPara['TimeInterval']
    PaintCurve      = PaintPara['Curve']

    BunchPara = config['BunchPara']

    # 将 check 中的键值映射到 BunchPara 中
    BunchPara['PlotFlag']     = check['CheckBunch']
    BunchPara['PlotRMS']      = check['PlotRMS']
    BunchPara['BmapPATHName'] = BmapPATHName
    BunchPara['SEO']          = SEOPATHName
    BunchPara["InjectEk"]     = Ek_MeV
    BunchPara["InjPosition"]  = Azimuth_start

    # 将 Paint 中的键值映射到 BunchPara 中
    BunchPara["PaintEnable"]       = PaintEnable
    BunchPara["PaintMaxNum"]       = PaintMaxNum
    BunchPara["PaintTimeInterval"] = PaintTimeInterval
    BunchPara["PaintCurve"]        = PaintCurve

    # 将 Emap 中的键值映射到 BunchPara 中
    BunchPara["EmapPATHName"] = EmapPATHName

    # =========================
    #  生成粒子束坐标分布
    # =========================
    if restart_bunch is not None:
        # 读取输入分布, 从断点继续传输
        if not os.path.exists(restart_bunch):
            raise FileNotFoundError(f"Error: restart_bunch file '{restart_bunch}' not found.")
        print(f"Loading restart bunch from {restart_bunch}")

        # 判断文件类型
        if restart_bunch.endswith('.npz'):
            array_dist_p = np.load(restart_bunch)['particles']
        else:
            array_dist_p = np.loadtxt(restart_bunch, delimiter=',')

        array_dist_v       = FFAG_ConversionTools().ConvertPrzek2Vrzek(array_dist_p)
        array_dist_v_boris = FFAG_ConversionTools().ConvertPrzek2Vrzek_boris(array_dist_p)

    else:
        # ---------- ★ DA 扫描设置：直接在脚本中定义 ★ ----------
        # 单位：发射度 emittance_* 是几何发射度 [pi mm mrad]
        DAScanPara = {
            "emittance_start": 10.0,     # 最小发射度
            "emittance_stop":  2000.0,   # 最大发射度
            "num_particles":   500,       # 发射度采样点数（单粒子数）
            "spacing_power":   0.9,      # <1：大发射度更密，小发射度更稀
        }

        # 要确保 BunchPara 里有注入时间（若 JSON 中没有，可以在此设一个默认值）
        if "InjTimeNanoSec" not in BunchPara:
            BunchPara["InjTimeNanoSec"] = 0.0  # ns，按需调整

        print("--- Generating DA-study bunch (same εx, εz; dense at large ε) ---")
        array_dist_p = GenerateDAStudyBunch(BunchPara, DAScanPara)

        # 这里所有粒子的 inj_t 一样，按 inj_t 排序不会改变顺序，但保持和普通模式一致
        inj_time_idx   = FFAG_ManageBunchAttribute().Attribute['inj_t']
        sorted_indices = np.argsort(array_dist_p[:, inj_time_idx])
        array_dist_p   = array_dist_p[sorted_indices]

        # 坐标转换
        array_dist_v       = FFAG_ConversionTools().ConvertPrzek2Vrzek(array_dist_p)
        array_dist_v_boris = FFAG_ConversionTools().ConvertPrzek2Vrzek_boris(array_dist_p)

        # 分配 Global_ID
        gid  = np.arange(array_dist_p.shape[0])
        attr = FFAG_ManageBunchAttribute().Attribute
        array_dist_v[:,       attr['Global_ID']] = gid
        array_dist_v_boris[:, attr['Global_ID']] = gid

    # =========================
    # StepDump / PositionDump 配置
    # =========================
    dumps_manager = Dumps()
    for dump_config in config['DumpPara']['modules']:
        if dump_config['type'] == 'StepDump':
            step_dump = StepDump(
                dump_config['start_time'],
                dump_config['end_time'],
                dump_config['interval_time'],
                dump_config['tracked_particle_id_global'],
                save_folder=dump_config['save_folder']
            )
            dumps_manager.add_dump(step_dump)
        elif dump_config['type'] == 'PositionDump':
            for dump_azimuth in dump_config['dump_azimuth']:
                position_dump = PositionDump(
                    dump_azimuth,
                    dump_config['start_time'],
                    dump_config['end_time'],
                    dump_config['num_particles_to_dump_global'],
                    save_folder=dump_config['save_folder']
                )
                dumps_manager.add_dump(position_dump)
        elif dump_config['type'] == 'StepDumpBunch':
            bunch_dump = StepDumpBunch(
                dump_config['start_time'],
                dump_config['end_time'],
                dump_config['interval_time'],
                save_folder=dump_config['save_folder']
            )
            dumps_manager.add_dump(bunch_dump)
        elif dump_config['type'] == 'PositionDumpBunch':
            for dump_azimuth in dump_config['dump_azimuth']:
                bunch_dump = PositionDumpBunch(
                    dump_azimuth,
                    dump_config['start_time'],
                    dump_config['end_time'],
                    save_folder=dump_config['save_folder']
                )
                dumps_manager.add_dump(bunch_dump)

    # =========================
    #          场 & SEO
    # =========================
    SEO_inj, BmapPath, SEO_info = LoadSEOParams(os.path.join(SEOPATHName, "SEO_ini.txt"), Ek_MeV)
    SEO_r_info = np.loadtxt(os.path.join(SEOPATHName, "SEO_r.txt"))
    SEO_fi_axis, SEO_Ek_axis, SEO_r_matrix = SEO_r_info[0, 1:], SEO_r_info[1:, 0], SEO_r_info[1:, 1:]

    GlobalParameters = FFAG_GlobalParameters()
    BMap = FFAG_Bfield_analytical(os.path.join(BmapPath, "config_Bmap.json"), max_order)
    EMap = FFAG_EField_new(EmapPATHName, config['Emap']['enable'])

    GlobalParameters.AddBMap(BMap)
    GlobalParameters.AddEMap(EMap)
    GlobalParameters.AddSEOInfo(SEO_info)
    if config['Aperture']['enable']:
        GlobalParameters.AddAperture(
            config['Aperture']['Rmin'], config['Aperture']['Rmax'],
            config['Aperture']['Zmin'], config['Aperture']['Zmax']
        )

    # =========================
    #     跟踪参数 & 起始相位
    # =========================
    t_start = -200.0 * 1e-9
    LocalParameters = dict()
    LocalParameters['stop_condition'] = config['track']['stop_condition']
    LocalParameters['time_step']      = config['track']['time_step']
    LocalParameters['step_dumps']     = dumps_manager
    LocalParameters['enable_SC']      = config['SC']['enable_SC']
    LocalParameters['SC_type']        = config['SC']['SC_type']
    SC_grid_size = tuple(config['SC']['grid_size_rfz'])
    LocalParameters['SC_step']        = config['SC']['step']
    solver_type = config['track']['solver_type']

    if restart_bunch is not None:
        t_start = array_dist_p[0, -1]
    else:
        inj_phase = (EMap.acc_phi0
                     - (EMap.gap_azimuths[0] - Azimuth_start) * EMap.harmonic
                     - (0 - t_start) / SEO_inj[3] * (np.pi * 2) * EMap.harmonic)
        # RF 相位写进数组
        array_dist_v[:, 9]       = inj_phase
        array_dist_v_boris[:, 9] = inj_phase

    # =========================
    #       创建求解器并跟踪
    # =========================
    rk = FFAG_RungeKutta()

    if solver_type == 0:
        BunchObj = FFAG_Bunch(array_dist_v, marcosize=BunchPara['ParticleDensity'], sc_grid_size=SC_grid_size)
        BunchObj.SEO_fi_axis   = SEO_fi_axis
        BunchObj.SEO_Ek_axis   = SEO_Ek_axis
        BunchObj.SEO_r_matrix  = SEO_r_matrix
        BunchObj.SEO_perimeter = SEO_info[:, 8]
        if restart_bunch is not None:
            BunchObj.SurvivedNum_Local = np.sum(BunchObj.LocalBunch[:, 8] == 1)
        rk.rk4_solve_dt_bunch3_withSC(
            FunctionForAccelerationBunch_dt,
            t_start, BunchObj,
            GlobalParameters, LocalParameters
        )

    elif solver_type == 1:
        BunchObj = FFAG_Bunch(array_dist_v_boris, marcosize=BunchPara['ParticleDensity'])
        BunchObj.SEO_fi_axis   = SEO_fi_axis
        BunchObj.SEO_Ek_axis   = SEO_Ek_axis
        BunchObj.SEO_r_matrix  = SEO_r_matrix
        BunchObj.SEO_perimeter = SEO_info[:, 8]
        if restart_bunch is not None:
            BunchObj.SurvivedNum_Local = np.sum(BunchObj.LocalBunch[:, 8] == 1)
        rk.rk4_solve_dt_bunch3_withSC_boris(
            FunctionForAccelerationBunch_dt,
            t_start, BunchObj,
            GlobalParameters, LocalParameters
        )
    else:
        raise ValueError("solver_type must be 0 or 1")


if __name__ == "__main__":
    # 默认使用 config_trackDA.json
    main("./config_trackDA.json")

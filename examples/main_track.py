import argparse
import os
import json
import numpy as np
from numba import set_num_threads
set_num_threads(1)
from FFAG_Distribution import GenerateBunches, LoadSEOParams
from FFAG_track import FFAG_RungeKutta, FunctionForAccelerationBunch_dt
from FFAG_ParasAndConversion import FFAG_GlobalParameters, FFAG_ConversionTools
from FFAG_Field import FFAG_EField_new, FFAG_Bfield_analytical, FFAG_EField_spiral, FFAG_BField_Error
from FFAG_Bunch import FFAG_Bunch, FFAG_ManageBunchAttribute
from FFAG_dump import StepDump, StepDumpBunch, PositionDump, Dumps, PositionDumpBunch
import matplotlib.pyplot as plt
import pyfftw


# 从 JSON 文件读取配置
def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config

def wrap_0_2pi(x):
    return x % (2.0 * np.pi)

def calc_injection_rf_phase(EMap, Azimuth_start, t_start, T0, Ek_inj, gap_idx=0):
    phi_start = wrap_0_2pi(Azimuth_start)

    if EMap.GapType == "fixed":
        phi_gap = EMap.gap_azimuths[gap_idx]
    elif EMap.GapType == "spiral":
        phi_gap0 = EMap.phi_gap_interp_unwrapped(Ek_inj)
        phi_gap = phi_gap0 + (EMap.gap_azimuths[gap_idx] - EMap.gap_azimuths[0])
    else:
        raise ValueError("EMap.type must be 'fixed' or 'spiral'")

    # phi_gap = wrap_0_2pi(phi_gap)
    # dphi_orbit = wrap_0_2pi(phi_gap - phi_start)
    dphi_orbit = phi_gap - phi_start

    inj_phase = (
        EMap.acc_phi0
        - EMap.harmonic * dphi_orbit
        - EMap.harmonic * 2.0 * np.pi * (0.0 - t_start) / T0
    )

    return inj_phase

# 主函数
def main(config_file, restart_bunch=None):
    # 读取 JSON 配置
    # config_file='./config_track.json'
    config = load_config_from_json(config_file)

    # 从配置中获取参数
    BmapPATHName = config['BmapAndSEO']['maps']
    max_order = config['BmapAndSEO']['max_order']
    SEOPATHName = os.path.join(BmapPATHName, config['BmapAndSEO']['SEO_FileName'])
    EmapPATHName = config['Emap']['maps']
    Ek_MeV = config['track']['start_EkMeV']
    Azimuth_start = config['track']['start_azimuth']
    check = config['check']

    # 粒子束参数
    PaintPara = config['Paint']
    PaintEnable = PaintPara['enable']
    PaintMaxNum = PaintPara['MaxBunchNum']
    PaintTimeInterval = PaintPara['TimeInterval']
    PaintCurve = PaintPara['Curve']

    # 粒子束参数
    BunchPara = config['BunchPara']

    # 将 check 中的键值映射到 BunchPara 中
    BunchPara['PlotFlag'] = check['CheckBunch']  # 对应 JSON 中的 check['CheckBunch']
    BunchPara['PlotRMS'] = check['PlotRMS']     # 对应 JSON 中的 check['PlotRMS']
    BunchPara['BmapPATHName'] = BmapPATHName
    BunchPara['SEO'] = SEOPATHName
    BunchPara["InjectEk"] = Ek_MeV
    BunchPara["InjPosition"] = Azimuth_start

    # 将 Paint 中的键值映射到 BunchPara中
    BunchPara["PaintEnable"] = PaintEnable
    BunchPara["PaintMaxNum"] = PaintMaxNum
    BunchPara["PaintTimeInterval"] = PaintTimeInterval
    BunchPara["PaintCurve"] = PaintCurve

    # 将Emap中的键值映射到BunchPara中
    BunchPara["EmapPATHName"] = EmapPATHName

    # 生成粒子束坐标分布
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

        # 获取 'inj_time' 在数组中的列索引
        inj_time_idx = FFAG_ManageBunchAttribute().Attribute['inj_t']
        # 对 array_dist_v 按照 inj_time 升序排序
        sorted_indices = np.argsort(array_dist_p[:, inj_time_idx])
        array_dist_p = array_dist_p[sorted_indices]  # 保持 array_dist_p 同步排序

        array_dist_v = FFAG_ConversionTools().ConvertPrzek2Vrzek(array_dist_p)
        array_dist_v_boris = FFAG_ConversionTools().ConvertPrzek2Vrzek_boris(array_dist_p)

    else:
        # 正常生成粒子束,从注入开始
        array_dist_p = GenerateBunches(BunchPara)

        # 获取 'inj_time' 在数组中的列索引
        inj_time_idx = FFAG_ManageBunchAttribute().Attribute['inj_t']

        # 对 array_dist_v 按照 inj_time 升序排序
        sorted_indices = np.argsort(array_dist_p[:, inj_time_idx])
        array_dist_p = array_dist_p[sorted_indices]  # 保持 array_dist_p 同步排序

        # # --------------------------------------------------------------
        # # 生成 22 个初始粒子：
        # # 11 个沿 x 方向线性分布（z=0）
        # # 11 个沿 z 方向线性分布（x=0）
        # # --------------------------------------------------------------
        #
        # x_min, x_max = -0.04, 0.04
        # z_min, z_max = -0.04, 0.04
        # n_x = 11
        # n_z = 11
        #
        # # 生成线性分布
        # x_offsets = np.linspace(x_min, x_max, n_x)  # 11 个 x
        # z_offsets = np.linspace(z_min, z_max, n_z)  # 11 个 z
        #
        # # 确保 array_dist_p 至少有 22 个粒子
        # assert array_dist_p.shape[0] >= (n_x + n_z), \
        #     f"array_dist_p 至少需要 {n_x + n_z} 个粒子！"
        #
        # # ==============================
        # # 前 11 个：沿 x 偏移，z=0
        # # ==============================
        # for i in range(n_x):
        #     pid = i
        #     array_dist_p[pid, 0] += x_offsets[i]  # r 方向偏移（假设你的 0 列是 x/r）
        #     # array_dist_p[pid, 2] = 0.0  # z 固定
        #     # 纵向/横向速度不变（保持初始相同动量）
        #
        # # ==============================
        # # 接下来的 11 个：沿 z 偏移，x=0
        # # ==============================
        # for j in range(n_z):
        #     pid = n_x + j
        #     # array_dist_p[pid, 0] = array_dist_p[0, 0]  # 同样的半径中心值
        #     array_dist_p[pid, 2] += z_offsets[j]  # z 偏移
        #     # 同样保持初始速度不变

        # array_dist_p=array_dist_p[:pid+1,:]

        array_dist_v = FFAG_ConversionTools().ConvertPrzek2Vrzek(array_dist_p)
        # P, Ek_MeV to v, Etot_J
        array_dist_v_boris = FFAG_ConversionTools().ConvertPrzek2Vrzek_boris(array_dist_p)
        # P, Ek_MeV to v, vf

        # 分配Global_ID
        array_dist_v[:, FFAG_ManageBunchAttribute().Attribute['Global_ID']] = np.arange(np.shape(array_dist_p)[0])
        array_dist_v_boris[:, FFAG_ManageBunchAttribute().Attribute['Global_ID']] = np.arange(np.shape(array_dist_p)[0])


    # plt.figure()
    # plt.scatter(array_dist_p[:, 0]*1000, array_dist_p[:, 1]*1000, s=2, color='b')
    # plt.xlabel("r (mm)")
    # plt.ylabel("pr (mrad)")

    # plt.figure()
    # plt.scatter(array_dist_p[:, 2]*1000, array_dist_p[:, 3]*1000, s=2, color='b')
    # plt.xlabel("z (mm)")
    # plt.ylabel("pz (mrad)")

    # plt.figure()
    # plt.scatter(array_dist_p[:, 0] * 1000, array_dist_p[:, 2] * 1000, s=2, color='b')
    # plt.xlabel("r (mm)")
    # plt.ylabel("z (mm)")

    # plt.figure()
    # plt.scatter(array_dist_p[:, 6], array_dist_p[:, 5], s=2)
    # plt.xlabel("injection time (ns)")
    # plt.ylabel("Ek (MeV)")
    # plt.xlim([-200,1000])


    # plt.show()

    # StepDump 配置
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
                save_folder=dump_config['save_folder'],
                filter_survive_flag=dump_config['filtter_survive_paticles']
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

    # Aperture参数

    # 加载 SEO 参数
    SEO_inj, BmapPath, SEO_info = LoadSEOParams(os.path.join(SEOPATHName, "SEO_ini.txt"), Ek_MeV)
    SEO_r_info = np.loadtxt(os.path.join(SEOPATHName, "SEO_r.txt"), skiprows=1)
    SEO_pr_info = np.loadtxt(os.path.join(SEOPATHName, "SEO_pr.txt"), skiprows=1)
    SEO_fi_axis, SEO_Ek_axis, SEO_r_matrix, SEO_pr_matrix = SEO_r_info[0, 1:], SEO_r_info[1:, 0], SEO_r_info[1:, 1:], SEO_pr_info[1:, 1:]

    GlobalParameters = FFAG_GlobalParameters()
    BMap = FFAG_Bfield_analytical(os.path.join(BmapPath,"config_Bmap.json"), max_order)

    # EMap = FFAG_EField_new(EmapPATHName, config['Emap']['enable'])
    header, _ = FFAG_EField_new.ParseEmapFile(EmapPATHName)
    gap_type = str(header["gap_model"]).strip().lower()
    if gap_type == "fixed":
        EMap = FFAG_EField_new(EmapPATHName, config['Emap']['enable'])
    elif gap_type == "spiral":
        EMap = FFAG_EField_spiral(EmapPATHName, config['Emap']['enable'])
    else:
        raise ValueError(f"Unknown Emap gap type: {gap_type!r}")

    with open (os.path.join(SEOPATHName, 'config_SEO.json'), "r") as SEOSetting:
        SEO_config = json.load(SEOSetting)
    BMapHarmonics = FFAG_BField_Error(SEO_config['BHarmonics']['harmonics'], SEO_config['BHarmonics']['enable'])

    GlobalParameters.AddBMap(BMap)
    GlobalParameters.AddEMap(EMap)
    GlobalParameters.AddSEOInfo(SEO_info)
    GlobalParameters.AddBHarmonics(BMapHarmonics)
    if config['Aperture']['enable']:
        GlobalParameters.AddAperture(config['Aperture']['Rmin'], config['Aperture']['Rmax'],
                                     config['Aperture']['Zmin'], config['Aperture']['Zmax'])

    # 初始化跟踪参数
    t_start = -200.0*1e-9
    LocalParameters = dict()
    LocalParameters['stop_condition'] = config['track']['stop_condition']
    LocalParameters['time_step'] = config['track']['time_step']
    LocalParameters['step_dumps'] = dumps_manager
    LocalParameters['enable_SC'] = config['SC']['enable_SC']
    LocalParameters['SC_type'] = config['SC']['SC_type']
    SC_grid_size = tuple(config['SC']['grid_size_rfz'])
    LocalParameters['SC_step'] = config['SC']['step']
    solver_type = config['track']['solver_type']

    if restart_bunch is not None:
        t_start = array_dist_p[0, -1]
    else:
        # inj_phase = (EMap.acc_phi0 -
        #              (EMap.gap_azimuths[0]-Azimuth_start) * EMap.harmonic -
        #              (0-t_start)/SEO_inj[3] * (np.pi*2) * EMap.harmonic)
        inj_phase = calc_injection_rf_phase(EMap,Azimuth_start,t_start,SEO_inj[3],SEO_inj[0],gap_idx=0)
        # 创建 FFAG_Bunch 对象并注入粒子
        array_dist_v[:, 9] = inj_phase
        array_dist_v_boris[:, 9] = inj_phase

        # array_dist_v_boris[:, 1] *= -1
        # array_dist_v_boris[:, 3] *= -1
        # array_dist_v_boris[:, 5] *= -1

    # 创建 Runge-Kutta 或 boris 求解器并运行跟踪
    rk = FFAG_RungeKutta()

    # # njit预热
    # #########################################################################
    # Bz_malloc0 = np.zeros(2000, dtype=np.float64)
    # Br_malloc0 = np.zeros(2000, dtype=np.float64)
    # Bf_malloc0 = np.zeros(2000, dtype=np.float64)
    # Ez_malloc0 = np.zeros(12000, dtype=np.float64)
    # Er_malloc0 = np.zeros(12000, dtype=np.float64)
    # Ef_malloc0 = np.zeros(12000, dtype=np.float64)
    # # Bz,_,_ = BMap.GetBfields(np.linspace(8.2,8.2,2000), np.linspace(0, np.pi/6,2000), np.linspace(0, 0.01,2000),np.ones(2000, dtype=bool),np.ones(2000, dtype=bool),SearchMod=False)
    # BMap.GetBfields(np.linspace(8.2,8.2,2000), np.linspace(0, np.pi/6,2000), np.linspace(0, 0.01,2000),
    #                 np.ones(2000, dtype=bool),np.ones(2000, dtype=bool),
    #                 Bz_malloc0, Br_malloc0, Bf_malloc0,
    #                 SearchMod=False)
    # EMap.Interpolation2D_EMap(np.linspace(8.2,8.2,12000), np.linspace(0, np.pi*2,12000), Ez_malloc0, Er_malloc0, Ef_malloc0)
    # #########################################################################

    if solver_type == 0:
        BunchObj = FFAG_Bunch(array_dist_v, marcosize=BunchPara['ParticleDensity'], sc_grid_size=SC_grid_size, BunchType='RK4')
        BunchObj.SEO_fi_axis = SEO_fi_axis
        BunchObj.SEO_Ek_axis = SEO_Ek_axis
        BunchObj.SEO_r_matrix = SEO_r_matrix
        BunchObj.SEO_pr_matrix = SEO_pr_matrix
        BunchObj.SEO_perimeter = SEO_info[:, 8]
        if restart_bunch is not None:
            # 恢复统计量
            BunchObj.restore_from_restart_matrix()
        rk.rk4_solve_dt_bunch3_withSC(FunctionForAccelerationBunch_dt,
                                      t_start, BunchObj,
                                      GlobalParameters, LocalParameters)

    elif solver_type == 1:
        BunchObj = FFAG_Bunch(array_dist_v_boris, marcosize=BunchPara['ParticleDensity'], sc_grid_size=SC_grid_size, BunchType='Boris')
        BunchObj.SEO_fi_axis = SEO_fi_axis
        BunchObj.SEO_Ek_axis = SEO_Ek_axis
        BunchObj.SEO_r_matrix = SEO_r_matrix
        BunchObj.SEO_pr_matrix = SEO_pr_matrix
        BunchObj.SEO_perimeter = SEO_info[:, 8]
        if restart_bunch is not None:
            # 恢复统计量
            BunchObj.restore_from_restart_matrix()
        rk.rk4_solve_dt_bunch3_withSC_boris(FunctionForAccelerationBunch_dt,
                                            t_start, BunchObj,
                                            GlobalParameters, LocalParameters)
    else:
        raise ValueError("solver_type must be 0 or 1")


if __name__ == "__main__":
    # main("./config_track.json", "./time_6452.56_rank_0.npz")
    main("./config_track.json")
    # # 设置命令行参数解析
    # parser = argparse.ArgumentParser(description="FFAG Simulation Tool")
    # parser.add_argument(
    #     "-j",
    #     type=str,
    #     required=True,
    #     help="Path to the JSON configuration file (e.g., config.json)"
    # )
    # parser.add_argument(
    #     "-r",
    #     type=str,
    #     default=None,
    #     help="Path to restart bunch file (optional)"
    # )
    #
    # # 解析命令行参数
    # args = parser.parse_args()
    #
    # # 调用主函数
    # main(args.j, args.r)


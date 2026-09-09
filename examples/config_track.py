import json
import time
import os
import numpy as np

cf = {}
###############################################################################
# 通用参数
config = dict()
config['date'] = time.time()  # 当前时间戳
config['MPI_process_num'] = 1
config['thread_num'] = 4

###############################################################################
# 跟踪参数
track = dict()
track['start_EkMeV'] = 300.0  # 初始动能 (MeV)
stop_condition = {'max_stepsN': 50000000,  # 最多跟踪多少step
                  'max_turn': 500.0, #  最多跟踪多少圈
                  'max_time': 358.475e-9 * 500.0}  #  最多跟踪多少时间(second)
track['time_start'] = -200.0*1e-9
track['time_step'] = 0.02e-9  #采用固定步长, 0.02ns --> 回旋周期约273ns --> 每圈约13650步, 每个cell约1137步
track['start_azimuth'] = 0.0  # 注入点方位角
track['stop_condition'] = stop_condition  # 3个终止条件满足任一个即终止
track['solver_type'] = 1  # 0--->RK4, 单步4阶精度但不保辛; 1--->Boris, 单步2阶精度但保辛, 速度稍快

###############################################################################
# SC参数
SC = dict()
SC['enable_SC'] = False  # True False 是否加空间电荷效应
SC['SC_type'] = 2
# SC_type=0: 直角坐标系网格+FFT(3d, 适用于小束团)
# SC_type=1: 柱坐标系网格+FFT(2.5d, 适用于长束团)
SC['grid_size_rfz'] = [128, 128, 128, 10]  # 网格尺寸, 直角坐标系为x y z, 柱坐标系为r z phi
SC['step'] = 20  # 多少步更新一次SC电场

###############################################################################
# 磁场 高阶非线性项 磁场谐波误差
BmapAndSEO = dict()
BmapAndSEO['maps'] = './Bmaps/OperaFieldMap'  # 磁场路径
# BmapAndSEO['maps'] = './Bmap_sp35'  # 磁场路径
BmapAndSEO['max_order'] = 0  # 展开阶数, n代表展开到(2*n+1阶) [0-->最高1阶, 1-->最高3阶, 2-->最高5阶]
#这里磁场谐波误差:
BmapAndSEO['SEO_FileName'] = 'resultsSEO'
# 1) 同一个Bmap下面有多组闭轨, 对应加入不同的磁场误差。每加一组磁场误差，可以算出相应的闭轨
# 2) 引入磁场误差 --> 用包含磁场误差的闭轨; 不加磁场误差 --> 用不含磁场误差的闭轨resultsSEO_5GsB4
# 3) 具体的磁场误差在config_SEO.py里面定义, 和闭轨数据一起包含在闭轨文件夹里resultsSEO_noError

###############################################################################
# 局部水平偏转引出 kicker（硬边界常量磁场模型）
# 区域内默认只叠加 Bz = sign * BL_Tm / L_eff_m；区域外为 0。
# 对当前正向环行的质子，负 Bz 产生向外的正径向偏转；Br=Bfi=0。
# phi 使用整圈全局方位角，不随主磁场按 sector 重复。
# 当前四块 EK 分布在 146.25 deg、168.75 deg 两个相邻直线节内，
# 每个直线节两块。按 600 MeV 质子闭轨的实际弧长布置：每块长 0.20 m，
# 同一直线节内两块之间留 0.20 m 净间隙；单块 5 mrad 对应 BL=0.0203305 T*m。
# 径向外边界取 OPERA map 的最大半径 12.2 m。
BumpKicker = dict()
BumpKicker['enable'] = True
BumpKicker['kickers'] = [
    {"name": "EK1", "phi_start_deg": 146.521143, "phi_span_deg": 0.971124,
     "BL_Tm": 0.0203305, "L_eff_m": 0.20,
     "rmin_m": 11.7, "rmax_m": 12.2, "sign": -1,
     "Br_T": 0.0, "Bfi_T": 0.0},
    {"name": "EK2", "phi_start_deg": 148.462148, "phi_span_deg": 0.968093,
     "BL_Tm": 0.0203305, "L_eff_m": 0.20,
     "rmin_m": 11.7, "rmax_m": 12.2, "sign": -1,
     "Br_T": 0.0, "Bfi_T": 0.0},
    {"name": "EK3", "phi_start_deg": 169.021176, "phi_span_deg": 0.971103,
     "BL_Tm": 0.0203305, "L_eff_m": 0.20,
     "rmin_m": 11.7, "rmax_m": 12.2, "sign": -1,
     "Br_T": 0.0, "Bfi_T": 0.0},
    {"name": "EK4", "phi_start_deg": 170.962138, "phi_span_deg": 0.968071,
     "BL_Tm": 0.0203305, "L_eff_m": 0.20,
     "rmin_m": 11.7, "rmax_m": 12.2, "sign": -1,
     "Br_T": 0.0, "Bfi_T": 0.0},
]

###############################################################################
# 电场配置
Emap = dict()
# Emap['maps'] = './Emap/FrequencyCurve_Lpaint_freqPaint.txt'  # 电场频率曲线(文件表头包含了电压，加速相位，加速腔位置等参数)
Emap['maps'] = './Emaps_test/FrequencyCurve_noRF.txt'
Emap['enable'] = False    # True False

###############################################################################
# 注入小束团参数
BunchPara = dict()
BunchPara['ParticleDensity'] = 0.25*1e11  # 每个宏粒子对应的实际粒子数
BunchPara['ParticleNum'] = 20000  # 每个子束团的宏粒子数
BunchPara['TransverseREmit'] = 18.0
BunchPara['TransverseZEmit'] = 18.0
# 这里的横向发射度是指横向RMS发射度=a, 对应1倍RMS椭圆面积为a pi.mm.mrad. 它和实际边界的关系为：
# 如果是kv分布，100%发射度椭圆=4RMS椭圆, 即kv分布的边界为发射度4a的椭圆；
# 如果是waterbag分布，100%发射度椭圆=6RMS椭圆, 即waterbag分布的边界为发射度6a的椭圆；
# 如果是gauss分布，86%发射度椭圆=4RMS椭圆, 即gauss分布的86%边界为发射度4a的椭圆；
BunchPara['LongitudeT'] = 72  # 纵向长度 (ns) waterbag: 100%包络椭圆的轴长。 gauss: 4RMS椭圆的轴长.72
BunchPara['LongitudeDEk'] = 8  # 纵向能散 (MeV) waterbag: 100%包络椭圆的轴长。 gauss: 4RMS椭圆的轴长.10
# 这里的纵向发射度是指实际边界，因为纵向没有beta alpha等数值，用边界值比较直观
# 纵向时间宽度和能散的实际最大宽度（guass为4RMS椭圆宽度）
BunchPara['InjTimeNanoSec'] = 0  # 初始注入时刻 (ns)
BunchPara['TransverseDistType'] = 'gauss'  # 横向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag')
BunchPara['LongitudeDistType'] = 'waterbag'
# 纵向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag', 'match', 'UniformGauss')

###############################################################################
# 涂抹
Paint = dict()
Paint['enable'] = False    # True False
Paint['MaxBunchNum'] = 100  # 注入子束团个数
Paint['TimeInterval'] = 358.475  # 时间间隔ns
Paint['Curve'] = './PaintCurves/Curve_linear_reverse.paint' # 涂抹曲线包含5列，第一列为时间，后面为小束团在相空间的偏移量

###############################################################################
# 孔径
Aperture = dict()
Aperture['enable'] = True    # True False
Aperture['Rmin'] = 10.0
Aperture['Rmax'] = 12.2
Aperture['Zmin'] = -0.10
Aperture['Zmax'] = 0.10

# ###############################################################################
# # field error
# BHarmonics = dict()
# BHarmonics['enable'] = False   # True False
# harmonic_1 = {
#     "m": 3, "amp_Gs": 2.0, "phase_deg": 30.0,
#     "rmin": 10.8, "rmax": 12.0,}
# harmonic_2 = {
#     "m": 2, "amp_Gs": 5.0, "phase_deg": 30.0,
#     "rmin": 11.2, "rmax": 11.4,}
# harmonic_3 = {
#     "m": 3,"amp_Gs": 12.0,"phase_deg": 30.0,
#     "rmin": 11.2,"rmax": 11.4,}
# BHarmonics["harmonics"] = [harmonic_1, ]


###############################################################################
# ------------------- 自动生成 TBT_Dump 配置 -------------------
DumpPara = dict()
TBT_Dump_list = []
start_records = list(range(0, 150, 10))   # 11, 111, 211, ... 直到3401
finish_records = [s + 2 for s in start_records]

for i, (start_record, finish_record) in enumerate(zip(start_records, finish_records)):
    dump = {
        "type": "PositionDumpBunch",
        "start_time": Paint['TimeInterval'] * 1e-9 * start_record,
        "end_time": Paint['TimeInterval'] * 1e-9 * finish_record,
        "dump_azimuth": [30.0, ],
        "save_folder": f"./output/simulation512_sc/TBT_Dump_{start_record}",
    }
    TBT_Dump_list.append(dump)

start_record, finish_record, num_record = 0.0, 450, 2000
FFT_Tune_Dump_0 = {
    "type": "PositionDump",  # turn by turn, 保存每个粒子为一个单独文件
    "start_time": Paint['TimeInterval']*1e-9 * start_record,
    "end_time": Paint['TimeInterval']*1e-9 * finish_record,
    "num_particles_to_dump_global": list(range(num_record)),  # 要保存粒子的ID
    "dump_azimuth": [0.1, ],  # 方位角 (度)
    "save_folder": "./output/simulation512_sc/FFT_Tune_Dump_0",  # 保存路径
}

# StepDumpBunch:
start_record, finish_record = 1.0, 3.0
RestartDump = {
    "type": "StepDumpBunch",  # 时间快照, 保存每个位置的bunch为一个单独文件
    "start_time": Paint['TimeInterval']*1e-9 * start_record - 350.0*1e-9/8,
    "end_time": Paint['TimeInterval']*1e-9 * finish_record,
    "interval_time": Paint['TimeInterval']*1e-9 * 1,
    "save_folder": "./output/simulation_paper_550/RestartDump",  # 保存路径
    "filtter_survive_paticles": True  # 保存所有粒子False or 保存存活粒子True
}

StepDump = {
    "type": "StepDump",  # 按时间间隔保存粒子 每个粒子单独保存为一个文件 按ID区分 可用于绘制单粒子轨迹
    # 采用追加写入的方式 且频繁读写 速度很慢 数据量大占用空间 仅能短时间使用
    "start_time": -1.02e-9, # 开始时间
    "end_time": 200.35e-9 * 1.0, # 终止时间
    "interval_time": 1.0e-9,  # 间隔多少时间保存一次
    "tracked_particle_id_global": list(range(10)),  # 全局保存的粒子数
    "save_folder": "./output/simulation_testdump2/FFT_Tune_Dump_0"  # 保存路径
}
# 汇总
DumpPara['modules'] = [FFT_Tune_Dump_0,*TBT_Dump_list,
]
# DumpPara['modules'] = [StepDump,
# ]



###############################################################################
# debug
check = dict()
check['CheckBunch'] = False # True False
check['PlotRMS'] = (1, 1, 4) # Gauss: (4, 4, 4), water bag:(6, 6, 2)

###############################################################################
# 整体配置
cf['config'] = config
cf['track'] = track
cf['SC'] = SC
cf['BmapAndSEO'] = BmapAndSEO
cf['BunchPara'] = BunchPara
cf['DumpPara'] = DumpPara
cf['check'] = check
cf['Emap'] = Emap
cf['Paint'] = Paint
cf['Aperture'] = Aperture
cf['BumpKicker'] = BumpKicker
# cf['BHarmonics'] = BHarmonics


# 保存 JSON 配置文件
if __name__ == "__main__":
    current_file_name = os.path.splitext(os.path.basename(__file__))[0] + ".json"
    with open(current_file_name, 'w') as f_out:
        json.dump(cf, f_out, indent=2, sort_keys=True)
    print(f"配置文件已保存为 {current_file_name}")

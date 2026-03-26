import os
import json
import time
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
stop_condition = {'max_stepsN': 90000000,  # 最多跟踪多少step
                  'max_turn': 4000.0, #  最多跟踪多少圈
                  'max_time': 365.64994e-9 * 4000.0}  #  最多跟踪多少时间(second)
track['time_start'] = -200.0*1e-9
track['time_step'] = 0.02e-9  #采用固定步长, 0.02ns --> 回旋周期约273ns --> 每圈约13650步, 每个cell约1137步
track['start_azimuth'] = 0.0  # 注入点方位角
track['stop_condition'] = stop_condition  # 3个终止条件满足任一个即终止
track['solver_type'] = 1  # 0--->RK4, 单步4阶精度但不保辛; 1--->Boris, 单步2阶精度但保辛, 速度稍快

###############################################################################
# SC参数
SC = dict()
SC['enable_SC'] = True  # True False 是否加空间电荷效应
SC['SC_type'] = 1
# SC_type=0: 直角坐标系网格+FFT(3d, 适用于小束团)
# SC_type=1: 柱坐标系网格+FFT(2.5d, 适用于长束团)
SC['grid_size_rfz'] = [128, 128, 128, 12]  # 网格尺寸, 直角坐标系为x y z, 柱坐标系为  phi r  z
SC['step'] = 20  # 多少步更新一次SC电场

###############################################################################
# 磁场和高阶非线性项配置
BmapAndSEO = dict()
BmapAndSEO['maps'] = '../Bmaps/Bmap_FD16'  # 磁场路径
BmapAndSEO['max_order'] = 2  # 展开阶数, n代表展开到(2*n+1阶) [0-->最高1阶, 1-->最高3阶, 2-->最高5阶]
#这里磁场谐波误差:
BmapAndSEO['SEO_FileName'] = 'resultsSEO_noError'
# 1) 同一个Bmap下面有多组闭轨, 对应加入不同的磁场误差。每加一组磁场误差，可以算出相应的闭轨
# 2) 引入磁场误差 --> 用包含磁场误差的闭轨; 不加磁场误差 --> 用不含磁场误差的闭轨
# 3) 具体的磁场误差在config_SEO.py里面定义, 和闭轨数据一起包含在闭轨文件夹里

###############################################################################
# 电场配置
Emap = dict()
Emap['maps'] = './Emap/FrequencyCurve_ramp30_1000_1000_varV.txt'  # 电场频率曲线(文件表头包含了电压，加速相位，加速腔位置等参数)
Emap['enable'] = True    # True False

###############################################################################
# 注入小束团参数
BunchPara = dict()
BunchPara['ParticleDensity'] = 1.0*1e8  # 每个宏粒子对应的实际粒子数
BunchPara['ParticleNum'] = 200  # 每个子束团的宏粒子数
BunchPara['TransverseREmit'] = 0.3  # 横向发射度 (pi*mm*mrad)
BunchPara['TransverseZEmit'] = 0.3  # 横向发射度 (pi*mm*mrad)
# 这里的横向发射度是指横向RMS发射度=a, 对应1倍RMS椭圆面积为a pi.mm.mrad. 它和实际边界的关系为：
# 如果是kv分布，100%发射度椭圆=4RMS椭圆, 即kv分布的边界为发射度4a的椭圆；
# 如果是waterbag分布，100%发射度椭圆=6RMS椭圆, 即waterbag分布的边界为发射度6a的椭圆；
# 如果是gauss分布，86%发射度椭圆=4RMS椭圆, 即gauss分布的86%边界为发射度4a的椭圆；
BunchPara['LongitudeT'] = 72.0  # 纵向长度 (ns) waterbag: 100%包络椭圆的时间坐标轴的长度。 gauss: 4RMS椭圆时间坐标轴的长度
BunchPara['LongitudeDEk'] = 1.0  # 纵向能散 (MeV) waterbag: 100%包络椭圆的能散坐标轴的长度。 gauss: 4RMS椭圆能散坐标轴的长度
# 这里的纵向发射度是指实际边界，因为纵向没有beta alpha等数值，用边界值比较直观
# 纵向时间宽度和能散的实际最大宽度（guass为4RMS椭圆宽度）
BunchPara['InjTimeNanoSec'] = 0  # 初始注入时刻 (ns)
BunchPara['TransverseDistType'] = 'gauss'  # 横向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag')
BunchPara['LongitudeDistType'] = 'UniformGauss'  # 纵向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag', 'match')
# 纵向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag', 'match', 'UniformGauss')

###############################################################################
# 涂抹
Paint = dict()
Paint['enable'] = True    # True False
Paint['MaxBunchNum'] = 1000  # 注入子束团个数
Paint['TimeInterval'] = 365.64994  # 时间间隔ns 365.64994
Paint['Curve'] = '../PaintCurves/Curve_r_up_linear_pz_up_linear.paint' # 涂抹曲线包含5列，第一列为时间，后面为小束团在相空间的偏移量

###############################################################################
# 孔径
Aperture = dict()
Aperture['enable'] = True    # True False
Aperture['Rmin'] = 10.3
Aperture['Rmax'] = 12.3
Aperture['Zmin'] = -0.14
Aperture['Zmax'] = 0.14

###############################################################################
# Dump 配置
DumpPara = dict()
# StepDumpBunch:
start_record, finish_record = 1008.0, 1011.0
RestartDump = {
    "type": "StepDumpBunch",  # 时间快照, 保存每个位置的bunch为一个单独文件
    "start_time": Paint['TimeInterval']*1e-9 * start_record,
    "end_time": Paint['TimeInterval']*1e-9 * finish_record,
    "interval_time": Paint['TimeInterval']*1e-9 * 1.0,
    "save_folder": "./output/simulation1/RestartDump",  # 保存路径
    "filtter_survive_paticles": True  # 保存所有粒子False or 保存存活粒子True
}

start_record, finish_record, num_record = 1010.0, 1510.0, 50000
FFT_Tune_Dump_0 = {
    "type": "PositionDump",  # turn by turn, 保存每个粒子为一个单独文件
    "start_time": Paint['TimeInterval']*1e-9 * start_record,
    "end_time": Paint['TimeInterval']*1e-9 * finish_record,
    "num_particles_to_dump_global": list(range(num_record)),  # 要保存粒子的ID
    "dump_azimuth": [30.0, ],  # 方位角 (度)
    "save_folder": "./output/simulation1/FFT_Tune_Dump_0",  # 保存路径
}

# ------------------- 自动生成 TBT_Dump 配置 -------------------
TBT_Dump_list = []
start_records = list(range(11, 3911, 100))   # 11, 111, 211, ... 直到3401
finish_records = [s + 2 for s in start_records]

for i, (start_record, finish_record) in enumerate(zip(start_records, finish_records)):
    dump = {
        "type": "PositionDumpBunch",
        "start_time": Paint['TimeInterval'] * 1e-9 * start_record,
        "end_time": Paint['TimeInterval'] * 1e-9 * finish_record,
        "dump_azimuth": [90.0, ],
        "save_folder": f"./output/simulation1/TBT_Dump_{start_record}",
    }
    TBT_Dump_list.append(dump)

# 汇总
DumpPara['modules'] = [
    *TBT_Dump_list,
    FFT_Tune_Dump_0,
]

###############################################################################
# debug
check = dict()
check['CheckBunch'] = False # True False
check['PlotRMS'] = (1.5, 1.5, 4) # Gauss: (4, 4, 4), water bag:(1.5, 1.5, 4)

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


# 保存 JSON 配置文件
if __name__ == "__main__":
    current_file_name = os.path.splitext(os.path.basename(__file__))[0] + ".json"
    with open(current_file_name, 'w') as f_out:
        json.dump(cf, f_out, indent=2, sort_keys=True)
    print(f"配置文件已保存为 {current_file_name}")


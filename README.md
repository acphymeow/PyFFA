# FFAG程序使用手册

## 目录

- [1. 引言](#1-引言)
- [2. 命令行](#2-命令行)
- [3. config文件说明](#3-config文件说明)
  - [3.1 config_Bmap.py磁场配置文件](#31-config_bmappy磁场配置文件)
  - [3.2 config_SEO.py闭轨计算配置](#32-config_seopy闭轨计算配置)
  - [3.3 config_Emap.py电场配置](#33-config_Emappy电场配置)
  - [3.4 多粒子跟踪](#34-多粒子跟踪)
- [4. 输出文件格式与绘图说明](#4-输出文件格式与绘图说明)

## 1. 引言

### 1.1 简介

本程序用于模拟粒子在加速器电磁场map中和空间电荷效应下的三维运动轨迹，主要功能有：
- 以中平面磁场 map 文件作为输入，利用静磁场无源无旋+泰勒展开，将中平面磁场外推形成3维磁场。
- RF电场加速
- 运动求解器有两种可以选择：
  - 4阶 Runge-Kutta 方法：单步4阶精度，不保辛；
  - Boris 方法：单步2阶精度，保辛。
- 使用FFT/ iFFT 加速的 PIC 算法进行2.5维空间电荷效应计算。
- 涂抹注入。

### 1.2 依赖库

- Python >= 3.12
- OpenMP
- MPI
- NumPy (Python包)
- SciPy (Python包)
- Numba (Python包)
- mpi4py (Python包)
- matplotlib (Python包)
- pyfftw (Python包)

### 1.3 源代码

-HPC

-CSNS Gitlab平台


### 1.4 更新记录
-2025.04.14 
- initial submit。
- 完成基本模拟功能: 束流光学，多粒子跟踪，3维空间电荷效应，涂抹注入。

-2025.06.05 
- 增加2.5维空间电荷效应。
- 性能优化：为函数内所有临时数组变量提前预分配和复用内存，提升计算性能。
- 性能优化：scipy.fft替换成fftw。
- 并行方法优化：纯MPI多进程换成MPI多进程+numba.prange多线程混合模式，节点内用prange多线程，不复制内存；节点之间用MPI多线程通信，每个节点复制一次内存。

-2025.06当前工作：增加磁场误差模块、增加相稳定区计算模块、高阶磁场插值性能优化、全程跟踪测试

---


## 2. 命令行

程序**项目目录**中，`main_*.py` 文件为可执行程序，`config_*.py` 文件用于配置参数，其生成的 `config_*.json` 文件作为可执行程序的输入。config_*.py配置文件的具体格式在下一节说明。

所有命令行如下：

```bash
source ./SetEnvCmd.sh              # CSNS HPC服务器加载虚拟环境

python config_Bmap.py             # 生成磁场配置
python main_Bmap.py -j config_Bmap.json     # 生成磁场 map

python config_SEO.py              # 设置闭轨计算参数，例如使用哪个磁场，计算哪些能量点
(mpirun -np 6) python main_SEO.py -j config_SEO.json   # 计算闭轨, 可使用 mpirun 进行并行运行, 也可直接运行

python config_Emap.py             # 生成电场配置
python main_Emap.py -j config_Emap.json     # 生成电场 map

# 个人电脑测试时：
python config_track.py            # 生成跟踪配置，包括初始小束团，涂抹,SC,电磁场map
python main_track.py -j config_track.json   # 多粒子跟踪
python main_MergeData.py output/simulation1/Bunch_Position/  # 计算完以后，合并多线程输出文件

# CSNS HPC上运行：
python config_track.py            # 生成跟踪配置，包括初始小束团，涂抹,SC,电磁场map
sbatch test.submit        # 在 CSNS HPC 服务器上提交任务

# -r选项可以进行断点续传：
python .\main_track.py -j .\config_track.json -r .\time_2596.84_rank_0.npz
python .\main_track.py -j .\config_track.json -r .\time_2596.84_rank_0.csv
# 其中.\time_2596.84_rank_0.npz或.\time_2596.84_rank_0.csv为t=2596.84ns断点时刻的Bunch文件
```


## 3. config文件说明

### 3.1 config_Bmap.py磁场配置文件

磁场map文件参数在config_Bmap.py文件中定义，需要包含以下字典变量。

machine为机器参数，包含注入引出能量两个参数，单位MeV。
```python
# machine parameters
machine = dict()
machine['energy_inj'] = 300
machine['energy_ext'] = 600
```

Bmap包含所有lattice参数，包括lattice类型，排列，场强等：
```python
# Bmap configuration
Bmap = dict()
Bmap['Type'] = 'SCALE'
# 可选：等时'ISOCHRONOUS', or 等比'SCALE'
Bmap['NSector'] = 12

Bmap['theta_step_rad'] = np.deg2rad(0.005)  # unit: rad
Bmap['rmin_max_step_m'] = (8.0, 9.3, 0.001)  # unit: m， 比实际Bmap范围大一些，避免迭代过程中超出Bmap范围

Bmap['interval_1'] = (7.5/30.0, 9.375/30.0, 2.8125/30.0, 2.8125/30.0, 7.5/30.0)  # unit: 1
Bmap['positive_or_negative_1'] = (0, 1.0, 0, -1.201/1.65, 0)  # unit: 1
Bmap['fringe_width_1'] = (0, 0.20, 0, 0.35, 0)  # unit: 1
Bmap['SpiralAngle_deg'] = 40.0  # unit: deg

Bmap['orbital_freq_MHz'] = 4.5  # unit: MHz, Type为等时'ISOCHRONOUS'时起效
Bmap['k_value'] = 5.714  # unit: 1, Type为等比'SCALE'时起效
Bmap['B0_max_T'] = 1.65  # unit: T, 参考半径处的B0, Type为等比'SCALE'时起效
Bmap['R0_m'] = 9.0   # unit: m, 参考半径, Type为等比'SCALE'时起效
```

| 参数名                  | 含义与说明                                                                                    |
|-----------------------|------------------------------------------------------------------------------------------|
| **Type**              | 定义磁场 map 类型，可选等时 `ISOCHRONOUS` 或等比 `SCALE`                                               |
| **NSector**           | Lattice 的周期数，表示磁场的周期数。本例为 12 个周期                                                         |
| **theta_step_rad**    | map柱坐标网格的方位角步长，单位为 rad。本例为 0.005°（已转换为 rad）                                              |
| **rmin_max_step_m**   | map柱坐标网格半径范围与步长，单位为 m。本例从 8.0 m 到 9.3 m，步长 0.001 m                                       |
| **interval_1**        | 一个周期划分为若干段，各段角宽度占比，本例分为5段，第1段占cell角宽度的7.5/30，第2段占cell角宽度的9.375/30                       |
| **positive_or_negative_1** | 每段的磁极极性：`1` 表示正磁极，`0` 为漂移段，负数为负磁极，负数数值代表相对强度，本例依次为漂移段，正磁极，漂移段，负磁极（场强为正磁极的1.201/1.65倍），漂移段 |
| **fringe_width_1**    | 边缘场宽度相对磁极平顶段的比例，如 `0.5` 表示 50% ，可以根据绘制的Bz-theta图微调，保证磁极间的场分布不相互干涉                        |
| **SpiralAngle_deg**   | 磁极的螺旋角，单位为度                                                                              |
| **orbital_freq_MHz**  | 仅等时型磁场生效，目标回旋频率，例如 4.5 MHz                                                               |
| **k_value**           | 仅等比型磁场生效，k 值，公式中的k                                                                       |
| **B0_max_T**          | 仅等比型磁场生效，参考半径处的磁场强度，公式中的B_0                                                              |
| **R0_m**              | 仅等比型磁场生效，参考半径值，公式中的r_0                                                                   |

对于 **等比型磁场**，磁场随半径的分布为：

$$
B(r) = B_0 \left(\frac{r}{r_0}\right)^k
$$

其中，$B_0$ 是参考半径 $r_0$ 处的磁场强度，$k$ 是梯度值。

对于 **等时型磁场**，磁场为关于 $r$ 的多项式形式：

$$
B(r) = a_0 + a_1 (r - r_{\text{min}}) + a_2 (r - r_{\text{min}})^2 + a_3 (r - r_{\text{min}})^3 + \cdots
$$

其中，$r_{\text{min}}$ 是起始半径，$a_0, a_1, a_2, \dots$ 是多项式的系数，用户指定目标频率，程序将自动搜索出一组满足要求的系数，使得回旋频率处处相等。


### 3.2 config_SEO.py闭轨计算配置

进行多粒子模拟前，还需要计算闭轨和光学参数，因为注入和跟踪过程中需要读取一些闭轨参数作为参考。

闭轨配置文件中包含以下参数
```python
# some general parameters
config = dict()
config['start_Ek'] = 300.0
config['end_Ek'] = 600.0
config['delta_Ek'] = 30.0
config['extra_Ek'] = ()
config['Bmap_path'] = './Bmap'
```

| 参数名                  | 含义与说明                                                       |
|-----------------------|-------------------------------------------------------------|
| **start_Ek**              | 起始能量点，单位MeV，本例为300MeV                                       |
| **end_Ek**           | 终止能量点，单位MeV，本例为600MeV                                       |
| **delta_Ek**    | 能量点间隔，单位MeV，本例为30MeV                                        |
| **extra_Ek**   | 是否有额外能量点，本例没有，可以加额外能量点，例如config['extra_Ek'] = (310,320,350,) |
| **Bmap_path**              | 使用的磁场                                                       |

闭轨数据计算结果会保存在**Bmap_path目录下resultsSEO文件夹中**

### 3.3 config_Emap.py电场配置

### 3.4 多粒子跟踪配置
多粒子跟踪的参数在config_track.py中定义, 包括注入的小束团信息，磁场map和闭轨信息，电场信息，涂抹曲线，空间电荷配置等，包含以下内容：

```python
# 跟踪参数
track = dict()
stop_condition = {'max_stepsN': 50000000,  # 最多跟踪多少step
                  'max_turn': 1000.5, #  最多跟踪多少圈
                  'max_time': float('inf')}  #  最多跟踪多少时间(second)

track['start_EkMeV'] = 300.0  # 初始动能 (MeV)
track['stop_condition'] = stop_condition  # 3个终止条件满足任一个即终止
track['time_step'] = 0.02e-9  #采用固定步长, 0.02ns --> 回旋周期约273ns --> 每圈约13650步, 每个cell约1137步
track['start_azimuth'] = 0.0  # 注入点方位角
track['solver_type'] = 1  # 0--->RK4, 单步4阶精度但不保辛; 1--->Boris, 单步2阶精度但保辛, 速度稍快
```

| 参数名                  | 含义与说明                               |
|-----------------------|-------------------------------------|
| **start_EkMeV**              | 注入粒子的动能，单位MeV，本例为300MeV。            |
| **stop_condition**           | 模拟终止条件，包括最多step数，最多圈数，最多时间3个条件，满足1个即终止。 |
| **time_step**    | 时间步长，单位s                            |
| **start_azimuth**   | 注入点方位角，即在哪里注入                       |
| **solver_type**              | 常微分方程求解器类型，0为4阶龙格库塔方法，1为Boris推进方法   |


然后是空间电荷效应配置参数
```python
# SC参数
SC = dict()
SC['enable_SC'] = True  # True False 是否加空间电荷效应
SC['SC_type'] = 1
# SC_type=0: 直角坐标系网格+FFT(3维, 适用于小束团)
# SC_type=1: 柱坐标系网格+FFT(2.5维, 适用于长束团)
SC['grid_size_rfz'] = [256, 256, 128]  # 网格尺寸, 直角坐标系为x y z, 柱坐标系为r z phi
```

| 参数名                  | 含义与说明                                                          |
|-----------------------|----------------------------------------------------------------|
| **enable_SC**              | 是否加空间电荷效应。                                                     |
| **SC_type**           | 空间电荷求解方法，0为直角坐标系网格+FFT(3维, 适用于小束团)， 1为柱坐标系网格+FFT(2.5维, 适用于长束团) |
| **grid_size_rfz**    | 各个方向的网格划分个数, 顺序为直角坐标系为x y z, 柱坐标系r phi z                       |

然后是磁场配置参数
```python
# 磁场和高阶非线性项配置
BmapAndSEO = dict()
BmapAndSEO['maps'] = './Bmap'  # 磁场路径
BmapAndSEO['max_order'] = 1  # 展开阶数, n代表展开到(2*n+1阶) [0-->最高1阶, 1-->最高3阶, 2-->最高5阶]
```

| 参数名                  | 含义与说明                                       |
|-----------------------|---------------------------------------------|
| **maps**              | 使用的磁场map的路径                                 |
| **max_order**           | 展开阶数，n代表最高展开到(2*n+1阶)， 因为Bz只有偶数阶，Br和Bf只有奇数阶 |

然后是电场配置参数
```python
# 电场配置
Emap = dict()
Emap['enable'] = True    # True False
Emap['maps'] = './Emap/FrequencyCurve.txt'  # 电场频率曲线(文件表头包含了电压，加速相位，加速腔位置等参数)
```

| 参数名                  | 含义与说明                |
|-----------------------|----------------------|
| **maps**              | 使用的电场map的路径，电场使用理论模型 |
| **enable**           | 是否加电场                |

然后是小束团配置参数
```python
# 注入小束团参数
BunchPara = dict()
BunchPara['ParticleDensity'] = 1.0e4  # 每个宏粒子对应的实际粒子数
BunchPara['ParticleNum'] = 100  # 每个子束团的宏粒子数
BunchPara['TransverseREmit'] = 0.1  # 横向发射度 (pi*mm*mrad)
BunchPara['TransverseZEmit'] = 0.1  # 横向发射度 (pi*mm*mrad)
BunchPara['LongitudeT'] = 0.3  # 纵向长度 (ns) 
BunchPara['LongitudeDEk'] = 0.004  # 纵向能散 (MeV) 
BunchPara['InjTimeNanoSec'] = 0  # 初始注入时刻 (ns)
BunchPara['TransverseDistType'] = 'gauss'  # 横向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag')
BunchPara['LongitudeDistType'] = 'gauss'  # 纵向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag')
```

| 参数名                  | 含义与说明                                                                     |
|-----------------------|---------------------------------------------------------------------------|
| **ParticleNum**              | 每个小束团的宏粒子数                                                                |
| **ParticleDensity**           | 每个宏粒子对应的实际粒子数                                                             |
| **TransverseREmit**    | 小束团的R方向（径向r-r'）发射度，单位π*mm*mrad，1倍RMS发射度                                   |
| **TransverseZEmit**   | 小束团的Z方向（轴向z-z'）发射度，单位π*mm*mrad，1倍RMS发射度                                   |
| **LongitudeT**              | 小束团纵向（t-Ek）时间长度，单位ns                                                      |
| **LongitudeDEk**              | 小束团纵向（t-Ek）能散，单位MeV                                                       |
| **InjTimeNanoSec**           | 注入第1个小束团的时刻，一般默认为0                                                        |
| **TransverseDistType**   | 小束团r-r',z-z'相空间分布类型，可选择'gauss', 'kv', 'waterbag', 'hollow_waterbag'4种     |
| **LongitudeDistType**              | 小束团t-Ek相空间分布类型，可选择'match', 'gauss', 'kv', 'waterbag', 'hollow_waterbag'5种 |

然后是涂抹过程配置参数
```python
# 涂抹配置
Paint = dict()
Paint['enable'] = True    # True False
Paint['MaxBunchNum'] = 100  # 注入子束团个数
Paint['TimeInterval'] = 273.35  # 时间间隔ns
Paint['Curve'] = './PaintCurves/Curve4.paint' # 涂抹曲线包含5列，第一列为时间，后面为小束团在相空间的偏移量
```

| 参数名                  | 含义与说明                                 |
|-----------------------|---------------------------------------|
| **enable**              | 是否涂抹                                  |
| **MaxBunchNum**           | 最多注入多少小束团                             |
| **TimeInterval**              | 每隔多少时间注入一个小束团                         |
| **Curve**           | 涂抹曲线包含5列，第一列为时间，后面为小束团在r-r'和z-z'相空间的偏移 |


然后是dump探测器配置参数，首先创建一个探测器对象，然后加入到DumpPara['modules']的探测器列表中，可以加多个探测器对象用来输出多组信息。
```python
# Dump 配置(探测器)
DumpPara = dict()

# 创建一个探测器对象:
PositionDumpBunch = {
    "type": "PositionDumpBunch",  # 粒子经过特定位置时保存整个束团, 保存每个位置的bunch为一个单独文件
    # 采用一次写入的方式  速度较快
    "start_time": 273.35e-9 * (-1.0),
    "end_time": 273.35e-9 * 10.0,
    "dump_azimuth": [5.6, 35.6, 65.6, 95.6],  # 保存间隔的方位角 (度)
    "save_folder": "./output/simulation1/Bunch_Position"  # 保存路径
}
# 将所有探测器加入到探测器列表中
DumpPara['modules'] = [PositionDumpBunch, ]
```

以下是PositionDumpBunch的参数说明

| 参数名                  | 含义与说明                                                      |
|-----------------------|------------------------------------------------------------|
| **type**              | 探测器类型，本例为PositionDumpBunch，表示探测器放在给定方位角处，输出穿过给定方位角的Bunch信息 |
| **start_time**           | 探测器起始时间                                                    |
| **end_time**              | 探测器结束时间                                                    |
| **dump_azimuth**           | 探测器放置在哪些方位角                                                |
| **save_folder**           | 输出文件到哪个目录                                                  |


根据在跟踪过程中保存粒子信息，有4种探测器类型：

| Dump 类型               | 文件输出方式      | 触发条件         | 适用场景                    |
|-----------------------|-----------|--------------|-------------------------|
| **PositionDumpBunch** | 每个束团一个文件  | 穿过给定方位角时记录一次   | 记录给定位置的束团信息（类似于PyORBIT） |
| **PositionDump**      | 每个粒子一个文件  | 穿过给定方位角时记录一次   | 记录粒子在特定位置运动演变           |
| **StepDumpBunch**          | 每个束团一个文件 | 每隔一定时间间隔记录一次 | 记录束团的时间演变，其输出文件也用于断点续传  |
| **StepDump**                  | 每个粒子一个文件 | 每隔一定时间间隔记录一次   | 记录粒子轨迹(文件很大非常耗时)        |

除了PositionDumpBunch以外，其他3种示例如下
```python
# 其他几种Dump探测器
# StepDump:
StepDump = {
    "type": "StepDump",  # 按时间间隔保存粒子 每个粒子单独保存为一个文件 按ID区分 可用于绘制单粒子轨迹
    # 采用追加写入的方式 且频繁读写 速度很慢 数据量大占用空间 仅能短时间使用
    "start_time": 273.35e-9 * (-1.0), # 开始时间
    "end_time": 273.35e-9 * 60.0, # 终止时间
    "interval_time": 50.0e-9,  # 间隔多少时间保存一次
    "num_particles_to_dump_global": 50,  # 全局保存的粒子数
    "save_folder": "./output/simulation1/Particle_time"  # 保存路径
}

# StepDumpBunch:
StepDumpBunch = {
    "type": "StepDumpBunch",  # 按时间间隔保存整个束团 按步数保存bunch为一个单独文件 按步数区分
    # 采用一次写入的方式 速度较快
    "start_time": 273.35e-9 * (-1.0), # 开始时间
    "end_time": 273.35e-9 * 60.0, # 终止时间
    "interval_time": 273.35e-9*10.5, # 每隔多少时间保存一次当前bunch
    "save_folder": "./output/simulation1/Bunch_time"  # 保存路径
}

# PositionDump:
PositionDump = {
    "type": "PositionDump",  # 粒子经过特定位置时保存, 每个粒子存为一个单独文件
    # 采用追加写入的方式  读写不如StepDump频繁  速度较慢
    "start_time": 273.35e-9 * (-1.0),
    "end_time": 273.35e-9 * 600.0,
    "dump_azimuth": (np.array([5.61, 35.61, 65.61, ])).tolist(),  # 保存间隔的方位角 (度)
    "num_particles_to_dump_global": 100,  # 全局保存的粒子数
    "save_folder": "./output/simulation1/Particle_Position"  # 保存路径
}
```


## 4. 输出文件格式与绘图说明

在模拟过程中，程序会输出磁场、电场、中间数据以及多个探测器模块记录的粒子信息，以下按模块进行说明。

### 4.1 磁场与电场文件格式

磁场与电场 map 通常保存在 `./Bmap/` 和 `./Emap/` 目录下，对应的配置文件分别为 `config_Bmap.json` 和 `config_Emap.json`。磁场为三维扩展 map，电场为基于频率曲线的理想模型（在给定位置生成一个矩形区域，矩形区域内加均匀电场）。

* 磁场 map：

  * 文件格式为 `.npz`，包含柱坐标网格下的 `Bz`, `Br`, `Bf`的展开系数

* 电场 map：

  * 格式为 `.txt`，表头包含频率、电压、加速相位、加速区域（矩形）位置，数据部分为频率曲线

---


### 4.2 闭轨与 Twiss 函数输出文件格式

程序在搜索SEO（静态平衡轨道，不同能量的闭轨） 与 Twiss 函数计算流程时，会在指定 Bmap 目录下生成 `resultsSEO/` 文件夹，包含如下分析数据文件：

| 文件名                        | 内容                               |
| -------------------------- |----------------------------------|
| `SEO_ini.txt`              | 不同能量闭轨的特征参数（Qr, Qz, freq, MeanR） |
| `SEO_r.txt`                | 不同能量闭轨的 r(φ)                     |
| `SEO_pr.txt`               | 不同能量闭轨的 pr(φ)                    |
| `BetaFuncR.txt`            | 径向 β<sub>r</sub>(φ) 曲线           |
| `BetaFuncZ.txt`            | 轴向 β<sub>z</sub>(φ) 曲线           |
| `AlphaFuncR.txt`           | 径向 α<sub>r</sub>(φ) 曲线           |
| `AlphaFuncZ.txt`           | 轴向 α<sub>z</sub>(φ) 曲线           |
| `Bz.txt`、`Br.txt`、`Bf.txt` | 闭轨上的磁场分量                         |

文件格式说明：

* 所有数据均为 ASCII 文本格式，表头注明单位（如 Ek 单位为 MeV，角度单位为 deg）
* 第一列为 能量(Ek)，第一行为 方位角(rad)，数据区域为不同能量点、不同方位角对应的物理量

下面给出绘制闭轨的示例方法，其他twiss参数文件格式类似：
```python
import numpy as np
import matplotlib.pyplot as plt

# 读取 r(φ) 数据, 绘制闭轨
data = np.loadtxt('./Bmap/resultsSEO/SEO_r.txt')

# 第0行是 phi，单位 rad；后续每行是对应能量点的 r(phi)
phi_rad = data[0, 1:]
phi_deg = np.rad2deg(phi_rad)
r_matrix = data[1:, 1:]  # shape: [n_energy, n_phi]
ek_list = data[1:, 0]    # 第1列是Ek（单位MeV）

# r vs fi
plt.figure(figsize=(8, 6))
for i, ek in enumerate(ek_list):
    plt.plot(phi_deg, r_matrix[i], label=f"Ek = {ek:.1f} MeV")
plt.xlabel('φ (deg)')
plt.ylabel('r (m)')
plt.title("Closed Orbit r(φ)")
plt.legend()
plt.grid(True)

# r*cos(fi) vs r*sin(fi)
plt.figure(figsize=(6, 6))
for i, ek in enumerate(ek_list):
    x = r_matrix[i] * np.cos(phi_rad)
    y = r_matrix[i] * np.sin(phi_rad)
    plt.plot(x, y, label=f"Ek = {ek:.1f} MeV")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title("Closed Orbits in x-y Plane")
plt.legend()
plt.axis('equal')
plt.grid(True)

plt.tight_layout()
plt.show()
```
---

### 4.3 探测器输出数据格式

不同探测器的输出文件格式完全一致，均为 `.npz` 格式的二进制数据，结构为 `n×17` 的二维数组，其中每行代表一个粒子状态，但**触发机制和数据堆叠方式不同**：

* `StepDump` / `StepDumpBunch`：按时间间隔触发，同一时间所有粒子堆叠或同一粒子按时间步堆叠
* `PositionDump` / `PositionDumpBunch`：按方位角触发，穿越探测角时记录粒子。其中PositionDumpBunch是类似于PyORBIT,在一个位置输出束团信息。

与 PyORBIT 的在线合并+打印不同，本程序采用 FLUKA 类似的方式：

* 每个进程独立输出本地 `.npz` 文件
* 后处理时统一合并为可读的ASCII文件`.csv` 文件

输出 `.npz` 或`.csv` 文件内容结构如下：

| 索引 | 变量名         | 单位  | 含义                       |
| -- | ----------- | --- |--------------------------|
| 0  | `r`         | m   | 径向位置                     |
| 1  | `pr`        | rad | 径向动量斜率 = arctan(vr / vf) |
| 2  | `z`         | m   | 轴向位置                     |
| 3  | `pz`        | rad | 轴向动量斜率 = arctan(vz / vf) |
| 4  | `fi`        | rad | 方位角                      |
| 5  | `Ek`        | MeV | 动能                       |
| 6  | `inj_t`     | s   | 注入时刻                     |
| 7  | `Inj_flag`  | -   | 是否已注入（1: 是, 0: 否）        |
| 8  | `Survive`   | -   | 是否存活（1: 是, 0: 丢失）        |
| 9  | `RF_phase`  | rad | 当前时刻电场的RF 相位（sin）        |
| 10 | `Esc_r`     | V/m | 空间电荷场 Er                 |
| 11 | `Esc_z`     | V/m | 空间电荷场 Ez                 |
| 12 | `Esc_fi`    | V/m | 空间电荷场 Efi                |
| 13 | `Bunch_ID`  | -   | 小束团编号                    |
| 14 | `Local_ID`  | -   | 当前MPI进程内编号               |
| 15 | `Global_ID` | -   | 全局编号                     |
| 16 | `t`         | s   | 当前记录时刻                   |

#### 输出文件命名规则：

```
prefix_start_{start_time}_end_{end_time}_rank_{rank_id}.npz
```

例如：

```
crossing_angle_5.6_start_0_end_1000_rank_1.npz
```

表示该文件记录的是 rank 1 进程在 0 到 1000 ns 时间段内，穿过 5.6° 方位角的粒子数据。

```
crossing_angle_5.6_start_0_end_1000_rank_merged.csv
```

表示该文件记录的是在 0 到 1000 ns 时间段内，穿过 5.6° 方位角的粒子数据, 各进程合并为一个文件。

输出文件示例：

```
output/simulation1/Bunch_Position/
├── crossing_angle_5.6_start_0_end_1000_rank_0.npz
├── crossing_angle_5.6_start_0_end_1000_rank_1.npz
└── ...
```
---

## 5. 数据合并与可视化工具

### 5.1 合并多线程 npz 文件为 CSV

每个线程会单独输出一个 `.npz` 文件，若需整体可视化分析，可使用脚本 `main_MergeData.py` 合并为 `.csv` 格式：

```bash
python main_MergeData.py output/simulation1/Bunch_Position/
```

输出结果保存在原目录的 `merged_files/` 子文件夹中，命名规则如下：

```
crossing_angle_5.6_start_0_end_1000_merged.csv
```

CSV 文件包含所有粒子属性，列顺序与 npz文件 中定义一致，最后一列为记录时间。

---
(20250605,未完)
# PyFFA lattice 动力学孔径（DA）扫描说明

主程序：`scripts/tool_scan_lattice_da.py`

## 1. 这里计算的 DA 是什么

本工具计算固定能量、无 RF、无空间电荷、无加速、无磁场误差谐波时的单粒子横向动力学孔径。它回答的是：给定 lattice、能量、物理孔径和跟踪圈数后，闭轨附近多大的横向初始振幅还能存活。

DA 不是束流的 RMS 发射度，也不是只看某一条水平或垂直初始线。扫描采用归一化振幅四分之一平面的极坐标网格：

```text
epsilon_r = rho^2 cos(theta)^2
epsilon_z = rho^2 sin(theta)^2
theta in [0, 90 deg]
```

每个方向从 `rho=0` 向外扫描。该方向的 DA 边界定义为从原点连续可达的最后一个稳定点；如果外侧出现孤立的稳定岛，不把它计入主 DA。

初始坐标由注入方位角处的闭轨和 Twiss 参数重建：

```text
x  = sqrt(beta * epsilon) cos(phi)
x' = -sqrt(epsilon / beta) [alpha cos(phi) + sin(phi)]
```

默认对四组横纵相位 `(0,0)、(90,0)、(0,90)、(90,90)` 分别跟踪；只有四组都存活，该极坐标点才判为稳定。这比单相位结果更保守。

## 2. 推荐计算流程

所有 lattice 必须使用完全相同的：

- 能量、注入方位角；
- 跟踪圈数和每圈积分步数；
- 径向、垂直物理孔径；
- 扫描上限、径向网格和极角网格；
- 初始相位组合和磁场 Taylor 展开阶数口径。

建议分三档：

本扫描器使用方位角 `phi` 作为自变量，因此应直接指定方位角步长，而不是从 Boris 的时间步换算。默认 `--phi-step-rad 0.001`；为保证每圈恰好走到 `2*pi`，程序实际取 `ceil(2*pi/0.001)=6284` 步/圈，对应实际步长约 `0.00099987 rad`。

建议分三档：

1. 快速排查：20--50 圈，`21 x 9` 网格，单相位，`h_phi=0.001 rad`。
2. lattice 筛选：200 圈，`41 x 19` 网格，默认四相位，`h_phi=0.001 rad`。
3. 最终结论：至少 1000 圈，`81 x 31` 网格，四相位或更密相位，先用 `h_phi=0.001 rad`，再用更小方位角步长检查收敛。

最终结果至少再用更密步数复算边界附近，确认 DA 指标变化小于预先约定的容差（建议 2%）。如果 `boundary_scan_limit_fraction` 不为 0，说明部分方向尚未找到损失边界，应增大 `--max-emit-pi-mm-mrad` 后重跑。

## 3. 命令示例

以下命令均从 PyFFA 根目录执行；运行前先执行 `SetEnvCmd.ps1`（Windows PowerShell）或 `source ./SetEnvCmd.sh`（Linux/HPC）。

单个 lattice 筛选：

```powershell
python project_reverse_injection/scripts/tool_scan_lattice_da.py `
  --bmap Bmaps/Bmap_FD16_k7p63_B1p65_R265_Fp0p75_s49p97 `
  --energy-mev 300 `
  --turns 200 `
  --phi-step-rad 0.001 `
  --max-emit-pi-mm-mrad 2000 `
  --n-radius 41 `
  --n-angle 19 `
  --aperture-r-min-m 10.8 `
  --aperture-r-max-m 12.0 `
  --aperture-z-abs-m 0.10 `
  --output-dir lattice_da_output
```

多个 lattice 用完全相同的口径比较时，重复写 `--bmap`：

```powershell
python project_reverse_injection/scripts/tool_scan_lattice_da.py `
  --bmap Bmaps/lattice_A `
  --bmap Bmaps/lattice_B `
  --bmap Bmaps/lattice_C `
  --energy-mev 300 `
  --turns 200 `
  --phi-step-rad 0.001 `
  --max-emit-pi-mm-mrad 2000 `
  --n-radius 41 `
  --n-angle 19 `
  --aperture-r-min-m 10.8 `
  --aperture-r-max-m 12.0 `
  --aperture-z-abs-m 0.10 `
  --output-dir lattice_da_comparison
```

输出目录非空时程序默认拒绝覆盖；确认要替换同名结果时再加 `--overwrite`。

### 串行运行 PPT 的 R/F 二参数 12 点网格

批处理入口：`scripts/tool_run_rf_grid_lattice_da_scans.py`。只扫描 PPT 网格里的两个 lattice 参数：

- DF field ratio：`R=0.265074 / 0.365074 / 0.465074 / 0.565074`；
- F 铁角增量：`F=0 / 0.5 / 0.75 deg`。

两者组合成 `3 行 x 4 列 = 12` 套 lattice。运行顺序按 PPT 版式逐行进行：每一行固定一个 F 值，四列依次为四个 R 值。所有案例固定使用 `max_order=2`，即加入到五阶的离中平面场项。程序不创建进程池，也不同时启动多个案例；高阶 map 生成和 DA 跟踪都一次只处理一个 lattice。

其余计算口径为：300 MeV、200 圈、`h_phi=0.001 rad`、`13 x 7` 极坐标网格、4 组初始相位、扫描上限 `4000 pi mm mrad`。此前三套 `max_order=0` 结果不能复用，12 套都必须用 `max_order=2` 重算。

先检查全部输入和串行顺序：

```powershell
python -u project_reverse_injection/scripts/tool_run_rf_grid_lattice_da_scans.py --dry-run
```

正式串行运行：

```powershell
python -u project_reverse_injection/scripts/tool_run_rf_grid_lattice_da_scans.py
```

默认输出到：

```text
project_reverse_injection/lattice_da_200turn_RF_grid_12_order2_300MeV/
  Bmap_FD16_.../da_survival_and_boundary.png
  lattice_da_summary.csv
  lattice_da_comparison.png
  rf_grid_da_summary.csv
  run_manifest.json
```

如果 Bmap 目录缺少 `Br/Bz/Bfi_order_1.npz` 或 `order_2.npz`，脚本会先调用与生产多进程版本公式一致的串行生成器，生成 `order=0/1/2` 文件，并把 `config_Bmap.json` 的 `expand_order` 更新为 2。只生成和检查 12 套高阶 map、不启动 DA 时可用：

```powershell
python -u project_reverse_injection/scripts/tool_run_rf_grid_lattice_da_scans.py --prepare-only
```

脚本支持 DA 断点续跑。需要重算已有的 `max_order=2` DA 结果时加：

```powershell
python -u project_reverse_injection/scripts/tool_run_rf_grid_lattice_da_scans.py --overwrite
```

高阶 map 已存在但需要强制重新生成时另加 `--regenerate-maps`。

## 4. 应该怎样展示

每个 lattice 自动输出：

- `da_survival_and_boundary.png`：左图是归一化振幅平面的存活圈数和保守 DA 边界；右图是边界随极角的变化。这是主结果图。
- `da_boundary.csv`：每个极角的边界、水平/垂直分量、相位 0 对应的物理位移，以及是否触及扫描上限。
- `da_polar_grid.csv`：每个极坐标点的综合稳定性、跨相位存活比例和最短存活圈数。
- `da_particles.csv`：每个初始相位粒子的初末状态、损失圈数和损失原因，便于追查。
- `da_summary.json`：该 lattice 的参数和汇总指标。

多 lattice 另外输出：

- `lattice_da_comparison.png`：边界曲线叠图及平均/最弱方向指标对比。
- `lattice_da_summary.csv`：汇总表，可直接用于 PPT。

汇报时建议同时给出：

1. 所有 lattice 的 DA 边界叠图；
2. 最弱方向 DA，即 `da_min_equivalent_emit_pi_mm_mrad`；
3. 平均边界振幅及其等效发射度；
4. 水平、垂直、45 度三个截距；
5. `boundary_scan_limit_fraction` 和数值收敛检查。

不要只用稳定点比例排名，因为该指标依赖扫描上限和网格取法；也不要只报平均 DA，因为狭窄方向往往决定实际可接受束流。

## 5. 当前适用范围

当前代码面向含 `config_Bmap.json`、`resultsSEO` 和 Taylor map 文件的 analytical PyFFA lattice。OPERA 3D lattice 尚未接入这个 DA 入口。粒子按 `phi` 为自变量使用现有 `FunctionForSEOUnified` 和 RK4 跟踪，损失判据为非有限状态、非法动量、径向越界或垂直越界。

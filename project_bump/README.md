# Bump / Kicker Extraction Project

本目录集中保存局部 kicker、单粒子引出、septum 布置、500 pi 束团分离和 kicker 排列比较脚本。核心跟踪器与磁场类仍位于仓库根目录的 `src/`，运行配置使用根目录 `config_track.json`。

## 运行方式

所有命令都应从 PyFFA 根目录执行。Windows PowerShell：

```powershell
.\SetEnvCmd.ps1
python .\project_bump\scripts\tool_plot_bmap_with_kickers.py
python .\project_bump\scripts\tool_track_extraction_kickers.py --turns 1
```

Linux/HPC：

```bash
source ./SetEnvCmd.sh
python project_bump/scripts/tool_track_extraction_kickers.py --turns 1
```

历史输出目录暂时保留在仓库根目录，例如 `extraction_single_particle_output/`、`extraction_bunch_septum_output/`、`extraction_kicker_split_scan/` 和 `extraction_kicker_three_straight_scan/`。这是为了保持已有脚本和结果引用不变。

## 当前物理模型

- `FFAG_BumpKickerField` 已接入 RK4/Boris 多粒子跟踪链。
- kicker 使用整圈全局方位角，不按主 Bmap 的 sector 周期重复。
- 当前为静态硬边界模型，没有时间波形和竖直 `z` 截止。
- 默认通过 `Bz = sign * BL_Tm / L_eff_m` 给出水平偏转，也支持固定 `Br_T` 和 `Bfi_T`。
- 正式配置为四块 `0.20 m` kicker，每块目标偏转 `5 mrad`；`config_track.json` 中 `BumpKicker.enable=true`。
- RF 和空间电荷在现有引出对比脚本中默认关闭。

## 脚本入口

- `scripts/tool_plot_bmap_with_kickers.py`：主磁场和 kicker 位置/增量场可视化。
- `scripts/tool_track_extraction_kickers.py`：600 MeV 单质子 kicker-on/off 生产 Boris 跟踪。
- `scripts/tool_plot_extraction_with_septum.py`：整环、引出轨迹和严格 0.5 m septum 布局。
- `scripts/tool_track_bunch_to_septum.py`：500 pi 水袋束团到 septum 的 kicker-on/off 对照。
- `scripts/tool_plot_bunch_extraction_three_views.py`：整环、方位角和 septum 入口三类视图。
- `scripts/tool_compare_kicker_splits.py`：两个直线节中的 `2+2`、`1+3`、`3+1`、`0+4` 比较。
- `scripts/tool_optimize_septum_per_kicker_split.py`：为每种排列独立优化 0.5 m septum。
- `scripts/tool_compare_kicker_splits_bunch_phase_space.py`：上述四种排列的 500 pi 束团入口相空间。
- `scripts/tool_compare_kicker_three_straights.py`：三个相邻直线节的 `2+1+1`、`1+2+1`、`1+1+2` 与六块 `2+2+2` 比较。
- `scripts/tool_compare_kicker_three_straights_bunch_phase_space.py`：三直线节方案的 500 pi 束团比较；支持 `--scan-dir`、`--labels 2+2+2` 和 `--reuse-samples`。

## 当前结论

- 正式两直线节 `2+2` 方案可安装；单粒子在优化 septum 的全长最小中心分离约 `44.1 mm`。
- 四种两直线节排列中，`0+4` 动力学分离最大，但在现有低场窗口和 `0.20 m` 净间隙约束下不可安装；只有 `2+2` 可合理安装。
- 三直线节方案均可安装，`1+2+1` 最好，但 500 pi 束团自由窗口约 `-41.1 mm`，仍差于两直线节 `2+2` 的约 `-32.5 mm`。
- 2026-09-09 的六块 `2+2+2` 计算位于 `extraction_kicker_three_straight_scan/run_20260909_155228`。三组各占 `0.600 m`，均可放入约 `0.882 m` 的低场窗；单粒子优化 septum 为展开方位角 `303.750--306.172 deg`，全长最小中心分离 `47.520 mm`。同一 500 pi 束团在入口的质心分离为 `48.174 mm`，完整径向自由窗口为 `-32.520 mm`，与两直线节 `2+2` 的 `48.784/-32.462 mm` 基本相同，仍不能无损分离。
- 当前四块固定 kicker 不能无损分离完整 500 pi 束团；真实结论还需计入 septum 厚度、束流包络、安装误差和闭轨误差。

更完整的历史数值和物理口径仍记录在仓库根目录 `AGENTS.md` 的“局部磁场 Kicker 模块”章节。

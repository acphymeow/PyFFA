# Comment 5：tune footprint 与共振线

本目录集中保存审稿意见 Comment 5 对应的 tune-shift 计算与共振线绘图脚本。审稿人建议在 tune footprint 中标出共振条件，并特别指出

\[
Q_x=3,
\qquad
Q_z=\frac{8}{3},
\]

其中第二条写成整数系数形式即为 `3Q_z=8`。本目录统一采用

\[
mQ_x+nQ_z=p,
\qquad
|m|+|n|\leq 3
\]

生成三阶及以下共振线，并去除几何上重复的直线。

## 已汇总的三类 tune 图

当前选中的样式为：共振线使用很浅的灰色，标签自动避开 tune footprint 主体、子图编号和图例；`Q_x=3` 与 `3Q_z=8` 均直接标出。参考输出保存在：

- Gaussian benchmark：`outputs/fig_tune_gauss_80_pic.png` 和 `outputs/fig_tune_gauss_80_theory.png`；
- application：`outputs/application_tune_shift_original.png` 和新生成的 `outputs/application_tune_shift_with_resonances.png/.pdf`；
- 纵向 SC benchmark：3D PIC/fixed-reference、LSC off/on 四种情况分别输出为四张独立图片，文件名以 `outputs/longitudinal_sc_tune_` 开头。

所有绘图脚本均保持“一张图对应一个独立 figure 和一组 PNG/PDF 文件”。脚本完成全部 `savefig` 后只调用一次 `plt.show()`，因此运行结束时会一次性显示本次生成的全部独立图片。

application 正式候选图以原始 `test_fft_tune6.py` 为计算底稿，使用全部 50,000 个历史 FFT dump 和每个粒子的全部时间记录。提取出的完整范围为 `Qx=2.5404--3.2215`、`Qz=2.0023--2.9984`；论文图仍使用原脚本的显示窗口 `Qx=2.3--3.3`、`Qz=1.9--2.9`，因此窗口外的少量点不会显示。新图与原论文图的 footprint 一致。纵向 SC 候选图使用 `04_lsc_tune_comparison` 保存的最终 nominal-60 NPZ，不重新引入早期或过时的 tune 数据。

## 脚本说明

| 脚本 | 作用 |
|---|---|
| `Benchmark_SC_tuneshift_gaussL.py` | 原项目 Gaussian tune-shift benchmark 的同名入口。完整实现保存在本目录的 `gaussian_tune_shift_baseline.py`，该入口直接运行它，避免维护两份重复代码。 |
| `Benchmark_SC_kvTune_with_resonances.py` | 独立的 KV--Gaussian tune benchmark 共振线版本。脚本自身包含理论计算、闭轨扣除和 FFT tune 提取，不导入根目录 `Benchmark_SC_kvTune.py`；在原 footprint 上叠加三阶及以下共振线，并输出独立 PNG/PDF。 |
| `gaussian_tune_shift_with_resonances.py` | 最终完整入口。读取 80 π mm mrad Gaussian benchmark 的逐粒子跟踪文件，扣除能量相关闭轨，使用 FFT 提取 `Q_x,Q_z`；随后计算对应的非线性理论 footprint，保存两组 CSV，并在 PIC/theory 图上叠加三阶及以下共振线。该脚本生成当前选中的参考图。 |
| `plot_resonance_lines_order3.py` | 轻量预览入口。直接读取已有 PIC/theory CSV，不重新执行逐粒子 tune 提取和理论跟踪；PIC 与 theory 各自生成独立图片。也可显示全部三阶共振线、只标注 `Q_x=3` 与 `3Q_z=8`，或生成指定 tune 窗口的纯共振图。用于快速选择线型，不替代完整计算。 |
| `gaussian_tune_shift_baseline.py` | 加共振线之前的 Gaussian tune-shift 基线脚本快照，用于追溯 tune 提取、理论计算和原图生成过程。其默认参数是早期 40 π mm mrad case，不作为 Comment 5 的最终绘图入口。 |
| `application_tune_shift_original_test_fft_tune6.py` | 原始 `test_fft_tune6.py` 的未修改快照，直接生成 manuscript 使用的 `fig_app_tuneshift.png`。保留用于核对数据路径、FFT 定义、绝对 tune 分支和原始画图范围，不作为材料包内的便携运行入口。 |
| `plot_application_tune_shift_with_resonances.py` | 在 `test_fft_tune6.py` 基础上的忠实整理版。保留时间排序、全部记录、`x+i p` 复信号 FFT、`1-tune` 转换和原水平分支判断；只增加稳健路径、CSV/PDF输出和浅灰共振线。读取 50,000 个逐粒子文件并保存 `results/application_tunes.csv`。 |
| `plot_longitudinal_sc_benchmark_tunes_with_resonances.py` | 纵向 SC benchmark 重画入口。读取 `04_lsc_tune_comparison` 最终 LSC-off/on NPZ，对 3D PIC/fixed-reference、LSC off/on 四种情况分别生成独立图片，并以同一线型叠加共振线。 |
| `paper_resonance_overlay.py` | 三类图共用的论文样式模块：浅灰共振线、重复线消除、标签避让，以及 `all/highlighted/none` 三种标签模式。 |
| `calculate_fig9_tune_depression.py` | 使用正文已经列出的三组最大空间电荷频移端点，计算 `Q_{x}/Q_{x0}` 和 `Q_{z}/Q_{z0}`；同时保存 PIC 与理论比值，供 Referee B Comment 11 的回复和 Fig. 9 图注使用。 |

## 运行方法

从仓库根目录运行完整 80 π mm mrad 版本：

```powershell
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/gaussian_tune_shift_with_resonances.py
```

运行原始 Gaussian tune-shift benchmark 基线入口：

```powershell
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/Benchmark_SC_tuneshift_gaussL.py
```

它需要以下已有数据：

- `../output/NewBenchmark/FFT_Tune_Gauss80_uniform72/*.npz`；
- `Bmap_FD16/resultsSEO_noError_X/SEO_r.txt`、`SEO_pr.txt`、`BetaFuncR.txt` 和 `BetaFuncZ.txt`。

新生成的 CSV 写入本目录的 `results/`，图片写入 `outputs/`。

只从已有 CSV 快速重画：

```powershell
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/plot_resonance_lines_order3.py --label-mode highlighted
```

运行 KV--Gaussian tune benchmark 的共振线版本：

```powershell
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/Benchmark_SC_kvTune_with_resonances.py
```

默认使用与原脚本一致的 `FFT_Tune_kv70` 数据和 tune 显示窗口，并采用 `gaussian_tune_shift_with_resonances.py` 的字体、共振线及标签风格，标注全部三阶及以下共振线；不添加 Gaussian 图中的蓝色水平或垂直参考线。可用 `--label-mode highlighted` 只标注 `Q_x=3` 与 `3Q_z=8`。

常用模式：

```powershell
# 标出窗口内全部一至三阶共振线
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/plot_resonance_lines_order3.py --label-mode all

# 只给 Qx=3 和 3Qz=8 添加文字标签
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/plot_resonance_lines_order3.py --label-mode highlighted

# 生成原 application tune 范围内的纯共振线图
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/plot_resonance_lines_order3.py --no-data --qx-limits 2.3 3.3 --qz-limits 1.9 2.9 --output-stem fig14_window_order3_resonance_only
```

完整重画 application tune footprint：

```powershell
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/plot_application_tune_shift_with_resonances.py
```

其默认输入为 `../output/FD16_paint_1_1/FFT_Tune_Dump_0`，默认使用每个粒子的全部记录，与原 `test_fft_tune6.py` 一致。快速检查可加 `--max-files 1000`；抽样 CSV 会使用单独文件名，不覆盖完整结果。

使用纵向 SC benchmark 的最终 NPZ 重画：

```powershell
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/plot_longitudinal_sc_benchmark_tunes_with_resonances.py
```

计算 Fig. 9 三个 emittance case 的 tune depression：

```powershell
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/calculate_fig9_tune_depression.py
```

这里把 tune footprint 的最大空间电荷频移端点作为代表性的 depressed tune。以正文采用的裸 tune `(Qx0,Qz0)=(3.23,2.81)` 为分母，PIC 三组结果分别为 `(0.909,0.882)`、`(0.933,0.916)` 和 `(0.953,0.941)`。

## 当前边界

- Gaussian 80 π mm mrad benchmark 已有数据叠加后的候选图。
- application tune footprint 已找到对应的 `FD16_paint_1_1` 原始 FFT dump，并基于 `test_fft_tune6.py` 的原始数值口径完成全部 50,000 粒子的正式重画。
- 纵向 SC benchmark 已完成最终 LSC on/off NPZ 的共振线版本；完整的 3D 数据读取、fixed-reference tracking 和 NPZ 生成链仍以 `04_lsc_tune_comparison` 为准，本目录只集中最终重画入口，避免维护两套重复物理代码。
- 三类候选图尚未替换 manuscript 中对应图片；本目录保存的是可供选择和替换的版本。

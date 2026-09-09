# 可复现脚本说明

本目录是面向 GitHub 的源码副本，固定路径为 `benchmarks/reproducibility/scripts/`。原始研究材料目录不是运行依赖，也不计划随本目录上传。这里只保留 Python 源码和 Markdown 说明；生成的图片、PDF、CSV、NPZ、文本结果及 Python/Numba 缓存由上级 `.gitignore` 排除。

本目录保存的是整理时采用的**完整核心源码快照**，不再只是跳转到原目录的 launcher。物理公式和数值参数未改；只增加了 `bundle_paths.py`，使脚本从任意工作目录运行时都能找到 PyFFA 根目录，并把新输出留在本材料包内。旧的 `run_*.py` 入口仍保留，便于与历史命令对应，但复现时建议直接运行下表中的完整脚本。

## 运行环境与数据

从仓库根目录运行，先执行 `SetEnvCmd.ps1`（Windows）或 `source ./SetEnvCmd.sh`（Linux/HPC）。主要依赖为 `numpy`、`scipy`、`matplotlib`、`numba`、`pyfftw` 和 `src/` 中的 PyFFA 模块。3D 对比还需要本地已有的 `HPC_dir/*` 数据和 `Bmap_FD16/resultsSEO_noError[_X]` 表；这些大型数据不包含在本 GitHub 源码目录中。

先做快速检查：

```powershell
python -m compileall -q benchmarks/reproducibility/scripts
```

`bundle_paths.py` 只负责定位仓库根目录、共享纵向模块和原始工作目录；它不包含任何物理计算。

## 1. Fig. 8：纵向电场与约 10% 偏差

建议顺序：

```powershell
python benchmarks/reproducibility/scripts/01_fig8_es/fig8_error_budget.py
python benchmarks/reproducibility/scripts/01_fig8_es/fig8_gaussian_kernel_test.py
python benchmarks/reproducibility/scripts/01_fig8_es/plot_fig8_improvement_steps.py
python benchmarks/reproducibility/scripts/01_fig8_es/plot_fig8_paper_uniform_disk.py
python benchmarks/reproducibility/scripts/01_fig8_es/plot_fig8_paper_gaussian_kernel.py
```

| 脚本 | 作用 |
|---|---|
| `test_sc_260508_Es_formula.py` | 基准底座：生成 3D Gaussian 长束团，装载 SEO 投影表，调用 `Build_2_5D_SliceGrid_sbs_new_fixProjection`，计算同一 disk kernel 的直接求和，并计算 Eq. (48) 的高精度 3D Gaussian 理论积分。其函数被下面三个脚本复用。 |
| `fig8_error_budget.py` | 误差预算主脚本。分别量化 solver 实现、纵向抽样噪声、横向核模型差、线密度形状差和 slice 离散误差。输出 `fig8_error_budget_output.txt`。 |
| `fig8_gaussian_kernel_test.py` | 验证 Gaussian 横向约化核及源 slice 平均；比较连续 Gaussian 卷积、离散 `NS=256` 结果和 Eq. (48)，并生成核形状图与最终 benchmark 图。 |
| `plot_fig8_improvement_steps.py` | 将 disk/raw、disk/smoothed、Gaussian/raw、Gaussian/smoothed 四种受控配置放在同一图中，展示每一步改动带来的独立影响。 |
| `plot_fig8_paper_uniform_disk.py` | 论文版 Fig. 8。保持生产代码的 uniform-disk kernel 和 `NS=256` 不变，仅绘制 smoothed 2.5D PIC 与 3D Gaussian 理论。脚本扫描 `6--20 cm` 的平滑尺度，当前参数区选用 RMS 最小且曲线明显变平滑的 `10 cm`；保存 PNG/PDF/CSV 后只调用一次 `plt.show()`。 |
| `plot_fig8_paper_gaussian_kernel.py` | 独立的 Gaussian-kernel 论文图版本，不修改生产 uniform-disk 求解器。使用 `NS=256`、Gaussian 源切片平均核，并扫描 `4--16 cm` 平滑尺度；当前选用 RMS 最小的 `12 cm`，相对 3D Gaussian 理论的 RMS 偏差为 `2.73%`。保存 PNG/PDF/CSV 后只调用一次 `plt.show()`。 |
| `run_error_budget.py`、`run_gaussian_kernel_test.py` | 历史 launcher；调用原始工作目录的同名权威脚本，不是材料包内源码复现的首选入口。 |

### Fig. 8 的控制变量设计

1. **求解器实现误差**：保持粒子、`rho_long`、`s_grid`、disk kernel 和 `NS` 全部相同，只把 FFT/卷积求解换成 `O(NS^2)` direct sum。二者 `relative L2=4.7e-10`，排除卷积公式/实现错误。
2. **shot noise**：保持 disk kernel、网格和理论参照不变，只比较 raw slice 线密度与 8 cm Gaussian-smoothed 线密度；另对同一平滑线密度做 200 次 Poisson 重采样交叉验证。
3. **横向模型差**：输入完全相同的理想 Gaussian 纵向线密度，只把 uniform-disk 横向核替换为 Eq. (48) 对应的 Gaussian 横向约化核，隔离 `disk versus Gaussian` 的确定性差异。
4. **纵向线密度形状差**：保持核和数值积分不变，只把理想 Gaussian 线密度换成实际有限样本、截断和闭轨投影后的线密度。
5. **slice 离散误差**：使用无噪声理想线密度和相同核，仅扫描 `NS=256/512/1024`，检查约 `ds^2` 的收敛。
6. **横向尺寸敏感性**：其余不变，仅扫描 `sigma_perp=15/20/25 mm`，量化尺寸口径对理论峰值的影响。

`fig8_gaussian_kernel_test.py` 的 Gaussian kernel 只用于 Fig. 8 benchmark；生产跟踪仍使用 finite-disk kernel。

为兼容当前 `Build_2_5D_SliceGrid_sbs_new_fixProjection` 接口，基准底座现在显式创建 `2*nz × 2*nx` 的 open-boundary FFT plan；这只是 plan 尺寸接口更新，不改变 `nx × nz` 物理网格、kernel 或电荷沉积。

## 2. `h>1` 周期连续性重建

建议顺序：

```powershell
python benchmarks/reproducibility/scripts/02_harmonic_continuity/test_core_bin_enclosing_arc.py
python benchmarks/reproducibility/scripts/02_harmonic_continuity/demo_core_bin_continuity_steps.py
```

以下脚本只用于历史追溯或辅助诊断：

```powershell
python benchmarks/reproducibility/scripts/02_harmonic_continuity/plot_fig5_core_bin_method.py
python benchmarks/reproducibility/scripts/02_harmonic_continuity/plot_fig6_s_continuity.py
python benchmarks/reproducibility/scripts/02_harmonic_continuity/test_density_support_continuity.py
python benchmarks/reproducibility/scripts/02_harmonic_continuity/plot_fig5_density_support_v2.py
```

脚本默认在保存结果后通过 `plt.show()` 一次性弹出全部三张验证图。无图形界面的批处理环境使用：

```powershell
python benchmarks/reproducibility/scripts/02_harmonic_continuity/test_core_bin_enclosing_arc.py --no-show
```

| 脚本 | 作用 |
|---|---|
| `02_harmonic_continuity/bunch_complement_common_cut.md` | 当前权威文字记录：梳理审稿人对 `h>1`、halo 和区间一致性的关注点，给出中文回复初稿、Section IV B 中文建议稿、合并后 Fig. 5 的 panel 逻辑与图注，并单独记录“一次建立全环 bins、逐束团复用”和边缘切片等内部实现信息。 |
| `demo_step01_generate_project_bins.py` | 入门分步脚本：先生成一个跨越 `0/2π` 的束团和一个普通束团；再执行 `phi_mod = phi mod 2π` 并显示 `[0,2π)` 上的等宽 histogram bins；最后周期平滑 histogram、找出 `h=2` 个最宽低密度区段，并将最宽区段中点设为公共切口。三步依次弹窗，暂不计算 occupied intervals。 |
| `test_core_bin_enclosing_arc.py` | 离线验证：从周期直方图寻找 `h` 个低密度间隔，以最宽间隔作为公共切口、其余间隔作为内部边界，再按预设电荷比例确定各束团的主要密度区域；覆盖全部 buckets 均占据的 `h=1/2/3`、跨界、内部低谷、halo stress、`q` 扫描和复杂度统计。其 workflow 图用于检查算法，不是论文最终图。 |
| `demo_core_bin_continuity_steps.py` | 合并后纵向连续化图的绘图入口。只建立一次全环周期 histogram，公共切口对齐到已有 bin edge，展开和逐束团处理直接复用同一组 bins。论文版四联图按“多圈坐标—周期密度与束团边界—展开及主要切片范围—常规与边缘切片”组织；密度、halo、公共切口和束团边界合并在同一 panel 中。具体切片数保留在脚本配置中供内部检查，但不显示在论文图中。 |
| `plot_fig5_core_bin_method.py` | 旧的 2×2 Fig. 5 候选图，现仅保留用于历史追溯。 |
| `plot_fig6_s_continuity.py` | 旧的独立 Fig. 6 绘图脚本。由于原 Fig. 5、6 已决定合并，该脚本不再作为论文最终图入口。 |
| `test_density_support_continuity.py` | 历史 density-support 辅助验证；它直接按密度阈值找连续区域，不执行当前“`h` 个间隔 + `h-1` 个内部边界”的最终流程。 |
| `plot_fig5_density_support_v2.py` | 上述历史 density-support 的替代图，不是最终 Fig. 5。 |
| `run_validation.py`、`run_figure.py` | 历史 launcher。 |

`h>1` 的离线算法证据以 `test_core_bin_enclosing_arc.py` 为准；合并后的 Fig. 5 由 `demo_core_bin_continuity_steps.py` 生成，并已作为 `figure_longitudinal_continuity_bins.pdf` 写入 manuscript。原 Fig. 6 删除，其脚本和其他早期 Fig. 5 脚本只作历史保留。论文表述、最终 panel 顺序和图注以 `bunch_complement_common_cut.md` 为准。

## 3. 纵向相空间：3D 与 1D 理论

直接运行：

```powershell
python benchmarks/reproducibility/scripts/03_longitudinal_phase_space/plot_longitudinal_phase_space_on_off.py
```

若本材料包中尚无 longitudinal cache，主脚本会先调用 `longitudinal_pair()` 生成 RF-only 和 RF+LSC 的配对理论轨迹，然后再绘图。

| 脚本 | 作用 |
|---|---|
| `plot_longitudinal_phase_space_on_off.py` | 主复现入口。读取 LSC on/off 3D TBT dump，在 turn `1,51,101,151,201,251` 与配对 1D 理论作相空间和 RMS/correlation 对比。 |
| `run_gauss_waterbag_coupling_prediction.py` | 生成 Gaussian transverse / waterbag longitudinal 配置的配对纵向驱动，并提供 `CACHE`、机器参数和 `longitudinal_pair()`。 |
| `run_coupled_sc_toy.py` | 完整横纵耦合 toy 底座：读取 SEO/Twiss、构造 lattice、执行横向跟踪并提取 FFT tune；相空间脚本复用其机器和跟踪函数。 |
| `run_comparison.py` | 历史 launcher。 |

共享源码位于 `shared_longitudinal/`：

| 脚本 | 作用 |
|---|---|
| `level5_bucket_sampled_tracking.py` | RF bucket 采样、SEO 频率表、纵向 RK4 推进和基本常数。 |
| `level47_3d_recipe_in_1d.py` | 与 3D 求解器一致的 128-slice finite-disk `disk_sum_1d` 场配方和 1D tracking。 |
| `level33_cache_3d_and_bare1D.py` | 从 3D TBT dump 去重/整理粒子，并生成 bare 1D 对照缓存；也是 level47 的数据加载依赖。 |

## 4. LSC on/off 引起的横向 tune-footprint 变化

从原始 3D 数据完整复现的顺序：

```powershell
python benchmarks/reproducibility/scripts/04_lsc_tune_comparison/compare_new_lsc_off_kernel_fix.py
python benchmarks/reproducibility/scripts/04_lsc_tune_comparison/compare_new_lsc_off_nominal60.py
python benchmarks/reproducibility/scripts/04_lsc_tune_comparison/compare_new_lsc_on_nominal60.py
python benchmarks/reproducibility/scripts/04_lsc_tune_comparison/plot_lsc_on_off_combined.py
```

最后一个脚本统一使用 `Qr=2.90--3.30`、`Qz=2.50--2.90`，并从各自 NPZ 的 `common_star` 同时设置左右 panel 的 red star。

| 脚本 | 作用 |
|---|---|
| `compare_new_lsc_off_kernel_fix.py` | 从旧/新 LSC-off FFT dump 重新提取 tune，验证每次 SC 更新 kernel 后的新数据，并保存后续 nominal-60 比较所需 NPZ。 |
| `compare_new_lsc_off_nominal60.py` | 最终 LSC-off 图：修复后 3D 对固定 `60e-6 m rad` toy；不从 3D 读取初始发射度或 `sigma_x/z`。 |
| `compare_new_lsc_on_nominal60.py` | 最终 LSC-on 图：修复后 3D 对同一 nominal-60 toy，并保留旧 3D 作为诊断参照。 |
| `plot_lsc_on_off_combined.py` | 将最终 on/off 两组结果合成统一坐标轴的 2×2 图，左右 panel 的 red star 严格共用同一坐标。 |
| `comparison_core.py` | 共享分析核心：3D 文件读取/去重、Hann-windowed zero-padded FFT tune、固定 toy 的 100-turn tracking、统计表和报告。 |
| `toy_tracking.py` | 固定横向 toy 的 lattice、Twiss、Gaussian action 分布、横向 SC kick 和 FFT tune 底层实现。 |
| `longitudinal_driver.py` | 为 on/off toy 生成逐圈纵向粒子/线密度驱动；on 使用 finite-disk LSC，off 使用 RF-only。 |
| `dispersion.py` | 从不同能量闭轨在 Frenet 法向上的投影计算 `Dx,Dx'`，供色散失配项使用。 |
| `plotting.py` | 两 panel tune footprint、统一范围、bare tune 与 common red star 的绘图函数。 |
| `beam_size.py` | 从 3D 第一条 tune record 的最密 phase slice 测量 `sigma_x`；仅用于诊断旧的 measured-sigma 口径，不进入最终 nominal-60 初值。 |
| `analyze_sigma_x_turn_by_turn.py` | 检查“3D 首记录时间窗口”和 toy turn-1 的 `sigma_x` 口径不一致；属于验证 1 的诊断脚本。 |
| `validate_2_gaussian_closure.py` | 控制变量验证 2：检查 3D slice 的横向 Gaussian closure 以及 toy 场形状假设。 |
| `validate_4_nonlinear_optics.py` | 验证 4：无 SC 裸晶格的幅度/能量非线性 detuning，并估算线性 toy 缺少的裸晶格修正。 |
| `validate_5_sc_kick_history.py` | 后续检查：比较 3D 与 toy 的逐圈/检查点横向 SC kick history，确认差异不是只来自最终 FFT。 |
| `run_lsc_off.py`、`run_lsc_on.py` | 历史 launcher。 |

### tune 比较的控制变量口径

- on/off 只改变纵向 SC 驱动；两者都保留横向 SC。
- 最终 toy 横向 RMS 发射度固定为 `60e-6 m rad`，不使用 3D 拟合尺寸调参。
- 3D 和 toy 使用同一类 Hann-windowed、zero-padded FFT estimator；窗口长度分别按实际可用记录写入输出表。
- 色散项由 toy 自己的纵向 slice 条件能散计算。
- `validate_2/4/5` 分别隔离 transverse Gaussian closure、裸晶格非线性和 SC kick history，避免把所有残差都归因于纵向模型。

## 5. Comment 5：tune footprint 共振线

本目录统一汇总三类 tune 图：Gaussian tune-shift benchmark、application tune footprint，以及纵向 SC benchmark 的 LSC on/off tune footprint。

最终 80 π mm mrad Gaussian tune-shift 图：

```powershell
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/gaussian_tune_shift_with_resonances.py
```

若只需使用已有 CSV 快速比较线型：

```powershell
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/plot_resonance_lines_order3.py --label-mode highlighted
```

application 与纵向 SC benchmark：

```powershell
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/plot_application_tune_shift_with_resonances.py
python benchmarks/reproducibility/scripts/05_tune_shift_resonances/plot_longitudinal_sc_benchmark_tunes_with_resonances.py
```

| 脚本 | 作用 |
|---|---|
| `gaussian_tune_shift_with_resonances.py` | Comment 5 的最终完整入口。重新提取 80 π mm mrad PIC tune、计算对应理论 footprint，并叠加 `mQ_x+nQ_z=p`、`|m|+|n|≤3` 的共振线。最终样式使用浅灰线和自动避让标签。 |
| `plot_resonance_lines_order3.py` | 从已有 CSV 快速重画；支持全部标签、只标 `Q_x=3`/`3Q_z=8`，以及 application tune 窗口的纯共振图。 |
| `gaussian_tune_shift_baseline.py` | 未加入共振线的历史基线，保留用于追溯，不作为最终图入口。 |
| `application_tune_shift_original_test_fft_tune6.py` | 原始 application tune 脚本的未修改快照；用于追溯 manuscript 中 `fig_app_tuneshift.png` 的直接来源。 |
| `plot_application_tune_shift_with_resonances.py` | 以 `test_fft_tune6.py` 为底稿，保留全部记录、时间排序、复信号 FFT 和原绝对 tune 分支；从 50,000 个 application dump 重新生成带共振线的候选图和 CSV。 |
| `plot_longitudinal_sc_benchmark_tunes_with_resonances.py` | 从 `04_lsc_tune_comparison` 的最终 nominal-60 NPZ 重画 LSC off/on 四联 tune 图。 |
| `paper_resonance_overlay.py` | 三类图共享的浅灰共振线与标签避让实现。 |
| `calculate_fig9_tune_depression.py` | 用 Fig. 9 正文已经列出的三组最大频移端点计算 depressed/bare tune 比值，并保存为 CSV。 |

三类原图和共振线候选图均保存在 `05_tune_shift_resonances/outputs/`。更完整的文件关系、输入数据与结果范围见 `05_tune_shift_resonances/README.md`。

## 6. Comment A-4：水平与竖直 RMS 包络

原 `Benchmark_SC_envlope_stable.py` 只从跟踪数据提取竖直尺寸，并使用了与全文不一致的 `\sigma_y`。材料包整理版同时提取水平 `\sigma_x` 和竖直 `\sigma_z`，分别输出 with-SC/no-SC 的四张独立图片，不在代码中合图：

```powershell
python benchmarks/reproducibility/scripts/06_rms_envelope/Benchmark_SC_envlope_stable.py
```

脚本保留根目录原版本的包络方程和数值口径，原文件未修改。详细参数、输出和无 SC case 的运行要求见 `06_rms_envelope/README.md`。

## 7. Comment B-4：原 Fig. 2、3、4 的组合图

推荐的一键入口为：

```powershell
python benchmarks/reproducibility/scripts/07_fig234_coordinate_systems/make_fig234_coordinate_systems.py
```

该入口先生成合并原 Fig. 2、3 的闭轨/坐标系 panel，再生成原 Fig. 4 的参考点投影 panel，最后输出 `outputs/fig234_ab_stitched.png`。三个绘图脚本均为完整源码快照，并已改为从任意工作目录定位 SEO 数据；运行时保存两个源图和最终组合图，只弹出最终组合图。该组合图现已作为 manuscript 的 Fig. 2，并在正文中分别引用两个 panel。具体文件关系见 `07_fig234_coordinate_systems/README.md`。

随机口径分开处理：Fig. 8 使用固定 `seed=42`，纵向相空间的配对 waterbag 使用固定种子以便逐粒子复现；最终 nominal-60 tune 入口则显式设置 `TOY_SEED=None` 和 `TOY_USE_RESULT_CACHE=False`，既不固定横向 toy 随机样本，也不读取旧 toy 结果缓存。因此频移统计的末位会在重复运行间轻微变化，论文采用值以本包 `results/` 快照为准。

## 最终与探索脚本的边界

材料包只复制了形成最终结论或解释残差所需的核心脚本。旧 measured-sigma 拟合、未收敛 realtime toy、stale-kernel 结果和早期 Fig. 8 smoothing/improved 草稿没有复制，避免它们被误认为最终证据。原始工作目录仍保留这些探索文件，便于必要时追溯。

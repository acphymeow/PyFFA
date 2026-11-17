# PyFFA Agent Notes

本文档用于后续连续性对话。新的协作者或 AI agent 进入项目时，先阅读本文件，再根据需要阅读 `tips.md`。

## 当前目录结构（2026-09-09）

- 正式核心模块位于 `src/`。
- 可执行入口与配置脚本位于 `examples/`。
- 基准/测试脚本位于 `benchmarks/`；其中 `benchmarks/reproducibility/scripts/` 保存从研究材料整理出的公开复现源码。通用辅助脚本位于 `tools/`。
- 轨迹反推项目位于 `project_reverse_injection/`；bump/kicker 引出项目位于 `project_bump/`，其正式脚本在 `project_bump/scripts/`。
- `_tmp` 和备份脚本位于 `legacy/`，不参与正式运行。
- 从项目根目录运行 `SetEnvCmd.ps1`（Windows PowerShell）或 `source ./SetEnvCmd.sh`（Linux/HPC），临时把 `src/` 与 `examples/` 加入 `PYTHONPATH`；后者用于兼容工具脚本对 `main_track.load_bmap_auto` 的复用。
- 所有主流程仍从项目根目录启动，例如 `python examples/main_track.py -j config_track.json`；这样配置中的 `./Bmaps`、`./Emaps` 和输出相对路径保持原语义。

## 项目定位

PyFFA 是用于 FFAG 加速器模拟的 Python 程序，当前主链条包括：

- 磁场 map 生成：`examples/config_Bmap.py` -> `config_Bmap.json` -> `examples/main_Bmap.py`
- 闭轨与 Twiss/SEO：`examples/config_SEO.py` -> `config_SEO.json` -> `examples/main_SEO.py`
- RF 电场曲线：`examples/config_Emap.py` -> `config_Emap.json` -> `examples/main_Emap.py`
- 多粒子跟踪：`examples/config_track.py` -> `config_track.json` -> `examples/main_track.py`

当前上下文有两条重点：

- 多粒子跟踪中的空间电荷模块，尤其是 `SC_type=1` 的 2.5D slice-by-slice PIC/FFT 链条。
- OPERA 3D 磁场接入后的 SEO/Twiss 与后续 `examples/main_track.py` 跟踪链条统一。

## 当前工作状态

- 本地目录已经初始化 Git 仓库，并设置过 `origin=https://github.com/acphymeow/PyFFA.git`。
- 暂时不要处理 GitHub 同步、force push 或覆盖远端历史。
- 新增了 `.gitignore`，用于排除缓存、IDE 文件、输出数据和生成的 JSON/map 文件。
- `examples/main_SEO.py`、`examples/main_Emap.py`、`examples/main_track.py` 已恢复命令行参数读取。
- `examples/main_track.py` 的起始时间改为读取 `config['track']['time_start']`。

## SC 模块现状

`src/` 中的正式文件已经按 `legacy/FFAG_Bunch_tmp.py` 和 `legacy/FFAG_SC_new_tmp.py` 合并了一部分更新：

- `src/FFAG_MathTools.py` 已包含 `Bilinear_interp_2D_vect_uniform_3var`。
- `src/FFAG_SC_new.py` 已包含：
  - `ComputeSliceBoundsAndCoordsSBS_merged_s`
  - 新版 `total_field_1d_numba`
  - `interpolate_fields_to_particles_SBS_numba_withEfi`
  - `make_s_continuous_mpi`
  - `compute_local_coordinates_from_projection`
  - `compute_local_coordinates_from_projection_at_energy`
- `src/FFAG_Bunch.py` 已按 tmp 主体更新：
  - 保留 `Build_2_5D_SliceGrid_sbs`
  - 加入 `Build_2_5D_SliceGrid_sbs_new`
  - 加入 `Build_2_5D_SliceGrid_sbs_new_fixProjection`
  - 开放边界 zero-padding FFT 使用外部传入的 padded FFTW plan；`kernel_hat` 不缓存，每次 SC 调用按当前 `dx,dz` 重建
  - 保留 `Build_2_5D_SliceGrid_sbs_GaussianToy_uniformL`
  - 保留 `Build_2_5D_SliceGrid_sbs_GaussianToy`
  - 不保留 `Build_2_5D_SliceGrid_sbs_LinearFast`
- `src/FFAG_track.py` 中 `SC_type=2` 已改为调用 GaussianToy，而不是 LinearFast。

验证过：

```bash
python -m compileall -q src examples benchmarks tools project_reverse_injection/scripts project_bump/scripts
python -c "import FFAG_Bunch, FFAG_track; print('imports ok')"
python examples/main_track.py --help
```

## 重要注意事项

1. `SC_type=1` 当前默认调用 `Build_2_5D_SliceGrid_sbs_new_fixProjection`（2026-08-05 起，`src/FFAG_track.py` RK4 121 行 / Boris 353 行；旧版 `sbs` 不再被跟踪链使用）。
2. `Build_2_5D_SliceGrid_sbs_new_fixProjection` 需要额外 SEO 数据：
   - `SEO_s_matrix`
   - `SEO_pr0_geom_matrix`
3. `examples/main_track.py` 已给 `BunchObj.SEO_s_matrix` 和 `BunchObj.SEO_pr0_geom_matrix` 赋值（254–256 行调 `FFAG_SC_new.convert_seo_pr_to_pr0_geom` / `build_seo_s_matrix` 构造，336–337 / 352–353 行 RK4/Boris 两处赋值）。
4. fixProjection 跟踪链已打通，smoke test `tools/tool_smoke_sc_fixProjection.py` 通过；完整 3D 多粒子验证已跑（HPC `Longitudinal_0p0/0p5/0p7/1p0`，200 kV，2026-08-09）并与 1D 理论解完成对比（详见 `PRAB_papers/longitudinal_sc_level0/PRAB_revision_progress.deepseek.md` §九/§十八）。
5. 纵向空间电荷场 `Es` 的当前状态（2026-08-06 核实）：
   - 三个 SBS 求解器都会计算 `Ef_map`（`total_field_1d_numba`，有限半径均匀带电圆盘核，默认 `a_radius=0.04 m`，不乘任何 gamma 因子）。
   - 旧版 `sbs` / `sbs_new` 的插值函数 `interpolate_fields_to_particles_SBS_numba` 中写回 `Es` 的行仍被注释（`# mat_in[idx_out, 12] = ef_val`），但这两个求解器已不再被跟踪链调用，不影响结果。
   - 跟踪链使用的 `fixProjection` 通过 `interpolate_fields_to_particles_SBS_numba_withEfi` 真正把 Es 旋转进 `(Er, Efi)` 写入列 10/12，**纵向 SC 已进入跟踪**。
6. SC 调用频率：跟踪主循环中每 `SC['step']`（`examples/config_track.py` 中当前为 20）步重算一次 SC 场，宏步之间对端点场做线性外推写回列 10/11/12；SC 场在单步 RK4/Boris 推进内冻结。
7. FFT 生命周期：`src/FFAG_track.py` 在循环外创建 `(2*Nz,2*Nx)` FFTW plan 并传入 `Build_2_5D_SliceGrid_sbs_new_fixProjection`；`src/FFAG_Bunch.py` 在每次 SC 调用内用当前 `dx,dz` 生成 Green kernel。plan 可复用，`kernel_hat` 不可跨 SC 更新复用。
8. 当前配置路径仍需统一：
   - `examples/config_Bmap.py` 输出 Bmap 路径
   - `examples/config_SEO.py` 读取 Bmap 路径
   - `examples/config_Emap.py` 读取 Bmap/SEO 路径
   - `examples/config_track.py` 读取 Bmap/SEO/Emap 路径

## Referee A-1/A-2 最终口径

- 统一审稿材料包：`PRAB_papers/referee_response_longitudinal_sc_bundle/README.md`。其中逐条整理 Referee A 6 条、Referee B 12 条，并集中提供 Fig. 8、`h>1`、纵向相空间和 LSC-induced transverse tune footprint 的脚本入口与结果快照。
- Fig. 8 纵向 `Es` 峰值区约 10% 偏低的权威记录：`PRAB_papers/commentA2_fig8_es_benchmark/README.md`，详细摘要见 `PRAB_papers/longitudinal_sc_level0/PRAB_revision_progress.deepseek.md` §十五。solver 与同 disk 核 direct sum 的 `relL2=4.7e-10`；峰值账本为横向模型 `-3.2%`、NS=256 离散 `-2.4%`、统计噪声 `±~5%`。Gaussian slice-averaged kernel + 8 cm 平滑将 rms 偏差降至 `2.86%`。这是 benchmark 改进，未自动替换生产纵向动力学。
- `h>1` 连续化的权威记录：`PRAB_papers/harmonic_continuity_offline_20260811/README.md`，详细摘要见 deepseek §十六。最终方案是“RF bucket 辅助分区 + 核心密度 bins 连续包围弧 + 单一公共切口”；已完成 `h=1/2/3/4`、部分填充、跨 `0/C`、内部低谷、halo 和 `q` 敏感性离线测试，但尚未接入生产 MPI/PIC，只能写为 proposed/generalized construction。

## PRAB 横纵向 SC 耦合最终状态（2026-08-17）

完整进度先读：

- 简版：`PRAB_papers/PRAB_revision_progress.md` §8
- 详细版：`PRAB_papers/longitudinal_sc_level0/PRAB_revision_progress.deepseek.md` §二十
- 最终方法与结果：`PRAB_papers/transverse_sc_tune_lsc_on_off_comparison/README.md`
- toy model：`PRAB_papers/transverse_longitudinal_sc_toy_20260813/README.md`

### slice 独立参考闭轨

`Build_2_5D_SliceGrid_sbs_new_fixProjection` 现采用两阶段投影：

1. 用束团 MPI 全局平均能量对应的共同闭轨周长 `C_ref` 构造 `s_mod`，完成周期连续化并固定 `slice_indices`；
2. MPI 归约每个 slice 的平均能量 `slice_mean_Ek`，让该 slice 内全部粒子共享这条能量相关闭轨的切向基；
3. 保持第一阶段的纵向网格和 slice 编号不变，仅在最终 slice 参考基中重建横向坐标与统计量。

物理口径：不同 slice 可因纵向 SC 引起的平均能量差使用不同闭轨，但同一 slice 的横向 Poisson 网格必须使用同一参考轨道。

### 已修复的有效长度问题

`ComputeSliceBoundsAndCoordsSBS_merged_s` 现在返回 `slice_indices[:N]`，使预分配容量与当前已注入/存活粒子的有效长度分离。修复已同步到根目录及两套 V10，并通过短缓冲区 `np.bincount`、编译和独立导入测试；该问题已经关闭。

### V10 on/off 配对约定

- `PyFFAG_V10`：横向 SC + 纵向 SC。
- `PyFFAG_V10_LscOff`：保留横向 SC，只去掉局部纵向场 `es_loc` 对全局 `(Er,Efi)` 的贡献。
- 两套目录的非缓存源文件当前只应在 `FFAG_SC_new.py` 的以下两行不同；若出现其他差异，先停止并审计：

```python
# on
er_global   = (ex_loc + pr0 * es_loc) * inv_norm
ephi_global = (-pr0 * ex_loc + es_loc) * inv_norm

# off
er_global   = ex_loc * inv_norm
ephi_global = -pr0 * ex_loc * inv_norm
```

不能把 `Esc_fi` 整体清零；`-pr0*ex_loc*inv_norm` 是横向场旋转后的方位分量。

### FFT kernel 修复与最终数据

旧开放边界实现复用由早期 `dx,dz` 生成的 `kernel_hat`，导致后续横向场偏弱、垂直场偏强。当前约定：

- `(2*Nz,2*Nx)` padded FFTW plan 在 `FFAG_track.py` 跟踪循环外创建并传入 SC 求解器；
- `FFAG_Bunch.py` 每次 SC 调用都用当前 `dx,dz` 重建 Green kernel 和 `kernel_hat`；
- 同一次 SC 调用所有 slice 共用 `dx,dz`，不同 SC 更新之间允许改变；
- 主项目 `src/`、V10 on、V10 off 的对应 `FFAG_track.py` / `FFAG_Bunch.py` 已同步。

最终 3D 数据：

- LSC off：`HPC_dir/Newdata_coupling_withoutLsc100`，25000 个 tune 文件，90 点有效 FFT 窗口；
- LSC on：`HPC_dir/Newdata_coupling_withLsc100`，25000 个文件均为 100 条完整记录。

### 接受的 nominal-60 theory 对比

理论直接使用 `epsilon_r=epsilon_z=60e-6 m rad`，不从 3D 读取发射度或 `sigma_x/z`。注入匹配束斑为 `sigma_x,beta=13.731532 mm`、`sigma_z=15.829606 mm`；一阶色散宽度由理论纵向分布的 slice 条件能散计算。理论不固定随机数，末位统计量可轻微变化，引用以保存 CSV 为准。

| case | mean Qr | Qr p01 | Qr p05--p95 | mean Qz | Qz p01 | Qz p05--p95 |
|---|---:|---:|---:|---:|---:|---:|
| LSC off, corrected 3D | 3.120977 | 3.045560 | 0.117072 | 2.696757 | 2.597624 | 0.172739 |
| LSC off, nominal-60 theory | 3.113461 | 3.024125 | 0.133445 | 2.674534 | 2.574057 | 0.160269 |
| LSC on, corrected 3D | 3.177538 | 3.133025 | 0.071708 | 2.766099 | 2.704568 | 0.109344 |
| LSC on, nominal-60 theory | 3.175201 | 3.126860 | 0.077474 | 2.749933 | 2.696987 | 0.084822 |

kernel 修复后 LSC-on 的 mean-Qr error 从 `0.00920` 降至 `0.00234`，Qr-p01 error 从 `0.01725` 降至 `0.00617`。固定 theory 缺少非线性裸晶格幅度频移；60-pi Gaussian 的平均修正约为 `(-0.00076,+0.01423)`，修正后的 on theory 均值 `(3.17444,2.76416)` 与 3D 只差 `(+0.00309,+0.00194)`。用户已接受该结果，不再继续 measured-sigma 或 realtime-feedback 调参路线。

纵向相空间在 turn `1,51,101,151,201,251` 与理论吻合：on 的 `sigma_s/sigma_E` 最大相对差为 `0.91%/1.40%`，off 为 `0.43%/1.08%`。主入口为 `PRAB_papers/transverse_sc_tune_lsc_on_off_comparison/README.md`；下一位 agent 不应恢复 `simulation1` 旧端点、旧预测表或未收敛实时 toy 作为论文结论。

## RF/Emap 电压 bug 修复记录（2026-08-07）

- 问题：`main_track.py` 实例化 E 场时固定使用 `FFAG_EField_new`（按 `gap_model` 分派到 `FFAG_EField_spiral` 的代码曾被注释）；且 `FFAG_EField_new.Interpolation2D_EMap` 多 gap 循环中，`emap_kernel`（`FFAG_Utils.py`）对未命中粒子清零，会抹掉前一个 gap 写入的场 → 只有最后一个 gap 生效，多 gap 运行时实际电压减半（HPC `Longitudinal_0p0/0p5/1p0` 三次运行实际 V_turn=100 kV 而非配置的 200 kV，由 f_s 测量 9.8 kHz vs 14.41 kHz 发现）。
- 修复：`emap_kernel` 未命中不再清零（调用方统一预清零）；`FFAG_EField_new.Interpolation2D_EMap` 循环前 `fill(0)` + `Malloc_RF_shift.fill(-1)` + `EnableFlag=False` 直接返回；`main_track.py` 恢复按 `gap_model` 分派（fixed→`FFAG_EField_new`，spiral→`FFAG_EField_spiral`）。
- 验证工具：`tools/tool_check_emap_voltage.py`（生产插值路径测每个 gap 的 ∫E·dl）。
- 注意：`HPC_dir/PyFFAG_V9/` 副本带同样 bug，重跑 HPC 前必须同步修复。

## 开发原则

- 不直接覆盖 GitHub 远端历史。
- 不做大范围重构，先保证物理链条清晰和可验证。
- 少写防御性代码，按默认格式开发，偏离默认格式直接报错，让配置错误尽早暴露：
  - 配置键用直接索引（如 `Emap['SEO_folder']`），不用 `dict.get(key, default)` 静默兜底；
  - 路径按唯一约定拼接，不做"先试 A 再试 B"的多候选回退；
  - 不为旧配置写法保留兼容分支；缺文件/缺键时抛带完整路径或键名的异常。
  - 例：`examples/main_Emap.py` 的 `resolve_seo_dir` 只接受 `Bmap_fold/SEO_folder`，缺 `SEO_ini.txt` 即 `FileNotFoundError`。
- 每次改 SC 算法后，至少执行：

```bash
python -m compileall -q src examples benchmarks tools project_reverse_injection/scripts project_bump/scripts
python -c "import FFAG_Bunch, FFAG_SC_new, FFAG_track; print('imports ok')"
python examples/main_track.py --help
```

- 对空间电荷模块的修改应优先保持以下接口稳定：
  - `LocalBunch[:, 10] = Esc_r`
  - `LocalBunch[:, 11] = Esc_z`
  - `LocalBunch[:, 12] = Esc_fi`

## 局部磁场 Kicker 模块（2026-09-03）

本节提到的 `tool_*.py` 均位于 `project_bump/scripts/`；项目运行与脚本索引见 `project_bump/README.md`。

- 已从相邻 `PyFFAG_Bump` 项目移植 `FFAG_BumpKickerField`，并接入当前项目的 RK4/Boris 多粒子跟踪链。
- kicker 使用整圈全局方位角，不随主 Bmap 按 sector 周期重复；作用区由 `phi_start_deg`、`phi_span_deg`、`rmin_m`、`rmax_m` 定义。
- 默认竖直场为 `Bz = sign * BL_Tm / L_eff_m`，也可用 `Br_T`、`Bfi_T` 配置固定的其他柱坐标场分量；场在区域内直接叠加到主磁场。
- 当前模型是静态硬边界模型，没有时间波形，也没有竖直 `z` 截止。
- `config_track.py` 和 `config_track.json` 当前配置四块水平偏转引出 kicker，`BumpKicker.enable=True`：EK1/EK2 位于 146.25 deg 直线节，EK3/EK4 位于相邻的 168.75 deg 直线节。按 600 MeV 质子 SEO 闭轨的实际弧长布置，每块 `L_eff=0.20 m`，同一直线节内净间隙 `0.20 m`；单块目标偏转 `5 mrad`，取 `BL=0.0203305 T*m`、`Bz=-0.1016525 T`，显式设 `Br=Bfi=0`。kicker 径向范围为 `11.7--12.2 m`，外边界与 OPERA map 最大半径一致；对当前正向环行质子，负 `Bz` 对应向外径向偏转。
- `tool_plot_bmap_with_kickers.py` 可直接读取 `config_track.json`，兼容 analytical/OPERA 3D Bmap，输出两幅直角坐标局部图：主场与无填充 kicker 边界、kicker-only `Delta Bz`；即使 kicker 开关关闭也会显示配置位置。
- `tool_track_extraction_kickers.py` 从 600 MeV SEO 闭轨（默认全局 `phi=140 deg`）构造单质子，使用生产 Boris 链分别做 kicker-on/off 跟踪，关闭 RF/SC，并输出 CSV、JSON 和直角坐标对比图；`--turns N` 可指定圈数，当前无波形模型等效于在整个跟踪区间保持恒定的平顶脉冲，粒子每圈经过时都会重复受 kick。构造 Boris 初值时必须把 SEO 的归一化径向动量 `pr=P_r/P` 转为轨迹角 `asin(pr)`，不可直接把 SEO `pr` 当角度。脚本同时保留 `r-rSEO` 图，另存实际 `R` 对外孔径的直接判据图、最接近孔径处的局部直角坐标放大图；多圈时还输出第 1 圈与最后一圈同相位比较及逐圈峰值图。2026-09-03 的历史 `2 mrad` 结果：`dt=0.01 ns` 单圈 `Rmax=12.0442398 m`，`dt=0.02 ns` 连续平顶 5 圈 `Rmax=12.0521585 m`，均未越过 `R=12.2 m`。当前 `5 mrad` 配置以 `dt=0.02 ns` 跟踪：单圈 `Rmax=12.0818975 m`、孔径余量 `118.103 mm`；连续平顶 5 圈仍未引出，五圈最大 `R=12.1040937 m`、最小余量 `95.906 mm`，均在第 2 圈；逐圈 kick 受 tune/相位影响发生叠加与抵消，不是线性累加。
- 固定极性位置优化试验（2026-09-04）：曾把两组临时移到 `78.75 deg` 与 `348.75 deg` 直线节，并把后组反装为正极性。按 600 MeV Twiss 相位折算，四块空间相干度约 `99.52%`（等效 `3.98/4` 块同相）；但 `dt=0.02 ns` 生产 Boris 跟踪仍为有界多圈振荡：单圈 `Rmax=12.0910742 m`，5 圈 `Rmax=12.1035889 m`，未越过 `12.2 m`。因此移动静态固定极性 kicker 只能优化单圈合成，不能改变 `Qr=3.307` 导致的每圈约 `110.52 deg` 相位旋转，也不能产生跨圈单调累积。试验输出保存在 `extraction_single_particle_output/run_20260904_000136` 和 `run_20260904_000229`；正式配置已恢复原来的两个相邻直线节、四块负极性布局。
- 若配合 septum，引出判据不应再用“粒子越过 OPERA map/外孔径 `R=12.2 m`”，而应使用 septum 位置的 kicked-orbit 与 600 MeV SEO 闭轨分离。原正式布局四块各 `5 mrad` 的第一圈中，EK4 后最大向外分离约 `71.33 mm`（`phi=183.97 deg`，但处于主磁场区）；低场长直线候选区 `phi=191.267--195.739 deg` 长约 `0.927 m`、`|Bz|<0.012 T`，分离约 `40.0--48.9 mm`。其中最优连续 `0.2/0.5/0.8 m` septum 窗口的最小分离分别约 `46.93/44.05/41.20 mm`。最终是否擦 septum 必须再结合挡板厚度、闭轨束与引出束的横向包络及安装/闭轨误差，不能只用单粒子中心轨迹下结论。
- `tool_plot_extraction_with_septum.py` 读取 `config_track.json` 和已有 kicker-on CSV，画整圈磁场/闭轨/第一圈引出轨迹/kicker/septum 布局及 septum 局部直角坐标放大。默认按严格的 `0.500 m` septum 中心线弧长，从 `phi=191.267 deg` 开始；末端通过轨迹采样点之间的线性插值落在约 `193.689 deg`，中心轨迹分离约 `44.08--48.91 mm`。当前输出为 `extraction_single_particle_output/run_20260903_222702/single_particle_600MeV_septum_layout_0p5m.png`。图中绿色 septum 只表示位于两条中心轨迹中间的零厚度中心线，不包含真实挡板厚度和束流包络；按用户要求不再画右下沿程曲线。
- `tool_track_bunch_to_septum.py` 用生产 Boris 链将水平 Courant--Snyder 水袋束团从 `phi=140 deg` 跟踪到上述 0.5 m septum，并用相同初态做 kicker-on/off 对照。项目口径中 `500 pi mm mrad` 解释为水平几何边界发射度 `epsilon_x=500e-6 m rad`（不再额外乘 pi）；默认 4096 个水袋粒子加 720 个精确边界粒子，竖直坐标为零，RF/SC 关闭。2026-09-04 结果在 `extraction_bunch_septum_output/run_20260904_023555`：注入点 `beta_x=5.97945 m`、`alpha_x=0.99166`，最大 `|x|=54.678 mm`；4816 个粒子均到达并存活。现有零厚度 septum 中心线下，水袋中 kicked 束全程位于外侧的比例为 `83.33%`，未踢束全程位于内侧的比例同为 `83.33%`；完整 500pi 边界在 septum 全长均重叠，最小可用挡板窗口为 `-32.89 mm`（负号表示两束包络重叠），所以当前 4x5 mrad 布局无法无损分离 500pi 束团，即使挡板厚度为零也不行。
- `tool_plot_bunch_extraction_three_views.py` 从上述多粒子 NPZ 结果生成三张独立图：`01_ring_layout.png`（整环磁场、闭轨、中心引出轨迹、4 个 kicker、septum）、`02_one_turn_r_vs_azimuth_septum_location.png`（从 `140--500 deg` 的一整圈 `r-azimuth` 图，叠加 OPERA `Bz`、16 个 sector 边界/直线节、kicker 和 septum 区间；说明 septum 选在 EK4 下游下一低场直线节，中心分离 `44.1--48.9 mm`、`max|base Bz|=0.0028 T`）和 `03_septum_entrance_distribution_500pi.png`（入口局部水平相空间；横轴为相对挡板径向位置，纵轴为相对挡板方向的径向发散角）。入口位置包络重叠约 `32.46 mm`。旧 `02_septum_zoom_500pi.png` 已按用户要求删除。
- `tool_compare_kicker_splits.py` 比较四块 `0.20 m`、各 `5 mrad`、固定负极性 kicker 在前/后两个相邻直线节的 `2+2`、`1+3`、`3+1`、`0+4` 分配；保持现有 `0.20 m` 磁铁间净间隙，正式配置不改。600 MeV 单粒子生产 Boris 一圈结果在 `extraction_kicker_split_scan/run_20260906_174544`。按 SEO 上 `|Bz|<0.012 T`，前后低场窗各约 `0.882 m`：`2+2` 可安装；`1+3` 后组三块超低场窗 `173 mm`；`3+1` 前组三块超 `174 mm`；`0+4` 后组超 `573 mm`。三块若要留在低场窗，组内间隙需降到不超过约 `113 mm`；四块则需降到约 `9 mm`，不具备合理安装余量。
- `tool_optimize_septum_per_kicker_split.py` 为每种 kicker 排列独立扫描最后一块 kicker 下游各低场直线节，选择严格 `0.500 m`、全长 `|Bz|<0.012 T` 且使中心轨迹全长最小外向分离最大的 septum 窗口。最优展开方位区间 / 全长最小中心分离为：`2+2: 191.250--193.672 deg / 44.114 mm`、`1+3: 191.250--193.672 deg / 48.076 mm`、`3+1: 393.750--396.172 deg / 50.567 mm`（环上 `33.750--36.172 deg`）、`0+4: 416.250--418.671 deg / 57.346 mm`（环上 `56.250--58.671 deg`）。`02_kicker_split_one_turn_r_vs_azimuth.png` 已改为四个子图分别标各自优化后的 septum，R 轴自动聚焦闭轨/引出轨迹；`03_optimized_septum_location_scan.png` 显示各候选低场直线节的目标函数。
- 同一脚本还生成普通二维图 `05_kicker_split_delta_r_vs_azimuth.png`：横轴为一圈展开方位角，纵轴为有符号的 `Delta r=r-r_SEO [mm]`，四个子图使用相同纵轴范围，并分别标出 sector/直线节、kicker 和各自优化后的 septum；黑色水平虚线为 `Delta r=0` 的 600 MeV 闭轨基准。此前的圆形雷达式图已按用户要求删除。
- `tool_compare_kicker_splits_bunch_phase_space.py` 对上述四种排列在各自优化后的 septum 入口跟踪同一组 `4096` 水袋 + `720` 精确边界粒子的 500pi 束团，RF/SC 关闭；每种排列的零厚度 blade 都重新放在各自 kicked/off 质心中点。结果图为同目录 `04_kicker_split_optimized_septum_entrance_phase_space_500pi.png`，摘要为 `bunch_phase_space_summary.json`。入口中心分离 / 完整径向自由窗口分别为：`2+2: 48.78/-32.46 mm`、`1+3: 49.95/-31.01 mm`、`3+1: 49.15/-30.30 mm`、`0+4: 58.19/-20.67 mm`；四种完整 500pi 包络仍全部重叠。`0+4` 动力学分离最好但安装严重不可行；在当前 `0.20 m` 间隙约束下仍只有 `2+2` 能放入低场直线节。
- 三个相邻直线节的四块 kicker 比较由 `tool_compare_kicker_three_straights.py` 完成，排列为 `2+1+1`、`1+2+1`、`1+1+2`，仍采用每块 `0.20 m/5 mrad`、固定负极性、同节内 `0.20 m` 间隙。三处低场窗均约 `0.882 m`，双 kicker 组占 `0.600 m`，所以三种排列全部可安装。600 MeV 单粒子结果在 `extraction_kicker_three_straight_scan/run_20260909_095802`：独立优化的 `0.5 m` septum 全长最小中心分离分别为 `29.307 mm @ 303.750--306.174 deg`、`38.348 mm @ 303.750--306.173 deg`、`29.229 mm @ 416.250--418.674 deg`；`1+2+1` 最好。脚本支持 `--run-dir` 从已有轨迹续画而不重复跟踪。
- `tool_compare_kicker_three_straights_bunch_phase_space.py` 对这三种三直线节排列使用同一组 `4096+720` 粒子的 500pi 水袋/精确边界束团，RF/SC 关闭，并在各自优化的 septum 入口比较 kicker-on/off。入口质心分离 / 完整径向自由窗口为：`2+1+1: 32.92/-47.69 mm`、`1+2+1: 39.03/-41.10 mm`、`1+1+2: 27.56/-51.57 mm`。三种完整束团包络均重叠；最佳 `1+2+1` 也差于两直线节正式 `2+2` 的 `48.78/-32.46 mm`，说明把四次踢分散到这三个相邻直线节并未改善横向相位合成。输出图为同目录 `04_three_straight_septum_entrance_phase_space_500pi.png`。
- `tool_plot_three_straight_local_xy.py` 从三直线节扫描结果生成各方案优化 septum 处的直角坐标图，视野覆盖完整一个 `22.5 deg` cell，并在上下游各留 `2.5 deg`；底图为 OPERA 中平面 `Bz`，叠加 600 MeV SEO、踢后中心轨迹和零厚度 0.5 m septum。`run_20260909_095802/05_three_straight_local_xy_field_trajectory.png` 中三种四-kicker方案的 septum 入口中心分离为 `2+1+1: 32.883 mm`、`1+2+1: 39.797 mm`、`1+1+2: 29.280 mm`。脚本支持 `--scan-dir`、`--labels` 和 `--output`。
- 六块 `2+2+2` 方案（2026-09-09）已加入上述两个三直线节脚本，结果位于 `extraction_kicker_three_straight_scan/run_20260909_155228`。三组分别从 `146.521/169.021/191.521 deg` 开始，每组两块 `0.20 m` kicker 加 `0.20 m` 净间隙，总占长 `0.600 m`，均可安装在约 `0.882 m` 的低场窗内。600 MeV 单粒子一圈的 `Rmax=12.080868 m`；独立优化的 `0.5 m` septum 为展开方位角 `303.750--306.172 deg`，全长最小中心分离 `47.520 mm`。同一 `4096+720` 粒子、500pi 水袋/边界束团在入口的质心分离 / 完整径向自由窗口为 `48.174/-32.520 mm`，水袋位于挡板正确一侧的比例为 on `85.06%`、off `86.67%`。与两直线节 `2+2` 的 `48.784/-32.462 mm` 相比没有形成实质改善，完整束团仍重叠。
- 600 MeV外侧磁场检查（2026-09-15）：`tool_plot_outer_field_behavior.py` 输出 `extraction_kicker_three_straight_scan/run_20260909_095802/06_outer_field_radial_behavior_600MeV.png`。600 MeV SEO径向范围为 `11.789473--12.020094 m`，最大半径距OPERA `rmax=12.2 m` 约 `180 mm`。在 `r=12.2 m`，单 cell 中平面 `Bz` 仍横跨 `-0.989185--+1.839846 T`，场包络总体向外增强而非衰减到零；直线节中心场虽接近零，但磁铁区仍为强场。`src/FFAG_Field.py` 的生产插值在 `r>rmax` 将 `Bz/Br/Bfi` 直接置零，且当前 `config_track.json` 外孔径也是 `12.2 m`，所以边界外零场是数值截断而非真实 fringe。现有SEO只到600 MeV；按550--600 MeV最大闭轨半径增量线性外推，达到12.2 m约在735 MeV，但此数只能作量级估计，不能用于设计。研究能量驱动的外移引出前需要扩大OPERA径向范围并覆盖真实漏场，然后重算高能闭轨、tune、稳定性和束团包络。

## 3D OPERA 磁场当前状态

近期加入了真实 3D OPERA 磁场模型支持，目标是让跟踪器同时兼容：

- `FFAG_Bfield_analytical`：原有解析 FFAG 场 + 高阶 Taylor 数值 map 修正。
- `FFAG_Bfield_3D_opera`：OPERA 导出的单 cell 三维场表。

主要修改点：

- `FFAG_Field.py`
  - `FFAG_Bfield_analytical` 增加 `field_type = "analytical"`。
  - 新增 `Opera3DFieldMapMetadata`，读取并检查 `Bmaps/OperaFieldMap/FD16.json` 元数据。
  - 新增 `FFAG_Bfield_3D_opera`，读取 `FD16.npz` 中的 `br/bphi/bz`，并提供与解析场一致的 `GetBfields(...)` 接口。
  - 新增 3D 表格场插值 numba helper，包括全局方位角到 OPERA 表格角度的映射，以及 `(y,r,phi)` 三线性插值。
  - 跟踪插值在 OPERA 最大半径处仍使用 map 最外层数据；`r > r_max` 时 `Bz/Br/Bfi` 全部置零，不再截断到最外层径向数据。SEO `SearchMod=True` 的径向磁墙逻辑不受影响。
  - 已移除早期测试类/模式：`opera3d_toy`、`taylor_map` 相关测试残留不再作为正式分支使用。
- `FFAG_track.py`
  - 跟踪中直接按 `Bmap.field_type` 分支，不再封装成统一 helper。
  - `analytical`：使用 cell-local `fi_vect = fi_global mod sector_period` 调用 `GetBfields(...)`，之后按 `max_order` 叠加 `AddHigherOrderBmaps(...)`。
  - `opera3d`：使用全局 `fi_global` 调用 `GetBfields(...)`，由 `FFAG_Bfield_3D_opera` 按 OPERA 表格 `phi_min` 重新折叠到真实 cell；不再额外叠加 Taylor map。
  - `FunctionForSEO(...)` 和 `FunctionForSEOVect(...)` 中也已经做了同样的 OPERA 角度兼容；`opera3d` 用全局 `fi`，其它场仍用 `fi_vect_mod`。
- 测试与可视化
  - `tools/tool_test_Opera3D_field.py`：当前保留的 OPERA 3D 数据读取和基础可视化检查。
  - 历史解析场/OPERA 联合对比工具已经不在当前工作区；相关 `+3.7 deg` 画图相位只作为历史记录，不属于生产磁场查询。

当前数据位置：

- 解析/Taylor map：`Bmaps/Bmap_FD16/`
- OPERA 3D 数据：`Bmaps/OperaFieldMap/FD16.json`、`FD16.npz`、`FD16.TABLE`、`FD16.grid`

常用验证：

```bash
python -m py_compile src/FFAG_Field.py src/FFAG_track.py tools/tool_test_Opera3D_field.py
python -c "import FFAG_Field, FFAG_track; print('imports ok')"
python tools/tool_test_Opera3D_field.py --field-dir Bmaps/OperaFieldMap
```

## SEO / 3D 场接口清理当前状态

近期对 SEO 右端函数、RK4 step 和 3D 场接口做了精简，当前约定如下：

- `FFAG_track.py`
  - `FunctionForSEO(...)` 和 `FunctionForSEOVect(...)` 现在都是薄壳，统一调用 `FunctionForSEOUnified(...)`。
  - `FunctionForSEOVectBz(...)` 已删除；Beta/Twiss 链条改为直接使用 `FunctionForSEOUnified(...)`。
  - `rk4_step_2(...)` 已删除；`rk4_step(..., return_bfield=True)` 负责返回 `(dr_step, Bz, Br, Bfi)`。
  - `rk4_solve_vect_3DMatrix(...)` 通过 `rk4_step(..., return_bfield=True)` 收集 `Bz/Br/Bfi` 轨迹。
- `FFAG_Field.py`
  - 正式 Bmap 接口保留 `GetBz(...)` 和 `GetBfields(...)`。
  - `GetBr(...)`、`GetBfi(...)` 已删除，当前仓库内无调用点。
  - `GetBz(..., SearchMod=True)` 是 SEO 中平面搜索使用的径向磁墙接口：`r < r_min` 给 `-Binf`，`r > r_max` 给 `+Binf`。
  - `GetBfields(..., SearchMod=...)` 不设置磁墙，解析场和 OPERA 3D 场均保持这一语义。
- OPERA 3D 方位角约定：
  - `FFAG_Bfield_3D_opera.GetBz(...)` 和 `GetBfields(...)` 的 `fi` 都必须传全局方位角 `fi_global`。
  - OPERA 类内部按真实 `phi_min` 折叠到单 cell 表格范围。
  - analytical 场仍使用 `fi_global mod sector_period`。
- `FunctionForSEOUnified(...)` 对 OPERA 3D 的处理：
  - 中平面闭轨搜索，即 `z == 0` 且 `pz == 0`，使用 `GetBz(..., SearchMod=True)`。
  - Twiss/Beta 扰动或需要完整 3D 场时，使用 `GetBfields(..., SearchMod=False)`，保留 `Br/Bfi`。

当前链条统一状态：

- `examples/main_SEO.py` 已可按 `Bmap_path` 自动识别 analytical Bmap 目录或 OPERA 3D 数据目录。
- `src/FFAG_BetaFunc.py` 已可从 SEO 的 Bmap 路径自动恢复 analytical 或 OPERA 3D Bmap。
- `examples/main_track.py` 已有 `load_bmap_auto(...)`（Bmap 目录下存在 `config_Bmap.json` 时用 analytical，否则按 OPERA 3D 文件夹加载），Bmap 自动识别链条已统一。

## SEO 当前跑通状态与后续重点

- `examples/config_SEO.py` 当前可直接设置 `Bmap_path = './Bmaps/OperaFieldMap'`；`examples/main_SEO.py` 会自动加载 OPERA 3D 场，不需要手动写 `FD16` 或 `Nsectors=16`。
- OPERA 3D 的 `FFAG_Bfield_3D_opera` 会自动从唯一 `.json/.npz` 文件对识别 dataset，并优先从 `phi_max_deg - phi_min_deg` 推断 cell 周期数。
- 已做过轻量 smoke test：`SearchSEOUsingInitialEkVect(300.0, h=0.01, verbose=True)` 可收敛，返回约 `r0=11.4540848937 m, pr0=-0.1376984208`。正式精度仍以 `python examples/main_SEO.py -j config_SEO.json` 的 `h=0.001` 运行结果为准。
- 当前闭轨搜索约定：
  - 无 `BHarmonics` 时，最终闭轨条件按一个 cell 搜索；保存 `SEO_r.txt/SEO_pr.txt` 时重新跟踪一整圈。
  - 有 `BHarmonics` 时，粗搜索仍从一个 cell 开始，但最终闭轨精修按一整圈。
  - `src/FFAG_BetaFunc.py` 当前 Twiss/Beta 计算按一整圈 map，并输出一整圈曲线。
- 已知后续要优先处理：
  - `GetQrQz(...)` 仍按一个 cell 计算 tune 并乘 `Nsectors`；若 `BHarmonics.enable=True` 或场不严格 cell 周期，应改用整圈 tune 逻辑。

## OPERA 3D 反向注入轨迹当前状态

- 详细记录见 `tips.md` 的 “OPERA 3D 300MeV 反向注入轨迹记录”。
- 当前脚本：
  - `project_reverse_injection/scripts/tool_inj_gap_opera3d.py`：从 300 MeV OPERA SEO 结果中找低场直线节/foil 点。
  - `project_reverse_injection/scripts/tool_reverse_inj_opera3d.py`：在 foil 点构造单粒子，转 Boris 变量后反转速度列 `[1, 3, 5]`，用正时间积分得到剥离前反推轨迹。
- 当前默认 foil 点约为：
  - `phi=11.632312 deg`
  - `r=11.2874182201 m`
  - `pr=0.0728759830`
  - `Ek=300 MeV`
- 反推暂不考虑 SC、RF、BHarmonics。
- 已在反推脚本中设置外半径孔径 `APERTURE_R_MAX_M = 12.0`，并且后处理只保存 `R<=12 m` 的正式结果。
- 当前典型输出位于 `opera3d_reverse_track_output/`：
  - `opera3d_300MeV_reverse_injection_Rle12_track.csv`
  - `opera3d_300MeV_reverse_injection_Rle12_coords_fields.csv`
  - `opera3d_300MeV_reverse_injection_Rle12_r_fi_bz.csv`
  - `opera3d_300MeV_reverse_injection_on_Bz.png`
  - `opera3d_300MeV_reverse_injection_Bz_along_s.png`
- `Rle12_r_fi_bz.csv` 是兼容旧 `1p05.csv` 的三列格式：`r,fi,bz`。
- 最近一次验证结果：`R<=12 samples=220`，`Rmax=11.9925528726 m`，路径长约 `4.284428 m`。

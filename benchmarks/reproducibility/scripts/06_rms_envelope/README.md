# Comment A-4：RMS 横向包络

审稿人对原 Fig. 11 提出了两个独立要求：

1. 将竖直坐标的记号由 `y` 统一为全文使用的 `z`，因此纵轴应写为 `\sigma_z`；
2. 同时给出对应的水平 RMS 包络 `\sigma_x`。

因此，仅把原图的纵轴从 `\sigma_y` 改成 `\sigma_z` 并不能完整回答该意见。论文候选结果应同时包含水平和竖直包络。当前脚本不在 Python 中合图，而是分别输出 with-SC/no-SC 的 `\sigma_x` 和 `\sigma_z` 四张独立图片，方便随后在 LaTeX 中排版。

## 脚本

`Benchmark_SC_envlope_stable.py` 是仓库根目录同名脚本的材料包整理版。原文件没有改动。整理版保留原包络方程、初值和 Euler–Cromer 积分口径，并作了以下调整：

- 跟踪数据同时计算 `sigma_x=std(r)` 和 `sigma_z=std(z)`；在固定方位角处，减去平均半径后 `std(r)` 与局部水平 RMS 尺寸相同；
- 理论部分复用原脚本已经求出的两个横向包络；
- 分别输出 with-SC/no-SC 的水平、竖直四张独立图片，不在脚本中组合；
- 每张图复用原始脚本的 `10×4` 尺寸、字体、蓝色理论曲线、红色黑边星形跟踪点、线宽、网格和 300 dpi；
- 水平两图统一使用 `1.2--2.8 cm`，竖直两图统一使用 `1.5--3.5 cm`，便于直接比较空间电荷开关前后的包络变化；
- 四张图统一使用 `ylim=[1.2,3.5] cm`，便于直接比较包络幅度；
- 每个 crossing 文件只读取一次，并仅加载 `r,z,phi` 三列；
- 路径改为相对仓库根目录自动解析，并增加统一参数区和 CSV 数值输出。

脚本顶部的“用户参数区”直接指定两个 case：

- `output/Benchmark_env_2e14/merged_files`：with SC；
- `output/Benchmark_env_2e13/merged_files`：without SC。

with-SC 理论使用 `2.0e14` 个质子对应的线电荷；without-SC 理论令空间电荷项为零。脚本不使用命令行参数。

从仓库根目录运行当前 with-SC case：

```powershell
python benchmarks/reproducibility/scripts/06_rms_envelope/Benchmark_SC_envlope_stable.py
```

直接运行后输出到本目录的 `outputs/`：

- `fig_rms_env_sc_x.png`；
- `fig_rms_env_sc_z.png`；
- `fig_rms_env_nosc_x.png`；
- `fig_rms_env_nosc_z.png`；
- `rms_envelope_with_sc_tracking.csv`；
- `rms_envelope_without_sc_tracking.csv`。

若只想快速检查少量 crossing 文件，在脚本参数区把 `MAX_FILES=None` 临时改为正整数；正式出图时恢复为 `None`。

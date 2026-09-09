# Reverse Injection Project

本目录集中保存 PyFFA 的 300 MeV 剥离注入点轨迹反推、解析磁场参数扫描、OPERA 3D 场反推和动力学孔径（DA）研究。

## 文档入口

- `AGENTS.md`：反推方法、物理约定、脚本入口、参数扫描和结果交接的完整说明。
- `DA_README.md`：动力学孔径的定义、扫描参数、收敛口径和批量运行方法。
- `lattice_da_200turn_R265_R365_R465_300MeV/README.md`：对应一组正式 200 圈 DA 结果。
- `lattice_da_preview_R265_R365_300MeV/PREVIEW_README.md`：早期预扫描结果说明。

## 运行方式

所有命令都应从 PyFFA 根目录执行。Windows PowerShell：

```powershell
.\SetEnvCmd.ps1
python .\project_reverse_injection\scripts\tool_inj_gap_opera3d.py
python .\project_reverse_injection\scripts\tool_reverse_inj_opera3d.py
```

Linux/HPC：

```bash
source ./SetEnvCmd.sh
python project_reverse_injection/scripts/tool_reverse_inj_opera3d.py
```

不要把当前工作目录切换到 `scripts/`；脚本中的 `./Bmaps`、`./outputs` 和结果目录均以 PyFFA 根目录为基准。

## 主要代码

- `scripts/tool_inj_gap_opera3d.py`：从 OPERA SEO 结果寻找低场直线节和 foil 点。
- `scripts/tool_reverse_inj_opera3d.py`：OPERA 3D 场中的反向注入轨迹。
- `scripts/tool_reverse_inj_compare_k_analytical.py`：多个解析场方案对比。
- `scripts/tool_scan_reverse_R_F_offsets_grid.py`：R/F/offset 网格扫描。
- `scripts/tool_scan_reverse_R_F_offsets_wide.py`：大负 offset 扩展扫描。
- `scripts/tool_scan_lattice_da.py`：固定能量横向动力学孔径扫描。
- `scripts/tool_run_rf_grid_lattice_da_scans.py`：R/F 网格的串行 DA 批处理。

核心 `FFAG_*` 模块仍由仓库根目录的 `src/` 提供，`main_SEO` 由 `examples/` 提供。

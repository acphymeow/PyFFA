# Opera 三维磁场数据格式与 PyFFA 读取指南

## 1. 文件命名与处理流程

本流程统一使用以下文件名：

```text
FD16.grid    # Opera 输入坐标网格，内容采用 Opera TABLE 格式
FD16.json    # 网格、坐标系和数组排列信息
FD16.table   # Opera 导出的坐标与磁场数据
FD16.npz     # 转换后的 PyFFA 三维磁场数据
```

处理流程：

```text
tool_generate_FD16_grid.py
    -> FD16.grid + FD16.json
    -> Opera 打开已求解的 OP3 文件
    -> 运行 COMI，生成 FD16.table
    -> tool_convert_FD16_field.py
    -> FD16.npz
```

说明：上面两个脚本名表示通用数据准备流程，当前仓库并未附带这两个同名脚本。仓库现有的 `tools/tool_generate_FD_cell_opera_table.py` 只负责生成当前 FD cell 的 OPERA 查询表，不能替代完整的 `FD16.table -> FD16.npz` 转换步骤。

所有网格范围、步长、模型原点和数组尺寸均以 `tool_generate_FD16_grid.py` 的参数区及其生成的 `FD16.json` 为准。本文档不重复固定这些参数值。

`.grid` 仅用于区分输入网格与输出磁场表，其内部仍采用标准 Opera TABLE 格式。

---

## 2. 坐标系

加速器全局坐标采用柱坐标：

```text
(R, phi, z)
```

Opera 模型采用局部直角坐标：

```text
(X_opera, Y_opera, Z_opera)
```

设模型原点在全局柱坐标中的位置为：

```text
(R0, phi0, z0)
```

当前坐标轴约定为：

```text
X_opera：径向
Y_opera：竖直方向
Z_opera：切向
```

从全局柱坐标转换到 Opera 局部坐标：

```text
delta_phi = phi - phi0

X_opera = R*cos(delta_phi) - R0
Y_opera = z - z0
Z_opera = R*sin(delta_phi)
```

在模型原点处应满足：

```text
X_opera = 0
Y_opera = 0
Z_opera = 0
```

模型原点参数由 `FD16.json` 中的 `coordinate_system` 字段读取。

---

## 3. 网格定义

网格由以下三个一维坐标轴组成：

```text
y_axis       # 竖直坐标
r_axis       # 全局半径
phi_axis     # 全局方位角
```

网格参数包括：

```text
y_min, y_max, dy
r_min, r_max, dr
phi_min, phi_max, dphi
```

对应点数为：

```text
ny
nr
nphi
```

三维数组形状统一为：

```python
(ny, nr, nphi)
```

总点数为：

```python
ny * nr * nphi
```

网格生成采用：

```python
yy, rr, phi = np.meshgrid(y_axis, r_axis, phi_axis, indexing="ij")
```

因此数组轴顺序和点排列顺序为：

```text
axis_order  : y, r, phi
point_order : Y -> R -> phi
flatten     : C order
```

实际网格范围、步长、点数和数组形状均写入 `FD16.json`，后续转换脚本不得重新手动填写。

---

## 4. FD16.grid 格式

`FD16.grid` 必须包含 Opera TABLE 标准表头：

```text
<total_points> 1 1 2
1 X [METRE]
2 Y [METRE]
3 Z [METRE]
0
X1 Y1 Z1
X2 Y2 Z2
...
```

其中：

```text
<total_points> = ny * nr * nphi
```

表头共 5 行，数值数据从第 6 行开始。

不要在第一行之前添加注释、说明文字或其他自定义表头，否则 Opera 可能无法读取。

---

## 5. FD16.json 内容

`FD16.json` 保存以下信息：

```text
format
version
coordinate_system
grid
array_layout
coordinate_table
```

主要字段结构：

```json
{
    "coordinate_system": {
        "model_origin_radius_m": "...",
        "model_origin_azimuth_deg": "..."
    },
    "grid": {
        "y_min_m": "...",
        "y_max_m": "...",
        "dy_m": "...",
        "ny": "...",
        "r_min_m": "...",
        "r_max_m": "...",
        "dr_m": "...",
        "nr": "...",
        "phi_min_deg": "...",
        "phi_max_deg": "...",
        "dphi_deg": "...",
        "nphi": "..."
    },
    "array_layout": {
        "shape": ["ny", "nr", "nphi"],
        "axis_order": ["y", "r", "phi"],
        "flatten_order": "C",
        "point_order": "Y -> R -> phi"
    }
}
```

转换脚本读取数组形状时应使用：

```python
ny, nr, nphi = metadata["array_layout"]["shape"]
```

不能假设 `shape` 位于 JSON 顶层。

---

## 6. Opera COMI 导出

在 Opera 中打开对应的已求解结果文件，然后执行：

```text
TABLE INFILE='FD16.grid' OUTFILE='FD16.table' COLUMNS=6 F1=X UNIT1T=LENGU F2=Y UNIT2T=LENGU F3=Z UNIT3T=LENGU F4=BX UNIT4T=FLUXU F5=BY UNIT5T=FLUXU F6=BZ UNIT6T=FLUXU UNIT7T=1 UNIT8T=1 UNIT9T=1 UNIT10T=1 UNIT11T=1 UNIT12T=1
```

`FD16.table` 的六列依次为：

```text
X_opera
Y_opera
Z_opera
BX_opera
BY_opera
BZ_opera
```

Opera 导出的 `BX、BY、BZ` 是局部直角坐标分量，不是加速器全局柱坐标分量。

---

## 7. 磁场分量转换

令：

```text
delta_phi = phi - phi0
```

局部直角坐标磁场转换为全局柱坐标磁场：

```text
BR   = BX_opera*cos(delta_phi) + BZ_opera*sin(delta_phi)
Bphi = -BX_opera*sin(delta_phi) + BZ_opera*cos(delta_phi)
Bz   = BY_opera
```

在模型中心方位角 `phi = phi0` 处，应满足：

```text
BR   = BX_opera
Bphi = BZ_opera
Bz   = BY_opera
```

这是检查坐标转换方向的重要条件。

---

## 8. FD16.table 读取

Opera TABLE 的数据起始标志为单独一行：

```text
0
```

转换脚本先查找该行，再读取后续数值数据。

读取后应检查：

```python
expected_points = ny * nr * nphi
```

并确认：

```python
data.shape[0] == expected_points
data.shape[1] >= 6
```

如果找不到 `0` 行，应先查看 `FD16.table` 的前若干行，不要直接猜测 `skiprows`。

---

## 9. 三维数组恢复

读取的六列数据按以下方式恢复：

```python
x_opera = data[:, 0].reshape(ny, nr, nphi)
y_opera = data[:, 1].reshape(ny, nr, nphi)
z_opera = data[:, 2].reshape(ny, nr, nphi)

bx_opera = data[:, 3].reshape(ny, nr, nphi)
by_opera = data[:, 4].reshape(ny, nr, nphi)
bz_opera = data[:, 5].reshape(ny, nr, nphi)
```

数组索引含义：

```text
[iy, ir, iphi]
```

即：

```text
[竖直位置索引, 半径索引, 方位角索引]
```

---

## 10. FD16.npz 内容

转换后的 `FD16.npz` 建议保存：

```text
y_axis
r_axis
phi_axis_deg
phi_axis_rad

x_opera
y_opera
z_opera

bx_opera
by_opera
bz_opera

br
bphi
bz
```

PyFFA 中读取：

```python
import numpy as np

field = np.load("FD16.npz")

y_axis = field["y_axis"]
r_axis = field["r_axis"]
phi_axis_rad = field["phi_axis_rad"]

br = field["br"]
bphi = field["bphi"]
bz = field["bz"]
```

---

## 11. 配套脚本

完整代码分别保存在：

```text
tool_generate_FD16_grid.py
tool_convert_FD16_field.py
```

其中：

- `tool_generate_FD16_grid.py`：在参数区设置实际网格和模型原点，生成 `FD16.grid` 与 `FD16.json`；
- `tool_convert_FD16_field.py`：只从 `FD16.json` 读取网格信息，将 `FD16.table` 转换为 `FD16.npz`。

文档不复制脚本中的具体参数，以避免脚本更新后文档参数失效。

---

## 12. 检查要点

1. `FD16.grid` 第一行点数必须等于 `ny * nr * nphi`。
2. `FD16.grid` 必须包含 5 行 Opera TABLE 表头。
3. `FD16.table` 应至少包含 `X Y Z BX BY BZ` 六列。
4. `FD16.table` 的数据行数必须与 `FD16.json` 中的网格尺寸一致。
5. 三维数组形状必须与 `metadata["array_layout"]["shape"]` 一致。
6. reshape 顺序必须与网格生成时的 `C order` 一致。
7. 在 `phi = phi0` 处检查 `BR=BX_opera`、`Bphi=BZ_opera`、`Bz=BY_opera`。
8. 修改网格或模型原点后，应重新生成 `FD16.grid`、`FD16.json`、`FD16.table` 和 `FD16.npz`。

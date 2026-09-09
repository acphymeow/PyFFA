# Comment B-4：合并原 Fig. 2、3、4

审稿人建议将原 Fig. 2、3、4 合并，并在正文中引用，以帮助读者理解全局柱坐标、沿能量相关闭轨重建的局部坐标，以及粒子到参考闭轨的投影关系。

## 推荐入口

从仓库任意目录运行：

```powershell
python benchmarks/reproducibility/scripts/07_fig234_coordinate_systems/make_fig234_coordinate_systems.py
```

脚本依次生成：

- `outputs/fig23_combined_proj.png`：合并原 Fig. 2、3。两条能量相关闭轨及束团说明不存在单一固定参考轨道；局部基沿闭轨变化，并与全局柱坐标基进行对照。
- `outputs/figure_global2local_panelb.png`：原 Fig. 4 的投影构造。同方位角参考点与法向投影点分别给出径向位移和局部横向坐标的定义。
- `outputs/fig234_ab_stitched.png`：建议用于论文的最终两联组合图。两个 panel 采用紧凑的左右排列，并裁去外围空白；程序保存三个文件后只弹出最终组合图。

输入数据为 `Bmap_FD16/resultsSEO_noError/SEO_r.txt`。路径由材料包的 `bundle_paths.py` 自动定位，不依赖运行时工作目录。

## 文件说明

| 文件 | 作用 |
|---|---|
| `make_fig234_coordinate_systems.py` | 一键生成两个源 panel 并拼接为最终图；推荐使用。 |
| `plot_fig23_combined_proj.py` | 原 Fig. 2+3 的完整绘图源码快照。 |
| `plot_fig4_panel_b.py` | 原 Fig. 4 投影构造的完整绘图源码快照。 |
| `stitch_fig23_fig4.py` | 不改变源图内容，只负责最终版式拼接和 panel 标号。 |

较早的三栏重画尝试未收入本复现目录；其信息密度较高，不作为这里的推荐版本。

## Manuscript 集成

2026-08-28 已将 `outputs/fig234_ab_stitched.png` 复制为 manuscript 的 `figure_coordinate_systems_combined.png`，替换原 Fig. 2、3、4。合并图现为 Fig. 2，正文分别引用 (a) 的能量相关局部坐标和 (b) 的法向投影构造。

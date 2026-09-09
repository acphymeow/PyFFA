# -*- coding: utf-8 -*-
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from test_inj_bunch import load_config_from_json, build_full_ring_bmap


# =========================================================
# 1. 单个粒子文件关于 x=0 镜像
# =========================================================
def mirror_particle_file_x0(src_path, dst_path):
    data_npz = np.load(src_path)
    particles = data_npz['particles'].copy()

    # 第4列为 fi
    # 关于 x=0 镜像:
    # x = r cos(fi), y = r sin(fi)
    # x' = -x, y' = y
    # 因此 fi' = pi - fi
    particles[:, 4] = (np.pi - particles[:, 4]) % (2.0 * np.pi)

    save_dict = {}
    for key in data_npz.files:
        if key == 'particles':
            save_dict[key] = particles
        else:
            save_dict[key] = data_npz[key]

    np.savez(dst_path, **save_dict)


# =========================================================
# 2. 批量镜像整个文件夹
# =========================================================
def mirror_all_particle_files(src_folder, dst_folder, pattern='particle_*_rank_0.npz'):
    os.makedirs(dst_folder, exist_ok=True)

    particle_files = sorted(glob.glob(os.path.join(src_folder, pattern)))
    if len(particle_files) == 0:
        raise FileNotFoundError(f'在文件夹 {src_folder} 中没有找到 {pattern} 文件')

    for src_path in particle_files:
        dst_path = os.path.join(dst_folder, os.path.basename(src_path))
        mirror_particle_file_x0(src_path, dst_path)

    mirrored_files = sorted(glob.glob(os.path.join(dst_folder, pattern)))
    return particle_files, mirrored_files


# =========================================================
# 3. 从 config + npz 构造整圈磁场背景，并做 x=0 镜像
# =========================================================
def build_mirrored_bmap_from_config(npz_path, npz_key, bmap_json, total_cells, thin=10):
    data = np.load(npz_path)[npz_key]

    with open(bmap_json, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    Bmap_cfg = config_data['Bmap']
    rmin_max_step_m = Bmap_cfg["rmin_max_step_m"]
    theta_step_rad = Bmap_cfg["theta_step_rad"]
    nsector_from_json = int(Bmap_cfg["NSector"])

    if nsector_from_json != total_cells:
        print(
            f"[warning] config_Bmap.json 中 NSector={nsector_from_json}, "
            f"但当前使用 total_cells={total_cells}。"
        )

    rmin, rmax, rstep = rmin_max_step_m
    n_r_steps = int((rmax - rmin) / rstep) + 1
    n_fi_steps = int((2.0 * np.pi / total_cells) / theta_step_rad) + 1

    fi_axis_single_cell = np.linspace(0.0, 2.0 * np.pi / total_cells, n_fi_steps)
    r_axis = np.linspace(rmin, rmax, n_r_steps)

    if data.shape != (len(r_axis), len(fi_axis_single_cell)):
        raise RuntimeError(
            f"data.shape = {data.shape}, expected = ({len(r_axis)}, {len(fi_axis_single_cell)})"
        )

    data_full, fi_axis_full, mesh_x, mesh_y = build_full_ring_bmap(
        data=data,
        r_axis=r_axis,
        fi_axis_single_cell=fi_axis_single_cell,
        total_cells=total_cells,
        thin=thin
    )

    # 关于 x=0 镜像：x -> -x
    mesh_x_mirror = -mesh_x
    mesh_y_mirror = mesh_y.copy()

    return data_full, mesh_x_mirror, mesh_y_mirror, r_axis, fi_axis_full


# =========================================================
# 4. 稳健的 TwoSlopeNorm
# =========================================================
def make_bz_norm(data_full):
    vmin = np.nanmin(data_full)
    vmax = np.nanmax(data_full)

    if vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    return None


# =========================================================
# 5. 在 data_full(r, fi) 上插值 Bz
# =========================================================
def interpolate_bz_rphi(r_points, fi_points, r_axis, data_full):
    """
    在 data_full(r, fi) 上插值 Bz。

    假设:
    - data_full.shape = (len(r_axis), n_phi)
    - fi 方向覆盖 [0, 2pi)
    - fi 方向周期边界
    - fi 方向均匀网格
    """

    r_points = np.asarray(r_points)
    fi_points = np.asarray(fi_points)

    nr, nphi = data_full.shape

    if nr != len(r_axis):
        raise RuntimeError(
            f"data_full.shape[0] = {nr}, but len(r_axis) = {len(r_axis)}"
        )

    bz = np.full_like(r_points, np.nan, dtype=float)

    # r 越界的点给 NaN
    valid = (r_points >= r_axis[0]) & (r_points <= r_axis[-1])
    if not np.any(valid):
        return bz

    r_valid = r_points[valid]
    fi_valid = np.mod(fi_points[valid], 2.0 * np.pi)

    # ---------- r 方向索引 ----------
    i1 = np.searchsorted(r_axis, r_valid, side='right')
    i0 = i1 - 1

    i0 = np.clip(i0, 0, nr - 2)
    i1 = i0 + 1

    r0 = r_axis[i0]
    r1 = r_axis[i1]
    tr = (r_valid - r0) / (r1 - r0)

    # ---------- fi 方向周期索引 ----------
    dfi = 2.0 * np.pi / nphi
    u = fi_valid / dfi

    j0 = np.floor(u).astype(int) % nphi
    j1 = (j0 + 1) % nphi
    tf = u - np.floor(u)

    # ---------- 双线性插值 ----------
    b00 = data_full[i0, j0]
    b10 = data_full[i1, j0]
    b01 = data_full[i0, j1]
    b11 = data_full[i1, j1]

    bz_valid = (
        (1.0 - tr) * (1.0 - tf) * b00
        + tr * (1.0 - tf) * b10
        + (1.0 - tr) * tf * b01
        + tr * tf * b11
    )

    bz[valid] = bz_valid
    return bz


# =========================================================
# 6. 计算轨迹累计弧长 s
# =========================================================
def compute_s_along_trajectory(r_traj, fi_traj):
    """
    根据极坐标轨迹估算累计弧长 s。

    ds^2 = dr^2 + (r_mid dfi)^2
    """

    r_traj = np.asarray(r_traj)
    fi_unwrap = np.unwrap(np.asarray(fi_traj))

    dr = np.diff(r_traj)
    dfi = np.diff(fi_unwrap)
    r_mid = 0.5 * (r_traj[:-1] + r_traj[1:])

    ds = np.sqrt(dr ** 2 + (r_mid * dfi) ** 2)

    s = np.zeros_like(r_traj, dtype=float)
    s[1:] = np.cumsum(ds)

    return s


# =========================================================
# 7. 提取单条轨迹上的 Bz
# =========================================================
def get_bz_along_trajectory(
    particle_file,
    r_axis,
    data_full,
    field_mode='mirrored_background'
):
    """
    从单条轨迹中提取 Bz。

    field_mode:

    1. original
       直接在原始整圈 Bmap 中取:
       Bz = Bz(r, fi)

    2. mirrored_background
       和当前镜像显示的磁场背景一致。

       镜像背景使用:
       mesh_x_mirror = -mesh_x

       对于显示坐标 fi_display，对应原始 Bmap 的:
       fi_original = pi - fi_display

       所以取场:
       Bz = Bz(r, pi - fi)
    """

    data = np.load(particle_file)['particles']

    r_traj = data[:, 0]
    fi_traj = np.mod(data[:, 4], 2.0 * np.pi)

    if field_mode == 'original':
        fi_for_field = fi_traj

    elif field_mode == 'mirrored_background':
        fi_for_field = (np.pi - fi_traj) % (2.0 * np.pi)

    else:
        raise ValueError("field_mode must be 'original' or 'mirrored_background'")

    bz_traj = interpolate_bz_rphi(
        r_points=r_traj,
        fi_points=fi_for_field,
        r_axis=r_axis,
        data_full=data_full
    )

    s_traj = compute_s_along_trajectory(r_traj, fi_traj)

    return r_traj, fi_traj, s_traj, bz_traj


# =========================================================
# 8. 画单条轨迹上的 Bz(s)
#    文件数据顺序是 inner -> outer
#    但图中显示 outer -> inner
# =========================================================
def plot_bz_along_trajectory_outer_to_inner(
    particle_file,
    r_axis,
    data_full,
    field_mode='mirrored_background',
    save_path=None
):
    r_traj, fi_traj, s_traj, bz_traj = get_bz_along_trajectory(
        particle_file=particle_file,
        r_axis=r_axis,
        data_full=data_full,
        field_mode=field_mode
    )

    print(f"[raw trajectory order] {os.path.basename(particle_file)}")
    print(f"raw r_start = {r_traj[0]:.6f} m, raw r_end = {r_traj[-1]:.6f} m")

    # -----------------------------------------------------
    # 你的文件顺序是 inner -> outer
    # 第二张图为了显示 outer -> inner，这里整体反转
    # -----------------------------------------------------
    r_plot = r_traj[::-1]
    fi_plot = fi_traj[::-1]
    bz_plot = bz_traj[::-1].copy()

    valid_rows = ~np.isnan(bz_plot)
    r_plot = r_plot[valid_rows]
    fi_plot = fi_plot[valid_rows]
    bz_plot = bz_plot[valid_rows]
    if bz_plot.size == 0:
        raise ValueError('Bz trajectory contains no valid rows after removing NaN values')

    if save_path is None:
        particle_path = os.path.abspath(particle_file)
        particle_stem = os.path.splitext(os.path.basename(particle_path))[0]
        save_path = os.path.join(
            os.path.dirname(particle_path),
            f'{particle_stem}_r_fi_bz.csv'
        )

    save_path = os.path.abspath(save_path)
    save_folder = os.path.dirname(save_path)
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)

    plot_data = np.column_stack((r_plot, fi_plot, bz_plot))
    np.savetxt(
        save_path,
        plot_data,
        delimiter=',',
        header='r,fi,bz',
        comments='',
        fmt='%.12e'
    )
    print(f'[saved plot data] {save_path}')

    # 反转后重新计算 s，使 s=0 对应外侧起点
    s_plot = compute_s_along_trajectory(r_plot, fi_plot)

    print(f"[Bz plot order] outer -> inner")
    print(f"plot r_start = {r_plot[0]:.6f} m, plot r_end = {r_plot[-1]:.6f} m")

    if r_plot[0] > r_plot[-1]:
        print("plot direction: outside -> inside")
    else:
        print("plot direction: inside -> outside")

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(s_plot, bz_plot, lw=1.5)

    ax.scatter(
        s_plot[0],
        bz_plot[0],
        s=45,
        marker='o',
        label='start / outer'
    )

    ax.scatter(
        s_plot[-1],
        bz_plot[-1],
        s=45,
        marker='s',
        label='end / inner'
    )

    ax.set_xlabel('s along trajectory [m]')
    ax.set_ylabel('Bz [T]')
    ax.set_title(
        f'Bz along trajectory\n'
        f'{os.path.basename(particle_file)}\n'
        f'outer -> inner, field_mode = {field_mode}'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()


# =========================================================
# 9. 画镜像后的磁场背景 + 镜像轨迹 + 注入点
#    第一张图保持原样：注入点取 data[0]
# =========================================================
def plot_mirrored_bmap_with_trajectories(
    mirrored_particle_files,
    data_full,
    mesh_x_mirror,
    mesh_y_mirror,
    figsize=(8, 6),
    cmap='RdBu_r'
):
    norm = make_bz_norm(data_full)

    plt.figure(figsize=figsize)

    cf = plt.contourf(
        mesh_x_mirror,
        mesh_y_mirror,
        data_full,
        levels=120,
        cmap=cmap,
        norm=norm
    )

    inj_xy = None

    for i, particle_file in enumerate(mirrored_particle_files):
        data = np.load(particle_file)['particles']

        r_traj = data[:, 0]
        fi_traj = data[:, 4]

        x_traj = r_traj * np.cos(fi_traj)
        y_traj = r_traj * np.sin(fi_traj)

        plt.plot(
            x_traj,
            y_traj,
            lw=0.8,
            alpha=0.85,
            color='k',
            label='Mirrored trajectories' if i == 0 else None
        )

        if inj_xy is None:
            # 第一张图不改：
            # 你这是反推轨迹，data[0] 是内侧注入点
            inj_xy = (x_traj[0], y_traj[0])

    if inj_xy is not None:
        x_inj, y_inj = inj_xy

        plt.plot(
            x_inj,
            y_inj,
            marker='o',
            markersize=10,
            color='yellow',
            markeredgecolor='black',
            markeredgewidth=1.2,
            linestyle='None',
            label='Injection point'
        )

        plt.annotate(
            'Injection point',
            xy=(x_inj, y_inj),
            xytext=(12, 12),
            textcoords='offset points',
            fontsize=11,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.85),
            arrowprops=dict(arrowstyle='->', lw=1.0, color='black')
        )

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Mirrored trajectories over mirrored Bz map')
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    plt.colorbar(cf, label='Bz [T]', shrink=0.85, aspect=35, pad=0.02)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()


# =========================================================
# 10. 主程序
# =========================================================
def main():
    # -----------------------------------------------------
    # 输入/输出文件夹
    # -----------------------------------------------------
    bmap_folder = './Bmap_inj_strip_1p2'
    trajectory_folder = './output/reverse_track_test_strip_1p2'
    mirrored_trajectory_folder = './output/reverse_track_test_strip_1p2_mirror_x0'

    config_track_json = './config_track.json'

    npz_path = os.path.join(bmap_folder, 'Bz_order_0.npz')
    npz_key = 'Bz_Taylor_coeff'
    bmap_json = os.path.join(bmap_folder, 'config_Bmap.json')

    src_folder = trajectory_folder
    dst_folder = mirrored_trajectory_folder

    total_cells = 16
    thin = 10

    # 目前只是读取配置，后续没有直接使用
    _ = load_config_from_json(config_track_json)

    # -----------------------------------------------------
    # 1. 批量镜像粒子文件
    # -----------------------------------------------------
    raw_files, mirrored_files = mirror_all_particle_files(
        src_folder=src_folder,
        dst_folder=dst_folder,
        pattern='particle_*_rank_0.npz'
    )

    print(f'原始粒子文件数: {len(raw_files)}')
    print(f'镜像后保存到: {dst_folder}')

    # -----------------------------------------------------
    # 2. 构造整圈 Bmap 和镜像显示坐标
    # -----------------------------------------------------
    data_full, mesh_x_mirror, mesh_y_mirror, r_axis, fi_axis_full = build_mirrored_bmap_from_config(
        npz_path=npz_path,
        npz_key=npz_key,
        bmap_json=bmap_json,
        total_cells=total_cells,
        thin=thin
    )

    print(f'Bmap shape: {data_full.shape}')
    print(f'r range: {r_axis[0]:.6f} - {r_axis[-1]:.6f} m')
    print(f'fi points in full ring: {data_full.shape[1]}')

    # -----------------------------------------------------
    # 图 1：镜像轨迹 + 镜像 Bz 背景
    # 保持原样，注入点在内侧，取 data[0]
    # -----------------------------------------------------
    plot_mirrored_bmap_with_trajectories(
        mirrored_particle_files=mirrored_files,
        data_full=data_full,
        mesh_x_mirror=mesh_x_mirror,
        mesh_y_mirror=mesh_y_mirror,
        figsize=(8, 6)
    )

    # -----------------------------------------------------
    # 图 2：单条轨迹上的 Bz(s)
    # 文件是 inner -> outer
    # 图上反过来显示 outer -> inner
    # -----------------------------------------------------
    plot_bz_along_trajectory_outer_to_inner(
        particle_file=mirrored_files[0],
        r_axis=r_axis,
        data_full=data_full,
        field_mode='mirrored_background'
    )

    plt.show()


if __name__ == '__main__':
    main()

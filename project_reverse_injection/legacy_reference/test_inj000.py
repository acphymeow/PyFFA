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
        print(f"[warning] config_Bmap.json 中 NSector={nsector_from_json}, "
              f"但当前使用 total_cells={total_cells}。")

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

    return data_full, mesh_x_mirror, mesh_y_mirror


# =========================================================
# 4. 画初始相空间 (r, pr) 和 (z, pz)
# =========================================================
def plot_initial_phase_spaces(raw_particle_files):
    r0_all = []
    pr0_all = []
    z0_all = []
    pz0_all = []

    for particle_file in raw_particle_files:
        data = np.load(particle_file)['particles']
        r0_all.append(data[0, 0])
        pr0_all.append(data[0, 1])
        z0_all.append(data[0, 2])
        pz0_all.append(data[0, 3])

    plt.figure(figsize=(11, 4.6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(r0_all, pr0_all, s=8, color='b')
    ax1.set_xlabel('r')
    ax1.set_ylabel('pr')
    ax1.set_title('Initial phase space: (r, pr)')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(z0_all, pz0_all, s=8, color='r')
    ax2.set_xlabel('z')
    ax2.set_ylabel('pz')
    ax2.set_title('Initial phase space: (z, pz)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()


# =========================================================
# 5. 画原始最终 (r, pr) 相空间点
# =========================================================
def plot_final_phase_space_original(raw_particle_files):
    r_all = []
    pr_all = []

    for particle_file in raw_particle_files:
        data = np.load(particle_file)['particles']
        r_all.append(data[-1, 0])
        pr_all.append(data[-1, 1])

    plt.figure(figsize=(6, 5))
    plt.scatter(r_all, pr_all, s=8, color='b')
    plt.xlabel('r')
    plt.ylabel('pr')
    plt.title('Final phase-space points (original)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# =========================================================
# 6. 画镜像后最终 (r, pr) 相空间点
# =========================================================
def plot_final_phase_space_mirrored(mirrored_particle_files):
    r_all = []
    pr_all = []

    for particle_file in mirrored_particle_files:
        data = np.load(particle_file)['particles']
        r_all.append(data[-1, 0])
        pr_all.append(data[-1, 1])

    plt.figure(figsize=(6, 5))
    plt.scatter(r_all, pr_all, s=8, color='r')
    plt.xlabel('r')
    plt.ylabel('pr')
    plt.title('Final phase-space points (mirrored about x=0)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# =========================================================
# 7. 画镜像后的磁场背景 + 镜像后的所有轨迹 + 注入点
#    注入点取自轨迹第一点
# =========================================================
def plot_mirrored_bmap_with_trajectories(
    mirrored_particle_files,
    data_full,
    mesh_x_mirror,
    mesh_y_mirror,
    figsize=(8, 6),
    cmap='RdBu_r'
):
    norm = TwoSlopeNorm(
        vmin=data_full.min(),
        vcenter=0.0,
        vmax=data_full.max()
    )

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
            x_traj, y_traj,
            lw=0.8, alpha=0.85, color='k',
            label='Mirrored trajectories' if i == 0 else None
        )

        if inj_xy is None:
            inj_xy = (x_traj[0], y_traj[0])

    if inj_xy is not None:
        x_inj, y_inj = inj_xy

        plt.plot(
            x_inj, y_inj,
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
    plt.colorbar(cf, label='Bz (T)', shrink=0.85, aspect=35, pad=0.02)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()


# =========================================================
# 8. 主程序
# =========================================================
def main():
    config_track_json = './config_track.json'

    npz_path = './Bmap_inj/Bz_order_0.npz'
    npz_key = 'Bz_Taylor_coeff'
    bmap_json = './Bmap_inj/config_Bmap.json'
    # Bz_along_co = './Bmap_inj/Bz.txt'

    src_folder = './output/reverse_track_test'
    dst_folder = './output/reverse_track_test_mirror_x0'

    total_cells = 16
    thin = 10

    _ = load_config_from_json(config_track_json)

    raw_files, mirrored_files = mirror_all_particle_files(
        src_folder=src_folder,
        dst_folder=dst_folder,
        pattern='particle_*_rank_0.npz'
    )

    print(f'原始粒子文件数: {len(raw_files)}')
    print(f'镜像后保存到: {dst_folder}')

    data_full, mesh_x_mirror, mesh_y_mirror = build_mirrored_bmap_from_config(
        npz_path=npz_path,
        npz_key=npz_key,
        bmap_json=bmap_json,
        total_cells=total_cells,
        thin=thin
    )

    # 初始相空间
    plot_initial_phase_spaces(raw_files)

    # 终态相空间
    plot_final_phase_space_original(raw_files)
    plot_final_phase_space_mirrored(mirrored_files)

    # 轨迹 + 背景场图
    plot_mirrored_bmap_with_trajectories(
        mirrored_particle_files=mirrored_files,
        data_full=data_full,
        mesh_x_mirror=mesh_x_mirror,
        mesh_y_mirror=mesh_y_mirror,
        figsize=(8, 6)
    )

    plt.show()


if __name__ == '__main__':
    main()
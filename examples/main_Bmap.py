import os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from numpy.polynomial import Polynomial
from FFAG_Field import FFAG_Bfield_analytical
from FFAG_HighOrder import calculate_s, Generate_Bmap_Bz, BmapParams_to_ConvexParams, Generate_Bmaps_MultiProcess
from FFAG_HighOrder_new import Generate_Bmaps_new

def parse_args():
    parser = argparse.ArgumentParser(description="Process Bmap coefficients and generate plots.")
    parser.add_argument('-j', '--json', required=True, help="Path to the JSON configuration file.")
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 获取 JSON 文件所在目录
    json_dir = os.path.dirname(os.path.abspath(args.json))

    # 读取配置文件
    with open(args.json, 'r') as f:
        config_data = json.load(f)

    Bmap = config_data['Bmap']
    rmin_max_step_m = Bmap["rmin_max_step_m"]
    SpiralAngle_deg = Bmap["SpiralAngle_deg"]
    theta_step_rad = Bmap["theta_step_rad"]
    rmin, rmax, rstep = rmin_max_step_m[0], rmin_max_step_m[1], rmin_max_step_m[2]

    config_data_config = config_data['config']
    folder = config_data_config['folder']
    expand_order = config_data_config['expand_order']

    # 如果 folder 是相对路径，计算绝对路径
    if not os.path.isabs(folder):
        output_dir = os.path.join(json_dir, folder)  # 相对路径拼接到 JSON 文件目录
    else:
        output_dir = folder  # 已经是绝对路径，直接使用

    # 确保保存文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    convex_params = BmapParams_to_ConvexParams(Bmap)

    # 计算r轴步数
    n_r_steps = int((rmax - rmin) / rstep) + 1  # 确保包括rmax
    n_fi_steps = int((np.pi * 2 / Bmap["NSector"] - 0.0) / theta_step_rad) + 1  # 确保包括fi_max

    fi_axis = np.linspace(0, np.pi * 2 / Bmap["NSector"], n_fi_steps)
    r_axis = np.linspace(rmin, rmax, n_r_steps)

    s_values = calculate_s(r_axis, 0.0, SpiralAngle_deg)
    s_values_FitCoef = Polynomial.fit(r_axis, s_values, 5)

    # 调用函数并打印结果
    FFAG_Bfield_analytical(args.json,0,AddNumricalMap=False).UpdataConfig()

    Br_Taylor_matrices, Bz_Taylor_matrices, Bfi_Taylor_matrices = Generate_Bmaps_MultiProcess(expand_order, r_axis, fi_axis,
                                                                                 convex_params, s_values_FitCoef,
                                                                                 Bmap['coefficients_tupel'])

    # Br_Taylor_matrices, Bz_Taylor_matrices, Bfi_Taylor_matrices = Generate_Bmaps_new(expand_order,r_axis, fi_axis, SaveData[1:,1:])
    # 按每一阶分别保存 Br, Bz, Bfi的系数矩阵为单独的 .npz 文件
    for order in range(len(Br_Taylor_matrices)):
        np.savez(os.path.join(output_dir, f"Br_order_{order}.npz"), Br_Taylor_coeff=Br_Taylor_matrices[order])
        np.savez(os.path.join(output_dir, f"Bz_order_{order}.npz"), Bz_Taylor_coeff=Bz_Taylor_matrices[order])
        np.savez(os.path.join(output_dir, f"Bfi_order_{order}.npz"),
                 Bfi_Taylor_coeff=Bfi_Taylor_matrices[order])

    print(f"系数矩阵按阶数分别保存为 npz 文件到文件夹: {output_dir}")

    # # 定义表头信息
    # header = f'unitR=m unitFi=rad unitB=T Nsectors={Bmap["NSector"]}'
    #
    # # 保存 Bmap.txt文件 并添加表头
    # np.savetxt(os.path.join(output_dir, "Bmap.txt"), SaveData, header=header, comments='')
    # print(f"中平面 Bz 保存为 txt 文件到文件夹: {output_dir}")

    # 将极坐标转换为笛卡尔坐标
    mesh_f, mesh_r = np.meshgrid(fi_axis, r_axis)
    mesh_x = mesh_r * np.cos(mesh_f)
    mesh_y = mesh_r * np.sin(mesh_f)

    plt.figure(figsize=(18, 6))
    plt.scatter(np.rad2deg(fi_axis), Bz_Taylor_matrices[0][0, :], label=f"R={r_axis[0]}", s=2)
    plt.scatter(np.rad2deg(fi_axis), Bz_Taylor_matrices[0][np.size(Bz_Taylor_matrices[0], 0) // 2, :],
                label=f"R={r_axis[np.size(Bz_Taylor_matrices[0], 0) // 2]}", s=2)
    plt.scatter(np.rad2deg(fi_axis), Bz_Taylor_matrices[0][-1, :], label=f"R={r_axis[-1]}", s=2)
    plt.legend()
    plt.xlabel("fi(deg)", fontsize=18)
    plt.ylabel("Bz(T)", fontsize=18)

    # 创建三维图
    fig = plt.figure(figsize=(18, 6))

    # 绘制 Bz_Taylor_matrices[0] 的曲面图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(mesh_x, mesh_y, Bz_Taylor_matrices[0], cmap='viridis', edgecolor='none')
    ax1.set_title('Bz Taylor Coefficients (Order 0)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('Bz')

    # 绘制 Br_Taylor_matrices[0] 的曲面图
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(mesh_x, mesh_y, Br_Taylor_matrices[0], cmap='plasma', edgecolor='none')
    ax2.set_title('Br Taylor Coefficients (Order 0)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_zlabel('Br')

    # 绘制 Bfi_Taylor_matrices[0] 的曲面图
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(mesh_x, mesh_y, Bfi_Taylor_matrices[0], cmap='inferno', edgecolor='none')
    ax3.set_title('Bfi Taylor Coefficients (Order 0)')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_zlabel('Bfi')

    # 显示图像
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

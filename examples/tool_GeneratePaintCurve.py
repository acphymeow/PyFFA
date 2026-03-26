import numpy as np
import os
import matplotlib.pyplot as plt

# 输入文件路径
input_Bmap = '../Bmaps/Bmap_FD16'
Bmap_SEOPath = os.path.join(input_Bmap, 'resultsSEO_noError/SEO_ini.txt')

# 从 SEO_ini.txt 读取回旋频率，计算注入周期
SEO_data = np.loadtxt(Bmap_SEOPath, skiprows=2)
T_inj = 1 / SEO_data[0, 4] * 1e9  # 周期 (ns)

# 配置参数
MaxBunchNum = 1000
TimeInterval = T_inj

# 最大偏移量（单位：m / rad）
delta_r_max = 0.03
delta_pr_max = 0.00
delta_z_max = 0.00
delta_pz_max = 0.007

# 涂抹函数定义
def apply_profile(frac, profile):
    if isinstance(profile, str):
        if profile == 'up_linear':
            return frac
        elif profile == 'down_linear':
            return 1 - frac
        elif profile == 'up_square':
            return frac ** 2
        elif profile == 'down_square':
            return 1 - frac ** 2
        elif profile == 'up_sqrt':
            return np.sqrt(frac)
        elif profile == 'down_sqrt':
            return 1 - np.sqrt(frac)
        elif profile.startswith('up_power_'):
            n = float(profile.replace('up_power_', ''))
            return frac ** n
        elif profile.startswith('down_power_'):
            n = float(profile.replace('down_power_', ''))
            return 1 - frac ** n
        else:
            raise ValueError(f"未知 profile 类型: {profile}")
    else:
        raise TypeError("profile 必须是 str 类型")

# 输出目录
output_dir = './PaintCurves'
os.makedirs(output_dir, exist_ok=True)

# 多组 profile 配置，每个维度单独定义
profile_sets = [
    {
        'r':  'up_power_0.2',
        'pr': 'up_linear',
        'z':  'down_linear',
        'pz': 'down_linear'
    },
    # {
    #     'r':  'up_linear',
    #     'pr': 'up_linear',
    #     'z':  'down_linear',
    #     'pz': 'down_power_3.0'
    # },
    # {
    #     'r':  'up_linear',
    #     'pr': 'up_linear',
    #     'z':  'down_linear',
    #     'pz': 'down_power_0.5'
    # },
    {
        'r':  'up_power_0.3',
        'pr': 'up_linear',
        'z':  'down_linear',
        'pz': 'down_linear'
    },
]
# profile_sets = [
#     {
#         'r':  'up_linear',
#         'pr': 'up_linear',
#         'z':  'up_linear',
#         'pz': 'up_linear'
#     }
# ]

# 曲线收集用于绘图
dr_profiles = []
dpz_profiles = []

for profile in profile_sets:
    data = []
    dr_list = []
    dpz_list = []

    for i in range(MaxBunchNum):
        t = i * TimeInterval
        frac = i / (MaxBunchNum - 1)

        dr  = delta_r_max  * apply_profile(frac, profile['r'])
        dpr = delta_pr_max * apply_profile(frac, profile['pr'])
        dz  = delta_z_max  * apply_profile(frac, profile['z'])
        dpz = delta_pz_max * apply_profile(frac, profile['pz'])

        data.append([t, dr, dpr, dz, dpz])
        dr_list.append(dr)
        dpz_list.append(dpz)

    # 生成安全的文件名
    name_parts = [f"{k}_{v.replace('.', '_')}" for k, v in profile.items()]
    profile_name = "__".join(name_parts)
    output_path = os.path.join(output_dir, f"Curve_{profile_name}.paint")

    # 保存文件
    header = "time_ns\tdelta_r(m)\tdelta_pr(rad)\tdelta_z(m)\tdelta_pz(rad)"
    np.savetxt(output_path, data, fmt='%.6f', delimiter='\t', header=header, comments='')
    print(f"✅ 曲线文件已保存: {output_path}")

    # 收集用于绘图
    dr_profiles.append((profile_name, dr_list))
    dpz_profiles.append((profile_name, dpz_list))

# 绘制 delta_r 曲线
plt.figure(figsize=(10, 6))
x = np.arange(MaxBunchNum)
for name, dr in dr_profiles:
    plt.plot(x, dr, label=name)
plt.xlabel('Bunch Index')
plt.ylabel('delta_r (m)')
plt.legend(fontsize='small')
plt.grid(True)
plt.tight_layout()

# 绘制 delta_pz 曲线
plt.figure(figsize=(10, 6))
for name, dpz in dpz_profiles:
    plt.plot(x, dpz, label=name)
plt.xlabel('Bunch Index')
plt.ylabel('delta_pz (rad)')
plt.legend(fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.show()





# import numpy as np
# import os
#
# # 输出目录和文件名
# output_dir = './PaintCurves'
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, 'Curve_linear_reverse.paint')
# # output_path = os.path.join(output_dir, 'Curve_linear.paint')
# input_Bmap = './Bmap'
# Bmap_SEOPath = os.path.join(input_Bmap, 'resultsSEO/SEO_ini.txt')
#
# # 从 SEO_ini.txt 读取回旋频率，计算注入周期
# SEO_data = np.loadtxt(Bmap_SEOPath, skiprows=2)
# T_inj = 1 / SEO_data[0, 4] * 1e9  # 周期 (ns)
#
# # 配置参数
# MaxBunchNum = 1000
# TimeInterval = T_inj
#
# # 最大偏移量（单位：m / rad）
# delta_r_max = 0.03
# delta_pr_max = 0.00
# delta_z_max = 0.00
# delta_pz_max = 0.007
#
# # 方向：'up' (0→max), 'down' (max→0)
# direction = {
#     'r':   'down',
#     'pr':  'up',
#     'z':   'up',
#     'pz':  'up'
# }
#
# # 数据生成
# data = []
# for i in range(MaxBunchNum):
#     t = i * TimeInterval
#     frac = i / (MaxBunchNum - 1)
#     dr  = delta_r_max  * (1 - frac) if direction['r']  == 'down' else delta_r_max  * frac
#     dpr = delta_pr_max * (1 - frac) if direction['pr'] == 'down' else delta_pr_max * frac
#     dz  = delta_z_max  * (1 - frac) if direction['z']  == 'down' else delta_z_max  * frac
#     dpz = delta_pz_max * (1 - frac) if direction['pz'] == 'down' else delta_pz_max * frac
#     data.append([t, dr, dpr, dz, dpz])
#
# # 表头
# header = "time_ns\tdelta_r(m)\tdelta_pr(rad)\tdelta_z(m)\tdelta_pz(rad)"
#
# # 写入文件
# np.savetxt(output_path, data, fmt='%.6f', delimiter='\t', header=header, comments='')
#
# print(f"✅ 涂抹曲线已保存至: {output_path}")

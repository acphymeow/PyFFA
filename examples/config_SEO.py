import json
import time
import os
import numpy as np

config_data = {}
###############################################################
# some general parameters
config = dict()
config['start_Ek'] = 280.0
config['end_Ek'] = 600.0
config['delta_Ek'] = 20
config['extra_Ek'] = ()
config['Bmap_path'] = '../Bmaps/Bmap_FD16'    # 读A文件夹里的磁场
config['SEO_folder'] = 'resultsSEO_noError'    # 闭轨结果放在文件夹A里面，子文件夹B下面

###############################################################################
# field error
BHarmonics = dict()
BHarmonics['enable'] = False    # True False
harmonic_1 = {
    "m": 3, "amp_Gs": 0.1, "phase_deg": 30.0,
    "rmin": 10.8, "rmax": 12.0,}
harmonic_2 = {
    "m": 2, "amp_Gs": 5.0, "phase_deg": 30.0,
    "rmin": 11.2, "rmax": 11.4,}
harmonic_3 = {
    "m": 3,"amp_Gs": 12.0,"phase_deg": 30.0,
    "rmin": 11.2,"rmax": 11.4,}
BHarmonics["harmonics"] = [harmonic_1, ]

# ready to write out
config_data['config'] = config
config_data['BHarmonics'] = BHarmonics

# 保存 JSON 配置文件
if __name__ == "__main__":
    current_file_name = os.path.splitext(os.path.basename(__file__))[0] + ".json"
    with open(current_file_name, 'w') as f_out:
        json.dump(config_data, f_out, indent=2, sort_keys=True)
    print(f"配置文件已保存为 {current_file_name}")

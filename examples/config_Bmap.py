import json
import time
import os
import numpy as np
from FFAG_ISO import CheckISOBmapConfigNew


config_data = {}
###############################################################
# some general parameters
config = dict()
config['date'] = time.time()
config['folder'] = '../Bmaps/Bmap_FD16'
config['expand_order'] = 2

###############################################################
# machine parameters
machine = dict()
machine['energy_inj'] = 300
machine['energy_ext'] = 600

###############################################################
# Bmap configuration
Bmap = dict()
Bmap['Type'] = 'SCALE'  # 可选：等时'ISOCHRONOUS', or 等比'SCALE'
Bmap['NSector'] = 16

Bmap['theta_step_rad'] = np.deg2rad(0.005)  # unit: rad, 方位角步长
Bmap['rmin_max_step_m'] = (10.8, 12.2, 0.001)  # unit: m， 比实际Bmap范围大一些，避免迭代过程中超出Bmap范围
#alpha = 0.4478
Bmap['interval_1'] = (5.625/22.5, 5.9083/22.5, 2.38755/22.5, 2.95415/22.5, 5.625/22.5)
Bmap['positive_or_negative_1'] = (0, 1.0, 0, -0.465074, 0)  # unit: 1
Bmap['fringe_width_1'] = (0, 1.17, 0, 1.17, 0)  # unit: 1
Bmap['SpiralAngle_deg'] = 49.973388

Bmap['orbital_freq_MHz'] = 4.5  # unit: MHz, Type为等时'ISOCHRONOUS'时起效
Bmap['k_value'] = 7.630003
Bmap['B0_max_T'] = 1.65  # unit: T, 参考半径处的B0, Type为等比'SCALE'时起效
Bmap['R0_m'] = 12.0   # unit: m, 参考半径, Type为等比'SCALE'时起效

###############################################################
###############################################################
# Bz configuration, not defined by the user
BzInfo = dict()
BzInfo['IsoCoef'] = None  # not defined by the user
BzInfo['IsoError'] = None  # not defined by the user
# Bmap['B0_T'] = Bmap['B0_max_T'] / (Bmap['rmin_max_step_m'][1]**Bmap['k_value'])  # not defined by the user
Bmap['B0_T'] = Bmap['B0_max_T'] / (Bmap['R0_m']**Bmap['k_value'])  # not defined by the user

# ready to write out
config_data['config'] = config
config_data['machine'] = machine
config_data['Bmap'] = Bmap
config_data['BzInfo'] = BzInfo

if __name__ == "__main__":
    coefficients_tupel = CheckISOBmapConfigNew(config_data)
    config_data['Bmap']['coefficients_tupel'] = coefficients_tupel
    config_data['Bmap']['Bmean'] = None
    config_data['Bmap']['r_axis'] = None
    current_file_name = os.path.splitext(os.path.basename(__file__))[0] + ".json"
    with open(current_file_name, 'w') as f_out:
        json.dump(config_data, f_out, indent=2, sort_keys=True)


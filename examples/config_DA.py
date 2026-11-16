import json
import os
import time


cf = {}

###############################################################################
# Standard fixed-energy transverse dynamic aperture scan.
#
# This default is a production-style single-lattice DA run. Add more Bmap
# folders to bmap_dirs when comparing multiple lattices.

config = dict()
config["date"] = time.time()

DA = dict()
DA["bmap_dirs"] = ["./Bmaps/Bmap_FD16_k7p63_B1p65_R465_Fp0p75_s49p97",]
DA["seo_folder"] = "resultsSEO"  # SEO/Twiss subfolder under each Bmap directory.
DA["output_dir"] = "./lattice_da_30turn_300MeV"
DA["energy_mev"] = 300.0
DA["azimuth_deg"] = 0.0
DA["turns"] = 300


DA["phi_step_rad"] = 0.001
DA["steps_per_turn"] = None
DA["max_emit_pi_mm_mrad"] = 2000.0
DA["n_radius"] = 41
DA["n_angle"] = 19
DA["phase_pairs_deg"] = "0,0;90,0;0,90;90,90"
DA["max_order"] = None
DA["aperture_r_min_m"] = None
DA["aperture_r_max_m"] = None
DA["aperture_z_abs_m"] = 0.10
DA["overwrite"] = True
DA["make_phase_space_plot"] = True
DA["numba_threads"] = 4  # None uses Numba default; set an integer to control parallel kernels.

cf["config"] = config
cf["DA"] = DA


if __name__ == "__main__":
    current_file_name = os.path.splitext(os.path.basename(__file__))[0] + ".json"
    with open(current_file_name, "w", encoding="utf-8") as f_out:
        json.dump(cf, f_out, indent=2, sort_keys=True)
    print(f"配置文件已保存为 {current_file_name}")

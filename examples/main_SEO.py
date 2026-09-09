import os
import json
import argparse
import copy
import numpy as np
from FFAG_track import FFAG_RungeKutta, FFAG_SearchSEO, FunctionForAccelerationBunch_dt
from FFAG_Field import FFAG_Bfield_analytical, FFAG_Bfield_3D_opera, FFAG_BField_Error
from FFAG_ParasAndConversion import FFAG_GlobalParameters, FFAG_ConversionTools
from FFAG_BetaFunc import FFAG_BetaFuncCalc


def load_bmap_auto(bmap_path, max_order=0):
    analytical_config = os.path.join(bmap_path, "config_Bmap.json")
    if os.path.exists(analytical_config):
        return FFAG_Bfield_analytical(analytical_config, max_order)

    return FFAG_Bfield_3D_opera(field_dir=bmap_path)


def main(EkRange, coeff_matrices_path, SEO_folder, BHarmonicsConfig):

    # # Load BMap data
    BMapData = load_bmap_auto(coeff_matrices_path, max_order=0)
    BMapHarmonics = FFAG_BField_Error(BHarmonicsConfig['harmonics'],BHarmonicsConfig['enable'])

    # Generate global parameters
    GlobalParas = FFAG_GlobalParameters()
    GlobalParas.AddBMap(BMapData)
    GlobalParas.AddBHarmonics(BMapHarmonics)
    GlobalParas.SEO_SaveFold = SEO_folder

    # Search SEO
    SEO_FilePath = FFAG_SearchSEO(GlobalParas).SearchSEOsControllerVect(EkRange)
    # SEO_FilePath = "./Bmap_FD16/resultsSEO_noError"

    # Calculate beta functions
    FFAG_BetaFuncCalc().CalcBetaFuncXPX(SEO_FilePath, EkRange)

    return SEO_FilePath




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFAG Simulation Command-Line Tool")
    parser.add_argument("-j", "--json_path", type=str, required=True, help="Path to config_SEO.json")
    args = parser.parse_args()

    with open(args.json_path, 'r') as f:
        config_data = json.load(f)

    start_Ek = config_data['config']['start_Ek']
    end_Ek = config_data['config']['end_Ek']
    delta_Ek = config_data['config']['delta_Ek']
    extra_Ek = config_data['config']['extra_Ek']
    Bmap_path = config_data['config']['Bmap_path']
    SEO_folder = config_data['config']['SEO_folder']
    BHarmonicsConfig = config_data['BHarmonics']

    EkRange = np.arange(start_Ek, end_Ek + delta_Ek, delta_Ek)
    EkRange = np.concatenate((EkRange, np.array(extra_Ek)))
    EkRange = np.sort(EkRange)

    # Run the main function with parsed arguments
    SEO_FilePath = main(EkRange, Bmap_path, SEO_folder, BHarmonicsConfig)

    with open(os.path.join(SEO_FilePath, "config_SEO.json"), 'w') as f_out:
        json.dump(config_data, f_out, indent=2, sort_keys=True)

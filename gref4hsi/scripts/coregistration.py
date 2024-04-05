import configparser
import os



# Function called to apply standard processing on a folder of files
def main(iniPath, viz = False):
    config = configparser.ConfigParser()
    config.read(iniPath)

    # Establish reference data
    path_orthomosaic_reference = config['Absolute Paths']['orthomosaic_reference']
    path_dem_reference = config['Absolute Paths']['dem_path']

    # Establish match data (HSI), including composite and anc data
    path_composites_match = config['Absolute Paths']['rgb_composite_folder']
    path_anc_match = config['Absolute Paths']['anc_folder']

    print("\n################ Coregistering: ################")

    # Iterate the RGB composites
    files = sorted(os.listdir(path_composites_match))
    n_files= len(files)
    file_count = 0
    for file_count, filename in enumerate(files):
        # Start with resampling the reference data to the match data

    
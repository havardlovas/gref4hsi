# Standard python library
import configparser
import sys
import os
from collections import namedtuple

# Local resources
from scripts import georeference_mod
from scripts import orthorectification
from lib import parsing_utils, specim_parsing_utils
from scripts import visualize
from scripts.modulate_config import prepend_data_dir_to_relative_paths


import numpy as np
"""
This script is meant to be used for testing the processing pipeline of airborne HI data.
""" 




# Settings associated with orthorectification of datacube
SettingsPreprocess = namedtuple('SettingsPreprocessing', ['dtype_datacube', 
                                                                        'lines_per_chunk', 
                                                                        'specim_raw_missions_dir', 
                                                                        'mission_name',
                                                                        'cal_dir',
                                                                        'reformatted_missions_dir',
                                                                        'rotation_matrix_hsi_to_body',
                                                                        'translation_hsi_to_body',
                                                                        'config_file_name'])

config_specim_preprocess = SettingsPreprocess(dtype_datacube = np.float32, 
                                # The data type for the datacube
                            lines_per_chunk= 2000, 
                            # Raw datacube is chunked into this size
                            specim_raw_missions_dir = 'D:/Specim/Missions/2022-08-31-Remøy/',
                            # Folder containing several missions
                            mission_name = '2022-08-31_0800_HSI',
                            # The particular mission (only thing changing?)
                            cal_dir = 'D:/Specim/Lab_Calibrations/', 
                            # Calibration directory holding all calibrations at all binning levels
                            reformatted_missions_dir = 'D:/HyperspectralDataAll/HI/',
                            # The fill value for empty cells (select values not occcuring in cube or ancillary data)
                            rotation_matrix_hsi_to_body = np.array([[0, 1, 0],
                                                                    [-1, 0, 0],
                                                                    [0, 0, 1]]),
                            # Boolean being expressing whether to rectify only composite (true) or data cube and composite (false). True is fast.
                            translation_hsi_to_body = np.array([0, 0, 0]),
                            # For large files, RAM issues could be a concern. For rectified files exeeding this size, data is written chunk-wize to a memory map.
                            config_file_name = 'configuration.ini')




DATA_DIR = config_specim_preprocess.reformatted_missions_dir + config_specim_preprocess.mission_name + '/'
config_file_mission = DATA_DIR + 'configuration.ini'


# Read config from a template (relative path):
config_path_template = 'data/config_examples/configuration.ini'

# Set the data directory for the mission, and create empty folder structure
prepend_data_dir_to_relative_paths(config_path=config_path_template, DATA_DIR=DATA_DIR)

# 







def main():
    ## Extract pose.csv and model.ply data from Agisoft Metashape (photogrammetry software) through API.
    ## Fails if you do not have an appropriate project.
    config = configparser.ConfigParser()
    config.read(config_file_mission)

    # The minimum for georeferencing is to parse 1) Mesh model and 2) The pose of the reference
    #specim_parsing_utils.main(config=config,
    #                          config_specim=config_specim_preprocess)
    
    config = parsing_utils.export_pose(config_file_mission)

    config = configparser.ConfigParser()

    config.read(config_file_mission)
    
    # Exports model
    parsing_utils.export_model(config_file_mission)

    ## Visualize the data 3D photo model from RGB images and the time-resolved positions/orientations
    
    visualize.show_mesh_camera(config, show_mesh = True, show_pose = True)

    # Georeference the line scans of the hyperspectral imager. Utilizes parsed data
    
    georeference_mod.main(config_file_mission)

    orthorectification.main(config_file_mission)

    #print('')


if __name__ == "__main__":
    main()
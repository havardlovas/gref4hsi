# Standard python library
import configparser
import sys
import os
from collections import namedtuple

module_path = 'C:/Users/haavasl/VSCodeProjects/hyperspectral_toolchain/src/'
if module_path not in sys.path:
    sys.path.append(module_path)

# Local resources
from scripts import georeference
from scripts import orthorectification
from lib import parsing_utils, specim_parsing_utils
from scripts import visualize
from scripts.config_utils import prepend_data_dir_to_relative_paths, customize_config


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
config_path_template = 'C:/Users/haavasl/VSCodeProjects/hyperspectral_toolchain/data/config_examples/configuration_specim.ini'

# Set the data directory for the mission, and create empty folder structure
prepend_data_dir_to_relative_paths(config_path=config_path_template, DATA_DIR=DATA_DIR)

# Non-default settings
custom_config = {'General':
                    {'mission_dir': DATA_DIR,
                    'model_export_type': 'geoid', # Ray trace onto geoid
                    'max_ray_length': 120}, # Max distance in meters from spectral imager to seafloor

                'Coordinate Reference Systems':
                    {'proj_epsg' : 25832, # The projected CRS for orthorectified data (an arctic CRS)
                    'geocsc_epsg_export' : 4978, # 3D cartesian system for earth consistent with GPS frame (but inconsistent with eurasian techtonic plate)
                    'dem_epsg' : 25832, # (Optional) If you have a DEM this can be used
                    'pos_epsg_orig' : 4978}, # The CRS of the positioning data we deliver to the georeferencing

                'Orthorectification':
                    {'resample_rgb_only': False, # Good choice for speed
                    'resolutionhyperspectralmosaic': 0.2, # Resolution in m
                    'raster_transform_method': 'north_east'}, # North-east oriented rasters.
                
                'HDF.raw_nav': {
                    'rotation_reference_type' : 'eul_ZYX', # The vehicle orientations are given in Yaw, Pitch, Roll from the NAV system
                    'is_global_rot' : False, # The vehicles orientations from NAV system are Yaw, Pitch, Roll
                    'eul_is_degrees' : True},
                'Absolute Paths': {
                    'geoid_path' : 'C:/Users/haavasl/VSCodeProjects/hyperspectral_toolchain/data/world/geoids/no_kv_HREF2018A_NN2000_EUREF89.tif',
                    #'geoid_path' : 'data/world/geoids/egm08_25.gtx',
                    #'dem_path' : 'D:/HyperspectralDataAll/HI/2022-08-31-060000-Remoy-Specim/Input/GIS/DEM_downsampled_deluxe.tif', 
                    # (above) The georeferencing allows processing using norwegian geoid NN2000 and worldwide EGM2008. Also, use of seafloor terrain models are supported. '
                    # At the moment refractive ray tracing is not implemented, but it could be relatively easy by first ray tracing with geoid+tide, 
                    # and then ray tracing from water
                    'tide_path' : 'D:/HyperspectralDataAll/HI/2022-08-31-060000-Remoy-Specim/Input/tidevann_nn2000_NMA.txt'}
                    # Tide data can be downloaded from https://www.kartverket.no/til-sjos/se-havniva
                    # Preferably it is downloaded with reference "NN2000" to agree with DEM
                
}

# Customizes the config file
customize_config(config_path=config_file_mission, dict_custom=custom_config)

def main():
    config = configparser.ConfigParser()
    config.read(config_file_mission)

    # This function parses raw specim data including (spectral, radiometric, geometric) calibrations and nav data
    # into an h5 file. The nav data is written to "raw/nav/" subfolders, whereas hyperspectral data and calibration data 
    # written to "processed/hyperspectral/" and "processed/calibration/" subfolders
    specim_parsing_utils.main(config=config,
                              config_specim=config_specim_preprocess)
    
    # Interpolates and writes the pose (of the vehicle body) to "processed/nav/". 
    # Also creates a ""

    # Be careful to not comment out this line since it calculates offsets
    config = parsing_utils.export_pose(config_file_mission)
    
    # Exports model
    parsing_utils.export_model(config_file_mission)

    # Commenting out the georeference step is fine if it has been done

    
    ## Visualize the data 3D photo model from RGB images and the time-resolved positions/orientations
    
    #visualize.show_mesh_camera(config, show_mesh = True, show_pose = True)

    # Georeference the line scans of the hyperspectral imager. Utilizes parsed data
    
    """georeference.main(config_file_mission)"""

    orthorectification.main(config_file_mission)

    #print('')


if __name__ == "__main__":
    main()
# Standard python library
import configparser
import sys
import os
import argparse
from collections import namedtuple
import pathlib

# This is very hard coded, but not necessary if Python does not know where to look
module_path = '/home/haavasl/VsCodeProjects/gref4hsi/gref4hsi/'
if module_path not in sys.path:
    sys.path.append(module_path)

# Local resources
from gref4hsi.scripts import georeference
from gref4hsi.scripts import orthorectification
from gref4hsi.utils import parsing_utils, specim_parsing_utils
from gref4hsi.scripts import visualize
from gref4hsi.utils.config_utils import prepend_data_dir_to_relative_paths, customize_config


import numpy as np
"""
This script is meant to be used for testing the processing pipeline of airborne HI data from the Specim AFX10 instrument.
"""




def main():
    # Set up argparse
    parser = argparse.ArgumentParser('georeference and rectify images')

    parser.add_argument('-s', '--specim_mission_folder',
                        type=str, 
                        default= r"/media/haavasl/Expansion/Specim/Missions/2024-02-19-Sletvik/slettvik_hopavaagen_202402191253_ntnu_hyperspectral_74m",
                        help='folder storing the hyperspectral data for the specific mission')

    parser.add_argument('-e', '--epsg_code', 
                        default=25832, 
                        type=int,
                        help='Coordinate Reference System EPSG code (e.g. 25832 for UTM 32)')

    parser.add_argument('-r', '--resolution_orthomosaic', 
                        type=float,
                        default=1, 
                        help='Resolution of the final processed orthomosaic in meters')

    parser.add_argument('-cal_dir', '--calibration_directory', 
                        type=str,
                        default="/media/haavasl/Expansion/Specim/Lab_Calibrations", 
                        help='Directory holding all spectral/radiometric/geometric calibrations for all binning values' )

    parser.add_argument('-c', '--config_file_yaml', 
                        default="",
                        help='File that contains the configuration \
                            parameters for the processing. \
                            If nothing, one is generated from template.\
                            can simply be one that was used for another mission.')

    parser.add_argument('-t', '--terrain_type', 
                        default="geoid", type=str,
                        help ='If terrain DEM is known, select "dem_file", and if not select "geoid".')

    parser.add_argument('-geoid', '--geoid_path', 
                        default="/media/haavasl/Expansion/Specim/Missions/2024-02-19-Sletvik/slettvik_hopavaagen_202402191253_ntnu_hyperspectral_74m/geoids/no_kv_HREF2018A_NN2000_EUREF89.tif", 
                        type=str,
                        help='If terrain DEM is not available.')

    parser.add_argument('-d', '--dem_path', 
                        default="/media/haavasl/Expansion/Specim/Missions/2024-02-19-Sletvik/slettvik_hopavaagen_202402191253_ntnu_hyperspectral_74m/dem/dem.tif", 
                        type=str,
                        help='A digital terrain model, if available. If none, the geoid will be used.')

    parser.add_argument('-v', '--enable_visualize', 
                        default=False, 
                        type = bool,
                        help='Visualize vehicle track and terrain model')


    args = parser.parse_args()

    # assigning the arguments to variables for simple backwards compatibility
    SPECIM_MISSION_FOLDER = args.specim_mission_folder
    EPSG_CODE = args.epsg_code
    RESOLUTION_ORTHOMOSAIC = args.resolution_orthomosaic
    CALIBRATION_DIRECTORY = args.calibration_directory
    CONFIG_FILE = args.config_file_yaml
    ENABLE_VISUALIZE = args.enable_visualize
    TERRAIN_TYPE = args.terrain_type
    DEM_PATH = args.dem_path
    GEOID_PATH = args.geoid_path

    # Settings associated with preprocessing of data from Specim Proprietary data to pipeline-compatible data
    SettingsPreprocess = namedtuple('SettingsPreprocessing', ['dtype_datacube', 
                                                                            'lines_per_chunk', 
                                                                            'specim_raw_mission_dir',
                                                                            'cal_dir',
                                                                            'reformatted_missions_dir',
                                                                            'rotation_matrix_hsi_to_body',
                                                                            'translation_body_to_hsi',
                                                                            'config_file_name'])

    config_specim_preprocess = SettingsPreprocess(dtype_datacube = np.float32, # The data type for the datacube
                                lines_per_chunk= 2000,  # Raw datacube is chunked into this many lines. GB_per_chunk = lines_per_chunk*n_pixels*n_bands*4 bytes
                                specim_raw_mission_dir = SPECIM_MISSION_FOLDER, # Folder containing several mission
                                cal_dir = CALIBRATION_DIRECTORY,  # Calibration directory holding all calibrations at all binning levels
                                reformatted_missions_dir = os.path.join(SPECIM_MISSION_FOLDER, 'processed'), # The fill value for empty cells (select values not occcuring in cube or ancillary data)
                                rotation_matrix_hsi_to_body = np.array([[0, 1, 0],
                                                                        [-1, 0, 0],
                                                                        [0, 0, 1]]), # Rotation matrix R rotating so that vec_body = R*vec_hsi.
                                translation_body_to_hsi = np.array([0, 0, 0]), # Translation t so that vec_body_to_object = vec_hsi_to_object + t
                                # For large files, RAM issues could be a concern. For rectified files exeeding this size, data is written chunk-wize to a memory map.
                                config_file_name = 'configuration.ini')



    # Where to place the config
    DATA_DIR = config_specim_preprocess.reformatted_missions_dir
    config_file_mission = os.path.join(DATA_DIR, 'configuration.ini')


    # Read config from a template (relative path):
    if CONFIG_FILE != "":
        config_path_template = CONFIG_FILE
    else:
        config_path_template = '/home/haavasl/VsCodeProjects/gref4hsi/data/config_examples/configuration_specim.ini'


    # Set the data directory for the mission, and create empty folder structure
    prepend_data_dir_to_relative_paths(config_path=config_path_template, DATA_DIR=DATA_DIR)

    # Non-default settings
    custom_config = {'General':
                        {'mission_dir': DATA_DIR,
                        'model_export_type': TERRAIN_TYPE, # Ray trace onto geoid
                        'max_ray_length': 200}, # Max distance in meters from spectral imager to seafloor. Specim does not fly higher

                    'Coordinate Reference Systems':
                        {'proj_epsg' : EPSG_CODE, # The projected CRS UTM 32, common on mainland norway
                        'geocsc_epsg_export' : 4978, # 3D cartesian system for earth consistent with GPS frame (but inconsistent with eurasian techtonic plate)
                        'dem_epsg' : EPSG_CODE, # (Optional) If you have a DEM this can be used
                        'pos_epsg_orig' : 4978}, # The CRS of the positioning data we deliver to the georeferencing

                    'Orthorectification':
                        {'resample_rgb_only': True, # True can be good choice for speed during DEV
                        'resolutionhyperspectralmosaic': RESOLUTION_ORTHOMOSAIC, # Resolution in m
                        'raster_transform_method': 'north_east'}, # North-east oriented rasters.
                    
                    'HDF.raw_nav': {
                        'rotation_reference_type' : 'eul_ZYX', # The vehicle orientations are given in Yaw, Pitch, Roll from the NAV system
                        'is_global_rot' : False, # The vehicles orientations from NAV system are Yaw, Pitch, Roll
                        'eul_is_degrees' : True}, # And given in degrees
                    'Absolute Paths': {
                        'geoid_path' : GEOID_PATH,
                        #'geoid_path' : 'data/world/geoids/egm08_25.gtx',
                        'dem_path' : DEM_PATH,
                        # (above) The georeferencing allows processing using norwegian geoid NN2000 and worldwide EGM2008. Also, use of seafloor terrain models are supported. '
                        # At the moment refractive ray tracing is not implemented, but it could be relatively easy by first ray tracing with geoid+tide, 
                        # and then ray tracing from water
                        #'tide_path' : 'D:/HyperspectralDataAll/HI/2022-08-31-060000-Remoy-Specim/Input/tidevann_nn2000_NMA.txt'
                        }
                        # Tide data can be downloaded from https://www.kartverket.no/til-sjos/se-havniva
                        # Preferably it is downloaded with reference "NN2000" to agree with DEM
                    
    }

    if TERRAIN_TYPE == 'geoid':
        custom_config['Absolute Paths']['geoid_path'] = GEOID_PATH
        #'geoid_path' : 'data/world/geoids/egm08_25.gtx'
    elif TERRAIN_TYPE == 'dem_file':
        custom_config['Absolute Paths']['dem_path'] = DEM_PATH

    # Customizes the config file according to settings
    customize_config(config_path=config_file_mission, dict_custom=custom_config)


    config = configparser.ConfigParser()
    config.read(config_file_mission)

    # This function parses raw specim data including (spectral, radiometric, geometric) calibrations and nav data
    # into an h5 file. The nav data is written to "raw/nav/" subfolders, whereas hyperspectral data and calibration data 
    # written to "processed/hyperspectral/" and "processed/calibration/" subfolders
    """specim_parsing_utils.main(config=config,
                              config_specim=config_specim_preprocess)"""
    
    # Interpolates and reformats the pose (of the vehicle body) to "processed/nav/" folder.
    config = parsing_utils.export_pose(config_file_mission)
    
    # Exports model
    parsing_utils.export_model(config_file_mission)

    # Commenting out the georeference step is fine if it has been done

    
    ## Visualize the data 3D photo model from RGB images and the time-resolved positions/orientations
    if ENABLE_VISUALIZE:
        visualize.show_mesh_camera(config, show_mesh = True, show_pose = True, ref_frame='ENU')

    # Georeference the line scans of the hyperspectral imager. Utilizes parsed data
    #georeference.main(config_file_mission)

    orthorectification.main(config_file_mission)


if __name__ == "__main__":
    main()
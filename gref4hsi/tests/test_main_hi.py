# Standard python library
import configparser
import sys
import os

# Local resources
from scripts import georeference
from scripts import orthorectification
from utils import parsing_utils
from gref4hsi.utils import visualize
from gref4hsi.utils.config_utils import prepend_data_dir_to_relative_paths

"""
This script is meant to be used for testing the processing pipeline of airborne HI data.
""" 
DATA_DIR = 'D:/HyperspectralDataAll/HI/2022-05-27-053505-NyAlesund-Flight8/'
config_file = DATA_DIR + 'configuration.ini'
# Set the data directory for the mission (locally where the data is stored)

prepend_data_dir_to_relative_paths(config_path=config_file, DATA_DIR=DATA_DIR)

config = configparser.ConfigParser()
config.read(config_file)

def main():
    ## Extract pose.csv and model.ply data from Agisoft Metashape (photogrammetry software) through API.
    ## Fails if you do not have an appropriate project.

    # The minimum for georeferencing is to parse 1) Mesh model and 2) The pose of the reference
    
    #config = parsing_utils.export_pose(config_file)

    config = configparser.ConfigParser()

    config.read(config_file)

    # Exports model
    #parsing_utils.export_model(config_file)

    ## Visualize the data 3D photo model from RGB images and the time-resolved positions/orientations
    
    #visualize.show_mesh_camera(config, show_mesh = True, show_pose = True)

    # Georeference the line scans of the hyperspectral imager. Utilizes parsed data
    
    #georeference.main(config_file)

    orthorectification.main(config_file)

    #print('')


if __name__ == "__main__":
    main()
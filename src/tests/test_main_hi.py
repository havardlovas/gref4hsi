# Standard python library
import configparser
import sys
import os

# Local resources
from scripts                 import georeference_mod
from scripts                 import orthorectification
from lib                     import parsing_utils
from scripts                 import visualize
from scripts.modulate_config import prepend_data_dir_to_relative_paths

"""
This script is meant to be used for testing the processing pipeline of airborne HI data.
""" 

#DATA_DIR    = 'D:/Ocean Color Colab/Processing Step4/Flight_1/'
DATA_DIR    = 'D:/Ocean Color Colab/Processing Step4/Flight_5/'
#DATA_DIR    = 'D:/Ocean Color Colab/Processing Step4/Flight_8/'
config_file = DATA_DIR + 'configuration.ini'
# Set the data directory for the mission (locally where the data is stored)

prepend_data_dir_to_relative_paths(config_path=config_file, DATA_DIR=DATA_DIR)

##Oliver: I belive this is not necessary
#config = configparser.ConfigParser()           #Oliver: Create obj. "configParser", / Update the configuration file with the new data directory (something gets written to the config file)
#config.read(config_file)                       #Oliver: Read the configuration file

def main():
    ## Extract pose.csv and model.ply data from Agisoft Metashape (photogrammetry software) through API.
    ## Fails if you do not have an appropriate project.

    # The minimum for georeferencing is to parse 1) Mesh model and 2) The pose of the reference

    #Oliver: This function updates the configuration.ini file with paths and writes the pose.csv file in the "Intermediate" folder and modifies the H5 files (adding "processed")
    config = parsing_utils.export_pose(config_file)

    #Oliver: I think this is not necessary!
#    config = configparser.ConfigParser()                #Oliver: config file was updated by the last function (export_pose) -> read it again
#    config.read(config_file)

    # Exports model
    parsing_utils.export_model(config_file)

    ## Visualize the data 3D photo model from RGB images and the time-resolved positions/orientations

    #visualize.show_mesh_camera(config, show_mesh = True, show_pose = True)

    # Georeference the line scans of the hyperspectral imager. Utilizes parsed data

    georeference_mod.main(config_file) #Must be done at least once for each dataset

    orthorectification.main(config_file)
    print('')

if __name__ == "__main__":
    main()
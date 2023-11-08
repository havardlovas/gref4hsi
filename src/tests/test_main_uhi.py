import configparser
import os

from lib import parsing_utils
from scripts import georeference
from scripts import visualize
from scripts.modulate_config import prepend_data_dir_to_relative_paths

# The configuration file stores the settings for georeferencing
cwd = os.getcwd()
config_file = cwd + '/data/Skogn21012021/configuration.ini'
print(config_file)
# Set the data directory for the mission (locally where the data is stored)
prepend_data_dir_to_relative_paths(config_path=config_file)

# TODO: update config.ini automatically with paths for simple reproducability
config = configparser.ConfigParser()
config.read(config_file)

def main():
    ## Extract pose.csv and model.ply data from Agisoft Metashape (photogrammetry software) through API.
    ## Fails if you do not have an appropriate project.

    # The minimum for georeferencing is to parse 1) Mesh model and 2) The pose of the reference
    #config = parsing_utils.export_pose(config_file)
    #parsing_utils.agisoft_export_model(config_file)

    ## Visualize the data 3D photo model from RGB images and the time-resolved positions/orientations
    #visualize.show_mesh_camera(config)

    # Georeference the line scans of the hyperspectral imager. Utilizes parsed data
    georeference.main(config_file, mode='georeference', is_calibrated=True)
    # Alternatively mode = 'calibrate'
    # georeference_mod.main(config_file, mode='calibrate', is_calibrated=True)


if __name__ == "__main__":
    main()
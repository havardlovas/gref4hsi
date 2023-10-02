import configparser

import georeference
import visualize

# Import the configuration file and read it into dictionary
home_path = 'C:/Users/haavasl/PycharmProjects/hyperspectral_toolchain'
data_dir = home_path + '/data/Skogn21012021'
config_file = data_dir + '/configuration.ini'
config = configparser.ConfigParser()
config.read(config_file)

def main():
    ## Extract pose.csv and model.ply data from Agisoft Metashape (photogrammetry software) through API.
    ## Fails if you do not have an appropriate project. Commented out
    # agisoft_extract.main(config_file)

    # Visualize the data 3D photo model from RGB images and the time-resolved positions/orientations
    visualize.show_mesh_camera(config)
    # Georeference the line scans of the hyperspectral imager. Depends only on time-resolved positions.
    georeference.main(config_file, mode='georeference', is_calibrated=True)
    # Alternative mode = 'calibrate'

if __name__ == "__main__":
    main()
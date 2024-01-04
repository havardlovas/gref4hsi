import configparser
import os

from lib import parsing_utils
from scripts import georeference
from scripts import visualize
from scripts.modulate_config import prepend_data_dir_to_relative_paths
from scripts import extract_rgb_from_h5

from scripts.image_color_correction import GrayWorldCorrection
# The configuration file stores the settings for georeferencing
config_file = 'D:/HyperspectralDataAll/UHI/2022-08-05-North-Pole-T1-S1/configuration.ini'
print(config_file)
# Set the data directory for the mission (locally where the data is stored)
prepend_data_dir_to_relative_paths(config_path=config_file)

# TODO: update config.ini automatically with paths for simple reproducability
config = configparser.ConfigParser()
config.read(config_file)

# Define the path to the h5-dir
h5dir = config['Absolute Paths']['h5dir']
# Define the write path
rgbimgpath_raw = config['Absolute Paths']['rgbimgpath']

extract_rgb_from_h5.main(folder_path=h5dir, save_path=rgbimgpath_raw)

rgbimgpath_corrected =  config['Absolute Paths']['rgbimgpathcorrected']

gwd = GrayWorldCorrection(image_dir= rgbimgpath_raw, image_dir_write=rgbimgpath_corrected)
# Calculate average image over all
gwd.calculate_avg_image_x()
# Calculate standard deviation image over all
gwd.calculate_sigma_image_x()
# As of now, the code only supports one intensity for all channels, but it could easily be modified to one per channel
mean_desired_intensity = 90
std_dev_desired_intensity = 30

gwd.grey_world_correction(mean_intensity=mean_desired_intensity, std_intensity=std_dev_desired_intensity)

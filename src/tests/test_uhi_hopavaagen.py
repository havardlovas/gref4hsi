import configparser
from parsing_utils import Hyperspectral
import georeference
import visualize
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# Define a custom sorting function to extract the numeric part of the filename
# Define a custom sorting function
def numerical_sort_key(filename):
    # Split the filename by '_' and '.' to extract the numeric part
    parts = filename.split('_')
    if len(parts) > 1:
        numeric_part = parts[1].split('.')[0]
        return int(numeric_part)
    return filename

# Import the configuration file and read it into dictionary
home_path = 'C:/Users/haavasl/PycharmProjects/hyperspectral_toolchain'
data_dir = home_path + '/data/Hopavagen24_10_2023'
# TODO: update config.ini automatically with paths for simple reproducability
config_file = data_dir + '/configuration.ini'
config = configparser.ConfigParser()
config.read(config_file)


def main():
    """
    Writes each *.h5 file (in "h5dir") radiance to a *.tif file (in "tiff_dir").
    """



    # Traverse through h5 dir to append the data to file

    h5dir = config['HDF']['h5dir']
    tiff_dir = 'E:/double-blueye/Radiance_composites/'

    concat_rgb_composite_img = None
    is_first_image_of_transect = True
    prev_chunk_number = 0
    prev_transect_name = ''
    files = os.listdir(h5dir)

    # TODO: Make numerical sort properly sort last element
    # Sort the filenames based on the numeric part
    sorted_filenames = sorted(files, key=numerical_sort_key, reverse=False)

    # TODO: Consider making list of lists (transect_chunks)
    print(sorted_filenames)
    print(len(sorted_filenames))
    count = 0

    for filename in sorted_filenames:
        if count > 0:
            # Find the interesting prefixes
            if filename.endswith('h5') or filename.endswith('hdf'):
                # Identify the total path and read data into Hyperspectral object
                path_hdf = h5dir + filename
                hyp = Hyperspectral(path_hdf, config)

                # Identify transect number
                filename_splitted_underscore = filename.split('_')

                # Involves
                curr_transect_name = filename_splitted_underscore[0] + '_'+ filename_splitted_underscore[1] + '_' +filename_splitted_underscore[2]
                # Chunk number (from 1-N)
                curr_chunk_number = int(filename_splitted_underscore[3].split('.')[0])


                # Convert data cube to radiance
                hyp.digital_counts_2_radiance(config)

                # Set custom RGB bands from *.ini file
                wl_red = float(config['General']['RedWavelength'])
                wl_green = float(config['General']['GreenWavelength'])
                wl_blue = float(config['General']['BlueWavelength'])


                wavelength_nm = np.array([wl_red, wl_green, wl_blue])
                band_ind_R = np.argmin(np.abs(wavelength_nm[0] - hyp.band2Wavelength))
                band_ind_G = np.argmin(np.abs(wavelength_nm[1] - hyp.band2Wavelength))
                band_ind_B = np.argmin(np.abs(wavelength_nm[2] - hyp.band2Wavelength))

                radiance_composite_rgb = hyp.dataCubeRadiance[:, :, [band_ind_R, band_ind_G, band_ind_B]]


                # Determine if current chunk_number is one greater then the previous
                print(curr_transect_name + '_' + str(curr_chunk_number))
                if curr_chunk_number - prev_chunk_number == 1:
                    print('')
                else:
                    # Then we should save the concatenated image.
                    tiff.imwrite(tiff_dir + prev_transect_name + '_concat.tif', concat_rgb_composite_img)
                    # Reset the is_first_image_of_transect
                    is_first_image_of_transect = True
                #print(curr_transect_name + '_' + str(prev_chunk_number))



                # Concatenating
                if is_first_image_of_transect:
                    concat_rgb_composite_img = radiance_composite_rgb
                    is_first_image_of_transect = False
                    tiff.imwrite(tiff_dir + curr_transect_name + '_' + str(curr_chunk_number) + '.tif', radiance_composite_rgb)
                else: # At a change of transects
                    concat_rgb_composite_img = np.concatenate((concat_rgb_composite_img, radiance_composite_rgb), axis=0)
                    tiff.imwrite(tiff_dir + curr_transect_name + '_' + str(curr_chunk_number) + '.tif', radiance_composite_rgb)



                prev_chunk_number = curr_chunk_number
                prev_transect_name = curr_transect_name
                count += 1
                if count == len(sorted_filenames):
                    tiff.imwrite(tiff_dir + curr_transect_name + '_concat.tif', concat_rgb_composite_img)

        else:
            count += 1


if __name__ == "__main__":
    main()
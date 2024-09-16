import configparser
import os
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

# Local resources:
from gref4hsi.utils.gis_tools import GeoSpatialAbstractionHSI
from gref4hsi.utils.parsing_utils import Hyperspectral



def main(iniPath):
    config = configparser.ConfigParser()
    config.read(iniPath)

    # Paths for input data
    h5_folder = config['Absolute Paths']['h5_folder']

    # Paths for output data:
    # 1) The data cube locations
    envi_cube_dir = config['Absolute Paths']['orthorectified_cube_folder']

    # 2) The RGB composite locations
    rgb_composite_dir = config['Absolute Paths']['rgb_composite_folder']

    # 3) The ancillary data locations
    anc_dir = config['Absolute Paths']['anc_folder']

    # 3) The footprints
    footprint_dir = config['Absolute Paths']['footprint_folder']

    


    # The necessary data from the H5 file for resampling the datacube and composite (fwhm is optional)
    # ECEF point cloud from georeferencing.
    h5_folder_point_cloud_ecef = config['Georeferencing']['points_ecef_crs']

    # Radiance cube location depends on whether original data was calibrated:
    is_calibrated = eval(config['HDF.hyperspectral']['is_calibrated'])
    if is_calibrated:
        h5_folder_radiance_cube = config['HDF.hyperspectral']['datacube']
    else:
        # Calibrate by calling hyp = Hyperspectral()
        # This file has two datacubes and the suffix helps to identify the calibrated one.
        h5_folder_radiance_cube = config['HDF.hyperspectral']['dataCube'] + '_radiance'

    # Wavelength centers for all bands
    h5_folder_wavelength_centers = config['HDF.calibration']['band2wavelength']
    
    try:
        h5_folder_wavelength_widths = config['HDF.calibration']['fwhm']
    except:
        h5_folder_wavelength_widths = 'undefined'

    
    try:
        pixel_mask_by_footprint = eval(config['Orthorectification']['mask_pixel_by_footprint'])
        if pixel_mask_by_footprint:
            pixel_mask_method = 'footprint'
        else: # Defaults to this
            pixel_mask_method = 'nn'

    except:
        # Defaults to 'nn'
        pixel_mask_method = 'nn'



    # The necessary data (a dictionary) from H5 file for resampling ancillary data (uses the same grid as datacube)
    anc_dict = config['Ancillary']

    # Settings having to do with coordinate reference systems, described below
    SettingsCRS = namedtuple('SettingsOrthorectification', ['epsg_geocsc', 
                                                              'epsg_proj'])
    
    config_crs = SettingsCRS(epsg_geocsc=int(config['Coordinate Reference Systems']['geocsc_epsg_export']),
                             # The epsg code of the geocentric coordinate system (ECEF)
                             epsg_proj=int(config['Coordinate Reference Systems']['proj_epsg']),
                             # The epsg code of the projected coordinate system (e.g. UTM 32 has epsg 32632 for wgs 84 ellipsoid)
                             )

    # Settings associated with orthorectification of datacube
    SettingsOrtho = namedtuple('SettingsOrthorectification', ['ground_resolution', 
                                                                            'wl_red', 
                                                                            'wl_green', 
                                                                            'wl_blue',
                                                                            'nodata_value',
                                                                            'raster_transform_method',
                                                                            'resample_rgb_only',
                                                                            'chunk_size_cube_GB',
                                                                            'wavelength_unit',
                                                                            'radiometric_unit',
                                                                            'sensor_type',
                                                                            'interleave',
                                                                            'pixel_mask_method'])
    
    config_ortho = SettingsOrtho(ground_resolution = float(config['Orthorectification']['resolutionHyperspectralMosaic']), 
                                 # Rectified grid resolution in meters
                              wl_red = float(config['General']['red_wave_length']), 
                              # Red wavelength in <wavelength_unit>
                              wl_green = float(config['General']['green_wave_length']), 
                              # Green wavelength in <wavelength_unit>
                              wl_blue = float(config['General']['blue_wave_length']), 
                              # Blue wavelength in <wavelength_unit>
                              raster_transform_method = config['Orthorectification']['raster_transform_method'], 
                              # Describes what raster transform to use. Either "north_east" or "minimal_rectangle". 
                              # Latter is fastest and most memory efficient, but support is lacking for several GIS software.
                              nodata_value = np.int64(config['Orthorectification']['nodata']),
                              # The fill value for empty cells (select values not occcuring in cube or ancillary data)
                              resample_rgb_only = eval(config['Orthorectification']['resample_rgb_only']),
                              # Boolean being expressing whether to rectify only composite (true) or data cube and composite (false). True is fast.
                              chunk_size_cube_GB = float(config['Orthorectification']['chunk_size_cube_GB']),
                              # For large files, RAM issues could be a concern. For rectified files exeeding this size, data is written chunk-wize to a memory map.
                              wavelength_unit = config['General']['wavelength_unit'],
                              # Unit of wavelengths, often nanometer
                              radiometric_unit = config['General']['radiometric_unit'],
                              # Unit of radiance, often like <pfx1>Watts / ( [<pfx2>^2]meter^2 steradian <pfx3>meter). pfx means metric prefix.
                              # <pfx1> relates to energy and is 1, milli (10e-3) or micro (10e-6)
                              # <pfx2> relates to area and is 1 or centi (10e-2)
                              # <pfx3> relates to wavelength and is micro (10e-6) or nano (10e-9)
                              sensor_type = config['General']['sensor_type'],
                              # Brand and model
                              interleave = config['Orthorectification']['interleave'],
                              # ENVI interleave: either 'bsq', 'bip' or 'bil', see:
                              # https://envi.geoscene.cn/help/Subsystems/envi/Content/ExploreImagery/ENVIImageFiles.html
                              pixel_mask_method = pixel_mask_method
                              # When resampling how to mask nodata pixels: either 'nn' or 'footprint
                              )


    print("\n################ Orthorectifying: ################")
    files = sorted(os.listdir(h5_folder))
    # Filter out files that do not end with ".h5"
    h5_files = [file for file in files if file.endswith(".h5")]
    n_files= len(h5_files)    
    file_count = 0
    for filename in h5_files:
        if filename.endswith('h5') or filename.endswith('hdf'):

            progress_perc = 100*file_count/n_files
            print(f"Orthorectifying file {file_count+1}/{n_files}, progress is {progress_perc} %")
            # Path to hierarchical file
            h5_filename = h5_folder + filename

            # Read the 3D point cloud, radiance cube 
            # Extract the point cloud (if it was not georeferenced, it will throw an error)
            try:
                point_cloud_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                         dataset_name=h5_folder_point_cloud_ecef)
            except:
                print('Because chunk failed ray tracing, it is not orthorectified')
                continue # Move to next h5 file
            # Need the radiance cube for resampling
            if not is_calibrated:
                # load_datacube will calibrate and write radiance data cube to h5 file (if not already there)
                hyp = Hyperspectral(filename=h5_filename, config=config, load_datacube=True)
                del hyp
            
            # Todo, don't georeference actual cube, but use dataset
            radiance_cube = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                            dataset_name = h5_folder_radiance_cube)

        
            wavelengths = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                            dataset_name=h5_folder_wavelength_centers)
            try:
                fwhm = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                            dataset_name=h5_folder_wavelength_widths)
            except KeyError:
                fwhm = np.nan


            # The code below is independent of the h5 file format. The exception is the writing of ancillary data to a form of datacube
            # Generates an object for dealing with GIS operations
            gisHSI = GeoSpatialAbstractionHSI(point_cloud=point_cloud_ecef, 
                                              transect_string=filename.split('.')[0],
                                              config_crs=config_crs)

            # The point cloud is transformed to the projected system
            gisHSI.transform_geocentric_to_projected(config_crs=config_crs)

            # Calculate the footprint of hyperspectral data
            gisHSI.footprint_to_shape_file(footprint_dir=footprint_dir)
            
            
            
            # Resample imagery (RGB composite or both)#!
            gisHSI.resample_datacube(radiance_cube=radiance_cube,
                                     wavelengths=wavelengths,
                                     fwhm=fwhm,
                                     envi_cube_dir=envi_cube_dir,
                                     rgb_composite_dir=rgb_composite_dir,
                                     config_ortho=config_ortho)

            

            # The ancilliary data is read from h5 files and converted into a datacube
            if eval(config['Orthorectification']['resample_ancillary']): 
                gisHSI.resample_ancillary(h5_filename=h5_filename, 
                                        anc_dict = anc_dict, 
                                        anc_dir = anc_dir,
                                        interleave=config_ortho.interleave)
        
        file_count+=1
if __name__ == '__main__':
    args = sys.argv[1:]
    iniPath = args[0]
    main(iniPath)
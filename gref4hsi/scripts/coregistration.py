import configparser
import os

import spectral as sp

from gref4hsi.utils.gis_tools import GeoSpatialAbstractionHSI
from gref4hsi.utils.parsing_utils import Hyperspectral
"""
[Coordinate Reference Systems]
proj_epsg = 25832
geocsc_epsg_export = 4978
pos_epsg_orig = 4978
dem_epsg = 25832
"""




# Function called to apply standard processing on a folder of files
def main(iniPath, viz = False):
    config = configparser.ConfigParser()
    config.read(iniPath)

    # Set the coordinate reference systems
    epsg_proj = int(config['Coordinate Reference Systems']['proj_epsg'])
    epsg_geocsc = int(config['Coordinate Reference Systems']['geocsc_epsg_export'])

    # Establish reference data
    path_orthomosaic_reference_folder = config['Absolute Paths']['orthomosaic_reference_folder']
    orthomosaic_reference_fn = os.listdir(path_orthomosaic_reference_folder)[0] # Grab the only file in folder
    ref_ortho_path = os.path.join(path_orthomosaic_reference_folder, orthomosaic_reference_fn)

    dem_path = config['Absolute Paths']['dem_path']

    # Establish match data (HSI), including composite and anc data
    path_composites_match = config['Absolute Paths']['rgb_composite_folder']
    path_anc_match = config['Absolute Paths']['anc_folder']

    # Create a temporary folder for resampled reference orthomosaics and DEMs
    ref_resampled_gis_path = config['Absolute Paths']['ref_ortho_reshaped']
    if not os.path.exists(ref_resampled_gis_path):
        os.mkdir(ref_resampled_gis_path)

    # The necessary data from the H5 file for getting the positions and orientations.
        
    # Position is stored here in the H5 file
    h5_folder_position_ecef = config['HDF.processed_nav']['position_ecef']

    # Quaternion is stored here in the H5 file
    h5_folder_quaternion_ecef = config['HDF.processed_nav']['quaternion_ecef']

    # Timestamps here
    h5_folder_time_pose = config['HDF.processed_nav']['timestamp']

    

    print("\n################ Coregistering: ################")

    # Iterate the RGB composites
    hsi_composite_files = sorted(os.listdir(path_composites_match))
    n_files= len(hsi_composite_files)
    file_count = 0
    for file_count, hsi_composite_file in enumerate(hsi_composite_files):
        
        file_base_name = hsi_composite_file.split('.')[0]
        
        # The match data (hyperspectral)
        hsi_composite_path = os.path.join(path_composites_match, hsi_composite_file)
        print(hsi_composite_path)

        # Prior to matching the files are cropped to image grid of hsi_composite_path
        ref_ortho_reshaped_path = os.path.join(ref_resampled_gis_path, hsi_composite_file)
        # The DEM is also cropped to grid for easy extraction of data
        dem_reshaped = os.path.join(ref_resampled_gis_path, file_base_name + '_dem.tif')

        # 
        GeoSpatialAbstractionHSI.resample_rgb_ortho_to_hsi_ortho(ref_ortho_path, hsi_composite_path, ref_ortho_reshaped_path)

        GeoSpatialAbstractionHSI.resample_dem_to_hsi_ortho(dem_path, hsi_composite_path, dem_reshaped)

        # By comparing the hsi_composite with the reference rgb mosaic we get two feature vectors in the pixel grid and 
        # the absolute registration error in meters in global coordinates
        uv_vec_hsi, uv_vec_ref, diff_AE_meters, transform_pixel_projected  = GeoSpatialAbstractionHSI.compare_hsi_composite_with_rgb_mosaic(hsi_composite_path, 
                                                                                                                                           ref_ortho_reshaped_path)
        

        # At first the reference observations must be converted to a true 3D system, namely ecef
        ref_points_ecef = GeoSpatialAbstractionHSI.compute_reference_points_ecef(uv_vec_ref, 
                                                                                 transform_pixel_projected, 
                                                                                 dem_reshaped, 
                                                                                 epsg_proj, 
                                                                                 epsg_geocsc)




        
        # Next up we need to get the associated pixel number and frame number. Luckily they are in the same grid as the pixel observations
        
        anc_file_path = os.path.join(path_anc_match, file_base_name + '.hdr')
        anc_image_object = sp.io.envi.open(anc_file_path)
        anc_band_list = anc_image_object.metadata['band names']

        pixel_nr_grid = anc_image_object[:,:, anc_band_list.index('pixel_nr_grid')].squeeze()

        unix_time_grid = anc_image_object[:,:, anc_band_list.index('unix_time_grid')].squeeze()


        # Read the ecef position, quaternion and timestamp
        h5_filename = os.path.join(config['Absolute Paths']['h5_folder'], file_base_name + '.h5')

        # Extract the point cloud
        position_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                        dataset_name=h5_folder_position_ecef)
        # Need the radiance cube for resampling
        quaternion_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                        dataset_name=h5_folder_quaternion_ecef)
    
        time_pose = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                        dataset_name=h5_folder_time_pose)

        pixel_nr_vec, position_vec, quaternion_vec = GeoSpatialAbstractionHSI.compute_position_orientation_features(uv_vec_hsi, 
                                                                       pixel_nr_grid, 
                                                                       unix_time_grid, 
                                                                       position_ecef, 
                                                                       quaternion_ecef, 
                                                                       time_pose)





        # TODO: write the difference vector to file.

        print()
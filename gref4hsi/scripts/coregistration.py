import configparser
import os

import numpy as np
import spectral as sp

from gref4hsi.utils.gis_tools import GeoSpatialAbstractionHSI
from gref4hsi.utils.parsing_utils import Hyperspectral
import gref4hsi.utils.geometry_utils as geom_utils

def optimize_function_boresight(param, features_df):

    self.rot_x = param[0]  # Pitch relative to cam (Equivalent to cam defining NED and uhi BODY)
    self.rot_y = param[1]*0  # Roll relative to cam
    self.rot_z = param[2]  # Yaw relative to cam
    self.v_c = param[3]
    self.f = param[4]
    self.k1 = param[5]*0
    self.k2 = param[6]
    self.k3 = param[7]

    # Computes the directions in a local frame, or normalized scanline image frame
    x_norm = geom_utils.compute_camera_rays_from_parameters(pixel_nr=features_df['pixel_nr'],
                                                   rot_x=param[0],
                                                   rot_y=param[1],
                                                   rot_y=param[2], 
                                                   cx=param[3],
                                                   f=param[4],
                                                   k1=param[5],
                                                   k2=param[6],
                                                   k3=param[7])
    
    trans_hsi_body = np.array([trans_z, trans_y, trans_x])
    rot_hsi_body = np.array([rot_z, rot_y, rot_x]) * 180 / np.pi

    X_norm = geom_utils.reproject_world_points_to_hsi(trans_hsi_body, 
                                             rot_hsi_body, 
                                             pos_body, 
                                             rot_body, 
                                             points_world) # reprojects to the same frame

    errorx = x_norm - X_norm[:, 0]
    errory = -X_norm[:, 1]

    print(np.median(np.abs(errorx)))
    print(np.median(np.abs(errory)))
    #if np.median(np.abs(errorx)) < 0.01:
    #import matplotlib.pyplot as plt
##
    #plt.scatter(calObj.pixel, errorx)
    #plt.scatter(errorx, errory)




    #print(np.median(np.abs(errorx)))
    #print(np.median(np.abs(errory)))



    return np.concatenate((errorx.reshape(-1), errory.reshape(-1)))




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


    ref_gcp_path = config['Absolute Paths']['ref_gcp_path']
    

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
    n_files = len(hsi_composite_files)
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
        anc_nodata = float(anc_image_object.metadata['data ignore value'])

        pixel_nr_grid = anc_image_object[:,:, anc_band_list.index('pixel_nr_grid')].squeeze()
        unix_time_grid = anc_image_object[:,:, anc_band_list.index('unix_time_grid')].squeeze()

        # Remove the suffixes added in the orthorectification
        suffixes = ["_north_east", "_minimal_rectangle"]
        for suffix in suffixes:
            if file_base_name.endswith(suffix):
                file_base_name_h5 = file_base_name[:-len(suffix)]

        # Read the ecef position, quaternion and timestamp
        h5_filename = os.path.join(config['Absolute Paths']['h5_folder'], file_base_name_h5 + '.h5')

        # Extract the point cloud
        position_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                        dataset_name=h5_folder_position_ecef)
        # Need the radiance cube for resampling
        quaternion_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                        dataset_name=h5_folder_quaternion_ecef)
    
        time_pose = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                        dataset_name=h5_folder_time_pose)
        
        print('')

        pixel_nr_vec, unix_time_vec, position_vec, quaternion_vec, feature_mask = GeoSpatialAbstractionHSI.compute_position_orientation_features(uv_vec_hsi, 
                                                                       pixel_nr_grid, 
                                                                       unix_time_grid, 
                                                                       position_ecef, 
                                                                       quaternion_ecef, 
                                                                       time_pose,
                                                                       nodata = anc_nodata)
        
        # Mask the reference points accordingly
        ref_points_vec = ref_points_ecef[feature_mask,:]

        # Now we have computed the GCPs
        gcp_dict = {'file_count': pixel_nr_vec*0 + file_count,
                    'pixel_nr': pixel_nr_vec, 
                    'unix_time': unix_time_vec,
                    'position_x': position_vec[:,0],
                    'position_y': position_vec[:,1],
                    'position_z': position_vec[:,2],
                    'quaternion_x': quaternion_vec[:,0],
                    'quaternion_y': quaternion_vec[:,1],
                    'quaternion_z': quaternion_vec[:,2],
                    'quaternion_w': quaternion_vec[:,3],
                    'reference_points_x': ref_points_vec[:,0],
                    'reference_points_y': ref_points_vec[:,1],
                    'reference_points_z': ref_points_vec[:,2]}
        
        import pandas as pd

        # Convert to a dataframe
        gcp_df = pd.DataFrame(gcp_dict)

        # Maybe write this dataframe to a 
        if file_count==0:
            gcp_df_all = gcp_df
        else:
            gcp_df_all = pd.concat([gcp_df_all, gcp_df])

        

    gcp_df_all.to_csv(path_or_buf=ref_gcp_path)

    # Using the accumulated features, we can optimize for the boresight angles
    from scipy.optimize import least_squares
    from gref4hsi.utils.geometry_utils import CalibHSI


    cal_obj = CalibHSI(file_name_cal_xml=config['Absolute Paths']['hsi_calib_path'])

    param0 = np.array([cal_obj.rx, 
                       cal_obj.ry, 
                       cal_obj.rz, 
                       cal_obj.cx, 
                       cal_obj.f, 
                       cal_obj.k1, 
                       cal_obj.k2, 
                       cal_obj.k3])


    res = least_squares(fun = optimize_function, x0 = param0, args= (gcp_df_all,) , x_scale='jac', method='lm')

    
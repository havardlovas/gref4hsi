import configparser
import os
import time

import numpy as np
import spectral as sp

from gref4hsi.utils.gis_tools import GeoSpatialAbstractionHSI
from gref4hsi.utils.parsing_utils import Hyperspectral
import gref4hsi.utils.geometry_utils as geom_utils
from gref4hsi.utils.geometry_utils import CalibHSI, GeoPose

from scipy.spatial.transform import Rotation as RotLib
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import pandas as pd



def numerical_jacobian(fun, param, eps=1e-6, **kwargs):
  """
  Computes the Jacobian of a function using central differences.

  Args:
      fun: The function for which to compute the Jacobian.
      param: The parameters of the function.
      eps: A small epsilon value for numerical differentiation.

  Returns:
      A numpy array representing the Jacobian.
  """
  n_param = len(param)
  n_fun = len(fun(param, **kwargs))
  jacobian = np.zeros((n_fun, n_param))
  for i in range(n_param):
    param_pos = param.copy()
    param_neg = param.copy()
    param_pos[i] += eps
    param_neg[i] -= eps
    jacobian[:, i] = (fun(param_pos, **kwargs) - fun(param_neg, **kwargs)) / (2 * eps)
  return jacobian


def interpolate_time_nodes(time_from, value, time_to, method = 'linear'):
    # One simple option is to use Scipy's sortiment. The sparsity pattern should be included somehow..
    if method in ['nearest', 'linear', 'slinear', 'quadratic', 'cubic']:
        return interp1d(time_from, value, kind=method)(time_to)
    

def compose_pose_errors(param_time_var, time_nodes, unix_time_features, rot_body_ned, rot_ned_ecef, pos_body):
    """Takes a (4*n_node) vector of errors in X, Y, Z and Yaw, interpolates and composes (adds) it to the pose from the navigation data"""
    n_features = len(unix_time_features)

    # Interpolate to the right time
    err_interpolated = interpolate_time_nodes(time_nodes, 
                                                param_time_var.reshape((4, -1)), 
                                                time_to = unix_time_features, method='quadratic').transpose()

    # The errors in the interpolated form
    param_err_pos = err_interpolated[:, 0:3]
    param_err_eul_ZYX = np.vstack((err_interpolated[:, 3:4].flatten(), 
                                        np.zeros(n_features),
                                        np.zeros(n_features))).transpose()
    
    # To be left multiplied with attitude 
    rot_err_NED = RotLib.from_euler('ZYX', param_err_eul_ZYX, degrees=True)


    # Modify the poses with the parametrically interpolated values:
    pos_body += param_err_pos

    # Recall that we only estimate yaw errors
    rot_body_ned_corr = rot_err_NED * rot_body_ned

    # Convert orientation to ECEF
    rot_body_ecef = rot_ned_ecef * rot_body_ned_corr

    return pos_body, rot_body_ecef

def objective_fun_reprojection_error(param, features_df, param0, is_variab_parameter, time_nodes):
    
    #
    n_param_tot_static = len(is_variab_parameter)
    n_variab_param_static = sum(is_variab_parameter==1)
    param_vec_total = np.zeros(n_param_tot_static)
    
    # Static parameters (not time-varying)
    param_count = 0
    for i in range(n_param_tot_static):
        if bool(is_variab_parameter[i]):
            # take from the parameter vector (assuming that they are ordered the same way)
            param_vec_total[i] = param[param_count]
            param_count += 1
            
        else:
            # Take fixed parameters from param0 - the initial parameter guess
            param_vec_total[i] = param0[i]

    # Static parameters:

    # Boresight
    rot_x = param_vec_total[0]
    rot_y = param_vec_total[1]
    rot_z = param_vec_total[2]

    # Principal point
    cx = param_vec_total[3]

    # Focal length
    f = param_vec_total[4]

    # Distortions
    k1 = param_vec_total[5]
    k2 = param_vec_total[6]
    k3 = param_vec_total[7]

    # Lever arm
    trans_x = param_vec_total[8] 
    trans_y = param_vec_total[9]
    trans_z = param_vec_total[10] 

    
    
    # Lever arm
    trans_hsi_body = np.array([trans_z, trans_y, trans_x])
    
    # Boresight vector
    eul_ZYX_hsi_body = np.array([rot_z, rot_y, rot_x]) * 180 / np.pi

    # Convert to rotation object for convenience
    rot_hsi_body = RotLib.from_euler('ZYX', eul_ZYX_hsi_body, degrees=True)

    # The position of the vehicle body wrt ECEF
    pos_body = np.vstack((features_df['position_x'], 
                          features_df['position_y'], 
                          features_df['position_z'])).transpose()
    
    # The quaternion of the body with respect to NED
    quat_body_ned = np.vstack((features_df['quaternion_b_n_x'],
                          features_df['quaternion_b_n_y'], 
                          features_df['quaternion_b_n_z'],
                          features_df['quaternion_b_n_w'])).transpose()
    
    # The rotation from NED to ECEF
    quat_ned_ecef = np.vstack((features_df['quaternion_n_e_x'],
                          features_df['quaternion_n_e_y'], 
                          features_df['quaternion_n_e_z'],
                          features_df['quaternion_n_e_w'])).transpose()

    # Convert to rotation object for convenience
    rot_body_ned = RotLib.from_quat(quat_body_ned)
    rot_ned_ecef = RotLib.from_quat(quat_ned_ecef)

    # Whether to estimate time-varying errors
    if time_nodes is None:
        # Assumes that navigation-based poses are correct
        rot_body_ecef = rot_ned_ecef * rot_body_ned

    else:

        # If we try to estimate errors in the navigation data (these parameters are after the static parameters)
        param_time_var = param[n_variab_param_static:]

        # The parametric errors represent a handful of nodes and must be interpolated to the feature times
        unix_time_features = features_df['unix_time']

        # We compose them with (add them to) the position/orientation estimates from the nav system
        pos_body, rot_body_ecef = compose_pose_errors(param_time_var, time_nodes, unix_time_features, rot_body_ned, rot_ned_ecef, pos_body)




    # The reference points in ECEF (obtained from the reference orthomosaic)
    points_world_reference = np.vstack((features_df['reference_points_x'], 
                          features_df['reference_points_y'], 
                          features_df['reference_points_z'])).transpose()
    
    # We reproject the reference points to the normalized HSI image plane
    X_norm = geom_utils.reproject_world_points_to_hsi_plane(trans_hsi_body, 
                                             rot_hsi_body, 
                                             pos_body, 
                                             rot_body_ecef, 
                                             points_world_reference) # reprojects to the same frame
    
    # Reprojected observations in image plane
    x_norm_reproj = X_norm[:, 0]
    y_norm_reproj = X_norm[:, 1]
    
    # The observation is by definition in the scanline where y_norm = 0
    # Using the pixel numbers corresponding to the features
    # we convert the pixel number to an x-coordinate on the normalized HSI image plane
    pixel_nr = features_df['pixel_nr']
    x_norm = geom_utils.compute_camera_rays_from_parameters(pixel_nr=pixel_nr,
                                                   cx=cx,
                                                   f=f,
                                                   k1=k1,
                                                   k2=k2,
                                                   k3=k3)
    
    # At the time (features_df["unix_time"]) of an observation, the feature satisfies y_norm=0 (not the case for y_norm_reproj).
    y_norm = np.zeros(x_norm.shape)

    # The difference is expressed as
    errorx = np.array(x_norm - x_norm_reproj)
    errory = np.array(y_norm - y_norm_reproj)

    # Least squares expects a 1D function evaluation vector
    return np.concatenate((errorx.reshape(-1), errory.reshape(-1)))




# Function called to apply standard processing on a folder of files
def main(config_path, mode):
    config = configparser.ConfigParser()
    config.read(config_path)

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
    file_count = 0

    if mode == 'compare':
        print("\n################ Comparing to reference: ################")
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

            # Extract the ecef positions for each frame
            position_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                            dataset_name=h5_folder_position_ecef)
            # Extract the ecef orientations for each frame
            quaternion_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                            dataset_name=h5_folder_quaternion_ecef)
            # Extract the timestamps for each frame
            time_pose = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                            dataset_name=h5_folder_time_pose)
            
        

            pixel_nr_vec, unix_time_vec, position_vec, quaternion_vec, feature_mask = GeoSpatialAbstractionHSI.compute_position_orientation_features(uv_vec_hsi, 
                                                                        pixel_nr_grid, 
                                                                        unix_time_grid, 
                                                                        position_ecef, 
                                                                        quaternion_ecef, 
                                                                        time_pose,
                                                                        nodata = anc_nodata)
            
            rot_body = RotLib.from_quat(quaternion_vec)
            geo_pose = GeoPose(timestamps=unix_time_vec, 
                               rot_obj=rot_body, 
                               rot_ref='ECEF', 
                               pos=position_vec, 
                               pos_epsg=4978)
            
            # Divide into two linked rotations
            quat_body_to_ned = geo_pose.rot_obj_ned.as_quat()
            quat_ned_to_ecef = geo_pose.rot_obj_ned_2_ecef.as_quat()


            # Mask the reference points accordingly and the difference vector
            ref_points_vec = ref_points_ecef[feature_mask, :]
            diff_AE_valid = diff_AE_meters[feature_mask]

            diff_uv = uv_vec_hsi[feature_mask, :] - uv_vec_ref[feature_mask, :]

            # Now we have computed the GCPs with coincident metainformation
            gcp_dict = {'file_count': np.ones(pixel_nr_vec.shape)*file_count,
                        'pixel_nr': pixel_nr_vec, 
                        'unix_time': unix_time_vec,
                        'position_x': position_vec[:,0],
                        'position_y': position_vec[:,1],
                        'position_z': position_vec[:,2],
                        'quaternion_b_n_x': quat_body_to_ned[:,0],
                        'quaternion_b_n_y': quat_body_to_ned[:,1],
                        'quaternion_b_n_z': quat_body_to_ned[:,2],
                        'quaternion_b_n_w': quat_body_to_ned[:,3],
                        'quaternion_n_e_x': quat_ned_to_ecef[:,0],
                        'quaternion_n_e_y': quat_ned_to_ecef[:,1],
                        'quaternion_n_e_z': quat_ned_to_ecef[:,2],
                        'quaternion_n_e_w': quat_ned_to_ecef[:,3],
                        'reference_points_x': ref_points_vec[:,0],
                        'reference_points_y': ref_points_vec[:,1],
                        'reference_points_z': ref_points_vec[:,2],
                        'diff_absolute_error': diff_AE_valid,
                        'diff_u': diff_uv[:, 0],
                        'diff_v': diff_uv[:, 1]}
            


            # Convert to a dataframe
            gcp_df = pd.DataFrame(gcp_dict)

            # Maybe write this dataframe to a 
            if file_count==0:
                gcp_df_all = gcp_df
            else:
                gcp_df_all = pd.concat([gcp_df_all, gcp_df])

        gcp_df_all.to_csv(path_or_buf=ref_gcp_path)

    elif mode == 'calibrate':
        
        
        gcp_df_all = pd.read_csv(ref_gcp_path)

        # Errors in the x direction. We define the inlier interval by use of the interquartile rule
        u_err = gcp_df_all['diff_u']
        Q1_u = np.percentile(u_err, 25)
        Q3_u = np.percentile(u_err, 75)
        IQR_u = Q3_u-Q1_u
        lower_bound_u = Q1_u - 1.5*IQR_u
        upper_bound_u = Q3_u + 1.5*IQR_u

        # Errors in the x direction. We define the inlier interval by use of the interquartile rule
        v_err = gcp_df_all['diff_v']
        Q1_v = np.percentile(v_err, 25)
        Q3_v = np.percentile(v_err, 75)
        IQR_v = Q3_v-Q1_v
        lower_bound_v = Q1_v - 1.5*IQR_v
        upper_bound_v = Q3_v + 1.5*IQR_v

        # Simply mask by the interquartile rule for the north-east registration errors (unit is in pixels)
        feature_mask_IQR = np.all([u_err > lower_bound_u,
                                   u_err < upper_bound_u,
                                   v_err > lower_bound_v,
                                   v_err < upper_bound_v], axis=0)
        # These features are used
        df_gcp_filtered = gcp_df_all[feature_mask_IQR]
        
        
        print("\n################ Calibrating camera parameters: ################")
        # Using the accumulated features, we can optimize for the boresight angles, camera parameters and lever arms
        
        ## Defining the options:
        
        # Here we define what that is to be calibrated
        calibrate_dict = {'calibrate_boresight': False,
                          'calibrate_camera': False,
                          'calibrate_lever_arm': False,
                          'calibrate_cx': False,
                          'calibrate_f': True,
                          'calibrate_k1': False,
                          'calibrate_k2': True,
                          'calibrate_k3': True}
        
        calibrate_per_transect = True

        estimate_time_varying = True
        time_node_spacing = 10 # s. If spacing 10 and transect lasts for 53 s will attempt to divide into largest integer leaving

        # Localize the prior calibration and use as initial parameters as well as constants for parameter xx if calibrate_xx = False
        cal_obj_prior = CalibHSI(file_name_cal_xml=config['Absolute Paths']['hsi_calib_path'])

        # Whether the parameter is toggled for calibration
        is_variab_parameter = np.zeros(11).astype(np.int64)

        param0 = np.array([cal_obj_prior.rx, 
                                cal_obj_prior.ry, 
                                cal_obj_prior.rz, 
                                cal_obj_prior.cx, 
                                cal_obj_prior.f, 
                                cal_obj_prior.k1, 
                                cal_obj_prior.k2, 
                                cal_obj_prior.k3,
                                cal_obj_prior.tx,
                                cal_obj_prior.ty,
                                cal_obj_prior.tz])

        if calibrate_dict ['calibrate_boresight']:
            is_variab_parameter[0:3] = 1
        
        if calibrate_dict ['calibrate_camera']:
            is_variab_parameter[3:8] = 1
            if not calibrate_dict['calibrate_cx']:
                is_variab_parameter[3] = 0
            if not calibrate_dict['calibrate_f']:
                is_variab_parameter[4] = 0
            if not calibrate_dict['calibrate_k1']:
                is_variab_parameter[5] = 0
            if not calibrate_dict['calibrate_k2']:
                is_variab_parameter[6] = 0
            if not calibrate_dict['calibrate_k3']:
                is_variab_parameter[7] = 0

        if calibrate_dict ['calibrate_lever_arm']:
            is_variab_parameter[8:11] = 1
                
        # TODO: User should be able to specify which parameters to calibrate with dictionary above
        # These are the adjustable parameters
        param0_variab = param0[is_variab_parameter==1]


        if calibrate_per_transect:
            n_transects = 1 + (df_gcp_filtered['file_count'].max() - df_gcp_filtered['file_count'].min())
            
            # Calculate the optimal camera parameters and 
            for i in range(int(n_transects)):
                df_current = df_gcp_filtered[df_gcp_filtered['file_count'] == i]


                if estimate_time_varying:
                    times_samples = df_current['unix_time']
                    transect_duration_sec = times_samples.max() - times_samples.min()
                    number_of_nodes = int(np.floor(transect_duration_sec/time_node_spacing)) + 1

                    # Calculate the number of nodes. It divides the transect into equal intervals, 
                    # meaning that the intervals can be somewhat different at least for a small number of them.
                    time_nodes = np.linspace(start=times_samples.min(), 
                                        stop = times_samples.max(), 
                                        num = number_of_nodes)
                    
                    # One parameter set for X, Y, Z and one for rotZ
                    param0_variab_tot = np.concatenate((param0_variab, np.zeros(4*number_of_nodes)), axis=0)

                    # Calculate the Jacobian (for exploiting sparsity)
                    kwargs = {'features_df': df_current,
                              'param0': param0,
                              'is_variab_parameter': is_variab_parameter,
                              'time_nodes': time_nodes}
                    J = numerical_jacobian(fun = objective_fun_reprojection_error, param = param0_variab_tot, **kwargs)

                    sparsity_perc = 100*((J==0).sum()/J.size)
                    print(f'Jacobian computed with {sparsity_perc:.0f} % zeros')

                    
                    from scipy.sparse import lil_matrix
                    sparsity = lil_matrix(J.shape, dtype=int)
                    sparsity[J != 0] = 1
                else:
                    time_nodes = None
                    param0_variab_tot = param0_variab

                n_features = df_current.shape[0]
                
                res_pre_optim = objective_fun_reprojection_error(param0_variab_tot, 
                                                      df_current, 
                                                      param0, 
                                                      is_variab_parameter,
                                                      time_nodes
                                                      )

                abs_err = np.sqrt(res_pre_optim[0:n_features]**2 + res_pre_optim[0:n_features]**2)
                median_error = np.median(abs_err)*cal_obj_prior.f
                print(f'Original MAE rp-error is {median_error:.1f} pixels')

                # Optimize the transect legs
                time_start  = time.time()
                res = least_squares(fun = objective_fun_reprojection_error, 
                                x0 = param0_variab_tot, 
                                args= (df_current, param0, is_variab_parameter, time_nodes, ), 
                                x_scale='jac',
                                jac_sparsity=sparsity)
                
                optim_dur = time.time() - time_start

                # Absolute reprojection errors
                abs_err = np.sqrt(res.fun[0:n_features]**2 + res.fun[0:n_features]**2)
                median_error_pix = np.median(abs_err)*cal_obj_prior.f

                print(f'Optimized MAE rp-error is {median_error_pix:.1f} pixels')
                print(f'Optimization time was {optim_dur:.0f} sec')
                print(f'Number of nodes was {number_of_nodes}')
                print(f'Number of features was {n_features}')
                print('')
                

        else:
            
            n_features = df_gcp_filtered.shape[0]

            res_pre_optim = objective_fun_reprojection_error(param0_variab, df_gcp_filtered, param0, is_variab_parameter)
            median_error_x = np.median(np.abs(res_pre_optim[0:n_features]))
            median_error_y = np.median(np.abs(res_pre_optim[n_features:2*n_features]))
            print(f'Original rp-error in x is {median_error_x} and in y is {median_error_y}')

            res = least_squares(fun = objective_fun_reprojection_error, 
                                x0 = param0_variab, 
                                args= (df_gcp_filtered, param0, is_variab_parameter, ), 
                                x_scale='jac',
                                method = 'lm')
            
            median_error_x = np.median(np.abs(res.fun[0:n_features]))*74
            median_error_y = np.median(np.abs(res.fun[n_features:2*n_features]))*74
            print(f'Optimized MAE rp-error in x is {median_error_x} and in y is {median_error_y}')

    
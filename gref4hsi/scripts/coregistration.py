import configparser
import os
import time
from glob import glob

import numpy as np
import spectral as sp

from gref4hsi.utils.gis_tools import GeoSpatialAbstractionHSI
from gref4hsi.utils.parsing_utils import Hyperspectral
import gref4hsi.utils.geometry_utils as geom_utils
from gref4hsi.utils.geometry_utils import CalibHSI, GeoPose

from scipy.spatial.transform import Rotation as RotLib
from scipy.optimize import least_squares
from scipy.interpolate import interp1d, RBFInterpolator
import pandas as pd
from scipy.sparse import lil_matrix
import pymap3d as pm
import matplotlib.pyplot as plt
from pykrige import OrdinaryKriging
from sklearn.utils import resample
from scipy.signal import medfilt





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


def assemble_jacobian_pattern(is_variab_param_intr, is_variab_param_extr, n_features):


    n_param_tot_static = len(is_variab_param_intr)
    n_variab_param_static = sum(is_variab_param_intr==1)
    n_adjustable_dofs = sum(is_variab_param_extr==1)
    n_variab_param_time_varying = n_adjustable_dofs*n_features # Work with flattened vectors

    n_cols = n_variab_param_static + n_variab_param_time_varying
    n_rows = 2*n_features + n_variab_param_time_varying # Latter is added to penalize errors


    sparsity = lil_matrix((n_rows, n_cols), dtype=int)
    
    # Iterate Static parameters (not time-varying)
    param_count = 0
    for i in range(n_param_tot_static):
        if bool(is_variab_param_intr[i]):
            # take from the parameter vector (assuming that they are ordered the same way)

            # All camera parameters affect the reprojection error in x
            sparsity[0:n_features, param_count] = 1

            # These params rx, ry, rz, f, tx, ty, tz affect the reprojection error in y
            if i in [0, 1, 2, 4, 8, 9, 10]:
                sparsity[n_features:2*n_features, param_count] = 1
            
            param_count += 1
    
    
    param_count_ext = 0
    for i in range(6):
        if bool(is_variab_param_extr[i]):
            # take from the parameter vector (assuming that they are ordered the same way)

            row_list_x = np.arange(n_features)
            row_list_y = n_features + np.arange(n_features)
            col_list = n_variab_param_static + param_count_ext*n_features + np.arange(n_features)
            


            row_list_err = 2*n_features + np.arange(n_variab_param_time_varying)
            col_list_err = n_variab_param_static + np.arange(n_variab_param_time_varying)


            # A time varying error only affects it's corresponding feature
            sparsity[row_list_x, col_list] = 1
            sparsity[row_list_y, col_list] = 1
            sparsity[row_list_err, col_list_err] = 1

            param_count_ext += 1


    return sparsity



def interpolate_time_nodes(time_from, value, time_to, method = 'linear'):
    """Interpolates the parameters (values) from the time nodes (time_from) to any array of interpolatable time_to array

    :param time_from: The time of the time_nodes (error parameters)
    :type time_from: ndarray (n,)
    :param value: The values of the position and orientation errors at the time_nodes
    :type value: ndarray (6, n)
    :param time_to: The queried time points for interpolation
    :type time_to: ndarray (m,)
    :param method: ['nearest', 'linear', 'slinear', 'quadratic', 'cubic'], defaults to 'linear'
    :type method: str, optional
    :return: _description_
    :rtype: interpolated values (6, n)
    """
    # One simple option is to use Scipy's sortiment of interpolation options. The sparsity pattern should be included somehow..
    if method in ['nearest', 'linear', 'slinear', 'quadratic', 'cubic']:



        vals = interp1d(time_from, value, kind=method, bounds_error = False, fill_value = np.nan)(time_to)

        # A Bit cluttery extrapolation with nearest on edges (where there are no features)

        if np.isnan(vals[0,-1]) or np.isnan(vals[0,0]):
            time_to_ex = time_to[np.isnan(vals[0,:])]

            vals_ex = interp1d(time_from, value, kind='nearest', fill_value = "extrapolate")(time_to_ex)

            vals[:, np.isnan(vals[0,:])] = vals_ex

        return vals
    elif method in ['gaussian']:
        
        

        # A resampling is done to avoid any 
        x, Y = resample(time_from, value.T, n_samples = 1000, random_state=0)

        vals_mu = np.zeros((time_to.size, 6))
        vals_std = np.zeros((time_to.size, 6))

        for i in range(6):

            y = Y[:, i].T

            if not np.all(y==0):

                uk = OrdinaryKriging(x, np.zeros(x.shape), y, variogram_model="gaussian")

                y_pred, y_sigmasq = uk.execute("grid", time_to, np.array([0.0]))

                # Medfilt used to remove unlikely outlier points
                vals_mu[:,i] = medfilt(np.squeeze(y_pred))

                # According to https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/generated/pykrige.ok.OrdinaryKriging.html
                # execute returns variances
                vals_std[:,i] = np.sqrt(medfilt(np.squeeze(y_sigmasq)))
            else:
                pass


        
        return vals_mu, vals_std

    elif method in ['none']:
        # Means that values need no interpolation
        if time_from.size == time_to.size:
            return value
        else:
            return interpolate_time_nodes(time_from, value, time_to, method = 'linear')


def gaussian_plot(unix_time_scans, time_nodes, param_pose_tot):


    sec_from_transect_start = unix_time_scans-unix_time_scans.min()

    vals_mu, vals_sigma = interpolate_time_nodes(time_nodes,
                                            param_pose_tot,
                                            time_to = unix_time_scans, 
                                            method='gaussian')
    
    x_pred = sec_from_transect_start

    x = time_nodes - unix_time_scans.min()

    Y = param_pose_tot.T

    y1 = Y[:, 0]
    y2 = Y[:, 1]
    y3 = Y[:, 2]
    y4 = Y[:, 5]

    y1_pred = vals_mu[:, 0]
    y2_pred = vals_mu[:, 1]
    y3_pred = vals_mu[:, 2]
    y4_pred = vals_mu[:, 5]
    y1_std = vals_sigma[:, 0]
    y2_std = vals_sigma[:, 1]
    y3_std = vals_sigma[:, 2]
    y4_std = vals_sigma[:, 5]



    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # Adjust figsize for desired plot size

    # Plot data on each subplot
    axes[0, 0].scatter(x, y1, label="Points")
    axes[0, 0].plot(x_pred, y1_pred, label="Predicted North error")
    axes[0, 0].fill_between(
        x_pred,
        y1_pred - y1_std,
        y1_pred + y1_std,
        alpha=0.3,
        label= "Standard Deviation North",
    )
    axes[0, 0].set_title("North error [m]")
    axes[0, 0].legend()

    axes[0, 1].scatter(x, y2, label="Points")
    axes[0, 1].plot(x_pred, y2_pred, label="Predicted East error")
    axes[0, 1].fill_between(
        x_pred,
        y2_pred - y2_std,
        y2_pred + y2_std,
        alpha=0.3,
        label= "Standard Deviation East",
    )
    axes[0, 1].set_title("East error [m]")
    axes[0, 1].legend()

    axes[1, 0].scatter(x, y3, label="Points")
    axes[1, 0].plot(x_pred, y3_pred, label="Predicted Down error")
    axes[1, 0].fill_between(
        x_pred,
        y3_pred - y3_std,
        y3_pred + y3_std,
        alpha=0.3,
        label= "Standard Deviation Down",
    )
    axes[1, 0].set_title("Down error [m]")
    axes[1, 0].legend()

    axes[1, 1].scatter(x, y4, label="Points")
    axes[1, 1].plot(x_pred, y4_pred, label="Predicted yaw")
    axes[1, 1].fill_between(
        x_pred,
        y4_pred - y4_std,
        y4_pred + y4_std,
        alpha=0.3,
        label= "Standard Deviation yaw",
    )
    axes[1, 1].set_title("Yaw error [deg]")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def compose_pose_errors(param_pose_tot, time_nodes, unix_time_features, rot_body_ned, rot_ned_ecef, pos_body, time_interpolation_method, pos_err_ref_frame, sigma_param, plot_error = False):
    """Takes a (6*n_node) vector of errors, interpolates and composes (adds) them to the pose from the navigation data"""
    n_features = len(unix_time_features)

    # Interpolate to the right time
    if time_interpolation_method != 'gaussian':
        err_interpolated = interpolate_time_nodes(time_nodes, 
                                                param_pose_tot,
                                                time_to = unix_time_features, 
                                                method=time_interpolation_method).transpose()
    else:
        err_mu, err_sigma = interpolate_time_nodes(time_nodes, 
                                                param_pose_tot,
                                                time_to = unix_time_features, 
                                                method=time_interpolation_method)
        
        # Statistics the engineering way. What are the non-diagonal elements of the covariance matrix anyway
        W_nav = np.matlib.repmat(1/sigma_param**2, err_mu.shape[0], 1)
        W_gp = 1/err_sigma**2

        err_interpolated = err_mu*W_gp / (W_gp + W_nav)


        

    # The position errors make up the three first
    param_err_pos = err_interpolated[:, 0:3]

    # The err has columns 3-5 roll, pitch, yaw, while from_euler('ZYX',...) requires yaw pitch roll order
    param_err_eul_ZYX = np.vstack((err_interpolated[:, 5].flatten(), 
                                        err_interpolated[:, 4].flatten(),
                                        err_interpolated[:, 3].flatten())).transpose()
    
    # To be left multiplied with attitude 
    rot_err_NED = RotLib.from_euler('ZYX', param_err_eul_ZYX, degrees=True)

    if plot_error == True:
        # Try plotting the error in position as func of time
        if pos_err_ref_frame == 'ecef':
            lat, lon, _ = pm.ecef2geodetic(pos_body[:, 0], pos_body[:, 1], pos_body[:, 2])

            err_ecef_x = param_err_pos[:,0]
            err_ecef_y = param_err_pos[:,1]
            err_ecef_z = param_err_pos[:,2]

            # Rotate error vector to NED for intuition
            e, n, u = pm.ecef2enuv(err_ecef_x, err_ecef_y, err_ecef_z, lat0 = lat.mean(), lon0=lon.mean())
        elif pos_err_ref_frame == 'ned':
            err_ned_x = param_err_pos[:,0]
            err_ned_y = param_err_pos[:,1]
            err_ned_z = param_err_pos[:,2]
            n = err_ned_x
            e = err_ned_y
            u = -err_ned_z
        
        # Plot the optimized curves
        if time_interpolation_method == 'gaussian':
            gaussian_plot(unix_time_features, time_nodes, param_pose_tot)
            
        
        
    
    



    # Modify the poses with the parametrically interpolated values:
    if pos_err_ref_frame == 'ecef':
        pos_body += param_err_pos
    elif pos_err_ref_frame == 'ned':
        # Add the ecef-transformed error
        pos_body += rot_ned_ecef.apply(param_err_pos)

    # Recall that we only estimate yaw errors
    rot_body_ned_corr = rot_err_NED * rot_body_ned

    # Convert orientation to ECEF
    rot_body_ecef = rot_ned_ecef * rot_body_ned_corr

    return pos_body, rot_body_ecef


def calculate_intrinsic_param(is_variab_param_intr, param, param0, as_calib_obj = False):
    """Based on which parameters were toggled for adjustments it fills the 11 element intrinsic camera parameter vector

    :param is_variab_param_intr: _description_
    :type is_variab_param_intr: bool
    :param param: _description_
    :type param: _type_
    :param param0: _description_
    :type param0: _type_
    :return: _description_
    :rtype: _type_
    """
    n_param_tot_static = len(is_variab_param_intr)
    param_vec_total = np.zeros(n_param_tot_static)
    
    # Static parameters (not time-varying)
    param_count = 0
    for i in range(n_param_tot_static):
        if bool(is_variab_param_intr[i]):
            # take from the parameter vector (assuming that they are ordered the same way)
            param_vec_total[i] = param[param_count]
            param_count += 1
            
        else:
            # Take fixed parameters from param0 - the initial parameter guess
            param_vec_total[i] = param0[i]
    

    if as_calib_obj:
        # If data is to be returned as dictionary
        calib_dict = {'rx': param_vec_total[0],
                      'ry': param_vec_total[1],
                      'rz': param_vec_total[2],
                      'cx': param_vec_total[3],
                      'f': param_vec_total[4],
                      'k1': param_vec_total[5],
                      'k2': param_vec_total[6],
                      'k3': param_vec_total[7],
                      'tx': param_vec_total[8],
                      'ty': param_vec_total[9],
                      'tz': param_vec_total[10],
                      'width': -1} # Must be set elsewhere
        
        return calib_dict
    else:
        # If data is to be returned as vector
        return param_vec_total


def calculate_pose_param(is_variab_param_extr, is_variab_param_intr, param, sigma=None):
    """Calculate a (6,n) pose vector from the parameter vector

    :param is_variab_param_extr: Boolean array saying which pose degrees of freedom are variable (to be adjusted)
    :type is_variab_param_extr: ndarray (6,) bool
    :param is_variab_param_intr: Boolean array saying which intrinsic/static parameters are variable (to be adjusted) 
    :type is_variab_param_intr: ndarray (11,) bool
    :param param: parameter vector
    :type param: ndarray (n_variab_param_static + 6*n,) 
    :return: The (6, n) pose vector where n is the number of time nodes
    :rtype: ndarray (6, n) 
    """

    n_variab_param_static = sum(is_variab_param_intr==1)

    # The number of pose degrees of freedom to estimate. Estimating posX, posY and yaw would give n_pose_dofs = 3
    n_pose_dofs = is_variab_param_extr.sum()
    param_time_var = param[n_variab_param_static:].reshape((n_pose_dofs, -1))

    # The errors default to zero
    param_pose_tot = np.zeros((6, param_time_var.shape[1]))

    var_dof_count = 0
    weighted_error_term = np.zeros(param_time_var.shape)

    # How about making the weighted error term into a lengthy vector
    # That is interpolated from param_time_var
    #time_vec_error_term = np.arange(time)
    for i in range(6): # Iterate 3 positions and 3 orientations
        # i=0 is posx, i=1 is posy, i=2 is posz, i=3 is roll, i=4 is pitch, and i=5 is yaw
        if is_variab_param_extr[i]:
            # If a degree of freedom is to be tuned, we insert the error parameters from the current vector
            param_pose_tot[i,:] = param_time_var[var_dof_count,:]

            # Penalize approximately
            if sigma is not None:
                if i < 3:
                    # Penalize position errors with 10 m (allows for some biases as well)
                    weighted_error_term[var_dof_count, :] = param_time_var[var_dof_count,:]/sigma[i] # Assuming uncorrelated errors
                if i > 3:
                    # TODO: Calculate NED equivalent errors? 
                    weighted_error_term[var_dof_count, :] = param_time_var[var_dof_count,:]/sigma[i] # Assuming uncorrelated errors


            var_dof_count += 1
    

    return param_pose_tot, weighted_error_term

def objective_fun_reprojection_error(param, features_df, param0, is_variab_param_intr, is_variab_param_extr, time_nodes, time_interpolation_method, pos_err_ref_frame, sigma_obs, sigma_param):
    """_summary_

    :param param: _description_
    :type param: _type_
    :param features_df: _description_
    :type features_df: _type_
    :param param0: _description_
    :type param0: _type_
    :param is_variab_param_intr: Array describing which of the 11 intrinsic parameters to be calibrated (boresight, lever arm, cam calib)
    :type is_variab_param_intr: ndarray (11,) bool
    :param is_variab_param_extr: Array describing which of the 6 intrinsic pose time series to be calibrated (posX, posY, posZ, roll, pitch, yaw)
    :type is_variab_param_extr: ndarray (11,) bool
    :param time_nodes: _description_
    :type time_nodes: _type_
    :param time_interpolation_method: _description_
    :type time_interpolation_method: _type_
    :return: _description_
    :rtype: _type_
    """
    
    #

    param_vec_intr = calculate_intrinsic_param(is_variab_param_intr, param, param0)

    # Boresight
    rot_x = param_vec_intr[0]
    rot_y = param_vec_intr[1]
    rot_z = param_vec_intr[2]

    # Principal point
    cx = param_vec_intr[3]

    # Focal length
    f = param_vec_intr[4]

    # Distortions
    k1 = param_vec_intr[5]
    k2 = param_vec_intr[6]
    k3 = param_vec_intr[7]

    # Lever arm
    trans_x = param_vec_intr[8] 
    trans_y = param_vec_intr[9]
    trans_z = param_vec_intr[10] 

    
    
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
        
        # Calculate the 6 dof pose error parameters (non-adjustable parameters rows are zero)
        param_pose_tot, weighted_error_term = calculate_pose_param(is_variab_param_extr, is_variab_param_intr, param, sigma = sigma_param)

        # The parametric errors represent a handful of nodes and must be interpolated to the feature times
        unix_time_features = features_df['unix_time']

        # We compose them with (add them to) the position/orientation estimates from the nav system
        pos_body, rot_body_ecef = compose_pose_errors(param_pose_tot, time_nodes, unix_time_features, rot_body_ned, rot_ned_ecef, pos_body, time_interpolation_method, pos_err_ref_frame, sigma_param)




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
    errorx = np.array(x_norm - x_norm_reproj)*f
    errory = np.array(y_norm - y_norm_reproj)*f


    errorx_norm = errorx / sigma_obs[0]
    errory_norm = errory / sigma_obs[1]


    # Least squares expects a 1D function evaluation vector
    err_vec = np.concatenate((errorx_norm.reshape(-1), errory_norm.reshape(-1), weighted_error_term.reshape(-1)))
    return err_vec

def filter_gcp_by_registration_error(u_err, v_err, method = 'iqr'):
    """Filters the matched point based on the registration error in pixels in x/east (u_err) and y/north (v_err). 
        Returns a binary mask of 

    :param u_err: error in pixels in x/east (u_err)
    :type u_err: ndarray (n,)
    :param v_err: error in pixels in y/north (v_err)
    :type v_err: ndarray (n,)
    :param method: The filtration method, defaults to 'iqr'
    :type method: str, optional
    :return: Binary mask that can be used for masking e.g. a dataframe or numpy array
    :rtype: ndarray (n,)
    """
    if method == 'iqr':
        # Errors in the x direction. We define the inlier interval by use of the interquartile rule
        Q1_u = np.percentile(u_err, 25)
        Q3_u = np.percentile(u_err, 75)
        IQR_u = Q3_u-Q1_u
        lower_bound_u = Q1_u - 1.5*IQR_u
        upper_bound_u = Q3_u + 1.5*IQR_u

        # Errors in the x direction. We define the inlier interval by use of the interquartile rule
        Q1_v = np.percentile(v_err, 25)
        Q3_v = np.percentile(v_err, 75)
        IQR_v = Q3_v-Q1_v
        lower_bound_v = Q1_v - 1.5*IQR_v
        upper_bound_v = Q3_v + 1.5*IQR_v

        # Simply mask by the interquartile rule for the north-east registration errors (unit is in pixels)
        feature_mask = np.all([u_err > lower_bound_u,
                                    u_err < upper_bound_u,
                                    v_err > lower_bound_v,
                                    v_err < upper_bound_v], axis=0)
    else:
        print('This method does not exist')
        NotImplementedError
    return feature_mask


def calculate_cam_and_pose_from_param(h5_filename, param, features_df, param0, is_variab_param_intr, is_variab_param_extr, time_nodes, time_interpolation_method, pos_err_ref_frame, h5_paths, plot_err_vec, sigma_obs, sigma_param):

    # Seems wize to return
    # Return in a dictionary allowing for simple dumping to XML file (exept width)
    if is_variab_param_intr.sum() > 0:
        calib_param_intr = calculate_intrinsic_param(is_variab_param_intr, param, param0, as_calib_obj = True)
    else:
        # Make identifiable that this has not been calibrated
        calib_param_intr = None


    # Read the original poses and time stamps from h5
        
    position_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                    dataset_name=h5_paths['h5_folder_position_ecef'])
    # Extract the ecef orientations for each frame
    quaternion_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                    dataset_name=h5_paths['h5_folder_quaternion_ecef'])
    # Extract the timestamps for each frame
    time_pose = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                    dataset_name=h5_paths['h5_folder_time_pose'])
    
    

    if time_nodes is not None:

        # Calculate the pose vector
        param_pose_tot, _ = calculate_pose_param(is_variab_param_extr, is_variab_param_intr, param)

        # Minor code block for decomposing orientation into BODY-NED and NED-ECEF rotations
        rot_body = RotLib.from_quat(quaternion_ecef)
        geo_pose = GeoPose(timestamps=time_pose, 
                            rot_obj=rot_body, 
                            rot_ref='ECEF', 
                            pos=position_ecef, 
                            pos_epsg=4978)
        rot_body_ned = geo_pose.rot_obj_ned
        rot_ned_ecef = geo_pose.rot_obj_ned_2_ecef

        # Compose the error parameter vector into the original pose estimates

        
        pos_body_corrected, rot_body_ecef_corrected = compose_pose_errors(param_pose_tot, 
                        time_nodes, 
                        time_pose, 
                        rot_body_ned, 
                        rot_ned_ecef, 
                        position_ecef, 
                        time_interpolation_method,
                        plot_error = plot_err_vec,
                        pos_err_ref_frame=pos_err_ref_frame,
                        sigma_param=sigma_param)
        
        quaternion_body_ecef_corrected = rot_body_ecef_corrected.as_quat()
    else:
        # If no corrections should be applied, set to none for identifiability
        pos_body_corrected = None
        rot_body_ecef_corrected = None
        
    
    return calib_param_intr, pos_body_corrected, quaternion_body_ecef_corrected


    
    


    
    


    # Read the correction data from param

# Function called to apply standard processing on a folder of files
def main(config_path, mode):
    config = configparser.ConfigParser()
    config.read(config_path)

    # Set the coordinate reference systems
    epsg_proj = int(config['Coordinate Reference Systems']['proj_epsg'])
    epsg_geocsc = int(config['Coordinate Reference Systems']['geocsc_epsg_export'])

    # Establish reference data
    path_orthomosaic_reference_folder = config['Absolute Paths']['orthomosaic_reference_folder']
    orthomosaic_reference_fn = [f for f in os.listdir(path_orthomosaic_reference_folder) if f.endswith('tif')][0] # Grab the only file in folder that ends with *.tif
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

    # Location for the calibrated Cal file:calib_file_coreg
    calib_file_coreg = config['Absolute Paths']['calib_file_coreg']

    

    # The necessary data from the H5 file for getting the positions and orientations.
        
    # Position is stored here in the H5 file
    h5_folder_position_ecef = config['HDF.processed_nav']['position_ecef']

    # Quaternion is stored here in the H5 file
    h5_folder_quaternion_ecef = config['HDF.processed_nav']['quaternion_ecef']

    # Modified positions after coregistration
    h5_folder_position_ecef_coreg = config['HDF.coregistration']['position_ecef']

    # Modified quaternions after coregistration
    h5_folder_quaternion_ecef_coreg = config['HDF.coregistration']['quaternion_ecef']

    # Timestamps for each hyperspectral frame
    h5_folder_time_pose = config['HDF.processed_nav']['timestamp']

    h5_paths = {'h5_folder_position_ecef': h5_folder_position_ecef,
                'h5_folder_quaternion_ecef': h5_folder_quaternion_ecef,
                'h5_folder_time_pose': h5_folder_time_pose, 
                'h5_folder_position_ecef_coreg': h5_folder_position_ecef_coreg,
                'h5_folder_quaternion_ecef_coreg': h5_folder_quaternion_ecef_coreg}

    

    print("\n################ Coregistering: ################")

    # Iterate the RGB composites
    #hsi_composite_files = sorted(os.listdir(path_composites_match))
    hsi_composite_paths = sorted(glob(os.path.join(path_composites_match, "*.tif")))
    hsi_composite_files = [os.path.basename(f) for f in hsi_composite_paths]
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
            n_cols_df = pixel_nr_vec.size
            gcp_dict = {'file_count': np.ones(n_cols_df)*file_count,
                        'h5_filename': np.repeat(h5_filename, n_cols_df),
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
        
        # Read Comparative data
        gcp_df_all = pd.read_csv(ref_gcp_path)

        # Registration error in pixels in x-direction (u_err) and y-direction (v_err)
        u_err = gcp_df_all['diff_u']
        v_err = gcp_df_all['diff_v']

        # Filter outliers according to the interquartile method using the registration errors
        feature_mask = filter_gcp_by_registration_error(u_err, v_err, method = 'iqr')
        
        # These features are used
        df_gcp_filtered = gcp_df_all[feature_mask]
        
        
        print("\n################ Calibrating camera parameters: ################")
        # Using the accumulated features, we can optimize for the boresight angles, camera parameters and lever arms
        
        ## Defining the options:
        
        # Here we define what that is to be calibrated
        calibrate_dict = {'calibrate_boresight': False,
                          'calibrate_camera': True,
                          'calibrate_lever_arm': False,
                          'calibrate_cx': False,
                          'calibrate_f': False,
                          'calibrate_k1': False,
                          'calibrate_k2': False,
                          'calibrate_k3': False}
        
        """calibrate_dict = {'calibrate_boresight': True,
                          'calibrate_camera': True,
                          'calibrate_lever_arm': True,
                          'calibrate_cx': True,
                          'calibrate_f': True,
                          'calibrate_k1': True,
                          'calibrate_k2': True,
                          'calibrate_k3': True}"""
        
        calibrate_per_transect = True
        estimate_time_varying = True
        time_node_spacing = 10 # s. If spacing 10 and transect lasts for 53 s will attempt to divide into largest integer leaving
        # TODO: fix gaussian interpolation
        time_interpolation_method = 'linear'

        node_partition = 'all_features' # ['temporal', 'feature', 'all_features']

        pos_err_ref_frame = 'ned' # ['ecef' or 'ned']

        # loss_function = 'soft_l1'

        # Select which time varying degrees of freedom to estimate errors for
        calibrate_dict_extr = {'calibrate_pos_x': True,
                          'calibrate_pos_y': True,
                          'calibrate_pos_z': True,
                          'calibrate_roll': False,
                          'calibrate_pitch': False,
                          'calibrate_yaw': True}
        
        """calibrate_dict_extr = {'calibrate_pos_x': True,
                          'calibrate_pos_y': True,
                          'calibrate_pos_z': True,
                          'calibrate_roll': True,
                          'calibrate_pitch': True,
                          'calibrate_yaw': True}"""


        # Whether to plot the error vectors as functions of time

        plot_err_vec_time = True
        plot_node_spacing = False


        


        # Localize the prior calibration and use as initial parameters as well as constants for parameter xx if calibrate_xx = False
        cal_obj_prior = CalibHSI(file_name_cal_xml=config['Absolute Paths']['hsi_calib_path'])

        # Whether the parameter is toggled for calibration
        # Both these 
        is_variab_param_intr = np.zeros(11).astype(np.int64)

        # Whether the DOF is toggled for calibration
        is_variab_param_extr = np.zeros(6).astype(np.int64)

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
            is_variab_param_intr[0:3] = 1
        
        if calibrate_dict ['calibrate_camera']:
            is_variab_param_intr[3:8] = 1
            if not calibrate_dict['calibrate_cx']:
                is_variab_param_intr[3] = 0
            if not calibrate_dict['calibrate_f']:
                is_variab_param_intr[4] = 0
            if not calibrate_dict['calibrate_k1']:
                is_variab_param_intr[5] = 0
            if not calibrate_dict['calibrate_k2']:
                is_variab_param_intr[6] = 0
            if not calibrate_dict['calibrate_k3']:
                is_variab_param_intr[7] = 0

        if calibrate_dict ['calibrate_lever_arm']:
            is_variab_param_intr[8:11] = 1
                
        

        if estimate_time_varying:
            # Define which dofs are to be adjusted in optimization
            if calibrate_dict_extr['calibrate_pos_x']:
                is_variab_param_extr[0] = 1

            if calibrate_dict_extr['calibrate_pos_y']:
                is_variab_param_extr[1] = 1
            
            if calibrate_dict_extr['calibrate_pos_z']:
                is_variab_param_extr[2] = 1
            
            if calibrate_dict_extr['calibrate_roll']:
                is_variab_param_extr[3] = 1
            
            if calibrate_dict_extr['calibrate_pitch']:
                is_variab_param_extr[4] = 1
            
            if calibrate_dict_extr['calibrate_yaw']:
                is_variab_param_extr[5] = 1
        
        # Based on which options were toggled, the number of adjustable dofs can be computed
        n_adjustable_dofs = is_variab_param_extr.sum()

        # These are the adjustable parameters (not time-varying)
        param0_variab = param0[is_variab_param_intr==1]

        # Set the keyword arguments for optimization
        kwargs = {'features_df': None,
                    'param0': param0,
                    'is_variab_param_intr': is_variab_param_intr,
                    'time_nodes': None,
                    'is_variab_param_extr': is_variab_param_extr,
                    'time_interpolation_method': time_interpolation_method,
                    'pos_err_ref_frame': pos_err_ref_frame,
                    'sigma_obs': np.array([1, 1]), # across track [pixels], along track [pixels]
                    'sigma_param': np.array([2, 2, 5, 0.1, 0.1, 2]), # north [m], east [m], down [m], roll [deg], pitch [deg], yaw [deg]
                    }

        # The sometimes natural think to do
        if calibrate_per_transect:
            # Number of transects is found from data frame
            n_transects = 1 + (df_gcp_filtered['file_count'].max() - df_gcp_filtered['file_count'].min())
            
            # Iterate through transects
            if plot_err_vec_time:
                iter = np.arange(n_transects)
            if plot_node_spacing:
                iter = [160, 80, 50, 30, 20, 15, 10, 7, 5, 3] #
                transect_nr = 4
            for i in iter:
                # Selected Transect
                if plot_node_spacing:

                    time_node_spacing = i
                else:
                    transect_nr = i
                
                
                df_current_unsorted = df_gcp_filtered[df_gcp_filtered['file_count'] == transect_nr]

                df_current = df_current_unsorted.sort_values(by='unix_time')
                
                # Update the feature info from specific transect
                kwargs['features_df'] = df_current

                n_features = df_current.shape[0]

                # Read out the file name corresponding to file index i
                h5_filename = df_current['h5_filename'].iloc[0]
                

                # If we are to estimate that which is time varying
                if estimate_time_varying:
                    ## The time range is defined by the transect time:
                    #times_samples = df_current['unix_time']

                    # Extract the timestamps for each frame
                    time_pose = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                                    dataset_name=h5_folder_time_pose)

                    # We can use the feature time:
                    time_arr_sorted_features = np.array(sorted(df_current['unix_time']))
                    
                    transect_duration_sec = time_arr_sorted_features.max() - time_arr_sorted_features.min()

                    # Number of nodes calculated from this (except when using "All features")
                    number_of_nodes = int(np.floor(transect_duration_sec/time_node_spacing)) + 1

                    # The time varying parameters are in total the number of dofs times number of nodes
                    param0_time_varying = np.zeros(n_adjustable_dofs*number_of_nodes)

                    # The time-varying parameters are stacked after the intrinsic parameters.
                    # This vector only holds parameters that will be adjusted
                    param0_variab_tot = np.concatenate((param0_variab, 
                                                        param0_time_varying), axis=0)
                    
                    
                    
                    if node_partition == 'temporal':
                        # It divides the transect into equal intervals time-wise, 
                        # meaning that the intervals can be somewhat different at least for a small number of them.
                        time_nodes = np.linspace(start=time_pose.min(), 
                                            stop = time_pose.max(), 
                                            num = number_of_nodes)
                        
                    elif node_partition == 'feature':
                        # We divide to have an equal number of features in each interval
                        idx_nodes = np.round(np.linspace(start=0, 
                                            stop = n_features-1, 
                                            num = number_of_nodes)).astype(np.int64)

                        # Select corresponding time stamps
                        time_nodes = time_arr_sorted_features[idx_nodes]

                        #At ends we asign the min/max of the pose to avoid extrapolation
                        time_nodes[0] = time_pose.min()
                        time_nodes[-1]  = time_pose.max()

                    elif node_partition == 'all_features':
                        n_f = n_features
                        idx_nodes = np.round(np.linspace(start=0, 
                                            stop = n_f-1, 
                                            num = n_f)).astype(np.int64)

                        # Select corresponding time stamps to all features
                        time_nodes = time_arr_sorted_features[idx_nodes]

                        #At ends we asign the min/max of the pose to avoid extrapolation
                        #time_nodes[0] = time_pose.min()
                        #time_nodes[-1]  = time_pose.max()

                        # No interpolation is needed
                        kwargs['time_interpolation_method'] = 'none'

                        # Redefine these
                        # Number of nodes calculated from this
                        number_of_nodes = time_nodes.size

                        # The time varying parameters are in total the number of dofs times number of nodes
                        param0_time_varying = np.zeros(n_adjustable_dofs*number_of_nodes)

                        # The time-varying parameters are stacked after the intrinsic parameters.
                        # This vector only holds parameters that will be adjusted
                        param0_variab_tot = np.concatenate((param0_variab, 
                                                            param0_time_varying), axis=0)
                        


                    # Update optimization kwarg
                    kwargs['time_nodes'] = time_nodes
                    
                    # Calculate the Jacobian for finding and exploiting sparsity
                    """J = numerical_jacobian(fun = objective_fun_reprojection_error, param = param0_variab_tot, **kwargs)

                    

                    sparsity_perc = 100*((J==0).sum()/J.size)

                    print(f'Jacobian computed with {sparsity_perc:.0f} % zeros')
                    
                    # Using list of list representation as recommended by Scipy's least squares
                    sparsity = lil_matrix(J.shape, dtype=int)

                    # Setting the non sparse elements (in theory they could be set for small values of J too)
                    sparsity[J != 0] = 1"""

                    sparsity = assemble_jacobian_pattern(is_variab_param_intr, is_variab_param_extr, n_features)
                else:
                    time_nodes = None
                    param0_variab_tot = param0_variab

                
                # Run once with initial parameters
                res_pre_optim = objective_fun_reprojection_error(param0_variab_tot, **kwargs)

                # Calculate the median absolute error in pixels
                SE = res_pre_optim[0:n_features]**2 + res_pre_optim[n_features:2*n_features]**2
                abs_err = np.sqrt(SE)
                
                # Histogram it?
                """plt.hist(abs_err[abs_err < 10], 50)
                plt.show()"""
                median_error_pix = np.median(abs_err)
                MAE_median = np.median(abs_err)

                rmse = np.sqrt(np.median(SE))
                print(f'Original MAE rp-error is {MAE_median:.1f} pixels')

                # Optimize the transect and record time duration
                time_start  = time.time()
                res = least_squares(fun = objective_fun_reprojection_error, 
                                x0 = param0_variab_tot, 
                                x_scale='jac',
                                jac_sparsity=sparsity,
                                kwargs=kwargs,
                                loss = 'linear')
                
                optim_dur = time.time() - time_start

                # Absolute reprojection errors
                SE = res.fun[0:n_features]**2 + res.fun[n_features:2*n_features]**2
                abs_err = np.sqrt(SE)
                median_error_pix = np.median(abs_err)

                """plt.hist(abs_err[abs_err < 10], 50)
                plt.show()"""

                MAE_median = np.median(abs_err)
                rmse = np.sqrt(np.median(SE))

                print(f'Optimized MAE rp-error is {MAE_median:.2f} pixels')
                print(f'Optimization time was {optim_dur:.0f} sec')
                print(f'Number of nodes was {number_of_nodes}')
                print(f'Number of features was {n_features}')
                print('')
                

                # TODO: the parameters should be injected into h5-file

                # Now, if all features were used to optimize, use gaussian interpolation (KRIGING actually)
                if node_partition == 'all_features':
                    kwargs['time_interpolation_method'] = 'gaussian'
                
                param_optimized = res.x
                camera_model_dict_updated, position_updated, quaternion_updated = calculate_cam_and_pose_from_param(h5_filename, param_optimized, **kwargs, h5_paths=h5_paths, plot_err_vec=plot_err_vec_time)



                # Now the data has been computed and can be written to h5:
                if position_updated is None:
                    pass
                else:
                    Hyperspectral.add_dataset(data=position_updated, 
                                            name = h5_folder_position_ecef_coreg,
                                            h5_filename=h5_filename)
                if position_updated is None:
                    pass
                else:
                    Hyperspectral.add_dataset(data = quaternion_updated, 
                                            name = h5_folder_quaternion_ecef_coreg,
                                            h5_filename = h5_filename)
                if camera_model_dict_updated is None:
                    pass
                else:
                    # Width is not a parameter and is inherited from the original file
                    camera_model_dict_updated['width'] = cal_obj_prior.w

                    CalibHSI(file_name_cal_xml= calib_file_coreg, 
                            mode = 'w',
                            param_dict = camera_model_dict_updated)
                

        else:
            
            n_features = df_gcp_filtered.shape[0]

            res_pre_optim = objective_fun_reprojection_error(param0_variab, df_gcp_filtered, param0, is_variab_param_intr)
            median_error_x = np.median(np.abs(res_pre_optim[0:n_features]))
            median_error_y = np.median(np.abs(res_pre_optim[n_features:2*n_features]))
            print(f'Original rp-error in x is {median_error_x} and in y is {median_error_y}')

            res = least_squares(fun = objective_fun_reprojection_error, 
                                x0 = param0_variab, 
                                args= (df_gcp_filtered, param0, is_variab_param_intr, ), 
                                x_scale='jac',
                                method = 'lm')
            
            median_error_x = np.median(np.abs(res.fun[0:n_features]))
            median_error_y = np.median(np.abs(res.fun[n_features:2*n_features]))
            print(f'Optimized MAE rp-error in x is {median_error_x} and in y is {median_error_y}')

    
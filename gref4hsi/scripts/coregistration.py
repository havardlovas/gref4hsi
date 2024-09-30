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
from sklearn.model_selection import train_test_split
from scipy.signal import medfilt

def _get_time_nodes(node_partition, df, h5_folder_time_scanlines, time_node_spacing):
    """Finds all the time nodes and scanline timestamps for each feature
    """

    # Extract the timestamps for each frame

    # Number of transects is found from data frame
    n_transects = 1 + (df['file_count'].max() - df['file_count'].min())
    
    # Iterate through transects
    # Do all
    iter = np.arange(n_transects)
    first_iter = True
    for i in iter:
        df_current_unsorted = df[df['file_count'] == i]

        # Sort values by chronology
        df_current = df_current_unsorted.sort_values(by='unix_time')

        n_features = df_current.shape[0]
        try:
            # Read out the file name corresponding to file index i
            h5_filename = df_current['h5_filename'].iloc[0]
        except IndexError:
            continue

        time_scanlines = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                                dataset_name= h5_folder_time_scanlines)
        # We can use the feature time:
        time_arr_sorted_features = np.array(sorted(df_current['unix_time']))
        
        transect_duration_sec = time_arr_sorted_features.max() - time_arr_sorted_features.min()

        transect_duration_sec = time_scanlines.max() - time_scanlines.min()

        # Number of nodes calculated from this (except when using "All features")
        number_of_nodes = int(np.floor(transect_duration_sec/time_node_spacing)) + 1

        if node_partition == 'temporal':
                            # It divides the transect into equal intervals time-wise, 
                            # meaning that the intervals can be somewhat different at least for a small number of them.
                            time_nodes = np.linspace(start=time_scanlines.min(), 
                                                stop = time_scanlines.max(), 
                                                num = number_of_nodes)
                            
        elif node_partition == 'feature':
                            # We divide to have an equal number of features in each interval
                            idx_nodes = np.round(np.linspace(start=0, 
                                                stop = n_features-1, 
                                                num = number_of_nodes)).astype(np.int64)

                            # Select corresponding time stamps
                            time_nodes = time_arr_sorted_features[idx_nodes]

                            # At ends we asign the min/max of the pose to avoid extrapolation
                            time_nodes[0] = time_scanlines.min()
                            time_nodes[-1]  = time_scanlines.max()
        
        if first_iter:
            first_iter = False
            time_nodes_tot = time_nodes
            time_scanlines_tot = time_scanlines
        else:
            time_nodes_tot = np.concatenate((time_nodes_tot, time_nodes))
            time_scanlines_tot = np.concatenate((time_scanlines_tot, time_scanlines))
        
    return time_nodes_tot, time_scanlines_tot



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
        x, Y = resample(time_from, value.T, n_samples = 100, random_state=0)

        vals_mu = np.zeros((time_to.size, 6))
        vals_std = np.zeros((time_to.size, 6))

        for i in range(6):

            y = Y[:, i].T

            if not np.all(y==0):
                y_all = value[i,:]

                """n = np.var(y_all[1:-1] - y_all[0:-2])
                r = 80
                s = 2*n"""

                uk = OrdinaryKriging(x, np.zeros(x.shape), y,
                                     exact_values=False)
                
                y_pred, y_sigmasq = uk.execute("grid", time_to, np.array([0.0]))
                if i == 1:
                    print(uk.variogram_model_parameters)


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


def plot_estimated_errors(unix_time_scans, time_nodes=None, param_pose_tot=None, vals_mu=None, vals_sigma=None):


    sec_from_transect_start = unix_time_scans-unix_time_scans.min()
    x_pred = sec_from_transect_start
    y1_pred = vals_mu[:, 0]
    y2_pred = vals_mu[:, 1]
    y3_pred = vals_mu[:, 2]
    y4_pred = vals_mu[:, 5]

    """vals_mu, vals_sigma = interpolate_time_nodes(time_nodes,
                                            param_pose_tot,
                                            time_to = unix_time_scans, 
                                            method='gaussian')"""
    if vals_sigma is not None:
        if vals_sigma.shape[0] == 6:
            vals_sigma = vals_sigma.T
    if vals_mu.shape[0] == 6:  
        vals_mu = vals_mu.T
    
    

    

    if param_pose_tot is not None:

        x = time_nodes - unix_time_scans.min()


        Y = param_pose_tot.T

        y1 = Y[:, 0]
        y2 = Y[:, 1]
        y3 = Y[:, 2]
        y4 = Y[:, 5]

        

    if vals_sigma is not None:
        y1_std = vals_sigma[:, 0]
        y2_std = vals_sigma[:, 1]
        y3_std = vals_sigma[:, 2]
        y4_std = vals_sigma[:, 5]



    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # Adjust figsize for desired plot size

    # Plot data on each subplot
    

    axes[0, 0].plot(x_pred, y1_pred, label="Predicted North error")

    if param_pose_tot is not None:
        axes[0, 0].scatter(x, y1, label="Points")

    if vals_sigma is not None:
        axes[0, 0].fill_between(
            x_pred,
            y1_pred - y1_std,
            y1_pred + y1_std,
            alpha=0.3,
            label= "Standard Deviation North",
        )
    axes[0, 0].set_title("North error [m]")
    axes[0, 0].set_ylim([-2, 2])
    axes[0, 0].legend()

    axes[0, 1].plot(x_pred, y2_pred, label="Predicted East error")
    

    if param_pose_tot is not None:
        axes[0, 1].scatter(x, y2, label="Points")

    if vals_sigma is not None:
        axes[0, 1].fill_between(
            x_pred,
            y2_pred - y2_std,
            y2_pred + y2_std,
            alpha=0.3,
            label= "Standard Deviation East",
        )
    axes[0, 1].set_ylim([-2, 2])
    axes[0, 1].set_title("East error [m]")
    axes[0, 1].legend()

    axes[1, 0].plot(x_pred, y3_pred, label="Predicted Down error")

    if param_pose_tot is not None:
        axes[1, 0].scatter(x, y3, label="Points")
    
    if vals_sigma is not None:
        axes[1, 0].fill_between(
            x_pred,
            y3_pred - y3_std,
            y3_pred + y3_std,
            alpha=0.3,
            label= "Standard Deviation Down",
        )
    axes[1, 0].set_title("Down error [m]")
    axes[1, 0].set_ylim([-5, 5])
    axes[1, 0].legend()


    

    axes[1, 1].plot(x_pred, y4_pred, label="Predicted yaw")

    if param_pose_tot is not None:
        axes[1, 1].scatter(x, y4, label="Points")

    if vals_sigma is not None:
        axes[1, 1].fill_between(
            x_pred,
            y4_pred - y4_std,
            y4_pred + y4_std,
            alpha=0.3,
            label= "Standard Deviation yaw",
        )
    axes[1, 1].set_title("Yaw error [deg]")
    axes[1, 1].set_ylim([-1, 1])
    axes[1, 1].legend()

    
    plt.tight_layout()
    
    #plt.show()




def compose_pose_errors(param_pose_tot, time_nodes, unix_time_features, rot_body_ned, rot_ned_ecef, pos_body, time_interpolation_method, pos_err_ref_frame, sigma_param, sigma_nodes = None, plot_error = False):
    """Takes a (6*n_node) vector of errors, interpolates and composes (adds) them to the pose from the navigation data"""
    n_features = len(unix_time_features)

    # Interpolate to the right time
    if time_interpolation_method != 'gaussian':
        
        # Interpolate the errors
        err_interpolated = interpolate_time_nodes(time_nodes, 
                                                param_pose_tot,
                                                time_to = unix_time_features, 
                                                method=time_interpolation_method).transpose()
        
        # If available interpolate standard deviations
        if sigma_nodes is not None:
            
            sigma = interpolate_time_nodes(time_nodes, 
                                                sigma_nodes,
                                                time_to = unix_time_features, 
                                                method=time_interpolation_method).transpose()
            
            # Asign a full dof vector of standard deviations
            sigma_all = np.zeros((sigma.shape[0], 6))
            dof_count = 0
            for i in range(6):
                if np.all(param_pose_tot[i,:] == 0):
                    pass
                else:
                    sigma_all[:, i] = sigma[:, dof_count]
                    dof_count += 1
            


    else:
        err_mu, err_sigma = interpolate_time_nodes(time_nodes, 
                                                param_pose_tot,
                                                time_to = unix_time_features, 
                                                method=time_interpolation_method)
        err_sigma = err_sigma.T
        err_mu = err_mu.T
        
        # Statistics the engineering way. What are the non-diagonal elements of the covariance matrix anyway

        # The weights for inverse variance weighted fusion:
        # https://en.wikipedia.org/wiki/Inverse-variance_weighting  

        # The statistically fused estimate
        err_interpolated = np.zeros(err_mu.shape)
        std_interpolated = np.zeros(err_mu.shape)
        for i in range(6):
            if np.all(param_pose_tot[i,:] == 0):
                pass
            else:
                W_nav = (np.matlib.repmat(1/sigma_param[i]**2, err_mu.shape[1], 1)).squeeze()*0
                W_gp = (1/err_sigma[i,:]**2).squeeze()
                y_i = err_mu[i,:].squeeze()

                # In theory it is a good fusion but there can be some unfortunate end effects
                err_interpolated[i,:] = y_i*W_gp / (W_gp + W_nav)
                std_interpolated[i,:] =  np.sqrt(1 / (W_gp + W_nav))
            
        err_interpolated = err_mu.T


        

    # The position errors make up the three first columns
    param_err_pos = err_interpolated[:, 0:3]

    # The orientation errors has columns 3-5 roll, pitch, yaw, while from_euler('ZYX',...) requires yaw pitch roll order
    param_err_eul_ZYX = np.vstack((err_interpolated[:, 5].flatten(), 
                                        err_interpolated[:, 4].flatten(),
                                        err_interpolated[:, 3].flatten())).transpose()
    
    
    # To be left multiplied with attitude 
    rot_err_NED = RotLib.from_euler('ZYX', param_err_eul_ZYX, degrees=True)

    if plot_error == True:
        # Try plotting the error in position as func of time
        
        # Plot the optimized curves
        if time_interpolation_method == 'gaussian':
            plot_estimated_errors(unix_time_features, time_nodes, param_pose_tot, vals_mu = err_interpolated, vals_sigma=err_sigma)
        else:
            if sigma_nodes is None:
                plot_estimated_errors(unix_time_features, vals_mu = err_interpolated)
            else:
                plot_estimated_errors(unix_time_features, vals_mu = err_interpolated, vals_sigma=sigma_all)
            


    # Modify the poses with the parametrically interpolated values:
    if pos_err_ref_frame == 'ecef':
        pos_body_corr = pos_body + param_err_pos
    elif pos_err_ref_frame == 'ned':
        # Add the ecef-transformed error. Is this right
        pos_body_corr = pos_body + rot_ned_ecef.apply(param_err_pos)

    # Corrected NED attitude
    rot_body_ned_corr = rot_err_NED * rot_body_ned

    # Convert orientation to ECEF
    rot_body_ecef_corr = rot_ned_ecef * rot_body_ned_corr

    return pos_body_corr, rot_body_ecef_corr


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


def calculate_pose_param(is_variab_param_extr, is_variab_param_intr, param, sigma=None, time_nodes = None, time_scanlines=None, time_interpolation_method=None):
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
    if sigma is not None:

        weighted_error_term_not_scaled_all = interpolate_time_nodes(time_nodes, 
                                                param_time_var,
                                                time_to = time_scanlines, 
                                                method = time_interpolation_method)

        weighted_error_term = np.zeros(weighted_error_term_not_scaled_all.shape)

        

        
    else:
        weighted_error_term = None
    # How about making the weighted error term into a lengthy vector
    # That is interpolated from param_time_var
    #time_vec_error_term = np.arange(time)
    for i in range(6): # Iterate 3 positions and 3 orientations
        # i=0 is posx, i=1 is posy, i=2 is posz, i=3 is roll, i=4 is pitch, and i=5 is yaw
        if is_variab_param_extr[i]:
            # If a degree of freedom is to be tuned, we insert the error parameters from the current vector
            param_pose_tot[i,:] = param_time_var[var_dof_count,:]

            # Penalize by calculating error per scanline
            if sigma is not None:
                if i < 3:
                    # Penalize position errors with 10 m (allows for some biases as well)
                    weighted_error_term[var_dof_count, :] = weighted_error_term_not_scaled_all[var_dof_count,:]/sigma[i] # Scale the errors
                if i > 3:
                    # TODO: Calculate NED equivalent errors? 
                    weighted_error_term[var_dof_count, :] = weighted_error_term_not_scaled_all[var_dof_count,:]/sigma[i] # Assuming uncorrelated errors


            var_dof_count += 1
    

    return param_pose_tot, weighted_error_term

def objective_fun_reprojection_error(param, features_df, param0, is_variab_param_intr, is_variab_param_extr, time_nodes, time_interpolation_method, pos_err_ref_frame, sigma_obs, sigma_param, time_scanlines):
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
    quat_body_ned = np.vstack((features_df['quat_body_to_ned_x'],
                          features_df['quat_body_to_ned_y'], 
                          features_df['quat_body_to_ned_z'],
                          features_df['quat_body_to_ned_w'])).transpose()
    
    # The rotation from NED to ECEF
    quat_ned_ecef = np.vstack((features_df['quat_ned_to_ecef_x'],
                          features_df['quat_ned_to_ecef_y'], 
                          features_df['quat_ned_to_ecef_z'],
                          features_df['quat_ned_to_ecef_w'])).transpose()

    # Convert to rotation object for convenience
    rot_body_ned = RotLib.from_quat(quat_body_ned)
    rot_ned_ecef = RotLib.from_quat(quat_ned_ecef)

    # Whether to estimate time-varying errors
    if np.all(is_variab_param_extr==0): # No variable extrinsic parameters
        # Assumes that navigation-based poses are correct
        rot_body_ecef_corr = rot_ned_ecef * rot_body_ned
        pos_body_corr = pos_body

    else:
        
        # Calculate the 6 dof pose error parameters (non-adjustable parameters rows are zero)
        param_pose_tot, weighted_error_term = calculate_pose_param(is_variab_param_extr, 
                                                                   is_variab_param_intr, 
                                                                   param, 
                                                                   sigma_param, 
                                                                   time_nodes, 
                                                                   time_scanlines, 
                                                                   time_interpolation_method)
        
        

        # The parametric errors represent a handful of nodes and must be interpolated to the feature times
        unix_time_features = features_df['unix_time']

        # We compose them with (add them to) the position/orientation estimates from the nav system
        pos_body_corr, rot_body_ecef_corr = compose_pose_errors(param_pose_tot, time_nodes, unix_time_features, rot_body_ned, rot_ned_ecef, pos_body, time_interpolation_method, pos_err_ref_frame, sigma_param)

        # Least squares expects a 1D function evaluation vector
        m = time_scanlines.size # Number of scanlines
        n = unix_time_features.size # Number of features
        k = is_variab_param_extr.sum() # Number of adjusted DOFs
        rho = (2*n/(k*m)) # Relationship between number of rp observations and penalty terms




    # The reference points in ECEF (obtained from the reference orthomosaic)
    points_world_reference = np.vstack((features_df['reference_points_x'], 
                          features_df['reference_points_y'], 
                          features_df['reference_points_z'])).transpose()
    
    # We reproject the reference points to the normalized HSI image plane
    X_norm = geom_utils.reproject_world_points_to_hsi_plane(trans_hsi_body, 
                                             rot_hsi_body, 
                                             pos_body_corr, 
                                             rot_body_ecef_corr, 
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

    # The node errors should be weighted heavily

    err_vec = np.concatenate((errorx_norm.reshape(-1), 
                              errory_norm.reshape(-1)))

    # Add penalty term if it exists
    if 'weighted_error_term' in locals():
        err_vec = np.concatenate((err_vec, 
                              np.sqrt(rho)*weighted_error_term.reshape(-1)))
    return err_vec

def filter_gcp_by_registration_error(u_err, v_err, method = 'iqr', hard_threshold_pix = None):
    """
    Filters the matched point based on the registration error in pixels in x/east (u_err) and y/north (v_err). 
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

        if hard_threshold_pix is not None:
            # Put a hard limit on filtration if IQR gives unreasonable filtering
            lower_bound_u = np.max((lower_bound_u, -hard_threshold_pix))
            lower_bound_v = np.max((lower_bound_v, -hard_threshold_pix))

            upper_bound_u = np.min((upper_bound_u, hard_threshold_pix))
            upper_bound_v = np.min((upper_bound_v, hard_threshold_pix))

        # Simply mask by the interquartile rule for the north-east registration errors (unit is in pixels)
        feature_mask = np.all([u_err > lower_bound_u,
                                    u_err < upper_bound_u,
                                    v_err > lower_bound_v,
                                    v_err < upper_bound_v], axis=0)
    else:
        print('This method does not exist')
        NotImplementedError
    return feature_mask


def calculate_cam_and_pose_from_param(h5_filename, param, features_df, param0, is_variab_param_intr, is_variab_param_extr, time_nodes, time_interpolation_method, pos_err_ref_frame, time_scanlines, h5_paths, plot_err_vec, sigma_obs, sigma_param, sigma_nodes):

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
    time_scanlines = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                    dataset_name=h5_paths['h5_folder_time_scanlines'])
    
    

    if time_nodes is not None:

        # Calculate the pose vector
        param_pose_tot, _ = calculate_pose_param(is_variab_param_extr, is_variab_param_intr, param)

        # Minor code block for decomposing orientation into BODY-NED and NED-ECEF rotations
        rot_body = RotLib.from_quat(quaternion_ecef)

        geo_pose = GeoPose(timestamps=time_scanlines, 
                            rot_obj=rot_body, 
                            rot_ref='ECEF', 
                            pos=position_ecef, 
                            pos_epsg=4978)
        
        rot_body_ned = geo_pose.rot_obj_ned
        rot_ned_ecef = geo_pose.rot_obj_ned_2_ecef

        # Compose the error parameter vector into the original pose estimates
        pos_body_corrected, rot_body_ecef_corrected = compose_pose_errors(param_pose_tot=param_pose_tot, 
                        time_nodes=time_nodes, 
                        unix_time_features=time_scanlines, 
                        rot_body_ned=rot_body_ned, 
                        rot_ned_ecef=rot_ned_ecef, 
                        pos_body=position_ecef, 
                        time_interpolation_method=time_interpolation_method,
                        plot_error = plot_err_vec,
                        pos_err_ref_frame=pos_err_ref_frame,
                        sigma_param=sigma_param,
                        sigma_nodes = sigma_nodes)
        
        quaternion_body_ecef_corrected = rot_body_ecef_corrected.as_quat()
    else:
        # If no corrections should be applied, set to none for identifiability
        pos_body_corrected = None
        rot_body_ecef_corrected = None
        
    
    return calib_param_intr, pos_body_corrected, quaternion_body_ecef_corrected


    
    


    
    


    # Read the correction data from param

# Function called to apply standard processing on a folder of files
def main(config_path, mode, is_calibrated, coreg_dict = {}):
    config = configparser.ConfigParser()
    config.read(config_path)

    # Set the coordinate reference systems
    epsg_proj = int(config['Coordinate Reference Systems']['proj_epsg'])
    epsg_geocsc = int(config['Coordinate Reference Systems']['geocsc_epsg_export'])

    resolution = float(config['Orthorectification']['resolutionhyperspectralmosaic'])

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
    h5_folder_time_scanlines = config['HDF.processed_nav']['timestamp']

    h5_paths = {'h5_folder_position_ecef': h5_folder_position_ecef,
                'h5_folder_quaternion_ecef': h5_folder_quaternion_ecef,
                'h5_folder_time_scanlines': h5_folder_time_scanlines, 
                'h5_folder_position_ecef_coreg': h5_folder_position_ecef_coreg,
                'h5_folder_quaternion_ecef_coreg': h5_folder_quaternion_ecef_coreg}

    

    print("\n################ Coregistering: ################")

    # Iterate the RGB composites
    #hsi_composite_files = sorted(os.listdir(path_composites_match))
    hsi_composite_paths = sorted(glob(os.path.join(path_composites_match, "*.tif")))
    hsi_composite_files = [os.path.basename(f) for f in hsi_composite_paths]
    count_valid = 0

    if mode == 'compare':
        print("\n################ Comparing to reference: ################")
        for file_count, hsi_composite_file in enumerate(hsi_composite_files):
            
            try:
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

                if is_calibrated:
                    # Extract data from coreg folder
                    position_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                                    dataset_name=h5_folder_position_ecef_coreg)
                    
                    quaternion_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                                    dataset_name=h5_folder_quaternion_ecef_coreg)

                else:
                    # Extract the ecef positions for each frame
                    position_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                                    dataset_name=h5_folder_position_ecef)
                    # Extract the ecef orientations for each frame
                    quaternion_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                                    dataset_name=h5_folder_quaternion_ecef)

                
                # Extract the timestamps for each frame
                time_scanlines = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                                dataset_name=h5_folder_time_scanlines)
                
            

                pixel_nr_vec, unix_time_vec, position_vec, quaternion_vec, feature_mask = GeoSpatialAbstractionHSI.compute_position_orientation_features(uv_vec_hsi, 
                                                                            pixel_nr_grid, 
                                                                            unix_time_grid, 
                                                                            position_ecef, 
                                                                            quaternion_ecef, 
                                                                            time_scanlines,
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
                            'quat_body_to_ned_x': quat_body_to_ned[:,0],
                            'quat_body_to_ned_y': quat_body_to_ned[:,1],
                            'quat_body_to_ned_z': quat_body_to_ned[:,2],
                            'quat_body_to_ned_w': quat_body_to_ned[:,3],
                            'quat_ned_to_ecef_x': quat_ned_to_ecef[:,0],
                            'quat_ned_to_ecef_y': quat_ned_to_ecef[:,1],
                            'quat_ned_to_ecef_z': quat_ned_to_ecef[:,2],
                            'quat_ned_to_ecef_w': quat_ned_to_ecef[:,3],
                            'reference_points_x': ref_points_vec[:,0],
                            'reference_points_y': ref_points_vec[:,1],
                            'reference_points_z': ref_points_vec[:,2],
                            'diff_absolute_error': diff_AE_valid,
                            'diff_u': diff_uv[:, 0],
                            'diff_v': diff_uv[:, 1]}
                


                # Convert to a dataframe
                gcp_df = pd.DataFrame(gcp_dict)

                # Concatanate frames
                if count_valid==0:
                    gcp_df_all = gcp_df
                    count_valid += 1
                else:
                    gcp_df_all = pd.concat([gcp_df_all, gcp_df])
            except:
                pass

        # Write points to a separate gcp_coreg.csv
        if not is_calibrated:
            gcp_df_all.to_csv(path_or_buf=ref_gcp_path)
        else:
            gcp_df_all.to_csv(path_or_buf=ref_gcp_path.split('.')[0] + '_coreg.csv')



    elif mode == 'calibrate':
        """Here we make adjustments to the extrinsic/intrinsic orientation parameters to minimize reprojection error"""
        # Read Comparative data
        
        # Separate files for calibrated and uncalibrated data
        if not is_calibrated:
            gcp_df_all = pd.read_csv(ref_gcp_path)
        else:
            gcp_df_all = pd.read_csv(ref_gcp_path.split('.')[0] + '_coreg.csv')

        # Registration error in pixels in x-direction (u_err) and y-direction (v_err)
        u_err = gcp_df_all['diff_u']
        v_err = gcp_df_all['diff_v']

        ## Defining the options:
        
        # Default options for camera calibration
        calibrate_dict_default = {'calibrate_boresight': False,
                          'calibrate_camera': False,
                          'calibrate_lever_arm': False,
                          'calibrate_cx': False,
                          'calibrate_f': False,
                          'calibrate_k1': False,
                          'calibrate_k2': False,
                          'calibrate_k3': False
                          }
        # Default options for estimating time varying errors
        calibrate_dict_extr_default = {'calibrate_pos_x': False,
                          'calibrate_pos_y': False,
                          'calibrate_pos_z': False,
                          'calibrate_roll': False,
                          'calibrate_pitch': False,
                          'calibrate_yaw': False}


        large_number = 1e12
        hard_threshold_m = coreg_dict.get('hard_threshold_m', large_number) # Hard coded threshold specifying the maximal registration error in meters for filtering. Only important when large portion outliers
        sigma_obs = coreg_dict.get('sigma_obs', np.array([1, 1])) # In pixels across and along track
        sigma_param = coreg_dict.get('sigma_param', np.array([2, 2, 5, 0.1, 0.1, 1])) # north [m], east [m], down [m], roll [deg], pitch [deg], yaw [deg]
        calibrate_dict = coreg_dict.get('calibrate_dict', calibrate_dict_default)
        calibrate_dict_extr = coreg_dict.get('calibrate_dict_extr', calibrate_dict_extr_default)
        time_node_spacing = coreg_dict.get('time_node_spacing', large_number) # s. A large number means one single node in time i.e. constant error in the DOF you are estimating
        time_interpolation_method = coreg_dict.get('time_interpolation_method', 'linear') # Interpolation method. Linear is currently recommended
        node_partition = coreg_dict.get('node_partition', 'temporal') # ['temporal', 'feature', 'all_features']. The partitioning scheme. Temporal makes equitemporal nodes, while "feature" makes nodes with equal feature count in each time segment
        pos_err_ref_frame = coreg_dict.get('pos_err_ref_frame', 'ned') # ['ecef' or 'ned']
        loss_function = coreg_dict.get('loss_function', 'soft_l1') # ['linear', 'huber', 'cauchy'..]. Least squares loss function for scipy least_squares implementation.
        calibrate_per_transect = coreg_dict.get('calibrate_per_transect', True) # TODO: Make this functional for Option False. An iterative concatenation would do the trick

        # Equivalent threshold in georaster pixel
        hard_threshold_pix = hard_threshold_m/resolution


        # Filter outliers according to the interquartile method using the registration errors
        feature_mask = filter_gcp_by_registration_error(u_err, v_err, 
                                                        method = 'iqr', 
                                                        hard_threshold_pix = hard_threshold_pix)

        # These features are used
        df_gcp_filtered = gcp_df_all[feature_mask]
        
       

        print("\n################ Calibrating camera parameters: ################")
        # Using the accumulated features, we can optimize for the boresight angles, camera parameters and lever arms
        
        
        
        
        
        estimate_time_varying = True

        

        

        

        val_mode = False # Whether to subset 50 percent of data to validation (i.e. not for fitting)

        # Whether to plot the error vectors as functions of time
        # TODO: remove these
        plot_err_vec_time = False
        plot_node_spacing = False

        # Localize the prior calibration and use as initial parameters as well as constants for parameter xx if calibrate_xx = False
        hsi_calib_path_original = config['Absolute Paths']['hsi_calib_path']
        if not is_calibrated:
            cal_obj_prior = CalibHSI(file_name_cal_xml=hsi_calib_path_original)
        else:
            try:
                calib_path = config['Absolute Paths']['calib_file_coreg']

                # Use coregistration adjusted version
                if os.path.exists(calib_path):
                    cal_obj_prior = CalibHSI(file_name_cal_xml=calib_path)
                    print('Using the coregistration-adjusted camera calibration')
                else:
                    cal_obj_prior = CalibHSI(file_name_cal_xml=hsi_calib_path_original)
            except:
                # Fall back to original calib file
                cal_obj_prior = CalibHSI(file_name_cal_xml=hsi_calib_path_original)


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
                    'sigma_obs': sigma_obs,
                    'sigma_param': sigma_param, # north [m], east [m], down [m], roll [deg], pitch [deg], yaw [deg]
                    }

        # What we tend to do here
        if calibrate_per_transect:
            # Number of transects is found from data frame
            n_transects = 1 + (df_gcp_filtered['file_count'].max() - df_gcp_filtered['file_count'].min())
            
            # Iterate through transects
            # Do all
            iter = np.arange(n_transects)
                
            for i in iter:
                # Selected Transect
                if plot_node_spacing:

                    time_node_spacing = i
                else:
                    transect_nr = i
                
                
                df_current_unsorted = df_gcp_filtered[df_gcp_filtered['file_count'] == transect_nr]

                # Sort values by chronology
                df_current = df_current_unsorted.sort_values(by='unix_time')
                
                # Update the feature info for optimization from specific transect
                kwargs['features_df'] = df_current

                n_features = df_current.shape[0]

                # Read out the file name corresponding to file index i
                try:
                    h5_filename = df_current['h5_filename'].iloc[0]
                except IndexError:
                    print('No Matches for image, moving on')
                    continue

                # If we are to estimate that which is time varying
                if estimate_time_varying:
                    ## The time range is defined by the transect time:
                    #times_samples = df_current['unix_time']

                    # Extract the timestamps for each frame
                    time_scanlines = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                                    dataset_name=h5_folder_time_scanlines)
                    kwargs['time_scanlines'] = time_scanlines

                    # We can use the feature time:
                    time_arr_sorted_features = np.array(sorted(df_current['unix_time']))
                    
                    transect_duration_sec = time_arr_sorted_features.max() - time_arr_sorted_features.min()

                    transect_duration_sec = time_scanlines.max() - time_scanlines.min()

                    # Number of nodes calculated from this (except when using "All features")
                    number_of_nodes = int(np.floor(transect_duration_sec/time_node_spacing)) + 1

                    # The time varying parameters are in total the number of dofs times number of nodes
                    param0_time_varying = np.zeros(n_adjustable_dofs*number_of_nodes)

                    # The time-varying parameters are stacked after the intrinsic parameters.
                    # This vector only holds parameters that will be adjusted
                    param0_variab_tot = np.concatenate((param0_variab, 
                                                        param0_time_varying), axis=0)
                    
                    # Should only use the training set:
                    

                    # Where we train:
                    # Training data
                    if val_mode:
                        idx = np.arange(df_current.shape[0])
                        idx_train, idx_val = train_test_split(idx, test_size=0.50, random_state=42)
                        kwargs['features_df'] = df_current.iloc[sorted(idx_train)]
                    else:
                        idx = np.arange(df_current.shape[0])
                        idx_train = idx
                        idx_val = idx_train
                    
                    
                    
                    if node_partition == 'temporal':
                        # It divides the transect into equal intervals time-wise, 
                        # meaning that the intervals can be somewhat different at least for a small number of them.
                        time_nodes = np.linspace(start=time_scanlines.min(), 
                                            stop = time_scanlines.max(), 
                                            num = number_of_nodes)
                        # Update optimization kwarg
                        kwargs['time_nodes'] = time_nodes

                        # Calculate the Jacobian for finding and exploiting sparsity
                        J = numerical_jacobian(fun = objective_fun_reprojection_error, param = param0_variab_tot, **kwargs)

                        

                        sparsity_perc = 100*((J==0).sum()/J.size)

                        print(f'Jacobian computed with {sparsity_perc:.0f} % zeros')
                        
                        # Using list of list representation as recommended by Scipy's least squares
                        sparsity = lil_matrix(J.shape, dtype=int)

                        # Setting the non sparse elements (in theory they could be set for small values of J too)
                        sparsity[J != 0] = 1
                        
                    elif node_partition == 'feature':
                        # We divide to have an equal number of features in each interval
                        idx_nodes = np.round(np.linspace(start=0, 
                                            stop = n_features-1, 
                                            num = number_of_nodes)).astype(np.int64)

                        # Select corresponding time stamps
                        time_nodes = time_arr_sorted_features[idx_nodes]

                        # At ends we asign the min/max of the pose to avoid extrapolation
                        time_nodes[0] = time_scanlines.min()
                        time_nodes[-1]  = time_scanlines.max()

                        # Update optimization kwarg
                        kwargs['time_nodes'] = time_nodes

                        
                        # Calculate the Jacobian for finding and exploiting sparsity
                        J = numerical_jacobian(fun = objective_fun_reprojection_error, param = param0_variab_tot, **kwargs)

                        

                        sparsity_perc = 100*((J==0).sum()/J.size)

                        print(f'Jacobian computed with {sparsity_perc:.0f} % zeros')
                        
                        # Using list of list representation as recommended by Scipy's least squares
                        sparsity = lil_matrix(J.shape, dtype=int)

                        # Setting the non sparse elements (in theory they could be set for small values of J too)
                        sparsity[J != 0] = 1

                    elif node_partition == 'all_features':
                        n_f = n_features
                        idx_nodes = np.round(np.linspace(start=0, 
                                            stop = n_f-1, 
                                            num = n_f)).astype(np.int64)

                        # Select corresponding time stamps to all features
                        time_nodes = time_arr_sorted_features[idx_nodes]

                        #At ends we asign the min/max of the pose to avoid extrapolation
                        #time_nodes[0] = time_scanlines.min()
                        #time_nodes[-1]  = time_scanlines.max()

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
                        
                        sparsity = assemble_jacobian_pattern(is_variab_param_intr, is_variab_param_extr, n_features)
                        


                    # Update optimization kwarg
                    kwargs['time_nodes'] = time_nodes
                    

                    
                else:
                    time_nodes = None
                    param0_variab_tot = param0_variab

                
                # Run once with initial using all features
                kwargs['features_df'] = df_current
                res_pre_optim = objective_fun_reprojection_error(param0_variab_tot, **kwargs)

                # Calculate the median absolute error in pixels
                SE = res_pre_optim[0:n_features]**2 + res_pre_optim[n_features:2*n_features]**2
                abs_err = np.sqrt(SE)
                MAE_median = np.median(abs_err)
                print(f'Original MAE median rp-error is {MAE_median:.2f} pixels')

                # Optimize the transect and record time duration
                time_start  = time.time()

                if val_mode:
                    # Prepare validation data
                    kwargs['features_df'] = df_current.iloc[sorted(idx_train)]
                else:
                    # Optimize using all data points
                    kwargs['features_df'] = df_current
                    n_train = n_features

                res = least_squares(fun = objective_fun_reprojection_error, 
                                x0 = param0_variab_tot, 
                                x_scale='jac',
                                jac_sparsity=sparsity,
                                kwargs=kwargs,
                                loss = loss_function)

                # Absolute reprojection errors
                n_train = idx_train.size
                SE = res.fun[0:n_train]**2 + res.fun[n_train:2*n_train]**2
                abs_err = np.sqrt(SE)
                MAE_mean = np.mean(abs_err)
                MAE_median = np.median(abs_err)
                rmse = np.sqrt(np.mean(SE))

                print(f'Optimized MAE median train set {MAE_median:.2f} pixels')
                print(f'Number of nodes was {number_of_nodes}')
                print(f'Number of features was {n_features}')
                print(f'Number of scanlines was {time_scanlines.size}')

                # Compute the median absolute error (more informative in outlier presense)
                x_err = res.fun[0:n_train]
                x_MAE = np.median(np.abs(x_err))

                y_err = res.fun[n_train:2*n_train]
                y_MAE = np.median(np.abs(y_err))

                ## Run the validation
                if val_mode:
                    kwargs['features_df'] = df_current.iloc[sorted(idx_val)]


                res_pre_optim = objective_fun_reprojection_error(res.x, **kwargs)

                # Calculate the median absolute error in pixels
                n = idx_val.size
                SE = res_pre_optim[0:n]**2 + res_pre_optim[n:2*n]**2
                abs_err = np.sqrt(SE)
                MAE_mean = np.mean(abs_err)
                MAE_median = np.median(abs_err)
                rmse_val = np.sqrt(np.mean(SE))

                
                print(f'MAE median error for VAL is {MAE_median:.2f} pixels')
                print('')


                

                
                

                
                
                # Now, if all features were used to optimize, use gaussian interpolation (KRIGING actually)
                if node_partition == 'all_features':
                    kwargs['time_interpolation_method'] = 'gaussian'
                
                param_optimized = res.x # Error curves

                camera_model_dict_updated, position_updated, quaternion_updated = calculate_cam_and_pose_from_param(h5_filename, 
                                                                                                                    param_optimized, 
                                                                                                                    **kwargs, 
                                                                                                                    h5_paths=h5_paths, 
                                                                                                                    plot_err_vec=plot_err_vec_time, 
                                                                                                                    sigma_nodes = None)

                # Now the data has been computed and can be written to h5:
                if not is_calibrated:
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
            
            n_features = df_gcp_filtered.shape[0] # All features

            # Sort values by chronology
            df_current = df_gcp_filtered.sort_values(by='unix_time')
            
            # Update the feature info for optimization from specific transect
            kwargs['features_df'] = df_current

            time_nodes, time_scanlines = _get_time_nodes(node_partition, df_gcp_filtered, h5_folder_time_scanlines, time_node_spacing)

            res_pre_optim = objective_fun_reprojection_error(param0_variab, df_gcp_filtered, param0, is_variab_param_intr, is_variab_param_extr, time_nodes, time_interpolation_method, pos_err_ref_frame, sigma_obs, sigma_param, time_scanlines)


            median_error_x = np.median(np.abs(res_pre_optim[0:n_features]))*resolution
            median_error_y = np.median(np.abs(res_pre_optim[n_features:2*n_features]))*resolution
            print(f'Original rp-error in x is {median_error_x} and in y is {median_error_y}')

            # The time varying parameters are in total the number of dofs times number of nodes
            number_of_nodes = time_nodes.size
            param0_time_varying = np.zeros(n_adjustable_dofs*number_of_nodes)

            # The time-varying parameters are stacked after the intrinsic parameters.
            # This vector only holds parameters that will be adjusted
            param0_variab_tot = np.concatenate((param0_variab, 
                                                param0_time_varying), axis=0)

            kwargs['time_scanlines'] = time_scanlines
            kwargs['time_nodes'] = time_nodes
            # Calculate the Jacobian for finding and exploiting sparsity
            J = numerical_jacobian(fun = objective_fun_reprojection_error, param = param0_variab_tot, **kwargs)
            
            # Using list of list representation as recommended by Scipy's least squares
            sparsity = lil_matrix(J.shape, dtype=int)

            # Setting the non sparse elements (in theory they could be set for small values of J too)
            sparsity[J != 0] = 1
            print(param0_variab_tot*180/np.pi)
            res = least_squares(fun = objective_fun_reprojection_error, 
                                x0 = param0_variab_tot, 
                                x_scale='jac',
                                jac_sparsity=sparsity,
                                kwargs=kwargs,
                                loss = loss_function)
            
            median_error_x = np.median(np.abs(res.fun[0:n_features]))*resolution
            median_error_y = np.median(np.abs(res.fun[n_features:2*n_features]))*resolution

            
            print(res.x*180/np.pi)
            print(f'Optimized MAE rp-error in x is {median_error_x} and in y is {median_error_y}')

    
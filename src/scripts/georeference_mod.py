# Built-ins
import configparser
import os
import sys

# Third party
from scipy.spatial.transform import Rotation as RotLib
import numpy as np
import pyvista as pv

# Local resources:
from scripts.geometry import CameraGeometry, CalibHSI
from lib.parsing_utils import Hyperspectral

def cal_file_to_rays(filename_cal, config):
        # See paper by Sun, Bo, et al. "Calibration of line-scan cameras for precision measurement." Applied optics 55.25 (2016): 6836-6843.
        # Loads line camera parameters for the hyperspectral imager from an xml file.

        # Certain imagers deliver geometry "per pixel". This can be resolved by fitting model parameters.
        calHSI = CalibHSI(file_name_cal_xml=filename_cal, config = config)
        f = calHSI.f
        u_c = calHSI.cx

        # Radial distortion parameters
        k1 = calHSI.k1
        k2 = calHSI.k2

        # Tangential distortion parameters
        k3 = calHSI.k3

        # Translation (lever arm) of HSI with respect to vehicle frame
        trans_x = calHSI.tx
        trans_y = calHSI.ty
        trans_z = calHSI.tz

        # Rotation of HSI (boresight) with respect to vehicle navigation frame (often defined by IMU or RGB cam)
        rot_x = calHSI.rx
        rot_y = calHSI.ry
        rot_z = calHSI.rz

        n_pix = calHSI.w

        # Define camera model array.
        u = np.arange(1, n_pix + 1)

        # Express uhi ray directions in uhi frame using line-camera model
        x_norm_lin = (u - u_c) / f

        x_norm_nonlin = -(k1 * ((u - u_c) / 1000) ** 5 + \
                          k2 * ((u - u_c) / 1000) ** 3 + \
                          k3 * ((u - u_c) / 1000) ** 2) / f

        x_norm = x_norm_lin + x_norm_nonlin

        p_dir = np.zeros((len(x_norm), 3))

        # Rays are defined in the UHI frame with positive z down
        p_dir[:, 0] = x_norm
        p_dir[:, 2] = 1

        rot_hsi_ref_eul = np.array([rot_z, rot_y, rot_x])

        rot_hsi_ref_obj = RotLib.from_euler(seq = 'ZYX',angles = rot_hsi_ref_eul, degrees=False)

        if config['General']['lever_arm_unit'] == 'mm':
            translation_hsi_ref = np.array([trans_x, trans_y, trans_z]) / 1000 # These are millimetres
        elif config['General']['lever_arm_unit'] == 'm':
            translation_hsi_ref = np.array([trans_x, trans_y, trans_z])

        intrinsic_geometry_dict = {'translation_hsi_ref': translation_hsi_ref,
                                   'rot_hsi_ref_obj': rot_hsi_ref_obj,
                                   'ray_directions_local': p_dir}

        return intrinsic_geometry_dict

def define_hsi_ray_geometry(pos_ref_ecef, quat_ref_ecef, time_pose, pos0, intrinsic_geometry_dict):
        # Instantiate a camera geometry object from the h5 pose data

        pos = pos_ref_ecef # Reference positions in ECEF offset by pos0
        rot_obj = RotLib.from_quat(quat_ref_ecef) # Reference orientations wrt ECEF
        
        ray_directions_local = intrinsic_geometry_dict['ray_directions_local']
        translation_hsi_ref = intrinsic_geometry_dict['translation_hsi_ref']
        rot_hsi_ref_obj = intrinsic_geometry_dict['rot_hsi_ref_obj']

        translation_ref_hsi =- translation_hsi_ref

        hsi_geometry = CameraGeometry(pos0=pos0, pos=pos, rot=rot_obj, time=time_pose, is_interpolated=True, use_absolute_position=True)
        
        
        hsi_geometry.intrinsicTransformHSI(translation_ref_hsi=translation_ref_hsi, rot_hsi_ref_obj = rot_hsi_ref_obj)

        hsi_geometry.defineRayDirections(dir_local = ray_directions_local)

        return hsi_geometry

def write_intersection_geometry_2_h5_file(hsi_geometry, config, h5_filename):
    # Add global points. projection equals intersections in ECEF
    
    points_global = hsi_geometry.projection # Use projected system for global description
    points_global_name = config['Georeferencing']['points_ecef_crs']
    Hyperspectral.add_dataset(data = points_global, name=points_global_name, h5_filename = h5_filename)

    # Add local points
    points_local = hsi_geometry.camera_to_seabed_local  # Use projected system for global description
    points_local_name = config['Georeferencing']['points_hsi_crs']
    Hyperspectral.add_dataset(data=points_local, name=points_local_name, h5_filename = h5_filename)

    # Add camera position
    position_hsi = hsi_geometry.PositionHSI  # Use projected system for global description
    position_hsi_name = config['Georeferencing']['position_ecef']
    Hyperspectral.add_dataset(data=position_hsi, name=position_hsi_name, h5_filename = h5_filename)

    # Add camera quaternion
    quaternion_hsi = hsi_geometry.RotationHSI.as_quat()  # Use projected system for global description
    quaternion_hsi_name = config['Georeferencing']['quaternion_ecef']
    Hyperspectral.add_dataset(data=quaternion_hsi, name=quaternion_hsi_name, h5_filename = h5_filename)

    # Add normals
    normals_local = hsi_geometry.normalsLocal # Use projected system for global description
    normals_local_name = config['Georeferencing']['normals_hsi_crs']
    Hyperspectral.add_dataset(data=normals_local, name=normals_local_name, h5_filename = h5_filename)

    # TODO: Makes sense to also write the view angles in a tangent plane (NED).

    # Add normals in NED
    normals_NED = hsi_geometry.normals_NED # Use projected system for global description
    normals_NED_name = config['Georeferencing']['normals_NED_crs']
    Hyperspectral.add_dataset(data=normals_NED, name=normals_NED_name, h5_filename = h5_filename)

    # Add theta_v the in-air view nadir angle
    theta_v = hsi_geometry.theta_v # Use projected system for global description
    theta_v_name = config['Georeferencing']['theta_v']
    Hyperspectral.add_dataset(data=theta_v, name=theta_v_name, h5_filename = h5_filename)

    # Add theta_s, the in-air sun nadir angle
    theta_s = hsi_geometry.theta_s # Use projected system for global description
    theta_s_name = config['Georeferencing']['theta_s']
    Hyperspectral.add_dataset(data=theta_s, name=theta_s_name, h5_filename = h5_filename)

    # Add phi_v the in-air view azimuth angle
    phi_v = hsi_geometry.phi_v # Use projected system for global description
    phi_v_name = config['Georeferencing']['phi_v']
    Hyperspectral.add_dataset(data=phi_v, name=phi_v_name, h5_filename = h5_filename)

    # Add phi_s, the in-air sun azimuth angle
    phi_s = hsi_geometry.phi_s # Use projected system for global description
    phi_s_name = config['Georeferencing']['phi_s']
    Hyperspectral.add_dataset(data=phi_s, name=phi_s_name, h5_filename = h5_filename)

    # Add time layer
    unix_time_grid = hsi_geometry.unix_time_grid # Use projected system for global description
    unix_time_grid_name = config['Georeferencing']['unix_time_grid']
    Hyperspectral.add_dataset(data=unix_time_grid, name=unix_time_grid_name, h5_filename = h5_filename)

    # Add tide layer
    hsi_tide_gridded = hsi_geometry.hsi_tide_gridded # Use projected system for global description
    hsi_tide_gridded_name = config['Georeferencing']['hsi_tide_gridded']
    Hyperspectral.add_dataset(data=hsi_tide_gridded, name=hsi_tide_gridded_name, h5_filename = h5_filename)


# Function called to apply standard processing on a folder of files
def main(iniPath):
    config = configparser.ConfigParser()
    config.read(iniPath)

    # Paths to 3D mesh ply file 
    path_mesh = config['Absolute Paths']['modelPath']

    # Directory of H5 files
    dir_r = config['Absolute Paths']['h5Dir']

    # The path to the XML file
    hsi_cal_xml = config['Absolute Paths']['HSICalibFile']

    # The path to the Tide file (if necessary)
    try:
        path_tide = config['Absolute Paths']['pathTide']
    except Exception as e:
        path_tide = 'Undefined'
    
    # Maximal allowed ray length
    max_ray_length = float(config['General']['maxRayLength'])

    mesh = pv.read(path_mesh)

    print('Georeferencing Images')

    for filename in sorted(os.listdir(dir_r)):
        if filename.endswith('h5') or filename.endswith('hdf'):
            # Path to hierarchical file
            path_hdf = dir_r + filename

            # Read h5 file
            hyp = Hyperspectral(path_hdf, config)

            # Using the cal file, we can define lever arm, boresight and local ray geometry (in dictionary)
            intrinsic_geometry_dict = cal_file_to_rays(filename_cal=hsi_cal_xml, config=config)

            
            # Define the rays in ECEF for each frame. Note that if there is no position offset, pos0 is a 1x3 of zeros
            hsi_geometry = define_hsi_ray_geometry(pos_ref_ecef = hyp.pos_ref, 
                                    quat_ref_ecef = hyp.quat_ref, 
                                    time_pose = hyp.pose_time, 
                                    pos0 = hyp.pos0, 
                                    intrinsic_geometry_dict = intrinsic_geometry_dict)

            
            hsi_geometry.intersectWithMesh(mesh = mesh, max_ray_length=max_ray_length)
            
            # Computes the view angles in the local NED. Computationally intensive as local NED is defined for each intersection
            hsi_geometry.compute_view_directions_local_tangent_plane()

            # Computes the sun angles in the local NED. Computationally intensive as local NED is defined for each intersection
            hsi_geometry.compute_sun_angles_local_tangent_plane()

            hsi_geometry.compute_tide_level(path_tide, tide_format = 'NMA')
            
            write_intersection_geometry_2_h5_file(hsi_geometry=hsi_geometry, config = config, h5_filename=path_hdf)

            print('Intersection geometry written to:\n {0}'.format(filename))


            
            print('Writing Point Cloud')
            hsi_geometry.writeRGBPointCloud(config = config, hyp = hyp, transect_string = filename.split('.')[0])

            #from scripts import visualize
            #visualize.show_projected_hsi_points(HSICameraGeometry=hsi_geometry, config=config, transect_string = filename.split('.')[0])

            












if __name__ == '__main__':
    args = sys.argv[1:]
    iniPath = args[0]
    main(iniPath)

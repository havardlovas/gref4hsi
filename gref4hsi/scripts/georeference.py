# Built-ins
import configparser
import json
import os
from pathlib import Path
import sys

# Third party
from scipy.spatial.transform import Rotation as RotLib
import numpy as np
import pyvista as pv
import h5py

# Lib resources:
from gref4hsi.utils.geometry_utils import CameraGeometry, CalibHSI
from gref4hsi.utils.parsing_utils import Hyperspectral
from gref4hsi.utils import visualize


def cal_file_to_rays(filename_cal):
        # See paper by Sun, Bo, et al. "Calibration of line-scan cameras for precision measurement." Applied optics 55.25 (2016): 6836-6843.
        # Loads line camera parameters for the hyperspectral imager from an xml file.

        # Certain imagers deliver geometry "per pixel". This can be resolved by fitting model parameters.
        calHSI = CalibHSI(file_name_cal_xml=filename_cal)
        f = calHSI.f
        cx = calHSI.cx

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

        # Number of pixels
        n_pix = calHSI.w

        # Define camera model array.
        u = np.arange(0, n_pix) + 0.5

        # Express uhi ray directions in uhi frame using line-camera model
        x_norm_lin = (u - cx) / f

        x_norm_nonlin = -(k1 * ((u - cx)) ** 5 + \
                          k2 * ((u - cx)) ** 3 + \
                          k3 * ((u - cx)) ** 2) / f

        x_norm = x_norm_lin + x_norm_nonlin

        p_dir = np.zeros((len(x_norm), 3))

        # Rays are defined in the HI frame with positive z down
        p_dir[:, 0] = x_norm
        p_dir[:, 2] = 1

        rot_hsi_ref_eul = np.array([rot_z, rot_y, rot_x])

        rot_hsi_ref_obj = RotLib.from_euler(seq = 'ZYX', angles = rot_hsi_ref_eul, degrees=False)

        translation_ref_hsi = np.array([trans_x, trans_y, trans_z])

        intrinsic_geometry_dict = {'translation_ref_hsi': translation_ref_hsi,
                                   'rot_hsi_ref_obj': rot_hsi_ref_obj,
                                   'ray_directions_local': p_dir}
        
        # Notably, one could compress the information by expressing the ray directions in the body frame

        return intrinsic_geometry_dict

def define_hsi_ray_geometry(pos_ref_ecef, quat_ref_ecef, time_pose, intrinsic_geometry_dict):
        """Instantiate a camera geometry object from the h5 pose data"""


        pos = pos_ref_ecef # Reference positions in ECEF
        rot_obj = RotLib.from_quat(quat_ref_ecef) # Reference orientations wrt ECEF
        
        ray_directions_local = intrinsic_geometry_dict['ray_directions_local']
        translation_ref_hsi = intrinsic_geometry_dict['translation_ref_hsi']
        rot_hsi_ref_obj = intrinsic_geometry_dict['rot_hsi_ref_obj']

        hsi_geometry = CameraGeometry(pos=pos, rot=rot_obj, time=time_pose, is_interpolated=True)
        
        
        hsi_geometry.intrinsicTransformHSI(translation_ref_hsi=translation_ref_hsi, rot_hsi_ref_obj = rot_hsi_ref_obj)

        hsi_geometry.defineRayDirections(dir_local = ray_directions_local)

        return hsi_geometry

def write_intersection_geometry_2_h5_file(hsi_geometry, config, h5_filename):
    # Write all intersection data (ancilliary) that could be relevant
    
    dict_ancilliary = config['Georeferencing']
    # Dictionary keys correspond to CameraGeometry attribute names (e.g. hsi_geometry.key), while values correspond to h5 data set paths.
    
    with h5py.File(h5_filename, 'a', libver='latest') as f:
        for attribute_name, h5_hierarchy_item_path in dict_ancilliary.items():
            if attribute_name != 'folder':
                if h5_hierarchy_item_path in f:
                    del f[h5_hierarchy_item_path]
                dset = f.create_dataset(name=h5_hierarchy_item_path, 
                                                data = getattr(hsi_geometry, attribute_name))


# Function called to apply standard processing on a folder of files
def main(iniPath, viz = False, use_coreg_param = False):
    config = configparser.ConfigParser()
    config.read(iniPath)

    # Paths to 3D mesh ply file 
    path_mesh = config['Absolute Paths']['model_path']

    # Directory of H5 files
    dir_r = config['Absolute Paths']['h5_folder']

    # Timestamps here
    h5_folder_time_pose = config['HDF.processed_nav']['timestamp']

    # Use the regular parameters from nav system and manufacturer:
    # The path to the XML file
    hsi_cal_xml = config['Absolute Paths']['hsi_calib_path']

    # Position is stored here in the H5 file
    h5_folder_position_ecef = config['HDF.processed_nav']['position_ecef']

    # Quaternion is stored here in the H5 file
    h5_folder_quaternion_ecef = config['HDF.processed_nav']['quaternion_ecef']

    if use_coreg_param:
        print('Using coregistred parameters for georeferencing')

        # Set the camera model to the calibrated one if it exists
        hsi_cal_xml_coreg = config['Absolute Paths']['calib_file_coreg']
        if os.path.exists(hsi_cal_xml_coreg):
             hsi_cal_xml = hsi_cal_xml_coreg

        # Optimized position
        h5_folder_position_ecef_coreg = config['HDF.coregistration']['position_ecef']

        # Quaternion 
        h5_folder_quaternion_ecef_coreg = config['HDF.coregistration']['quaternion_ecef']
        

    
    
    
    # The path to the Tide file (if necessary and available)
    try:
        path_tide = config['Absolute Paths']['tide_path']
    except Exception as e:
        path_tide = 'Undefined'
    
    # Maximal allowed ray length
    max_ray_length = float(config['General']['max_ray_length'])

    dem_per_transect = False

    try:

        if eval(config['General']['dem_per_transect']):

            dem_per_transect = True

            dem_folder_parent = Path(config['Absolute Paths']['dem_folder'])

            # Get all entries (files and directories)
            all_entries = dem_folder_parent.iterdir()

            # Filter for directories (excluding '.' and '..')
            transect_folders = [entry for entry in all_entries if entry.is_dir() and not entry.name.startswith('.')]

            mesh_dict = {'transect_name_list': [],
                        'mesh_list': [],
                        'mesh_trans_list': []}

            for transect_folder in transect_folders:
                transect_name = transect_folder.name

                mesh_dict['transect_name_list'].append(transect_name)

                

                transect_folder = str(transect_folder)
                path_mesh = os.path.join(transect_folder, 'model.vtk')

                mesh = pv.read(path_mesh)
                mesh_dict['mesh_list'].append(mesh)

                model_meta_path = path_mesh.split('.')[0] + '_meta.json' 
                with open(model_meta_path, "r") as f:
                    # Load the JSON data from the file
                    metadata_mesh = json.load(f)
                    mesh_off_x = metadata_mesh['offset_x']
                    mesh_off_y = metadata_mesh['offset_y']
                    mesh_off_z = metadata_mesh['offset_z']
                # Mesh is translated by this much
                mesh_trans = np.array([mesh_off_x, mesh_off_y, mesh_off_z]).astype(np.float64)
                mesh_dict['mesh_trans_list'].append(mesh_trans)

        else:
            Exception

    except:

        mesh = pv.read(path_mesh)

        model_meta_path = path_mesh.split('.')[0] + '_meta.json' 
        with open(model_meta_path, "r") as f:
            # Load the JSON data from the file
            metadata_mesh = json.load(f)
            mesh_off_x = metadata_mesh['offset_x']
            mesh_off_y = metadata_mesh['offset_y']
            mesh_off_z = metadata_mesh['offset_z']
            # Mesh is translated by this much
            mesh_trans = np.array([mesh_off_x, mesh_off_y, mesh_off_z]).astype(np.float64)

    
    print("\n################ Georeferencing: ################")
    files = sorted(os.listdir(dir_r))
    # Filter out files that do not end with ".h5"
    h5_files = [file for file in files if file.endswith(".h5")]
    n_files= len(h5_files)
    file_count = 0
    for filename in h5_files:
        if filename.endswith('h5') or filename.endswith('hdf'):
            print(filename)
            progress_perc = 100*file_count/n_files
            print(f"Georeferencing file {file_count+1}/{n_files}, progress is {progress_perc} %")

            # Path to hierarchical file
            h5_filename = dir_r + filename

            # Read h5 file
            hyp = Hyperspectral(h5_filename, config)

            if use_coreg_param:
                 try:
                    # Use the coregistred dataset if it exists
                    print('Using coregistred position')
                    pos_ref_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                    dataset_name= h5_folder_position_ecef_coreg)
                 except:
                    # If not use the original
                    pos_ref_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                    dataset_name= h5_folder_position_ecef)
                 try:
                    # Use the coregistred quaternion-dataset if it exists
                    print('Using coregistred quaternion')
                    quat_ref_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                    dataset_name= h5_folder_quaternion_ecef_coreg)
                 except:
                     # If not use the original
                    quat_ref_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                    dataset_name= h5_folder_quaternion_ecef)
            else:

                # Just use the regular navigation data
                # If not use the original
                pos_ref_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                dataset_name= h5_folder_position_ecef)
                # Extract the ecef orientations for each frame
                quat_ref_ecef = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                                dataset_name=h5_folder_quaternion_ecef)
            # Extract the timestamps for each frame
            time_pose = Hyperspectral.get_dataset(h5_filename=h5_filename,
                                                            dataset_name= h5_folder_time_pose)


            # Using the cal file, we can define lever arm, boresight and camera model geometry (in dictionary)
            intrinsic_geometry_dict = cal_file_to_rays(filename_cal=hsi_cal_xml)

            
            # Define the rays in ECEF for each frame. 
            hsi_geometry = define_hsi_ray_geometry(pos_ref_ecef, 
                                    quat_ref_ecef, 
                                    time_pose,
                                    intrinsic_geometry_dict = intrinsic_geometry_dict)

            

            # Determine which 3D model/mesh to use based on transect name
            if dem_per_transect:
                # First find out which transect filename is in and use that mesh
                string_list = mesh_dict['transect_name_list']
                mesh_index = next((i for i, string in enumerate(string_list) if string in filename), -1)
                mesh = mesh_dict['mesh_list'][mesh_index]
                mesh_trans = mesh_dict['mesh_trans_list'][mesh_index]


            try:
                # If intersection failed here, then skip data point
                hsi_geometry.intersect_with_mesh(mesh = mesh, max_ray_length=max_ray_length, mesh_trans = mesh_trans)
            except ValueError:
                print(f'Skipping transect chuck because of lacking intersections: {filename}')
                continue

            
            
            # Computes the view angles in the local NED. Computationally intensive as local NED is defined for each intersection
            hsi_geometry.compute_view_directions_local_tangent_plane()

            # Computes the sun angles in the local NED. Computationally intensive as local NED is defined for each intersection
            hsi_geometry.compute_sun_angles_local_tangent_plane()

            hsi_geometry.compute_tide_level(path_tide, tide_format = 'NMA')

            hsi_geometry.compute_elevation_mean_sealevel(source_epsg = config['Coordinate Reference Systems']['geocsc_epsg_export'], 
                                                         geoid_path = config['Absolute Paths']['geoid_path'])
            
            write_intersection_geometry_2_h5_file(hsi_geometry=hsi_geometry, config = config, h5_filename=h5_filename)

            hsi_geometry.write_rgb_point_cloud(config = config, hyp = hyp, transect_string = filename.split('.')[0], mesh_trans= mesh_trans)

            if viz:
                 visualize.show_projected_hsi_points(HSICameraGeometry=hsi_geometry, 
                                                     config=config, 
                                                     transect_string = filename.split('.')[0],
                                                     mesh_trans = mesh_trans)

            


            
            

            #from scripts import visualize
            #visualize.show_projected_hsi_points(HSICameraGeometry=hsi_geometry, config=config, transect_string = filename.split('.')[0])

        file_count+=1
            












if __name__ == '__main__':
    args = sys.argv[1:]
    iniPath = args[0]
    main(iniPath)

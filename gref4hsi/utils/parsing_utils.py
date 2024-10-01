# Python built-ins
import sys
import os
import configparser
from os import path
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as RotLib
from scipy.spatial.transform import Rotation
import h5py
from scipy.spatial.transform import Rotation as RotLib
import pandas as pd
import pymap3d as pm
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from pyproj import CRS, Transformer

# Local modules
from gref4hsi.utils.geometry_utils import CameraGeometry, GeoPose
from gref4hsi.utils.geometry_utils import rot_mat_ned_2_ecef, interpolate_poses
from gref4hsi.utils.geometry_utils import dem_2_mesh, crop_geoid_to_pose


class Hyperspectral:
    """
    Class for storing/accessing acquired hyperspectral data. One instance corresponds to a transect chunk (*.h5).
    """
    def __init__(self, filename, config, load_datacube = True):
        """
        Instantiate transect chunk object from chunk file name and config file.
        :param filename: string
        path to *.h5 file
        :param config: config type
        dictionary-like interface to configuration file
        """
        # Chunk filename
        self.name = filename
        # If errors arise due to improper dataset paths below, open the *.h5 file in hdfviewer and edit the path.
        with h5py.File(filename, 'r', libver='latest') as self.f:
            # TODO: Embed the paths from this special format into confic file
            try:
                dataCubePath = config['HDF.hyperspectral']['dataCube']
            except:
                pass
            
            try:
                timestampHyperspectralPath = config['HDF.hyperspectral']['timestamp']
                self.dataCubeTimeStamps = self.f[timestampHyperspectralPath][()]
            except:
                pass

            try:
                # The band center wavelengths
                band2WavelengthPath = config['HDF.calibration']['band2Wavelength']
                self.band2Wavelength = self.f[band2WavelengthPath][()]
            except:
                pass
            
            try:
                # The band widths (if they exist)
                fwhmPath = config['HDF.calibration']['fwhm']

                self.fwhm = self.f[fwhmPath][()]
            except:
                pass

            try:
                # These should come together in the case that radiometric calibration should be done
                radiometricFramePath = config['HDF.calibration']['radiometricFrame']
                darkFramePath = config['HDF.calibration']['darkFrame']
                exposureTimePath = config['HDF.hyperspectral']['exposureTime']
                

                self.t_exp = self.f[exposureTimePath][()] / 1000 # Assuming milliseconds

                if self.t_exp.size > 1:
                    self.t_exp = self.t_exp[0]

                self.darkFrame = self.f[darkFramePath][()]
                self.radiometricFrame = self.f[radiometricFramePath][()]
            except:
                pass
            
            try:
                RGBFramesPath = config['HDF.rgb']['rgbFrames']
                timestampRGBPath = config['HDF.rgb']['timestamp']
                try:
                    # Extract RGB image data if available
                    self.RGBTimeStamps = self.f[timestampRGBPath][()]
                    self.RGBImgs = self.f[RGBFramesPath][()]
                    self.n_imgs = self.RGBTimeStamps.shape[0]
                except (UnboundLocalError, KeyError):
                    pass
            except:
                pass
            
            

            if load_datacube:
                # To save memory, we made it an option to not load datacube
                self.dataCube = self.f[dataCubePath][()]

                self.n_scanlines = self.dataCube.shape[0]
                self.n_pix = self.dataCube.shape[1]
                self.n_bands = self.dataCube.shape[1]

            # Check if the dataset exists
            processed_nav_folder = config['HDF.processed_nav']['folder']
            
            # Read the position da
            if processed_nav_folder in self.f:
                try:
                    processed_nav_config = config['HDF.processed_nav']
                    self.pos_ref = self.f[processed_nav_config['position_ecef']][()]
                    self.quat_ref = self.f[processed_nav_config['quaternion_ecef']][()]
                    self.pose_time = self.f[processed_nav_config['timestamp']][()]
                except:
                    pass
        
        
        # This step must be conducted after closing h5 file
        if load_datacube:
            # Calculate radiance cube if this is cube is not calibrated
            self.digital_counts_2_radiance(config=config)
            



       

            



    def digital_counts_2_radiance(self, config):
        """Calibrate data """

        # Only calibrate if data it is not already done
        is_calibrated = eval(config['HDF.hyperspectral']['is_calibrated'])

        if is_calibrated:
            self.dataCubeRadiance = self.dataCube.astype(np.float32)

            # This is then the path of radiance
            radiance_cube_path = config['HDF.hyperspectral']['dataCube']
        else:
            self.dataCubeRadiance = np.zeros(self.dataCube.shape, dtype = np.float32)
            for i in range(self.dataCube.shape[0]):
                self.dataCubeRadiance[i, :, :] = (self.dataCube[i, :, :] - self.darkFrame) / (
                        self.radiometricFrame * self.t_exp)
            
            # Add the radiance dataset
            radiance_cube_path = config['HDF.hyperspectral']['dataCube'] + '_radiance'
            
            # Write the radiance data to the h5 file. Next time this is used is during orthorectification
            Hyperspectral.add_dataset(data = self.dataCubeRadiance, name=radiance_cube_path, h5_filename=self.name, overwrite=True)

        # For memory efficiency
        del self.dataCube
    
    @staticmethod
    def add_dataset(data, name, h5_filename, overwrite = True):
        """
        Method to write a dataset to the h5 file
        :param data: type any permitted (see h5py doc)
        The data to be written
        :param name: string
        The path/name of the dataset
        :param h5_filename: string
        The path to the h5_file
        :return: None
        """
        # The h5 file structure can be studied by unravelling the structure in Python or by using HDFview
        with h5py.File(h5_filename, 'a', libver='latest') as f:
            # Check if the dataset exists
            if overwrite:
                # Overwrite and create new dataset
                if name in f:
                    del f[name]

                dset = f.create_dataset(name=name, data = data)
            else:
                if name in f:
                    # Do nothing
                    pass
                else:
                    # Make new
                    dset = f.create_dataset(name=name, data = data)
                pass

    """def get_dataset(self, dataset_name):
        
        Method to return a dataset by the name
        :param dataset_name: string
        The h5 dataset path
        :return: dataset
        Returns the queried dataset. Could be many datatypes, but mainly numpy arrays in our usage.
        
        # The h5 file structure can be studied by unravelling the structure in Python or by using HDFview
        with h5py.File(self.name, 'a', libver='latest') as self.f:
            dataset = self.f[dataset_name][()]
        return dataset"""
    @staticmethod
    def get_dataset(h5_filename, dataset_name):
        """equivalent to the method, except it does not need an instance of a Hyperspectral object

        :param h5_filename: Path to h5/hdf file
        :type h5_filename: string
        :param dataset_name: h5 rooted path, e.g. processed/reflectance/remote_sensing_reflectance
        :type dataset_name: string
        :return: The dataset at the relevant location
        :rtype: numpy array or other
        """
        # The h5 file structure can be studied by unravelling the structure in Python or by using HDFview
        with h5py.File(h5_filename, 'a', libver='latest') as f:
            dataset = f[dataset_name][()]
        return dataset

class DataLogger:
    def __init__(self, filename, header):
        self.filename = filename
        # Generate file with header
        with open(filename, 'w') as fh:
            fh.write(header + '\n')

    def append_data(self, data):
        # Append data line to file
        with open(self.filename, 'a') as fh:
            # Generate data line to file
            if data[0] != None:
                line = ','.join([str(el) for el in data])

                # Append line to file
                fh.write(line + '\n')

def ardupilot_extract_pose(config, iniPath):
    
    """
        :param config: The processing configuration object (from */configuration.ini file)
        :param iniPath: The path to */configuration.ini file
        :return: 
    """
    
    # Attitude retrieved from an ardupilot CSV
    att_path = config['General']['ardupath'] + 'Att.csv'
    df_att = pd.read_csv(att_path)

    # Convert the DataFrame to a matrix (list of lists)
    matrix_att = np.array(df_att.values.tolist())
    
    # Rotations in roll-pitch-yaw define the rotation of the vehicle's body frame (i.e. IMU)
    Rotation = RotLib.from_euler("ZYX", np.flip(matrix_att[:,1:4], axis = 1), degrees=True)
    time_rot = matrix_att[:,0]
    
    # Position retrieved from an ardupilot CSV
    pos_path = config['General']['ardupath'] + 'pos.csv'
    df_pos = pd.read_csv(pos_path)
    matrix_pos = np.array(df_pos.values.tolist())

    time_pos = matrix_pos[:, 0]
    Position = matrix_pos[:, 1:4]

    linearPositionInterpolator = interp1d(time_pos, np.transpose(Position), fill_value='extrapolate')
    position_nav_interpolated = np.transpose(linearPositionInterpolator(time_rot))

    pose_path = config['General']['ardupath'] + 'pose.csv'

    config.set('General', 'pose_path', pose_path)

    epsg_proj = 4326 # Standard
    epsg_geocsc = config['General']['modelepsg']
    # Transform the mesh points to ECEF.

    geocsc = CRS.from_epsg(epsg_geocsc)
    proj = CRS.from_epsg(epsg_proj)
    transformer = Transformer.from_crs(proj, geocsc)

    points_proj = position_nav_interpolated

    lat = points_proj[:, 0].reshape((-1, 1))
    lon = points_proj[:, 1].reshape((-1, 1))
    hei = points_proj[:, 2].reshape((-1, 1))

    (x_ecef, y_ecef, z_ecef) = transformer.transform(xx=lat, yy=lon, zz=hei)

    pos_geocsc = np.concatenate((x_ecef.reshape((-1,1)), y_ecef.reshape((-1,1)), z_ecef.reshape((-1,1))), axis = 1)

    #dlogr = DataLogger(pose_path, 'CameraLabel, X, Y, Z, Roll, Pitch, Yaw, RotX, RotY, RotZ')



    data_matrix = np.zeros((len(time_rot), 10))
    for i in range(len(time_rot)):

        # Ardupilot delivers geodetic positions (EPSG::
        pos_geod = position_nav_interpolated[i, :]

        # Define rotation matrix between local tangential plane (NED, North-East-Down) and Earth-Centered-Earth-Fixed (ECEF).
        R_n_e = rot_mat_ned_2_ecef(lat=pos_geod[0], lon=pos_geod[1])

        Rotation_n = RotLib.from_matrix(R_n_e)

        Rotation_e = Rotation_n*Rotation[i] # Composing rotations

        roll = matrix_att[i, 1]
        pitch = matrix_att[i, 2]
        yaw = matrix_att[i, 3]

        

        rotz, roty, rotx = Rotation_e.as_euler("ZYX", degrees=True)

        pos = pos_geocsc[i,:]

        data_matrix[i,:] = np.array([time_rot[i], pos[0], pos[1], pos[2], roll, pitch, yaw, rotx, roty, rotz])

        #dlogr.append_data([time_rot[i], pos[0], pos[1], pos[2], roll, pitch, yaw, rotx, roty, rotz])



    headers = ['Timestamp', ' X', ' Y', ' Z', ' Roll', ' Pitch', ' Yaw', ' RotX', ' RotY', ' RotZ']

    # Create a DataFrame from the data_matrix and headers
    df = pd.DataFrame(data_matrix, columns=headers)

    # Save the DataFrame as a CSV file
    df.to_csv(pose_path, index=False)

    with open(iniPath, 'w') as configfile:
        config.write(configfile)
        
    return config


def reformat_h5_embedded_data_h5(config, config_file):
    """
                Parses pose from h5, interpolates and reformats. Writes the positions and orientations of hyperspectral frames to
                    dataset under the /Nav folder

                Args:
                    config_file: The *.ini file containing the paths and configurations.

                Returns:
                    None: The function returns nothing.
    """


    # Traverse through h5 dir to append the data to file
    h5_folder = config['Absolute Paths']['h5_folder']
    is_first = True
    for filename in sorted(os.listdir(h5_folder)):
        
        # Find the interesting prefixes
        if filename.endswith('h5') or filename.endswith('hdf'):
            # Identify the total path
            path_hdf = h5_folder + filename

            # Read out the data of a file
            hyp = Hyperspectral(path_hdf, config, load_datacube = False)


            # If rotation reference is quaternion:
            # Alternatives quat, eul_ZYX
            rotation_reference_type = config['HDF.raw_nav']['rotation_reference_type']
            if rotation_reference_type == 'quat':
                quaternion_ref = hyp.get_dataset(h5_filename=path_hdf, dataset_name=config['HDF.raw_nav']['quaternion'])
                if quaternion_ref.shape[0] == 4:
                    quaternion_ref = np.transpose(quaternion_ref)
                
            
                quaternion_convention  = config['HDF.raw_nav']['quaternion_convention']

                # If scalar-first convention
                if quaternion_convention == 'wxyz':
                    quat_temp = quaternion_ref
                    quaternion_ref = np.roll(quaternion_ref, shift=-1, axis=1)
                elif quaternion_convention == 'xyzw':
                    # We use the xyzw convention for scipy, see:
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
                    pass
                else:
                    raise ValueError("The quaternion convention must be defined")
            elif rotation_reference_type == 'eul_ZYX':
                eul_ref = hyp.get_dataset(h5_filename=path_hdf,dataset_name=config['HDF.raw_nav']['eul_ZYX'])
                if eul_ref.shape[0] == 3:
                    eul_ref = np.transpose(eul_ref)
                
                # From keyword in config file
                eul_is_degrees = eval(config['HDF.raw_nav']['eul_is_degrees'])
                
                ## Assuming It is given as roll, pitch, yaw
                eul_ZYX_ref = np.flip(eul_ref, axis = 1)
                quaternion_ref = RotLib.from_euler(angles = eul_ZYX_ref, seq = 'ZYX', degrees = eul_is_degrees).as_quat()
                
            else:
                raise ValueError("The rotation reference type must be defined as quat or eul_ZYX")


            is_global_rot = eval(config['HDF.raw_nav']['is_global_rot'])

            # Parse position
            position_ref = hyp.get_dataset(h5_filename=path_hdf,dataset_name=config['HDF.raw_nav']['position'])
            
            if position_ref.shape[0] == 3:
                position_ref = np.transpose(position_ref)


            timestamps_imu = hyp.get_dataset(h5_filename=path_hdf, dataset_name=config['HDF.raw_nav']['timestamp'])\
                .reshape(-1)


            # The rotation of the reference
            rot_obj = RotLib.from_quat(quaternion_ref)

            timestamp_hsi = hyp.dataCubeTimeStamps.reshape(-1)

            # Compute interpolated absolute positions positions and orientation:
            
            # In special case where data has already been interpolated
            if np.array_equal(timestamps_imu, timestamp_hsi):
                position_interpolated = position_ref
                quaternion_interpolated = quaternion_ref
                
            else: # Do interpolation
                position_interpolated, quaternion_interpolated = interpolate_poses(timestamp_from=timestamps_imu,
                                                                 pos_from=position_ref,
                                                                 rot_from=rot_obj,
                                                                 timestamps_to=timestamp_hsi)

            # If the original orientations are with respect to (North-East-Down) NED
            if not is_global_rot:
                rot_ref = 'NED'
            else:
                rot_ref = 'ECEF'

            # The positions supplied for the reference
            pos_epsg_orig = config['Coordinate Reference Systems']['pos_epsg_orig']

            # The positions supplied for exporting the model
            pos_epsg_export = config['Coordinate Reference Systems']['geocsc_epsg_export']

            # The interpolated rotation-object
            rot_interpolated = RotLib.from_quat(quaternion_interpolated)

            # For simple geo-pose for changing between formats.
            geo_pose_ref = GeoPose(timestamps=timestamp_hsi,
                                   rot_obj= rot_interpolated, 
                                   rot_ref=rot_ref,
                                   pos = position_interpolated, 
                                   pos_epsg=pos_epsg_orig)

            # Convert position to the epsg used for the 3D model
            geo_pose_ref.compute_geocentric_position(epsg_geocsc=pos_epsg_export)


            # Calculate the geodetic position using the WGS-84 (for conversion of orientations)
            epsg_wgs84 = 4326
            geo_pose_ref.compute_geodetic_position(epsg_geod=epsg_wgs84)

            # Calculate ECEF orientation
            geo_pose_ref.compute_geocentric_orientation()

            # For readability, extract the position and orientation wrt ECEF
            position_ref_ecef = geo_pose_ref.pos_geocsc
            quaternion_ref_ecef = geo_pose_ref.rot_obj_ecef.as_quat()

            # For stacking in CSV
            rot_vec_ecef = geo_pose_ref.rot_obj_ecef.as_euler('ZYX', degrees=True)

            partial_pose = np.concatenate((timestamp_hsi.reshape((-1,1)), position_ref_ecef, rot_vec_ecef), axis=1)
            if is_first:
                # Initialize
                total_pose = partial_pose
                # Ensure that flag is not raised again. Offsets should only be set once.
                is_first = False
            else:
                # Concatenate/stack poses
                total_pose = np.concatenate((total_pose, partial_pose), axis=0)
                is_first = False
            
            

            # Add camera position
            position_ref_name = config['HDF.processed_nav']['position_ecef']
            Hyperspectral.add_dataset(data=position_ref_ecef, name=position_ref_name, h5_filename=path_hdf)

            # Add camera quaternion
            quaternion_ref_name = config['HDF.processed_nav']['quaternion_ecef']
            Hyperspectral.add_dataset(data=quaternion_ref_ecef, name=quaternion_ref_name, h5_filename=path_hdf)

            # Add time stamps
            time_stamps_name = config['HDF.processed_nav']['timestamp']
            Hyperspectral.add_dataset(data=timestamp_hsi, name=time_stamps_name, h5_filename=path_hdf)


            


    headers = ['Timestamp', ' X', ' Y', ' Z', ' RotZ', ' RotY', ' RotX']

    # Create a DataFrame from the data_matrix and headers
    # Stores the entire pose path
    df = pd.DataFrame(total_pose, columns=headers)

    pose_path = config['Absolute Paths']['pose_path']

    try:
        # Save the DataFrame as a CSV file
        df.to_csv(pose_path, index=False)
    except:
        # Extract directory
        directory = os.path.dirname(pose_path)

        # Check if directory doesn't exist and create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        df.to_csv(pose_path, index=False)

# The main script for aquiring pose.        
def export_pose(config_file):
    """
        Parses pose and embeds it into the *.h5 file. Writes the positions and orientations of hyperspectral frames to
            dataset under the /Nav folder

        Args:
            config_file: The *.ini file containing the paths and configurations.

        Returns:
            config: The function returns config object to allow that it is modified.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    # This file handles pose exports of various types
    # As of now there are three types:

    # 1) Agisoft, h5_embedded and independent_file
    try:
        pose_export_type = config['General']['pose_export_type']
    except:
        # Default to h5_embedded
        pose_export_type = 'h5_embedded'

    if pose_export_type == 'h5_embedded':
        reformat_h5_embedded_data_h5(config=config,
                                              config_file=config_file)
    elif pose_export_type == 'independent_file':
        print('There is no support for this export functionality! Fix immediately!')
        config = -1
    else:
        print('File type: ' + pose_export_type + 'type is not defined!')
        config = -1

    
    
def export_model(config_file):
    """"""
    config = configparser.ConfigParser()
    config.read(config_file)

    # This file handles model exports of various types
    # As of now there are three types:
    # 1) Agisoft, *.ply file/DEM and None
    model_export_type = config['General']['model_export_type']

    if model_export_type == 'ply_file':
        # Nothing needs to be done then?
        pass
    elif model_export_type == 'dem_file':
        file_path_dem = config['Absolute Paths']['dem_path']
        file_path_3d_model = config['Absolute Paths']['model_path']
        file_path_geoid = config['Absolute Paths']['geoid_path']

        # Important to check if the DEM is given with respect to 'geoid' or 'ellipsoid'
        try:
            dem_ref = config['Coordinate Reference Systems']['dem_ref']
        except:
            # If entry is non-existent, it is assumed that it is wrt ellipsoid
            dem_ref = 'ellipsoid'
        # Make new only once.

        if os.path.exists(file_path_3d_model):
            print('3D model already exists and overwriting is not supported')
            pass
        else:
            try:
                
                # If for some reason you have better DEM data on a per-transect basis
                # E.g. with range measuring sensors underwater
                if eval(config['General']['dem_per_transect']):
                    dem_folder_parent = Path(config['Absolute Paths']['dem_folder'])

                    # Get all entries (files and directories)
                    all_entries = dem_folder_parent.iterdir()

                    # Filter for directories (excluding '.' and '..')
                    transect_folders = [entry for entry in all_entries if entry.is_dir() and not entry.name.startswith('.')]

                    for transect_folder in transect_folders:
                        transect_folder = str(transect_folder)
                        file_path_3d_model = os.path.join(transect_folder, 'model.vtk')
                        file_path_dem = os.path.join(transect_folder, 'dem.tif')
                        dem_2_mesh(path_dem=file_path_dem, model_path=file_path_3d_model, config=config)
                
                    
                    
            # Only one dem
            except Exception as e:
                if dem_ref == 'geoid':
                    # First add raster from DEM folder with Geoid
                    dem_ref_is_geoid = True
                else:
                    dem_ref_is_geoid = False

                dem_2_mesh(path_dem=file_path_dem, model_path=file_path_3d_model, config=config, dem_ref_is_geoid=dem_ref_is_geoid, path_geoid=file_path_geoid, config_file_path=config_file)

    elif model_export_type == 'geoid':
        file_path_dem = config['Absolute Paths']['dem_path']
        file_path_geoid = config['Absolute Paths']['geoid_path']
        file_path_3d_model = config['Absolute Paths']['model_path']

        # Crop the DEM to appropriate size based on poses and maximum ray length
        crop_geoid_to_pose(path_dem=file_path_dem, config=config, geoid_path=file_path_geoid)

        # Make into a 3D model
        dem_2_mesh(path_dem=file_path_dem, model_path=file_path_3d_model, config=config)







if __name__ == "__main__":
    # Here we could set up necessary steps on a high level. 
    args = sys.argv[1:]
    config_file = args[0]

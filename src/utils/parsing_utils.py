# Python built-ins
import pickle
import time
import sys
import os
import configparser
from os import path

# Third party libraries
import numpy as np
import Metashape as MS
import pandas as pd
from scipy.spatial.transform import Rotation as RotLib
from scipy.spatial.transform import Rotation
import h5py
from Metashape import Vector as vec
from scipy.spatial.transform import Rotation as RotLib
import pandas as pd
import pymap3d as pm
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from pyproj import CRS, Transformer

# Local modules
from utils.geometry_utils import CameraGeometry, GeoPose
from utils.geometry_utils import rot_mat_ned_2_ecef, interpolate_poses
from utils.geometry_utils import dem_2_mesh, crop_geoid_to_pose


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
                is_uhi = config['HDF']['is_uhi']

                
                if eval(is_uhi):
                    self.t_exp = self.f[exposureTimePath][()][0] / 1000  # Recorded in milliseconds
                else:
                    self.t_exp = self.f[exposureTimePath][()] / 1000

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

                # Calculate radiance cube if this is cube is not calibrated
                self.digital_counts_2_radiance(config=config)

            # Check if the dataset exists
            processed_nav_folder = config['HDF.processed_nav']['folder']
            
            # Read the position da
            if processed_nav_folder in self.f:
                try:
                    processed_nav_config = config['HDF.processed_nav']
                    self.pos_ref = self.f[processed_nav_config['position_ecef']][()]
                    self.pos0 = self.f[processed_nav_config['pos0']][()]
                    self.quat_ref = self.f[processed_nav_config['quaternion_ecef']][()]
                    self.pose_time = self.f[processed_nav_config['timestamp']][()]
                except:
                    pass
            



       

            



    def digital_counts_2_radiance(self, config):

        # Only calibrate if data it is not already done
        is_calibrated = eval(config['HDF.hyperspectral']['is_calibrated'])

        if is_calibrated == is_calibrated:
            self.dataCubeRadiance = self.dataCube.astype(np.float32)

            # Add the radiance dataset
            radiance_cube_path = config['HDF.hyperspectral']['dataCube']
        else:
            self.dataCubeRadiance = np.zeros(self.dataCube.shape, dtype = np.float32)
            for i in range(self.dataCube.shape[0]):
                self.dataCubeRadiance[i, :, :] = (self.dataCube[i, :, :] - self.darkFrame) / (
                        self.radiometricFrame * self.t_exp)
            
            # Add the radiance dataset
            radiance_cube_path = config['HDF.hyperspectral']['dataCube'] + '_radiance'
            
            # TODO: Write the radiance data to the h5 file
            Hyperspectral.add_dataset(data = self.dataCubeRadiance, name=radiance_cube_path, h5_filename=self.name)
        
        config.set('HDF.hyperspectral', 'radiance_cube', radiance_cube_path)

        # For memory efficiency
        del self.dataCube

        
            
            
    
    """def add_dataset(self, data, name):
        
        # The h5 file structure can be studied by unravelling the structure in Python or by using HDFview
        with h5py.File(self.name, 'a', libver='latest') as self.f:
            # Check if the dataset exists
            if name in self.f:
                del self.f[name]

            dset = self.f.create_dataset(name=name, data = data)
    """
    
    @staticmethod
    def add_dataset(data, name, h5_filename):
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
            if name in f:
                del f[name]

            dset = f.create_dataset(name=name, data = data)

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

    (xECEF, yECEF, zECEF) = transformer.transform(xx=lat, yy=lon, zz=hei)

    pos_geocsc = np.concatenate((xECEF.reshape((-1,1)), yECEF.reshape((-1,1)), zECEF.reshape((-1,1))), axis = 1)

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

        if i%1000 == 0:
            print(i)

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

def find_desired_chunk(chunkName, chunks, model_name):
    for chunk in chunks:
        if chunk.label in chunkName:
            cameras = chunk.cameras
            if not len(cameras):
                MS.app.messageBox("No camera in this chunk called {0}!".format(chunk.label))
                continue
            chunk.remove(chunk.markers)  # Clear all markers

            return chunk

def agisoft_export_pose(config, config_file):
    # Activating Licence
    print('Activating Licence')
    MS.License().activate("EE2Z6-O5ZVF-1JYNV-NKSRY-UXTGR")
    print('Licence Activated')

    doc = MS.Document()
    doc.read_only = False



    # When exporting, it is relevant to export data in a suitable format
    # CRSExport

    proj_string = config['Coordinate Reference Systems']['ExportEPSG']
    if proj_string != 'Local':
        crs_export = MS.CoordinateSystem(str(proj_string))
        local = False
    else:
        local = True

    # Define *.psx file
    path_psx = config['General']['pathPsx']
    doc.open(path_psx, read_only=False)

    chunk_str = config['General']['chunkName']
    chunkName = [chunk_str]

    # Output. CSV contains labeled roll, pitch, yaw
    csv_name = config['Absolute Paths']['pose_path']
    model_name = config['Absolute Paths']['model_path']

    chunks = doc.chunks

    # Extract the positions of cameras and the orientations
    if path.exists(csv_name):
        print('Pose File Already exists. Overwriting')
    # Find the relevant chunk within the project (with name according to chunkName).

    chunk = find_desired_chunk(chunkName=chunkName, chunks=chunks, model_name=model_name)

    # If CRS is local, adopt it from project
    if local:
        crs_export=chunk.crs

    # Iterate to find right chunk
    for chunk in chunks:
        # Locate the right chunk
        if chunk.label in chunkName:
            cameras = chunk.cameras
            count = 0
            # Mean position is used for mean shifting the 3D model to avoid unnecessarily large coordinates
            mean_pos = np.zeros(3)
            # TODO: Condider replacing Datalogger with Pandas implementation
            # Log data X,Y,Z are in selected geocentric coordinate system (a.k.a. ECEF)
            # Rotations are stored as local orientation (roll, pitch, yaw) and orientation w.r.t. ECEF (rotX...)
            # Encoding, decoding and abstraction of orientations are done with Rotation class from scipy.spatial.transform
            dlogr = DataLogger(csv_name, 'CameraLabel, X, Y, Z, Roll, Pitch, Yaw, RotX, RotY, RotZ')
            if not len(cameras):
                MS.app.messageBox("No camera in this chunk called {0}!".format(chunk.label))
                continue

            # Not sure why markers are removed here?
            chunk.remove(chunk.markers)

            # Chunk transform matrix (arbitrary and no transform if local CRS)
            T = chunk.transform.matrix

            # Global CRS is inherited from MS projects
            ecef_crs = chunk.crs.geoccs

            # Certain UHI projects use local/engineering CRS
            if ecef_crs is None:
                ecef_crs = MS.CoordinateSystem('LOCAL')
            for cam in cameras:
                # If the camera is aligned do something
                if cam.center != None: # Ugly condition formulation

                    # T describes the transformation from the arbitrary chunk crs to ECEF
                    # Cam center local describes camera origin in chunk crs
                    cam_center_local = cam.center
                    cam_center_global = T.mulp(cam_center_local)

                    # Position is projected onto projected geographic coordinates that is used in
                    pos = crs_export.project(cam_center_global)

                    # Not sure about this stuff
                    if np.max(np.abs(pos)) > 1000:
                        mean_pos[0] += pos[0]
                        mean_pos[1] += pos[1]
                        mean_pos[2] += pos[2]

                    count += 1


                    m = chunk.crs.localframe(cam_center_global)
                    # m is a transformation matrix at the point of

                    T_cam_chunk = cam.transform  # Transformation from cam to chunk euclidian space CES
                    T_chunk_world = T  # Transformation from chunk euclidian space to ECEF

                    cam_transform = T_chunk_world * T_cam_chunk

                    location_ecef = cam_transform.translation()  # Location in ECEF

                    rotation_cam_ecef = cam_transform.rotation()  # Rotation of camera with respect to ECEF

                    # Rotation from ECEF to ENU. This will be identity for local coordinate definitions
                    rotation_ecef_enu = ecef_crs.localframe(location_ecef).rotation()

                    rotation_cam_enu = rotation_ecef_enu * rotation_cam_ecef

                    pos = MS.CoordinateSystem.transform(location_ecef, source=ecef_crs, target=crs_export)

                    R_cam_ned = rotation_cam_enu * MS.Matrix().Diag([1, -1, -1])
                    R_cam_ecef = rotation_cam_ecef

                    # Extracting roll-pitch-yaw is a bit wierd due to axis conventions
                    r_scipy = Rotation.from_matrix(np.asarray(R_cam_ned).reshape((3, 3)))
                    yaw, pitch, roll = r_scipy.as_euler("ZXY", degrees=True)

                    # ECEF rotations are preferred
                    r_scipy_ecef = Rotation.from_matrix(np.asarray(R_cam_ecef).reshape((3, 3)))
                    rotz, roty, rotx = r_scipy_ecef.as_euler("ZYX", degrees=True)

                    # We append
                    dlogr.append_data([cam.label, pos[0], pos[1], pos[2], roll, pitch, -yaw, rotx, roty, rotz])

    mean_pos /= count

    config.set('General', 'offset_x', str(mean_pos[0]))
    config.set('General', 'offset_y', str(mean_pos[1]))
    config.set('General', 'offset_z', str(mean_pos[2]))

    # Exporting models with offsets might be convenient
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    return config

def agisoft_export_model(config_file):
    """
            Exports a model from an agisoft (metashape) project

            Args:
                config_file: The *.ini file containing the paths and configurations.

            Returns:
                config: The function returns config object to allow that it is modified.
        """
    MS.License().activate("XXXX-XXXX-XXXX-XXXX-XXXX")
    doc = MS.Document()
    doc.read_only = False

    config = configparser.ConfigParser()
    config.read(config_file)

    # When exporting, it is relevant to export data in a suitable format
    # CRSExport

    proj_string = 'EPSG::' + config['Coordinate Reference Systems']['ExportEPSG']
    if proj_string != 'Local':
        crs_export = MS.CoordinateSystem(str(proj_string))
        local = False
    else:
        local = True

    # Define *.psx file
    path_psx = config['General']['pathPsx']
    doc.open(path_psx, read_only=False)
    
    chunk_str = config['General']['chunkName']
    chunkName = [chunk_str]

    # Output.
    model_name = config['Absolute Paths']['model_path']

    chunks = doc.chunks
    # Extract camera model from project

    # Exporting models with offsets might be convenient
    offset_x = float(config['General']['offset_x'])
    offset_y = float(config['General']['offset_y'])
    offset_z = float(config['General']['offset_z'])

    # Extract the 3D model and texture file
    if path.exists(model_name):
        print('Model File Already exists. Overwriting')
    chunk = find_desired_chunk(chunkName=chunkName, chunks=chunks, model_name=model_name)
    if local:
        chunk.exportModel(path=model_name, crs=chunk.crs, shift=vec((offset_x, offset_y, offset_z)), save_texture = True, texture_format = MS.ImageFormat.ImageFormatPNG)
        
    else:
        chunk.exportModel(path=model_name, crs=crs_export, shift=vec((offset_x, offset_y, offset_z)), save_texture = True, texture_format = MS.ImageFormat.ImageFormatPNG)

def append_agisoft_data_h5(config):
    """
            Parses pose and embeds it into the *.h5 file. Writes the positions and orientations of hyperspectral frames to
                dataset under the /Nav folder

            Args:
                config_file: The *.ini file containing the paths and configurations.

            Returns:
                None: The function returns nothing.
    """
    # Retrieve the pose file (per RGB image, with RGB image name tags)
    filename_pose = config['Absolute Paths']['pose_path']
    pose = pd.read_csv(filename_pose, sep=',', header=0)

    # Offsets should correspond with polygon model offset
    off_x = float(config['General']['offset_x'])
    off_y = float(config['General']['offset_y'])
    off_z = float(config['General']['offset_z'])
    pos0 = np.array([off_x, off_y, off_z]).reshape([-1, 1])


    # Traverse through h5 dir to append the data to file
    h5_folder = config['Absolute Paths']['h5_folder']
    for filename in sorted(os.listdir(h5_folder)):
        # Find the interesting prefixes
        if filename.endswith('h5') or filename.endswith('hdf'):
            # Identify the total path
            path_hdf = h5_folder + filename

            # Read out the data of a file
            hyp = Hyperspectral(path_hdf, config)

            # Split file names
            filename_splitted = filename.split('_')
            transect_string = filename_splitted[2] + '_' + filename_splitted[3].split('.')[0]
            print(transect_string)
            # Interpolate
            # Select the images wiht a label containing the string
            relevant_images = pose.iloc[:, 0].str.contains(transect_string)
            ind_vec = np.arange(len(relevant_images))

            # The last part of the camera label holds the index
            rgb_ind = [int(pose["CameraLabel"].str.split("_")[i][-1]) for i in
                            ind_vec[relevant_images == True]]

            # The images containing the string
            agisoft_ind = ind_vec[relevant_images == True]

            # The relevant image indices (based on which images were matched during alignment)
            
            # 0-CameraLabel, 1-X, 2-Y, 3-Z, 4-Roll, 5-Pitch, 6-Yaw, 7-RotX, 8-RotY, 9-RotZ
            agisoft_poses = pose.values[agisoft_ind, 1:10]

            # Selects timestamps for RGB frames
            time_rgb = hyp.RGBTimeStamps[rgb_ind]

            # Selects timestamps for HSI frames
            time_hsi = hyp.dataCubeTimeStamps

            # Interpolate for the selected time stamps.
            # First find the time stamps within the RGB time stamps to avoid extrapolation

            minRGB = np.min(time_rgb)
            maxRGB = np.max(time_rgb)

            minInd = np.argmin(np.abs(minRGB - time_hsi))
            maxInd = np.argmin(np.abs(maxRGB - time_hsi))
            if time_hsi[minInd] < minRGB:
                minInd += 1
            if time_hsi[maxInd] > maxRGB:
                maxInd -= 1

            rotZXY = np.concatenate((agisoft_poses[:, 8].reshape((-1, 1)),
                                     agisoft_poses[:, 7].reshape((-1, 1)),
                                     agisoft_poses[:, 6].reshape((-1, 1))), axis=1)

            rot_obj = RotLib.from_euler("ZYX", rotZXY, degrees=True)
            # Establish a camera geometry object
            RGBCamera = CameraGeometry(pos0=pos0, pos=agisoft_poses[:, 0:3], rot=rot_obj, time=time_rgb)

            # A method of the object is interpolation
            RGBCamera.interpolate(time_hsi=time_hsi, minIndRGB=minInd, maxIndRGB=maxInd,
                                  extrapolate=True)

            # Add camera position
            position_hsi = RGBCamera.position_nav_interpolated


            # Add camera position
            

            # Add camera quaternion
            

            # Add time stamps
            
            
            position_ref_name = config['HDF.processed_nav']['position_ecef']
            hyp.add_dataset(data=position_hsi, name=position_ref_name)

            # Add camera quaternion
            quaternion_hsi = RGBCamera.rotation_nav_interpolated.as_quat()
            quaternion_ref_name = config['HDF.processed_nav']['quaternion_ecef']
            hyp.add_dataset(data=quaternion_hsi, name=quaternion_ref_name)

            # Add time stamp
            time_stamps = RGBCamera.time_hsi  # Use projected system for global description
            time_stamps_name = config['HDF.processed_nav']['timestamp']
            hyp.add_dataset(data=time_stamps, name=time_stamps_name)


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
                eul_is_degrees = eval(config['HDF.raw_nav']['eul_is_degrees'])
                print(eul_is_degrees)
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
            position_interpolated, quaternion_interpolated = interpolate_poses(timestamp_from=timestamps_imu,
                                                             pos_from=position_ref,
                                                             pos0=None,
                                                             rot_from=rot_obj,
                                                             timestamps_to=timestamp_hsi,
                                                             use_absolute_position = True)

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
                                   rot_obj= rot_interpolated, rot_ref=rot_ref,
                                   pos = position_interpolated, pos_epsg=pos_epsg_orig)

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

            if is_first:
                # Calculate some appropriate offsets in ECEF
                pos0 = np.mean(position_ref_ecef, axis=0)

                # For stacking in CSV
                rot_vec_ecef = geo_pose_ref.rot_obj_ecef.as_euler('ZYX', degrees=True)

                # 
                total_pose = np.concatenate((timestamp_hsi.reshape((-1,1)), position_ref_ecef, rot_vec_ecef), axis=1)
                # Ensure that flag is not raised again. Offsets should only be set once.
                is_first = False
            else:
                # Ensure that flag is not raised again. Offsets should only be set once.
                is_first = False
                rot_vec_ecef = geo_pose_ref.rot_obj_ecef.as_euler('ZYX', degrees=True)
                partial_pose = np.concatenate((timestamp_hsi.reshape((-1,1)), position_ref_ecef, rot_vec_ecef), axis=1)
                total_pose = np.concatenate((total_pose, partial_pose), axis=0)
            
            position_offset_name = config['HDF.processed_nav']['pos0']
            Hyperspectral.add_dataset(data=pos0, name=position_offset_name, h5_filename=path_hdf)

            # Add camera position
            position_ref_name = config['HDF.processed_nav']['position_ecef']
            Hyperspectral.add_dataset(data=position_ref_ecef - pos0, name=position_ref_name, h5_filename=path_hdf)

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
    # Save the DataFrame as a CSV file
    df.to_csv(pose_path, index=False)

    config.set('General', 'offset_x', str(pos0[0]))
    config.set('General', 'offset_y', str(pos0[1]))
    config.set('General', 'offset_z', str(pos0[2]))
    # Exporting models with offsets might be convenient
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    return config

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

    if pose_export_type == 'agisoft':
        # Just take in use the already existing parser
        config = agisoft_export_pose(config = config, config_file=config_file)
        # This gives a csv file with RGB poses, but they should be appended to each H5 file.
        append_agisoft_data_h5(config=config)

    elif pose_export_type == 'h5_embedded':
        config = reformat_h5_embedded_data_h5(config=config,
                                              config_file=config_file)
    elif pose_export_type == 'independent_file':
        print('There is no support for this export functionality! Fix immediately!')
        config = -1
    else:
        print('File type: ' + pose_export_type + 'type is not defined!')
        config = -1

    return config
    
    
def export_model(config_file):
    """"""
    config = configparser.ConfigParser()
    config.read(config_file)

    # This file handles model exports of various types
    # As of now there are three types:
    # 1) Agisoft, *.ply file/DEM and None
    model_export_type = config['General']['model_export_type']

    if model_export_type == 'agisoft':
        agisoft_export_model(config_file=config_file)
    elif model_export_type == 'ply_file':
        # Nothing needs to be done then?
        pass
    elif model_export_type == 'dem_file':
        file_path_dem = config['Absolute Paths']['dem_path']
        file_path_3d_model = config['Absolute Paths']['model_path']
        # Make new only once.

        dem_2_mesh(path_dem=file_path_dem, model_path=file_path_3d_model, config=config)

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

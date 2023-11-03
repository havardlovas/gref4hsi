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
from scripts.geometry import CameraGeometry, GeoPose
from scripts.geometry import rot_mat_ned_2_ecef, interpolate_poses
from scripts.gis_tools import dem_2_mesh, position_transform_ecef_2_llh


class Hyperspectral:
    """
    Class for storing/accessing acquired hyperspectral data. One instance corresponds to a transect chunk (*.h5).
    """
    def __init__(self, filename, config):
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
            # Special case of an old UHI format from 2017.
            # TODO: Embed the paths from this special format into confic file
            if eval(config['HDF']['isMarmineUHISpring2017']):
                # The raw data cube
                self.dataCube = self.f['uhi/pixels'][()]

                # Exposure time of HSI
                self.t_exp = self.f['uhi/parameters'][()][0, 0] / 1000

                # Time stamps of line scans
                self.dataCubeTimeStamps = self.f['uhi/parameters'][()][:, 2]

                # Correspondence describing the wavelength corresponding to each band
                self.band2Wavelength = self.f['uhi/calib'][()]

                # The time stamps corresponding to each RGB image
                self.RGBTimeStamps = self.f['rgb/parameters'][()][:, 2]

                # The actual RGB images
                # TODO: Make a class method for writing these to *.png files with apropriate name conventions. Including Exif tagging
                # TODO: Make a class method for performing grey-world correction of images
                self.RGBImgs = self.f['rgb/pixels'][()]

                # Check if the dataset exists
                if 'georeference' in self.f:
                    dir = 'georeference/'
                    self.normals_local = self.f[dir + 'normals_local'][()]
                    self.points_global = self.f[dir + 'points_global'][()]
                    self.points_local = self.f[dir + 'points_local'][()]
                    self.position_hsi = self.f[dir + 'position_hsi'][()]
                    self.quaternion_hsi = self.f[dir + 'quaternion_hsi'][()]
                # Check if the dataset exists
                if 'nav' in self.f:
                    dir = 'nav/'
                    self.pos_rgb = self.f[dir + 'position_rgb'][()]
                    self.quat_rgb = self.f[dir + 'quaternion_rgb'][()]
                    self.pose_time = self.f[dir + 'time_stamp'][()]

            else:
                dataCubePath = config['HDF.hyperspectral']['dataCube']
                exposureTimePath = config['HDF.hyperspectral']['exposureTime']
                timestampHyperspectralPath = config['HDF.hyperspectral']['timestamp']
                band2WavelengthPath = config['HDF.calibration']['band2Wavelength']
                radiometricFramePath = config['HDF.calibration']['radiometricFrame']
                darkFramePath = config['HDF.calibration']['darkFrame']
                RGBFramesPath = config['HDF.rgb']['rgbFrames']
                timestampRGBPath = config['HDF.rgb']['timestamp']

                self.dataCube = self.f[dataCubePath][()]

                try:
                    self.t_exp = self.f[exposureTimePath][()][0] / 1000  # Recorded in milliseconds
                except KeyError:
                    pass
                self.dataCubeTimeStamps = self.f[timestampHyperspectralPath][()]
                self.band2Wavelength = self.f[band2WavelengthPath][()]
                try:
                    self.darkFrame = self.f[darkFramePath][()]
                    self.radiometricFrame = self.f[radiometricFramePath][()]
                except KeyError:
                    pass


                try:
                    self.RGBTimeStamps = self.f[timestampRGBPath][()]
                    self.RGBImgs = self.f[RGBFramesPath][()]
                except (KeyError, TypeError):
                    pass


                # Check if the dataset exists
                #if 'georeference' in self.f:
                #    dir_geo = 'georeference/'
                #    self.normals_local = self.f[dir_geo + 'normals_local'][()]
                #    self.points_global = self.f[dir_geo + 'points_global'][()]
                #    self.points_local = self.f[dir_geo + 'points_local'][()]
                #    self.position_hsi = self.f[dir_geo + 'position_hsi'][()]
                #    self.quaternion_hsi = self.f[dir_geo + 'quaternion_hsi'][()]
#
                #if 'nav' in self.f:
                #    # Navigation data for the reference (Typically IMU or RGB cam)
                #    dir_nav = 'nav/'
                #    self.pos_rgb = self.f[dir_nav + 'position_rgb'][()]
                #    self.quat_rgb = self.f[dir_nav + 'quaternion_rgb'][()]
                #    self.pose_time = self.f[dir_nav + 'time_stamp'][()]





        self.n_scanlines = self.dataCube.shape[0]
        self.n_pix = self.dataCube.shape[1]
        self.n_bands = self.dataCube.shape[1]
        self.n_imgs = self.RGBTimeStamps.shape[0]

        # The maximal image size is typically 1920 by 1200. It is however cropped during recording
        # And there can be similar issues. The binning below is somewhat heuristic but general
        if self.n_pix > 1000:
            self.spatial_binning = 0
        elif self.n_pix > 500:
            self.spatial_binning = 1
        elif self.n_pix > 250:
            self.spatial_binning = 2

        if self.n_bands  > 600:
            self.spectral_binning = 0
        elif self.n_bands  > 250:
            self.spectral_binning = 1
        elif self.n_bands  > 125:
            self.spectral_binning = 2



    def digital_counts_2_radiance(self, config):

        # Special case
        if eval(config['HDF']['isMarmineUHISpring2017']):
            calibFolder = config['HDF']['calibFolder']
            if self.dataCube.shape[1] == 480:
                self.radiometricFrame = np.load(calibFolder + '/radiometricFrame480.npy')
                self.darkFrame = np.load(calibFolder + '/darkFrame480.npy')
                self.spatial_binning = 2
            elif self.dataCube.shape[1] == 960:
                self.radiometricFrame = np.load(calibFolder + '/radiometricFrame.npy')
                self.darkFrame = np.load(calibFolder + '/darkFrame.npy')
                self.spatial_binning = 1




        self.dataCubeRadiance = np.zeros(self.dataCube.shape, dtype = np.float64)
        for i in range(self.dataCube.shape[0]):
            self.dataCubeRadiance[i, :, :] = (self.dataCube[i, :, :] - self.darkFrame) / (
                    self.radiometricFrame * self.t_exp)

    def add_dataset(self, data, name):
        """
        Method to write a dataset to the h5 file
        :param data: type any permitted (see h5py doc)
        The data to be written
        :param name: string
        The path/name of the dataset
        :return: None
        """
        # The h5 file structure can be studied by unravelling the structure in Python or by using HDFview
        with h5py.File(self.name, 'a', libver='latest') as self.f:
            # Check if the dataset exists
            if name in self.f:
                del self.f[name]

            dset = self.f.create_dataset(name=name, data = data)

    def get_dataset(self, dataset_name):
        """
        Method to return a dataset by the name
        :param dataset_name: string
        The h5 dataset path
        :return: dataset
        Returns the queried dataset. Could be many datatypes, but mainly numpy arrays in our usage.
        """
        # The h5 file structure can be studied by unravelling the structure in Python or by using HDFview
        with h5py.File(self.name, 'a', libver='latest') as self.f:
            dataset = self.f[dataset_name][()]
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
    
    :param config: 
    :param iniPath: 
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
    PositionInterpolated = np.transpose(linearPositionInterpolator(time_rot))

    pose_path = config['General']['ardupath'] + 'pose.csv'

    config.set('General', 'posepath', pose_path)

    epsg_proj = 4326 # Standard
    epsg_geocsc = config['General']['modelepsg']
    # Transform the mesh points to ECEF.

    geocsc = CRS.from_epsg(epsg_geocsc)
    proj = CRS.from_epsg(epsg_proj)
    transformer = Transformer.from_crs(proj, geocsc)

    points_proj = PositionInterpolated

    lat = points_proj[:, 0].reshape((-1, 1))
    lon = points_proj[:, 1].reshape((-1, 1))
    hei = points_proj[:, 2].reshape((-1, 1))

    (xECEF, yECEF, zECEF) = transformer.transform(xx=lat, yy=lon, zz=hei)

    pos_geocsc = np.concatenate((xECEF.reshape((-1,1)), yECEF.reshape((-1,1)), zECEF.reshape((-1,1))), axis = 1)

    #dlogr = DataLogger(pose_path, 'CameraLabel, X, Y, Z, Roll, Pitch, Yaw, RotX, RotY, RotZ')



    data_matrix = np.zeros((len(time_rot), 10))
    for i in range(len(time_rot)):

        # Ardupilot delivers geodetic positions (EPSG::
        pos_geod = PositionInterpolated[i, :]

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
    csv_name = config['General']['posePath']
    model_name = config['General']['modelPath']

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

    config.set('General', 'offsetX', str(mean_pos[0]))
    config.set('General', 'offsetY', str(mean_pos[1]))
    config.set('General', 'offsetZ', str(mean_pos[2]))

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
    MS.License().activate("EE2Z6-O5ZVF-1JYNV-NKSRY-UXTGR")
    doc = MS.Document()
    doc.read_only = False

    config = configparser.ConfigParser()
    config.read(config_file)

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

    # Output.
    model_name = config['General']['modelPath']

    chunks = doc.chunks
    # Extract camera model from project

    # Exporting models with offsets might be convenient
    offsetX = float(config['General']['offsetX'])
    offsetY = float(config['General']['offsetY'])
    offsetZ = float(config['General']['offsetZ'])

    # Extract the 3D model and texture file
    if path.exists(model_name):
        print('Model File Already exists. Overwriting')
    chunk = find_desired_chunk(chunkName=chunkName, chunks=chunks, model_name=model_name)
    if local:
        chunk.exportModel(path=model_name, crs=chunk.crs, shift=vec((offsetX, offsetY, offsetZ)))
    else:
        chunk.exportModel(path=model_name, crs=crs_export, shift=vec((offsetX, offsetY, offsetZ)))

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
    filenamePose = config['General']['posePath']
    pose = pd.read_csv(filenamePose, sep=',', header=0)

    # Offsets should correspond with polygon model offset
    offX = float(config['General']['offsetX'])
    offY = float(config['General']['offsetY'])
    offZ = float(config['General']['offsetZ'])
    pos0 = np.array([offX, offY, offZ]).reshape([-1, 1])


    # Traverse through h5 dir to append the data to file
    h5dir = config['Absolute Paths']['h5dir']
    for filename in sorted(os.listdir(h5dir)):
        # Find the interesting prefixes
        if filename.endswith('h5') or filename.endswith('hdf'):
            # Identify the total path
            path_hdf = h5dir + filename

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
            position_hsi = RGBCamera.PositionInterpolated
            position_hsi_name = 'nav/position_rgb'
            hyp.add_dataset(data=position_hsi, name=position_hsi_name)

            # Add camera quaternion
            quaternion_hsi = RGBCamera.RotationInterpolated.as_quat()
            quaternion_hsi_name = 'nav/quaternion_rgb'
            hyp.add_dataset(data=quaternion_hsi, name=quaternion_hsi_name)

            # Add time stamp
            time_stamps = RGBCamera.time_hsi  # Use projected system for global description
            time_stamps_name = 'nav/time_stamp'
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

    # Offsets should correspond with polygon model offset
    offX = float(config['General']['offsetX'])
    offY = float(config['General']['offsetY'])
    offZ = float(config['General']['offsetZ'])
    if offX == -1:
        # Set pos0 to None to ensure that error is raised if it is used for anything
        pos0 = None

    # Traverse through h5 dir to append the data to file
    h5dir = config['Absolute Paths']['h5dir']
    is_first = True
    for filename in sorted(os.listdir(h5dir)):
        # Find the interesting prefixes
        if filename.endswith('h5') or filename.endswith('hdf'):
            # Identify the total path
            path_hdf = h5dir + filename

            # Read out the data of a file
            hyp = Hyperspectral(path_hdf, config)

            quaternion_ref = hyp.get_dataset(dataset_name=config['HDF.raw_nav']['quaternion'])
            if quaternion_ref.shape[0] == 4:
                quaternion_ref = np.transpose(quaternion_ref)
            is_global_rot = config['HDF.raw_nav']['is_global_rot']

            # Parse position
            position_ref = hyp.get_dataset(dataset_name=config['HDF.raw_nav']['position_ecef'])\
                .reshape((-1,3))
            if position_ref.shape[0] == 3:
                position_ref = np.transpose(position_ref)


            timestamps_imu = hyp.get_dataset(dataset_name=config['HDF.raw_nav']['timestamp'])\
                .reshape(-1)







            # The rotation of the reference
            rot_obj = RotLib.from_quat(quaternion_ref)

            timestamp_hsi = hyp.dataCubeTimeStamps\
                .reshape(-1)


            # Compute absolute positions positions and orientation:
            position_interpolated, quaternion_interpolated = interpolate_poses(timestamp_from=timestamps_imu,
                                                             pos_from=position_ref,
                                                             pos0=pos0,
                                                             rot_from=rot_obj,
                                                             timestamps_to=timestamp_hsi,
                                                             use_absolute_position = True)


            # If the original orientations are with respect to NED
            if not is_global_rot:
                rot_ref = 'NED'
            else:
                rot_ref = 'ECEF'

            # The positions supplied for the reference
            pos_epsg_orig = config['General']['pos_epsg_orig']

            # The positions supplied for exporting the model
            pos_epsg_export = config['Coordinate Reference Systems']['exportepsg']

            rot_interpolated = RotLib.from_quat(quaternion_interpolated)
            geo_pose_ref = GeoPose(timestamps=timestamp_hsi,
                                   rot_obj= rot_interpolated, rot_ref=rot_ref,
                                   pos = position_interpolated, pos_epsg=pos_epsg_orig)

            # Convert position to the epsg used for the 3D model
            geo_pose_ref.compute_geocentric_position(epsg_geocsc=pos_epsg_export)


            # calculate the geodetic position using the WGS-84 (for conversion of orientations)
            epsg_wgs84 = 4326
            geo_pose_ref.compute_geodetic_position(epsg_geod=epsg_wgs84)

            # calculate ECEF orientation
            geo_pose_ref.compute_geocentric_orientation()

            # For readability
            position_ref_ecef = geo_pose_ref.pos_geocsc

            quaternion_ref_ecef = geo_pose_ref.rot_obj_ecef.as_quat()

            condition_for_setting_offsets = (offX == -1 and is_first)

            if condition_for_setting_offsets:
                # Ensure that flag is not raised again. Offsets should only be set once.
                is_first = False
                # Calculate some appropriate offsets in ECEF
                pos0 = np.mean(position_ref_ecef, axis=0)



        # Add camera position
        position_ref_name = config['HDF.processed_nav']['quaternion_ref_ecef']
        hyp.add_dataset(data=position_ref_ecef, name=position_ref_name)

        # Add camera quaternion
        quaternion_ref_name = config['HDF.processed_nav']['quaternion_ref_ecef']
        hyp.add_dataset(data=quaternion_ref_ecef, name=quaternion_ref_name)

        # Add time stamps
        time_stamps_name = config['HDF.processed_nav']['timestamp_hsi']
        hyp.add_dataset(data=timestamp_hsi, name=time_stamps_name)

    config.set('General', 'offsetX', str(pos0[0]))
    config.set('General', 'offsetY', str(pos0[1]))
    config.set('General', 'offsetZ', str(pos0[2]))

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
    poseExportType = config['General']['poseExportType']

    if poseExportType == 'agisoft':
        # Just take in use the already existing parser
        config = agisoft_export_pose(config = config, config_file=config_file)
        # This gives a csv file with RGB poses, but they should be appended to each H5 file.
        append_agisoft_data_h5(config=config)

    elif poseExportType == 'h5_embedded':
        config = reformat_h5_embedded_data_h5(config=config,
                                              config_file=config_file)
    elif poseExportType == 'independent_file':
        print('There is no support for this export functionality! Fix immediately!')
        config = -1
    else:
        print('File type: ' + poseExportType + 'type is not defined!')
        config = -1

    return config
    
    
    
    
if __name__ == "__main__":
    # Here we could set up necessary steps on a high level. 
    args = sys.argv[1:]
    config_file = args[0]

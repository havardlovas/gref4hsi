

import Metashape as MS

from Metashape import Vector as vec
import pickle
import time
import sys
import numpy as np
from os import path
import configparser
from scipy.spatial.transform import Rotation


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


def extract_pose_MS(chunkName, chunks, csv_name, crs_export, config):
    # Define the data logger:

    # crs_export is a coordinate system based on configurations
    for chunk in chunks:

        if chunk.label in chunkName:
            cameras = chunk.cameras
            count = 0
            mean_pos = np.zeros(3)
            dlogr = DataLogger(csv_name, 'CameraLabel, X, Y, Z, Roll, Pitch, Yaw, RotX, RotY, RotZ')
            if not len(cameras):
                MS.app.messageBox("No camera in this chunk called {0}!".format(chunk.label))
                continue
            chunk.remove(chunk.markers)  # Clear all markers
            T = chunk.transform.matrix

            ecef_crs = chunk.crs.geoccs
            if ecef_crs is None:
                ecef_crs = MS.CoordinateSystem('LOCAL')
            for cam in cameras:

                if cam.center != None:

                    # T describes the transformation from the arbitrary crs to the


                    cam_center_local = cam.center
                    cam_center_global = T.mulp(cam_center_local)


                    # Position is projected onto projected geographic coordinates that is used in
                    pos = crs_export.project(cam_center_global)

                    if np.max(np.abs(pos)) > 1000:
                        mean_pos[0] += pos[0]
                        mean_pos[1] += pos[1]
                        mean_pos[2] += pos[2]

                    count += 1
                    # Now it is in the crs_export

                    m = chunk.crs.localframe(cam_center_global)
                    # m is a transformation matrix at the point of

                    T_cam_chunk = cam.transform # Transformation from cam to chunk euclidian space CES
                    T_chunk_world = T # Transformation from chunk euclidian space to ECEF

                    cam_transform = T_chunk_world*T_cam_chunk

                    location_ecef = cam_transform.translation() # Location in ECEF

                    rotation_cam_ecef = cam_transform.rotation() # Rotation of camera with respect to ECEF

                    # Rotation from ECEF to ENU. This will be identity for local coordinate definitions
                    rotation_ecef_enu = ecef_crs.localframe(location_ecef).rotation()

                    rotation_cam_enu = rotation_ecef_enu * rotation_cam_ecef

                    pos = MS.CoordinateSystem.transform(location_ecef, source=ecef_crs, target=crs_export)

                    R_cam_ned = rotation_cam_enu*MS.Matrix().Diag([1, -1, -1])
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
    return config


def extract_model_MS(chunkName, chunks, model_name):
    for chunk in chunks:
        if chunk.label in chunkName:
            cameras = chunk.cameras
            if not len(cameras):
                MS.app.messageBox("No camera in this chunk called {0}!".format(chunk.label))
                continue
            chunk.remove(chunk.markers)  # Clear all markers

            return chunk


def agisoft_export(iniPath):
    MS.License().activate("EE2Z6-O5ZVF-1JYNV-NKSRY-UXTGR")
    doc = MS.Document()
    doc.read_only = False

    config = configparser.ConfigParser()
    config.read(iniPath)

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

    if local:
        config = extract_pose_MS(chunkName=chunkName, chunks=chunks, csv_name=csv_name, crs_export=chunk.crs, config = config)
    else:
        config = extract_pose_MS(chunkName=chunkName, chunks=chunks, csv_name=csv_name, crs_export=crs_export, config = config)

    # Extract camera model from project

    # Exporting models with offsets might be convenient
    offsetX = float(config['General']['offsetX'])
    offsetY = float(config['General']['offsetY'])
    offsetZ = float(config['General']['offsetZ'])
    with open(iniPath, 'w') as configfile:
        config.write(configfile)


    # Extract the 3D model and texture file
    if path.exists(model_name):
        print('Model File Already exists. Overwriting')
    chunk = extract_model_MS(chunkName=chunkName, chunks=chunks, model_name=model_name)
    if local:
        chunk.exportModel(path=model_name, crs=chunk.crs, shift=vec((offsetX, offsetY, offsetZ)))
    else:
        chunk.exportModel(path=model_name, crs=crs_export, shift=vec((offsetX, offsetY, offsetZ)))


def agisoft_export_pose(iniPath):
    MS.License().activate("EE2Z6-O5ZVF-1JYNV-NKSRY-UXTGR")
    doc = MS.Document()
    doc.read_only = False

    config = configparser.ConfigParser()
    config.read(iniPath)

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

    if local:
        config = extract_pose_MS(chunkName=chunkName, chunks=chunks, csv_name=csv_name, crs_export=chunk.crs,
                                 config=config)
    else:
        config = extract_pose_MS(chunkName=chunkName, chunks=chunks, csv_name=csv_name, crs_export=crs_export,
                                 config=config)

    # Exporting models with offsets might be convenient
    with open(iniPath, 'w') as configfile:
        config.write(configfile)
    return config

def agisoft_export_model(iniPath):
    MS.License().activate("EE2Z6-O5ZVF-1JYNV-NKSRY-UXTGR")
    doc = MS.Document()
    doc.read_only = False

    config = configparser.ConfigParser()
    config.read(iniPath)

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
    chunk = extract_model_MS(chunkName=chunkName, chunks=chunks, model_name=model_name)
    if local:
        chunk.exportModel(path=model_name, crs=chunk.crs, shift=vec((offsetX, offsetY, offsetZ)))
    else:
        chunk.exportModel(path=model_name, crs=crs_export, shift=vec((offsetX, offsetY, offsetZ)))
if __name__ == "__main__":
    args = sys.argv[1:]
    iniPath = args[0]
    # Running parsing_utils is per now the same as running agisoft_extract
    agisoft_export(iniPath)

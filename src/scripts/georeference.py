import configparser
import os
import pickle
import sys
from scipy.spatial.transform import Rotation as RotLib
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import interp1d

from geometry import CameraGeometry, FeatureCalibrationObject, CalibHSI
from gis_tools import GeoSpatialAbstractionHSI
from parsing_utils import Hyperspectral


class RayCasting:
    def __init__(self, config):
        self.config = config
        # Offsets should correspond with polygon model offset
        self.offX = float(config['General']['offsetX'])
        self.offY = float(config['General']['offsetY'])
        self.offZ = float(config['General']['offsetZ'])
        self.pos0 = np.array([self.offX, self.offY, self.offZ]).reshape([-1, 1])


        self.hyp = []
        self.upsample_factor = int(config['Georeferencing']['upsamplingFactor'])
        self.step_imgs = int(config['Calibration']['stepImages'])

        self.max_ray_length = float(config['General']['maxRayLength'])

    def loadSeaBed(self, filenameSeaBed, filenameTexture = None):  # Loading Mesh in *.ply format
        
        if filenameTexture == None:
            self.mesh = pv.read(filenameSeaBed)
        else:
            print('Consult georeference_drone_imagery_multi_ray.py to see more about including a texture image')


    def load_cam_calibration(self, filenameCal, config):
        # See paper by Sun, Bo, et al. "Calibration of line-scan cameras for precision measurement." Applied optics 55.25 (2016): 6836-6843.
        # Loads line camera parameters for the hyperspectral imager from an xml file.

        # Certain imagers deliver geometry "per pixel". This can be resolved by fitting model parameters.
        calHSI = CalibHSI(file_name_cal=filenameCal, config = config)
        self.f = calHSI.f
        self.v_c = calHSI.cx

        # Radial distortion parameters
        self.k1 = calHSI.k1
        self.k2 = calHSI.k2

        # Tangential distortion parameters
        self.k3 = calHSI.k3

        # Translation (lever arm) of HSI with respect to vehicle frame
        self.trans_x = calHSI.tx
        self.trans_y = calHSI.ty
        self.trans_z = calHSI.tz

        # Rotation of HSI (boresight) with respect to vehicle frame
        self.rot_x = calHSI.rx
        self.rot_y = calHSI.ry
        self.rot_z = calHSI.rz

    def init_hsi_rays(self):

        # Define camera model array for 960 pixels and 480.
        self.v_not_binned = np.arange(1, self.hyp.n_pix*self.hyp.spatial_binning + 1)

        if self.hyp.spatial_binning != 1:
            self.v = np.mean(self.v_not_binned.reshape((-1, self.hyp.spatial_binning)), axis = 1)
        else:
            self.v = self.v_not_binned

        # Express uhi ray directions in uhi frame using line-camera model
        x_norm_lin = (self.v - self.v_c) / self.f

        x_norm_nonlin = -(self.k1 * ((self.v - self.v_c) / 1000) ** 5 + \
                          self.k2 * ((self.v - self.v_c) / 1000) ** 3 + \
                          self.k3 * ((self.v - self.v_c) / 1000) ** 2) / self.f

        self.x_norm = x_norm_lin + x_norm_nonlin

        self.p_dir = np.zeros((len(self.x_norm), 3))

        # Rays are defined in the UHI frame with positive z down
        self.p_dir[:, 0] = self.x_norm
        self.p_dir[:, 2] = 1

        self.HSICamera = self.RGBCamera

        rot_hsi_rgb = np.array([self.rot_z, self.rot_y, self.rot_x]) * 180/np.pi
        translation_hsi_rgb = np.array([self.trans_x, self.trans_y, self.trans_z]) / 1000 # These are millimetres

        self.HSICamera.intrinsicTransformHSI(translation_rgb_hsi=-translation_hsi_rgb, rot_hsi_rgb= rot_hsi_rgb)

        self.HSICamera.defineRayDirections(dir_local=self.p_dir)

        #import visualize
        #visualize.show_camera_geometry(self.HSICamera, self.config)


    def update_camera_model_param(self, param):
        # For the different binning, a slightly different v_c and focal length was applied
        self.rot_x = param[0]  # Pitch relative to cam (Equivalent to cam defining NED and uhi BODY)
        self.rot_y = param[1]  # Roll relative to cam
        self.rot_z = param[2]  # Yaw relative to cam
        self.v_c = param[3]
        self.f = param[4]
        self.k1 = param[5]
        self.k2 = param[6]
        self.k3 = param[7]

        self.init_camera_rays()

    def interpolate_poses(self):
        # Instantiate a camera geometry object from the h5 pose data

        pos = self.hyp.pos_rgb # RGB
        quat = RotLib.from_quat(self.hyp.quat_rgb) # RGB
        time_pose = self.hyp.pose_time # RGB

        # Create a dictionary with keyword arguments
        kwargs_dict = {
            "is_interpolated": "True",
            "is_offset": "True"
        }

        self.RGBCamera = CameraGeometry(pos0=self.pos0, pos=pos, rot=quat, time=time_pose, **kwargs_dict)
    def select_timestamps(self, mode):
        if mode == 'georeference': ## Geometric upsampling of footprints. Only for georeferencing.
            if self.upsample_factor != 1:
                a1 = np.arange(len(self.hyp.dataCubeTimeStamps))  # Original timestep number
                f_time = interp1d(a1, self.hyp.dataCubeTimeStamps, fill_value='extrapolate')  # Interpolation function
                a2 = np.arange(len(self.hyp.dataCubeTimeStamps) * self.upsample_factor) / self.upsample_factor - (
                        self.upsample_factor - 1) / (
                             2 * self.upsample_factor)
                self.time_hsi = f_time(a2)
            else:
                self.time_hsi = self.hyp.dataCubeTimeStamps
        ##
        elif mode == 'calibrate': ## Selection of one scan per RGB or even less depending on a variety of factors.
            self.ind_hsi = np.zeros(len(range(0, self.hyp.n_imgs, self.step_imgs)), dtype='int32')
            self.n_valid_rgb_imgs = len(self.img_ind.reshape(-1))
            for i in range(0, self.n_valid_rgb_imgs, self.step_imgs):
                ind_rgb = self.rgb_ind[i]
                # We pair hsi indices with rgb indices
                self.ind_hsi[int(i / self.step_imgs)] = np.argmin(
                    np.abs(self.hyp.dataCubeTimeStamps - self.time_rgb[ind_rgb]))

            self.time_hsi = self.hyp.dataCubeTimeStamps[self.ind_hsi] # Only closest time stamps should be utilized.
        elif mode == 'geotag':
            self.time_hsi = self.hyp.dataCubeTimeStamps
        else:
            print('There is no such mode')

    def ray_trace(self):
        # This function is run per H5 file
        #
        self.minInd = 0
        self.maxInd = self.hyp.n_scanlines
        self.init_hsi_rays()

        self.HSICamera.intersectWithMesh(mesh = self.mesh, max_ray_length=self.max_ray_length)

        print('Writing Point Cloud')
        self.HSICamera.writeRGBPointCloud(config = self.config, hyp = self.hyp, transect_string=self.transect_string)

        print('Point cloud written')
        import visualize
        visualize.show_projected_hsi_points(HSICameraGeometry=self.HSICamera, config=self.config, transect_string=self.transect_string)


        # Geospatial processing steps
        gisHSI = GeoSpatialAbstractionHSI(point_cloud=self.HSICamera.projection, transect_string=self.transect_string, datacube_indices=np.arange(self.minInd, self.maxInd), config=self.config)

        gisHSI.transform_geocentric_to_projected()

        gisHSI.footprint_to_shape_file()

        gisHSI.resample_datacube(self.hyp, rgb_composite=True, minInd=self.minInd, maxInd=self.maxInd, extrapolate = True)

        gisHSI.compare_hsi_composite_with_rgb_mosaic()

        self.gisHSI = gisHSI

    def map_back(self):
        # Seems unnecessary
        w_datacube = self.hyp.n_pix
        self.gisHSI.map_pixels_back_to_datacube(w_datacube=w_datacube)





def objective_function(param, calObj):
    # Computes the directions in a local frame
    calObj.computeDirections(param)

    # reprojects to the same frame
    calObj.reprojectFeaturesHSI()

    # Determination of across-track reprojection error
    errorx = calObj.x_norm - calObj.HSIToFeaturesLocal[:, 0]

    # Determination of along-track reprojection error
    errory = -calObj.HSIToFeaturesLocal[:, 1]



    return np.concatenate((errorx.reshape(-1), errory.reshape(-1)))



def main(iniPath, mode, is_calibrated):
    config = configparser.ConfigParser()
    config.read(iniPath)

    # Paths to pose csv, 3D model ply file and the directory of H5 files
    path_pose = config['General']['posePath']
    path_mesh = config['General']['modelPath']
    dir_r = config['HDF']['h5Dir']
    # The path to the XML file
    hsi_cal_xml_b1 = config['Georeferencing']['HSICalibFileB1']
    hsi_cal_xml_b2 = config['Georeferencing']['HSICalibFileB2']
    if is_calibrated != True:
        hsi_cal_xml = config['Calibration']['hsiCalibFileInit']
    rc = RayCasting(config)

    # Load mesh
    rc.loadSeaBed(filenameSeaBed=path_mesh)


    # Only relevant for calibration part of things
    if is_calibrated != True:
        calObj1 = FeatureCalibrationObject(type='camera calibration', config=config)
        calObj1.load_cam_calibration(filenameCal=hsi_cal_xml, config=config)

        calObj2 = FeatureCalibrationObject(type='camera calibration', config=config)
        calObj2.load_cam_calibration(filenameCal=hsi_cal_xml, config=config)
    else:
        # Not the most elegant approach. Could encode it somehow
        calObj1 = FeatureCalibrationObject(type='camera calibration', config=config)
        calObj1.load_cam_calibration(filenameCal=hsi_cal_xml_b1, config=config)

        calObj2 = FeatureCalibrationObject(type='camera calibration', config=config)
        calObj2.load_cam_calibration(filenameCal=hsi_cal_xml_b2, config=config)



    count = 0
    if mode == 'georeference':
        print('Georeferencing Images')
        first = True
        for filename in sorted(os.listdir(dir_r)):
            if filename.endswith('h5') or filename.endswith('hdf'):
                # TODO: What is the point of count?
                if count > -1:
                    is_uhi = config['HDF']['is_uhi']

                    if is_uhi == 'True':
                        filename_splitted = filename.split('_')

                        transect_string = filename_splitted[2] + '_' + filename_splitted[3].split('.')[0]
                        rc.transect_string = transect_string
                        print(transect_string)
                        path_hdf = dir_r + filename
                        # Read h5 file and assign to raycaster
                        rc.hyp = Hyperspectral(path_hdf, config)
                        rc.hyp.digital_counts_2_radiance(config)
                        print(transect_string + ' With binning ' + str(rc.hyp.spatial_binning))
                    else:
                        path_hdf = dir_r + filename
                        rc.hyp = HyperspectralHI(path_hdf, config)


                    # Load the appropriate calibration file. This is old legacy code when data is recorded with different.
                    # binning settings
                    if is_calibrated == True:
                        if rc.hyp.spatial_binning == 1:
                            rc.load_cam_calibration(filenameCal=hsi_cal_xml_b1, config=config)
                        if rc.hyp.spatial_binning == 2:
                            rc.load_cam_calibration(filenameCal=hsi_cal_xml_b2, config=config)
                    else:
                        rc.load_cam_calibration(filenameCal=hsi_cal_xml, config= config)


                    # The interpolation of poses can be done prior to calibration. We want a position and orientation
                    rc.interpolate_poses()

                    rc.ray_trace()
                    rc.map_back()

                    if is_calibrated == True:
                        # Append key info to data
                        dir = 'georeference/'

                        # Add global points
                        points_global = rc.gisHSI.points_proj # Use projected system for global description
                        points_global_name = dir + 'points_global'
                        rc.hyp.add_dataset(data = points_global, name=points_global_name)

                        # Add local points
                        points_local = rc.HSICamera.camera_to_seabed_local  # Use projected system for global description
                        points_local_name = dir + 'points_local'
                        rc.hyp.add_dataset(data=points_local, name=points_local_name)

                        # Add camera position
                        position_hsi = rc.HSICamera.PositionHSI  # Use projected system for global description
                        position_hsi_name = dir + 'position_hsi'
                        rc.hyp.add_dataset(data=position_hsi, name=position_hsi_name)

                        # Add camera quaternion
                        quaternion_hsi = rc.HSICamera.RotationHSI.as_quat()  # Use projected system for global description
                        quaternion_hsi_name = dir + 'quaternion_hsi'
                        rc.hyp.add_dataset(data=quaternion_hsi, name=quaternion_hsi_name)

                        # Add normals
                        normals_local = rc.HSICamera.normalsLocal # Use projected system for global description
                        normals_local_name = dir + 'normals_local'
                        rc.hyp.add_dataset(data=normals_local, name=normals_local_name)

                    if rc.hyp.spatial_binning == 1:
                        calObj1.appendGeometry(hsiGis=rc.gisHSI, cameraGeometry=rc.HSICamera, binning = rc.hyp.spatial_binning)
                        calObjPath1 = config['Calibration']['hsicalibObjPathB1']
                        file_cal1 = open(calObjPath1, 'wb')
                        pickle.dump(calObj1, file_cal1)
                    if rc.hyp.spatial_binning == 2:
                        calObj2.appendGeometry(hsiGis=rc.gisHSI, cameraGeometry=rc.HSICamera, binning = rc.hyp.spatial_binning)
                        calObjPath2 = config['Calibration']['hsicalibObjPathB2']
                        file_cal2 = open(calObjPath2, 'wb')
                        pickle.dump(calObj2, file_cal2)

                count += 1
                if count == 1:
                    break






    elif mode == 'calibrate':
        config = configparser.ConfigParser()
        config.read(iniPath)

        calObjPath1 = config['Calibration']['hsicalibObjPathB1']

        file1 = open(calObjPath1, 'rb')
        ## dump information to that file
        calObj1 = pickle.load(file1)

        #calObjPath2 = config['Calibration']['hsicalibObjPathB2']

        #file2 = open(calObjPath2, 'rb')
        #### dump information to that file
        #calObj2 = pickle.load(file2)


        if is_calibrated == False:
            print('Calibrating Boresight and Camera Model')

            # Step 1:
            param0 = np.array([calObj1.rot_x, calObj1.rot_y, calObj1.rot_z, calObj1.v_c, calObj1.f, calObj1.k1, calObj1.k2, calObj1.k3])

            from scipy.optimize import least_squares
            res = least_squares(fun = objective_function, x0 = param0, args= (calObj1,) , x_scale='jac', method='lm')


            file_name_calib = config['Georeferencing']['hsicalibfileb1']


            param_calib = res.x

            if eval(config['General']['isFlippedHSI']):
                param_calib[2] -= np.pi

            param_calib[5] = 0
            param_calib[1] = 0
            print(param_calib)
            CalibHSI(file_name_cal=file_name_calib, config = config, mode = 'w', param = param_calib)

            # Step 1:
            param02 = np.array(
                [calObj2.rot_x, calObj2.rot_y, calObj2.rot_z, calObj2.v_c, calObj2.f, calObj2.k1, calObj2.k2, calObj2.k3])

            from scipy.optimize import least_squares
            res2 = least_squares(fun=objective_function, x0=param02, args=(calObj2,), x_scale='jac', method='lm')

            file_name_calib2 = config['Georeferencing']['hsicalibfileb2']

            param_calib2 = res2.x

            if eval(config['General']['isFlippedHSI']):
                param_calib2[2] -= np.pi

            param_calib2[5] = 0
            param_calib2[1] = 0
            print(param_calib2)
            CalibHSI(file_name_cal=file_name_calib2, config=config, mode='w', param=param_calib2)
        else:
            import matplotlib.pyplot as plt
            resolution = float(config['Georeferencing']['resolutionhyperspectralmosaic'])
            print(np.median(calObj1.diff[calObj1.diff < 5])*resolution*1000)

            try:
                diff_tot = np.concatenate((calObj1.diff, calObj2.diff))
            except AttributeError:
                diff_tot = calObj1.diff



            print(np.median(diff_tot[diff_tot < 10]) * resolution * 1000)

            print(calObj1.diff)
            print(len(diff_tot))

            plot_figures_paper(fignum=2, data = diff_tot)
            plt.legend()
            plt.show()

            resolution = float(config['Georeferencing']['resolutionhyperspectralmosaic'])
            print(np.median(calObj2.diff[calObj2.diff < 5]) * resolution * 1000)
            print(len(calObj2.diff[calObj2.diff < 5]))
            plt.hist(calObj2.diff[calObj2.diff < 10] * resolution * 1000, 25)
            plt.show()
















    elif mode == 'geotag':
        # We may geotag images all images within H5 file
        print('Geotagging RGB poses for hyperspectral and RGB time stamps')
        # We only need one ray caster object, but probably we can append hyperspectral
        files = os.listdir(dir_r)
        for filename in sorted(files):
            if filename.endswith('h5') or filename.endswith('hdf'):

                path_hdf = dir_r + filename
                # Read h5 file and asign to raycaster
                rc.hyp = Hyperspectral(path_hdf, config)

                #rc.hyp.DN2Radiance(config)
                filename_splitted = filename.split('_')
                transect_string = filename_splitted[2] + '_' + filename_splitted[3].split('.')[0]
                print(transect_string)
                rc.interpolate_poses()
                print()

if __name__ == '__main__':
    args = sys.argv[1:]
    iniPath = args[0]
    main(iniPath)
